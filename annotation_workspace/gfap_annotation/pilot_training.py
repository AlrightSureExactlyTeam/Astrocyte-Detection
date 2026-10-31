"""Reproducible manual/weak multi-task pilot utilities.

No code in this module runs training on import. The command-line wrapper performs
validation unless --start-training is explicitly supplied.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .pilot_data import ImageJWeakPatchDataset, ManualPatchDataset
from .unet_smoke import masked_manual_segmentation_loss
from .workflow import PROJECT_ROOT
from src.train import build_model, get_device, load_init_weights, save_checkpoint, set_seed


def validate_pilot_config(config: dict) -> dict:
    """Reject ambiguous supervision, hyperparameters, and slide-leaking splits."""
    for section in ("experiment", "data", "model", "loss", "training", "validation", "logging"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    data, training = config["data"], config["training"]
    seed = config["experiment"].get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("experiment.seed must be a nonnegative integer")
    for name in ("batch_size", "epochs", "num_workers"):
        value = training.get(name)
        minimum = 0 if name == "num_workers" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"training.{name} must be an integer >= {minimum}")
    learning_rate = training.get("learning_rate")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("training.learning_rate must be finite and positive")
    if training.get("optimizer") != "adam":
        raise ValueError("Only the existing Adam pilot optimizer is supported")
    if training.get("save_checkpoint") is not False:
        raise ValueError("Pilot preparation requires save_checkpoint: false")
    patch = data.get("patch_size")
    stride = data.get("patch_stride")
    if patch != 256 or stride != 256 or data.get("image_size") != [256, 256]:
        raise ValueError("This simple pilot uses non-overlapping original-resolution 256×256 patches")
    if data.get("resize_whole_tiles") is not False:
        raise ValueError("Whole-tile resizing must remain disabled")
    if data.get("input_channels") != 1 or data.get("channel") != 1:
        raise ValueError("The validated GFAP pilot requires one input channel, channel=1")
    if config["model"].get("output_channels") != 4:
        raise ValueError("The existing U-Net requires three semantic logits plus one auxiliary logit")
    manual = config["loss"].get("manual", {})
    if manual.get("type") != "masked_cross_entropy" or manual.get("ignore_index") != 255:
        raise ValueError("Manual supervision must use masked CE with ignore_index=255")
    weak = config["loss"].get("weak", {})
    if weak.get("enabled"):
        if weak.get("type") != "positive_region_bce_with_logits":
            raise ValueError("V2 weak supervision must be positive-region BCE")
        if weak.get("outside_candidate_mask") != "ignore_not_background":
            raise ValueError("ImageJ-negative pixels must remain ignored")
        weight = weak.get("weight")
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Weak loss weight must be finite and positive")
    if config["validation"].get("formal_evaluation_enabled") is not False:
        raise ValueError("Pseudo-label pilots cannot enable formal pseudo-ground-truth evaluation")
    if config["validation"].get("allow_same_slide_tile_split") is not False:
        raise ValueError("Same-slide random tile splitting is forbidden")
    train_groups = set(data.get("manual_train_slide_groups", [])) | set(data.get("imagej_train_slide_groups", []))
    val_groups = set(data.get("manual_validation_slide_groups", [])) | set(data.get("imagej_validation_slide_groups", []))
    overlap = train_groups & val_groups
    if overlap:
        raise ValueError(f"Slide leakage between train and validation: {sorted(overlap)}")
    if not data.get("manual_train_slide_groups"):
        raise ValueError("Both pilots require manual training samples")
    if weak.get("enabled") and not data.get("imagej_train_slide_groups"):
        raise ValueError("V2 requires ImageJ training samples")
    return config


def load_pilot_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_pilot_config(config)


def build_pilot_datasets(config: dict, workspace_root: str | Path = PROJECT_ROOT) -> dict:
    data = config["data"]
    manifest = PROJECT_ROOT / data["manifest"]
    kwargs = {
        "patch_size": data["patch_size"], "stride": data["patch_stride"],
        "channel": data["channel"],
    }
    datasets = {
        "manual_train": ManualPatchDataset(
            manifest, workspace_root, slide_groups=set(data["manual_train_slide_groups"]), **kwargs
        )
    }
    if config["loss"]["weak"].get("enabled"):
        datasets["weak_train"] = ImageJWeakPatchDataset(
            manifest, workspace_root, slide_groups=set(data["imagej_train_slide_groups"]), **kwargs
        )
        if data["imagej_validation_slide_groups"]:
            datasets["weak_validation"] = ImageJWeakPatchDataset(
                manifest, workspace_root, slide_groups=set(data["imagej_validation_slide_groups"]), **kwargs
            )
    return datasets


def weak_positive_region_bce(outputs: dict, targets: dict) -> dict:
    """BCE on ImageJ-positive candidate pixels only; unknown zeros are ignored."""
    logits = outputs["center_logits"]
    foreground = targets["weak_foreground"].float()
    valid = targets["weak_valid_mask"].bool()
    if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape != foreground.shape or valid.shape != foreground.shape:
        raise ValueError("Weak logits, targets, and valid mask must be [B,1,H,W]")
    if not torch.equal(valid, foreground == 1):
        raise ValueError("Only ImageJ-positive pixels may be weak-valid")
    if valid.any() and not torch.all(foreground[valid] == 1):
        raise ValueError("Weak supervision is positive-only")
    if not valid.any():
        return {"loss": logits.sum() * 0.0, "per_pixel_loss": torch.zeros_like(logits), "valid_mask": valid}
    per_pixel = F.binary_cross_entropy_with_logits(logits, foreground, reduction="none")
    per_pixel = per_pixel * valid
    return {"loss": per_pixel[valid].mean(), "per_pixel_loss": per_pixel, "valid_mask": valid}


class PilotLossLogger:
    """Append-only JSONL loss logging with one record per optimization/eval batch."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("x", encoding="utf-8")

    def log(self, **record) -> None:
        clean = {
            key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value
            for key, value in record.items()
        }
        self.handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def save_prediction_visualization(
    path: str | Path,
    image: torch.Tensor,
    outputs: dict,
    targets: dict,
    sample_id: str,
    weak: bool = False,
) -> None:
    """Save input, supervision, prediction, and overlay without altering source data."""
    gray = image.detach().cpu()[0].numpy()
    gray = np.clip(gray, 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    truth = np.zeros_like(rgb)
    if weak:
        prediction_probability = torch.sigmoid(outputs["center_logits"].detach().cpu())[0, 0].numpy()
        prediction = np.repeat(prediction_probability[..., None], 3, axis=2)
        truth[targets["weak_valid_mask"].detach().cpu()[0].numpy()] = (0, 1, 0)
        foreground = prediction_probability >= 0.5
        color = np.zeros_like(rgb)
        color[foreground] = (0, 1, 0)
    else:
        labels = targets["semantic_mask"].detach().cpu().numpy()
        prediction_labels = outputs["semantic_logits"].detach().cpu()[0].argmax(0).numpy()
        prediction = np.zeros_like(rgb)
        for target, color in ((1, (0, 1, 0)), (2, (1, 0, 0))):
            truth[labels == target] = color
            prediction[prediction_labels == target] = color
        foreground = prediction_labels > 0
        color = prediction
    overlay = rgb.copy()
    overlay[foreground] = 0.45 * rgb[foreground] + 0.55 * color[foreground]
    panels = [np.clip(panel * 255, 0, 255).astype(np.uint8) for panel in (rgb, truth, prediction, overlay)]
    height, width = gray.shape
    canvas = np.zeros((height + 42, width * 4, 3), dtype=np.uint8)
    canvas[42:] = np.concatenate(panels, axis=1)
    cv2.putText(canvas, sample_id[:90], (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for index, heading in enumerate(("GFAP input", "Weak positives" if weak else "Manual labels", "Prediction", "Overlay")):
        cv2.putText(canvas, heading, (8 + width * index, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
    success, encoded = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not success:
        raise OSError("Could not encode prediction visualization")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded.tobytes())


def train_pilot(config: dict, save_final_checkpoint: bool = False) -> Path:
    """Explicit pilot runner with finite-loss checks and isolated run outputs."""
    validate_pilot_config(config)
    set_seed(config["experiment"]["seed"])
    device = get_device(config["training"]["device"])
    datasets = build_pilot_datasets(config)
    training = config["training"]
    model = build_model(config)
    if config["model"].get("init_weights"):
        load_init_weights(model, config["model"]["init_weights"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%f_") + uuid.uuid4().hex[:6]
    run_dir = PROJECT_ROOT / training["output_root"] / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config_snapshot.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    logger = PilotLossLogger(run_dir / config["logging"]["loss_log"])
    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    loader_kwargs = {
        "batch_size": training["batch_size"], "num_workers": training["num_workers"],
        "generator": generator,
    }
    manual_loader = DataLoader(datasets["manual_train"], shuffle=True, **loader_kwargs)
    weak_loader = DataLoader(datasets["weak_train"], shuffle=True, **loader_kwargs) if "weak_train" in datasets else None
    weak_weight = config["loss"]["weak"].get("weight", 0.0)
    history = {}
    final_losses = {}
    optimizer_steps = 0
    visualization_paths = []

    def record(epoch: int, phase: str, loss_name: str, loss: torch.Tensor) -> None:
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite {phase} loss at epoch {epoch}")
        logger.log(epoch=epoch, phase=phase, **{loss_name: value})
        history.setdefault((epoch, phase), []).append(value)
        final_losses[phase] = value

    def backward_checked(loss: torch.Tensor, epoch: int, phase: str) -> None:
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Non-finite {phase} loss before backward at epoch {epoch}")
        loss.backward()
        invalid = [
            name for name, parameter in model.named_parameters()
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
        ]
        if invalid:
            raise FloatingPointError(f"Non-finite gradients in {phase}: {invalid[:5]}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    try:
        for epoch in range(1, training["epochs"] + 1):
            model.train()
            for images, targets, _ in manual_loader:
                images = images.to(device)
                targets = {key: value.to(device) for key, value in targets.items()}
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = masked_manual_segmentation_loss(outputs, targets)["loss"]
                backward_checked(loss, epoch, "manual_train")
                optimizer.step()
                optimizer_steps += 1
                record(epoch, "manual_train", "manual_ce", loss)
            if weak_loader is not None:
                for images, targets, _ in weak_loader:
                    images = images.to(device)
                    targets = {key: value.to(device) for key, value in targets.items()}
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(images)
                    loss = weak_positive_region_bce(outputs, targets)["loss"] * weak_weight
                    backward_checked(loss, epoch, "weak_train")
                    optimizer.step()
                    optimizer_steps += 1
                    record(epoch, "weak_train", "weak_positive_bce", loss)
            if "weak_validation" in datasets:
                model.eval()
                val_loader = DataLoader(datasets["weak_validation"], batch_size=training["batch_size"], shuffle=False)
                with torch.no_grad():
                    for images, targets, _ in val_loader:
                        images = images.to(device)
                        targets = {key: value.to(device) for key, value in targets.items()}
                        loss = weak_positive_region_bce(model(images), targets)["loss"]
                        record(epoch, "weak_validation_monitor_only", "weak_positive_bce", loss)
        training_seconds = time.perf_counter() - training_started
        if config["logging"]["sample_visualizations"]:
            model.eval()
            for name, weak in (("manual_train", False), ("weak_train", True), ("weak_validation", True)):
                if name not in datasets:
                    continue
                for index in range(min(config["logging"]["visualization_samples"], len(datasets[name]))):
                    image, targets, metadata = datasets[name][index]
                    with torch.no_grad():
                        outputs = model(image.unsqueeze(0).to(device))
                    outputs = {key: value.cpu() for key, value in outputs.items()}
                    path = run_dir / "visualizations" / f"{name}_{index:02d}.png"
                    save_prediction_visualization(path, image, outputs, targets, metadata["sample_id"], weak=weak)
                    visualization_paths.append(str(path))
    finally:
        logger.close()
    checkpoint_path = None
    if save_final_checkpoint:
        checkpoint_path = run_dir / "checkpoints" / "final.pt"
        if checkpoint_path.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_path}")
        metrics = {"manual_train_loss": final_losses["manual_train"]}
        if "weak_train" in final_losses:
            metrics["weak_train_loss"] = final_losses["weak_train"]
        save_checkpoint(model, optimizer, training["epochs"], metrics, config, checkpoint_path)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"]["name"],
        "status": "completed",
        "technical_pilot_not_formal_evaluation": True,
        "hyperparameters": {
            "epochs": training["epochs"], "batch_size": training["batch_size"],
            "learning_rate": training["learning_rate"], "seed": config["experiment"]["seed"],
        },
        "dataset_patches": {name: len(dataset) for name, dataset in datasets.items()},
        "epoch_history": [
            {
                "epoch": epoch, "phase": phase, "batches": len(values),
                "mean_loss": sum(values) / len(values), "final_batch_loss": values[-1],
            }
            for (epoch, phase), values in sorted(history.items())
        ],
        "final_manual_train_loss": final_losses["manual_train"],
        "final_weak_train_loss": final_losses.get("weak_train"),
        "final_weak_validation_monitor_loss": final_losses.get("weak_validation_monitor_only"),
        "weak_train_loss_weight": weak_weight,
        "all_losses_finite": True,
        "all_gradients_finite": True,
        "optimizer_steps": optimizer_steps,
        "device": str(device),
        "peak_cuda_memory_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 3) if device.type == "cuda" else None,
        "peak_cuda_memory_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 3) if device.type == "cuda" else None,
        "total_training_seconds": round(training_seconds, 3),
        "total_run_seconds": round(time.perf_counter() - training_started, 3),
        "visualizations": visualization_paths,
        "checkpoint_saved": checkpoint_path is not None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "limitations": config["limitations"],
    }
    with (run_dir / "training_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return run_dir

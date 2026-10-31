"""Losses, supported-region evaluation, and explicit short runs for V3.

No training occurs on import. The command wrapper requires --start-training for
both the few-cell overfit and pilot modes. Unknown pixels never contribute as
negative/Background examples.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, default_collate

from .v3_data import MORPHOLOGY_IGNORE, SomaAnchoredCropDataset
from .v3_model import create_v3_model
from .workflow import PROJECT_ROOT
from src.train import get_device, set_seed

V3_CHECKPOINT_SCHEMA = "astrocyte_instance_v3_soma_anchored_v1"


def validate_v3_config(config: dict) -> dict:
    for section in ("experiment", "data", "model", "loss", "training", "runs", "evaluation"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing V3 configuration section: {section}")
    seed = config["experiment"].get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("experiment.seed must be a nonnegative integer")
    data = config["data"]
    if data.get("patch_size") != 256 or data.get("channel") != 1:
        raise ValueError("V3 pilot requires original-resolution 256x256 crops and GFAP channel=1")
    for key in ("use_source_semantic_background", "unlabeled_pixels_are_negative", "random_patch_split"):
        if data.get(key) is not False:
            raise ValueError(f"data.{key} must remain false")
    model = config["model"]
    if model.get("architecture") != "small_unet_shared_encoder":
        raise ValueError("Unsupported V3 architecture")
    if model.get("morphology_classes") != ["soma", "process"] or model.get("background_logit") is not False:
        raise ValueError("V3 morphology head must contain only Soma and Process")
    if not isinstance(model.get("embedding_dim"), int) or model["embedding_dim"] < 2:
        raise ValueError("model.embedding_dim must be >= 2")
    loss = config["loss"]
    for name in ("morphology_weight", "attraction_weight", "separation_weight"):
        value = loss.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"loss.{name} must be finite and nonnegative")
    margin = loss.get("cosine_margin")
    if not isinstance(margin, (int, float)) or not 0 < margin <= 2:
        raise ValueError("loss.cosine_margin must be in (0, 2]")
    training = config["training"]
    for name in ("batch_size", "num_workers"):
        value = training.get(name)
        minimum = 0 if name == "num_workers" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"training.{name} must be an integer >= {minimum}")
    lr = training.get("learning_rate")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("training.learning_rate must be finite and positive")
    if training.get("output_root") != "results/gfap_v3":
        raise ValueError("V3 outputs must remain isolated under results/gfap_v3")
    for name, run in config["runs"].items():
        if name == "smoke":
            continue
        if not isinstance(run.get("epochs"), int) or run["epochs"] <= 0 or run["epochs"] > run.get("maximum_allowed_epochs", run["epochs"]):
            raise ValueError(f"Invalid or excessive epoch count for {name}")
    evaluation = config["evaluation"]
    if evaluation.get("gt_soma_anchored_diagnostic") is not True or evaluation.get("free_instance_detection") is not False:
        raise ValueError("V3 evaluation must remain GT-Soma-anchored, not free instance detection")
    if evaluation.get("full_image_background_metrics") is not False or evaluation.get("counting_accuracy") is not False:
        raise ValueError("Unsupported full-image Background/counting metrics must remain disabled")
    return config


def load_v3_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return validate_v3_config(yaml.safe_load(handle))


def _zero(*tensors: torch.Tensor) -> torch.Tensor:
    return sum(tensor.sum() for tensor in tensors) * 0.0


def compute_v3_loss(outputs: dict, targets: dict, loss_config: dict) -> dict[str, torch.Tensor]:
    logits = outputs["morphology_logits"].float()
    embedding = outputs["instance_embedding"].float()
    labels = targets["morphology_mask"].long()
    valid = targets["morphology_valid_mask"].bool()
    association = targets["association_mask"].long()
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("morphology_logits must be [B,2,H,W]")
    if embedding.ndim != 4 or embedding.shape[0] != logits.shape[0] or embedding.shape[2:] != logits.shape[2:]:
        raise ValueError("instance_embedding must be [B,D,H,W] aligned with morphology logits")
    if labels.shape != logits.shape[:1] + logits.shape[2:] or valid.shape != labels.shape or association.shape != labels.shape:
        raise ValueError("V3 target shapes must be [B,H,W]")
    if not torch.equal(valid, labels != MORPHOLOGY_IGNORE):
        raise ValueError("Morphology valid mask must exactly equal labels != 255")
    if valid.any() and ((labels[valid] < 0) | (labels[valid] > 1)).any():
        raise ValueError("Only internal Soma=0 and Process=1 labels are valid")
    if (association < 0).any() or ((association > 0) & ~valid).any():
        raise ValueError("Association IDs must be positive and confined to confirmed morphology")
    if not valid.any():
        raise ValueError("V3 crop has no positive morphology supervision")
    morphology = F.cross_entropy(logits, labels, ignore_index=MORPHOLOGY_IGNORE, reduction="mean")
    normalized = F.normalize(embedding, p=2, dim=1, eps=1e-6)
    zero = _zero(logits, embedding)
    attractions = []
    anchors = []
    for batch_index in range(logits.shape[0]):
        for cell_id in torch.unique(association[batch_index]):
            cell_id = int(cell_id.item())
            if cell_id == 0:
                continue
            cell = association[batch_index] == cell_id
            soma = cell & (labels[batch_index] == 0)
            process = cell & (labels[batch_index] == 1)
            if not soma.any() or not process.any():
                raise ValueError(f"Association cell {cell_id} lacks an in-crop Soma or Process target")
            soma_vectors = normalized[batch_index][:, soma]
            process_vectors = normalized[batch_index][:, process]
            anchor = F.normalize(soma_vectors.mean(dim=1), p=2, dim=0, eps=1e-6)
            attractions.append((1 - (soma_vectors * anchor[:, None]).sum(dim=0)).mean())
            attractions.append((1 - (process_vectors * anchor[:, None]).sum(dim=0)).mean())
            anchors.append(anchor)
    attraction = torch.stack(attractions).mean() if attractions else zero
    separations = []
    for first in range(len(anchors)):
        for second in range(first + 1, len(anchors)):
            cosine = torch.dot(anchors[first], anchors[second])
            separations.append(F.relu(cosine - (1.0 - float(loss_config["cosine_margin"]))))
    separation = torch.stack(separations).mean() if separations else zero
    total = (
        float(loss_config["morphology_weight"]) * morphology
        + float(loss_config["attraction_weight"]) * attraction
        + float(loss_config["separation_weight"]) * separation
    )
    return {
        "total": total, "morphology": morphology, "attraction": attraction,
        "separation": separation, "association_cells": torch.tensor(float(len(anchors)), device=logits.device),
        "separation_pairs": torch.tensor(float(len(separations)), device=logits.device),
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def association_diagnostic(embedding: torch.Tensor, labels: torch.Tensor, association: torch.Tensor) -> tuple[dict, np.ndarray]:
    """Assign GT-supported pixels to GT-Soma embedding anchors; not detection."""
    if embedding.ndim != 3 or labels.ndim != 2 or association.ndim != 2:
        raise ValueError("association_diagnostic expects embedding [D,H,W] and labels/association [H,W]")
    if embedding.shape[1:] != labels.shape or association.shape != labels.shape:
        raise ValueError("association_diagnostic inputs must share spatial dimensions")
    embedding = F.normalize(embedding.detach().float().cpu(), p=2, dim=0, eps=1e-6)
    labels = labels.detach().cpu()
    association = association.detach().cpu()
    anchor_map: dict[int, torch.Tensor] = {}
    for cell_id in torch.unique(association):
        cell_id = int(cell_id.item())
        if cell_id == 0:
            continue
        soma = (association == cell_id) & (labels == 0)
        process = (association == cell_id) & (labels == 1)
        if soma.any() and process.any():
            anchor_map[cell_id] = F.normalize(embedding[:, soma].mean(dim=1), p=2, dim=0, eps=1e-6)
    predicted = np.zeros(labels.shape, dtype=np.int32)
    correct = total = correct_competitive = total_competitive = 0
    within_cosines, between_cosines = [], []
    ids = sorted(anchor_map)
    if ids:
        anchor_tensor = torch.stack([anchor_map[cell_id] for cell_id in ids], dim=1)  # D,N
        for cell_id in ids:
            process = (association == cell_id) & (labels == 1)
            vectors = embedding[:, process]
            similarities = anchor_tensor.T @ vectors  # N,pixels
            assignments = similarities.argmax(dim=0)
            predicted[process] = np.asarray(ids, dtype=np.int32)[assignments.numpy()]
            truth_index = ids.index(cell_id)
            correct += int((assignments == truth_index).sum())
            total += int(assignments.numel())
            within_cosines.append(similarities[truth_index].mean())
            others = [index for index in range(len(ids)) if index != truth_index]
            if others:
                between_cosines.append(similarities[others].mean())
            if len(ids) >= 2:
                correct_competitive += int((assignments == truth_index).sum())
                total_competitive += int(assignments.numel())
    metrics = {
        "association_pixels": total,
        "association_anchor_count": len(ids),
        "gt_soma_anchor_process_accuracy": _safe_ratio(correct, total),
        "gt_soma_anchor_process_accuracy_multianchor_crops": _safe_ratio(correct_competitive, total_competitive),
        "mean_within_process_anchor_cosine": float(torch.stack(within_cosines).mean()) if within_cosines else None,
        "mean_between_process_anchor_cosine": float(torch.stack(between_cosines).mean()) if between_cosines else None,
    }
    return metrics, predicted


def supported_region_metrics(outputs: dict, targets: dict) -> dict:
    logits = outputs["morphology_logits"].detach().float().cpu()
    prediction = logits.argmax(dim=1)
    labels = targets["morphology_mask"].detach().cpu()
    valid = targets["morphology_valid_mask"].detach().cpu().bool()
    result = {"valid_morphology_pixels": int(valid.sum())}
    for class_index, name in ((0, "soma"), (1, "process")):
        truth = valid & (labels == class_index)
        pred = valid & (prediction == class_index)
        intersection = int((truth & pred).sum())
        result[f"{name}_dice_supervised_support"] = _safe_ratio(2 * intersection, int(truth.sum() + pred.sum()))
        result[f"{name}_recall_confirmed_pixels"] = _safe_ratio(intersection, int(truth.sum()))
        result[f"predicted_{name}_fraction_of_supervised_support"] = _safe_ratio(int(pred.sum()), int(valid.sum()))
    association_results = []
    for batch_index in range(labels.shape[0]):
        metrics, _ = association_diagnostic(
            outputs["instance_embedding"].detach().cpu()[batch_index],
            labels[batch_index], targets["association_mask"].detach().cpu()[batch_index],
        )
        association_results.append(metrics)
    result["gt_soma_anchor_process_accuracy"] = _mean_optional(
        [item["gt_soma_anchor_process_accuracy"] for item in association_results]
    )
    result["gt_soma_anchor_process_accuracy_multianchor_crops"] = _mean_optional(
        [item["gt_soma_anchor_process_accuracy_multianchor_crops"] for item in association_results]
    )
    result["association_anchor_count"] = int(sum(item["association_anchor_count"] for item in association_results))
    return result


def _mean_optional(values: list) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def save_v3_checkpoint(model, optimizer, epoch: int, config: dict, metrics: dict, path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite V3 checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_schema": V3_CHECKPOINT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": int(epoch), "config": config, "metrics": metrics,
        "output_heads": {"morphology": ["soma", "process"], "embedding_dim": model.embedding_dim, "background": False},
        "evaluation_scope": "partial_supervision_gt_soma_anchored_not_free_instance_detection",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_v3_checkpoint(model, path: str | Path, device: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_schema") != V3_CHECKPOINT_SCHEMA:
        raise ValueError("Incompatible V3 checkpoint schema")
    if checkpoint.get("output_heads", {}).get("background") is not False:
        raise ValueError("V3 checkpoint unexpectedly declares Background supervision")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.handle = path.open("x", encoding="utf-8")

    def log(self, **values) -> None:
        clean = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu())
            clean[key] = value
        self.handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _targets_to_device(targets: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in targets.items()}


def _check_finite_losses(losses: dict, phase: str) -> None:
    for name in ("total", "morphology", "attraction", "separation"):
        if not bool(torch.isfinite(losses[name]).item()):
            raise FloatingPointError(f"Non-finite {phase} {name} loss")


def _check_gradients(model) -> None:
    invalid = [name for name, parameter in model.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
    if invalid:
        raise FloatingPointError(f"Non-finite V3 gradients: {invalid[:5]}")


def _clean_metrics(losses: dict, diagnostics: dict | None = None) -> dict:
    result = {name: float(losses[name].detach().cpu()) for name in ("total", "morphology", "attraction", "separation")}
    result["association_cells"] = int(losses["association_cells"].item())
    result["separation_pairs"] = int(losses["separation_pairs"].item())
    if diagnostics:
        result.update(diagnostics)
    return result


def _mean_records(records: list[dict]) -> dict:
    result = {}
    for key in records[0]:
        values = [row[key] for row in records if row.get(key) is not None]
        if values and isinstance(values[0], (int, float)):
            result[key] = float(np.mean(values))
        else:
            result[key] = None
    return result


def _write_rgb_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    if not success:
        raise OSError(f"Could not encode PNG: {path}")
    with path.open("xb") as handle:
        handle.write(encoded.tobytes())


def save_loss_curves(history: list[dict], path: Path) -> None:
    phases = sorted({row["phase"] for row in history})
    phase_colors = [(40, 100, 220), (220, 80, 45), (30, 150, 80)]
    canvas = np.full((790, 1040, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "V3 short-run losses (technical pilot, not formal evaluation)", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
    for panel_index, loss_name in enumerate(("total", "morphology", "attraction", "separation")):
        ox = 20 + (panel_index % 2) * 510
        oy = 50 + (panel_index // 2) * 365
        width, height = 490, 340
        left, top = ox + 48, oy + 34
        right, bottom = ox + width - 12, oy + height - 38
        cv2.putText(canvas, loss_name, (ox + 8, oy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (left, top), (right, bottom), (80, 80, 80), 1)
        relevant = [row for row in history if row.get(loss_name) is not None]
        values = [float(row[loss_name]) for row in relevant]
        epochs = [int(row["epoch"]) for row in relevant]
        minimum, maximum = min(values), max(values)
        span = max(maximum - minimum, 1e-8)
        epoch_span = max(max(epochs) - min(epochs), 1)
        for tick in range(5):
            y = int(top + (bottom - top) * tick / 4)
            value = maximum - span * tick / 4
            cv2.line(canvas, (left, y), (right, y), (225, 225, 225), 1)
            cv2.putText(canvas, f"{value:.3g}", (ox, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (70, 70, 70), 1, cv2.LINE_AA)
        for phase_index, phase in enumerate(phases):
            rows = [row for row in relevant if row["phase"] == phase]
            points = [(
                int(left + (int(row["epoch"]) - min(epochs)) / epoch_span * (right - left)),
                int(top + (maximum - float(row[loss_name])) / span * (bottom - top)),
            ) for row in rows]
            color = phase_colors[phase_index % len(phase_colors)]
            if len(points) > 1:
                cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(canvas, point, 3, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, phase, (left + phase_index * 145, oy + height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    _write_rgb_png(path, canvas)


def _instance_rgb(ids: np.ndarray) -> np.ndarray:
    palette = np.asarray([
        (35, 190, 80), (225, 65, 155), (55, 125, 235), (240, 165, 35),
        (145, 75, 210), (35, 190, 190), (220, 75, 65), (120, 200, 45),
        (230, 105, 200), (70, 85, 180), (180, 130, 55), (75, 175, 130),
    ], dtype=np.uint8)
    rgb = np.full((*ids.shape, 3), 184, dtype=np.uint8)
    for cell_id in np.unique(ids):
        if cell_id > 0:
            rgb[ids == cell_id] = palette[(int(cell_id) - 1) % len(palette)]
    return rgb


def save_v3_visualization(path: Path, image: torch.Tensor, outputs: dict, targets: dict, metadata: dict) -> None:
    gray = image.detach().float().cpu()[0].numpy()
    labels = targets["morphology_mask"].detach().cpu().numpy()
    association = targets["association_mask"].detach().cpu().numpy()
    uncertain = targets["uncertain_mask"].detach().cpu().numpy()
    conflict = targets["conflict_mask"].detach().cpu().numpy()
    probabilities = torch.softmax(outputs["morphology_logits"].detach().float().cpu(), dim=1)[0].numpy()
    association_metrics, predicted_association = association_diagnostic(
        outputs["instance_embedding"].detach().cpu()[0], targets["morphology_mask"].detach().cpu(),
        targets["association_mask"].detach().cpu(),
    )
    gt_morphology = np.full((*labels.shape, 3), 184, dtype=np.uint8)
    gt_morphology[labels == 0] = (25, 225, 65)
    gt_morphology[labels == 1] = (230, 40, 165)
    support = gt_morphology.copy()
    support[uncertain] = (255, 190, 15)
    support[conflict] = (25, 25, 25)
    original = np.repeat(np.clip(gray * 255, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    soma_probability = cv2.cvtColor(cv2.applyColorMap(np.clip(probabilities[0] * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    process_probability = cv2.cvtColor(cv2.applyColorMap(np.clip(probabilities[1] * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
    accuracy = association_metrics["gt_soma_anchor_process_accuracy"]
    info = np.full((256, 256, 3), 255, dtype=np.uint8)
    lines = [
        "V3 diagnostic scope", f"Center GT cell: {metadata['center_cell_id']}",
        f"GT anchors: {association_metrics['association_anchor_count']}",
        f"Process assignment: {accuracy:.3f}" if accuracy is not None else "Process assignment: n/a",
        "", "No Background supervision", "No free instance detection", "No validated cell count",
        "", "Assignment uses GT support", "and GT Soma anchors.", "Not a full-image prediction.",
    ]
    for index, line in enumerate(lines):
        cv2.putText(info, line, (8, 25 + index * 19), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (25, 25, 25), 1, cv2.LINE_AA)
    panels = [
        (original, "Original GFAP crop"),
        (gt_morphology, "GT morphology; gray = Unknown, not BG"),
        (_instance_rgb(association), "GT complete-cell instance IDs"),
        (soma_probability, "Predicted Soma probability - all pixels"),
        (process_probability, "Predicted Process probability - all pixels"),
        (_instance_rgb(predicted_association), "Pred assignment: GT support + GT Soma anchors"),
        (support, "Support: yellow Uncertain / black conflict"),
        (info, "Evaluation limitations"),
    ]
    title_height, header = 48, 36
    canvas = np.full((header + 2 * (256 + title_height), 1024, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "V3 prediction versus manual GT - partial-supervision diagnostic", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (15, 15, 15), 1, cv2.LINE_AA)
    for index, (panel, title) in enumerate(panels):
        x = (index % 4) * 256
        y = header + (index // 4) * (256 + title_height)
        cv2.putText(canvas, title, (x + 5, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (25, 25, 25), 1, cv2.LINE_AA)
        canvas[y + title_height:y + title_height + 256, x:x + 256] = panel
    _write_rgb_png(path, canvas)


def _build_dataset(config: dict, split: str, mode: str, augment: bool, max_crops: int | None = None):
    data = config["data"]
    return SomaAnchoredCropDataset(
        PROJECT_ROOT / data["manifest"], split=split, patch_size=data["patch_size"], channel=data["channel"],
        seed=config["experiment"]["seed"], augment=augment, jitter=data.get("jitter", 32), max_crops=max_crops,
    )


def _v3_collate(batch):
    """Collate tensors normally but preserve variable per-crop metadata as a list."""
    images, targets, metadata = zip(*batch)
    return default_collate(images), default_collate(targets), list(metadata)


def run_v3(config: dict, mode: str) -> Path:
    validate_v3_config(config)
    if mode not in {"smoke", "overfit", "pilot"}:
        raise ValueError(f"Unsupported run mode: {mode}")
    set_seed(config["experiment"]["seed"])
    device = get_device(config["training"]["device"])
    training = config["training"]
    run_config = config["runs"][mode]
    amp_enabled = bool(training.get("mixed_precision", False) and device.type == "cuda")
    stamp = datetime.now(timezone.utc).strftime(f"{mode}_%Y%m%dT%H%M%S_%f_") + uuid.uuid4().hex[:6]
    run_dir = PROJECT_ROOT / training["output_root"] / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config_snapshot.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    shutil.copyfile(PROJECT_ROOT / config["data"]["manifest"], run_dir / "dataset_manifest.csv")
    shutil.copyfile(PROJECT_ROOT / config["data"]["audit"], run_dir / "dataset_audit.json")

    max_crops = run_config.get("max_crops")
    train_dataset = _build_dataset(config, "train", mode, augment=mode == "pilot", max_crops=max_crops)
    validation_dataset = _build_dataset(config, "validation", mode, augment=False) if mode == "pilot" else None
    if train_dataset.source_images & (validation_dataset.source_images if validation_dataset else set()):
        raise ValueError("Source-image leakage detected")
    model = create_v3_model(
        in_channels=1, base_channels=config["model"]["base_channels"], embedding_dim=config["model"]["embedding_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    train_loader = DataLoader(
        train_dataset, batch_size=training["batch_size"], shuffle=mode != "smoke", num_workers=training["num_workers"],
        pin_memory=device.type == "cuda", generator=generator, collate_fn=_v3_collate,
    )
    val_loader = DataLoader(
        validation_dataset, batch_size=training["batch_size"], shuffle=False, num_workers=training["num_workers"],
        pin_memory=device.type == "cuda", collate_fn=_v3_collate,
    ) if validation_dataset else None
    logger = JsonlLogger(run_dir / "batch_log.jsonl")
    history: list[dict] = []
    visualization_paths: list[str] = []
    optimizer_steps = 0
    first_train_loss = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    training_seconds = 0.0
    checkpoint_path = None
    final_metrics = {}
    try:
        if mode == "smoke":
            model.train()
            images, targets, _ = next(iter(train_loader))
            images = images.to(device)
            targets = _targets_to_device(targets, device)
            before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(images)
            if outputs["morphology_logits"].shape != (images.shape[0], 2, *images.shape[2:]):
                raise ValueError("V3 morphology output shape mismatch")
            if outputs["instance_embedding"].shape != (images.shape[0], model.embedding_dim, *images.shape[2:]):
                raise ValueError("V3 embedding output shape mismatch")
            losses = compute_v3_loss(outputs, targets, config["loss"])
            _check_finite_losses(losses, "smoke")
            losses["total"].backward()
            _check_gradients(model)
            for name, parameter in model.named_parameters():
                if not torch.equal(before[name], parameter):
                    raise RuntimeError(f"Smoke backward unexpectedly updated parameter: {name}")
            diagnostics = supported_region_metrics(outputs, targets)
            final_metrics = _clean_metrics(losses, diagnostics)
            logger.log(epoch=0, phase="smoke_backward_no_optimizer_step", **final_metrics)
            checkpoint_path = save_v3_checkpoint(model, optimizer, 0, config, final_metrics, run_dir / "checkpoints/smoke.pt")
            restored = create_v3_model(1, config["model"]["base_channels"], config["model"]["embedding_dim"]).to(device)
            load_v3_checkpoint(restored, checkpoint_path, device)
            for name, parameter in restored.named_parameters():
                if not torch.equal(parameter, dict(model.named_parameters())[name]):
                    raise RuntimeError(f"Checkpoint reload mismatch: {name}")
            history.append({"epoch": 0, "phase": "smoke", **final_metrics})
        else:
            epochs = run_config["epochs"]
            scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
            training_started = time.perf_counter()
            for epoch in range(1, epochs + 1):
                model.train()
                train_dataset.set_epoch(epoch)
                train_records = []
                for images, targets, _ in train_loader:
                    images = images.to(device, non_blocking=True)
                    targets = _targets_to_device(targets, device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                        outputs = model(images)
                    losses = compute_v3_loss(outputs, targets, config["loss"])
                    _check_finite_losses(losses, "train")
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    _check_gradients(model)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training.get("gradient_clip_norm", 1.0))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_steps += 1
                    with torch.no_grad():
                        diagnostics = supported_region_metrics(outputs, targets)
                    record = _clean_metrics(losses, diagnostics)
                    if first_train_loss is None:
                        first_train_loss = record["total"]
                    train_records.append(record)
                    logger.log(epoch=epoch, phase="overfit_train" if mode == "overfit" else "train", **record)
                train_epoch = _mean_records(train_records)
                history.append({"epoch": epoch, "phase": "overfit_train" if mode == "overfit" else "train", **train_epoch})
                if val_loader is not None:
                    model.eval()
                    val_records = []
                    with torch.no_grad():
                        for images, targets, _ in val_loader:
                            images = images.to(device, non_blocking=True)
                            targets = _targets_to_device(targets, device)
                            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                                outputs = model(images)
                            losses = compute_v3_loss(outputs, targets, config["loss"])
                            _check_finite_losses(losses, "validation")
                            record = _clean_metrics(losses, supported_region_metrics(outputs, targets))
                            val_records.append(record)
                            logger.log(epoch=epoch, phase="validation", **record)
                    history.append({"epoch": epoch, "phase": "validation", **_mean_records(val_records)})
                print(json.dumps({"mode": mode, "epoch": epoch, **train_epoch}, ensure_ascii=False), flush=True)
            training_seconds = time.perf_counter() - training_started
            final_metrics = history[-1]
            checkpoint_path = save_v3_checkpoint(
                model, optimizer, epochs, config, final_metrics, run_dir / "checkpoints/final.pt",
            )
            model.eval()
            visualization_dataset = validation_dataset if validation_dataset is not None else _build_dataset(config, "train", mode, False, max_crops)
            for index in range(min(run_config.get("visualization_samples", 3), len(visualization_dataset))):
                image, targets, metadata = visualization_dataset[index]
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                        outputs = model(image.unsqueeze(0).to(device))
                path = run_dir / "visualizations" / f"{'validation' if validation_dataset else 'overfit_train'}_{index:02d}.png"
                path.parent.mkdir(exist_ok=True)
                save_v3_visualization(path, image, outputs, targets, metadata)
                visualization_paths.append(str(path))
    finally:
        logger.close()
    if history and mode != "smoke":
        with (run_dir / "epoch_history.csv").open("x", newline="", encoding="utf-8") as handle:
            fields = sorted({key for row in history for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        save_loss_curves(history, run_dir / "loss_curves.png")
    with (run_dir / "epoch_history.json").open("x", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    predicted_soma = final_metrics.get("predicted_soma_fraction_of_supervised_support")
    predicted_process = final_metrics.get("predicted_process_fraction_of_supervised_support")
    collapse_warning = bool(
        predicted_soma is not None and predicted_process is not None
        and max(predicted_soma, predicted_process) >= 0.999
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
        "status": "completed", "technical_pilot_not_formal_evaluation": True,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "mixed_precision_enabled": amp_enabled, "all_losses_finite": True, "all_gradients_finite": True,
        "optimizer_steps": optimizer_steps, "no_optimizer_step_smoke": mode == "smoke" and optimizer_steps == 0,
        "train_crops": len(train_dataset), "validation_crops": len(validation_dataset) if validation_dataset else 0,
        "train_source_images": len(train_dataset.crop_source_images),
        "validation_source_images": len(validation_dataset.crop_source_images) if validation_dataset else 0,
        "first_train_loss": first_train_loss, "final_metrics": final_metrics,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_reload_verified": mode == "smoke",
        "visualizations": visualization_paths,
        "peak_cuda_memory_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 3) if device.type == "cuda" else None,
        "peak_cuda_memory_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 3) if device.type == "cuda" else None,
        "training_seconds": round(training_seconds, 3), "total_run_seconds": round(time.perf_counter() - started, 3),
        "single_class_collapse_warning_on_supervised_support": collapse_warning,
        "background_supervision_used": False, "free_instance_detection_evaluated": False, "counting_accuracy_evaluated": False,
        "limitations": config["limitations"],
    }
    if mode == "overfit":
        summary["overfit_loss_reduction"] = first_train_loss - final_metrics["total"]
        summary["overfit_learning_check_passed"] = first_train_loss > final_metrics["total"]
    with (run_dir / "training_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return run_dir

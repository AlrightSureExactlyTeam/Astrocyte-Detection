"""Deployable U-Net inference runtime for Streamlit Cloud Live inference.

The functions below are verbatim copies of the validated implementations in
the local-only annotation workspace (annotation_workspace is not deployed):

- validate_pilot_config / load_pilot_config
  <- annotation_workspace/gfap_annotation/pilot_training.py
- display_image / load_checkpoint / predict_tile / prediction_statistics
  <- annotation_workspace/tools/compare_unet_pilots.py

Only the import locations changed; no algorithm, architecture, or weight
logic was modified. Model construction and image preprocessing still come
from the in-repo src package (src.train, src.preprocessing).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import yaml

from src.preprocessing import prepare_image
from src.train import build_model, load_init_weights

CLASS_NAMES = ("background", "soma", "process")


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


def display_image(raw: np.ndarray, channel: int = 1) -> np.ndarray:
    gray = raw[..., channel].astype(np.float32)
    low, high = np.percentile(gray, [1, 99.8])
    if high <= low:
        high = float(gray.max())
    gray = np.clip((gray - low) / max(high - low, 1e-12), 0, 1)
    return np.repeat(gray[..., None], 3, axis=2)


def load_checkpoint(run_dir: Path, device: torch.device):
    config = load_pilot_config(run_dir / "config_snapshot.yaml")
    model = build_model(config)
    checkpoint = run_dir / "checkpoints" / "final.pt"
    load_init_weights(model, checkpoint)
    model = model.to(device).eval()
    return model, config, checkpoint


def predict_tile(model, raw: np.ndarray, device: torch.device, patch_size: int = 256, batch_size: int = 4) -> np.ndarray:
    height, width = raw.shape[:2]
    if height % patch_size or width % patch_size:
        raise ValueError(f"Tile shape {raw.shape} is not divisible by {patch_size}")
    patches, coordinates = [], []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = prepare_image(raw[y:y + patch_size, x:x + patch_size], channel=1)
            patches.append(patch)
            coordinates.append((x, y))
    probabilities = np.empty((3, height, width), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.stack(patches[start:start + batch_size]).to(device)
            outputs = model(batch)
            logits = outputs["semantic_logits"]
            if logits.shape != (batch.shape[0], 3, patch_size, patch_size) or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid or non-finite semantic logits")
            batch_probability = torch.softmax(logits, dim=1).cpu().numpy()
            for offset, (x, y) in enumerate(coordinates[start:start + batch_size]):
                probabilities[:, y:y + patch_size, x:x + patch_size] = batch_probability[offset]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite stitched probabilities")
    if not np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-5):
        raise RuntimeError("Semantic probabilities do not sum to one")
    return probabilities


def prediction_statistics(labels: np.ndarray) -> dict:
    counts = {name: int((labels == index).sum()) for index, name in enumerate(CLASS_NAMES)}
    total = labels.size
    fractions = {name: counts[name] / total for name in CLASS_NAMES}
    present = [name for name in CLASS_NAMES if counts[name] > 0]
    dominant = max(fractions, key=fractions.get)
    return {
        "predicted_pixel_counts": counts,
        "predicted_pixel_fractions": fractions,
        "classes_present": present,
        "dominant_class": dominant,
        "dominant_fraction": fractions[dominant],
        "exact_single_class_collapse": len(present) == 1,
        "dominance_warning_95_percent": fractions[dominant] >= 0.95,
    }

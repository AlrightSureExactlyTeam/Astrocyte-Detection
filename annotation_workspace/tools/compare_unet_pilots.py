#!/usr/bin/env python
"""Reload V1/V2 checkpoints and compare aligned full-resolution GFAP predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE.parent
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(PROJECT_ROOT))

from gfap_annotation.pilot_training import load_pilot_config  # noqa: E402
from src.preprocessing import load_image, prepare_image  # noqa: E402
from src.train import build_model, get_device, load_init_weights  # noqa: E402

CLASS_NAMES = ("background", "soma", "process")
SELECTED_SAMPLES = [
    "GBM664_S5_Scan1_x043008_y109568_RGB",
    "GBM664_S5_Scan1_x043008_y111616_RGB",
    "GBM664_S5_Scan1_x044032_y110592_RGB",
    "GBM664_S5_Scan1_x047104_y115712_RGB",
    "GBM664_S5_Scan1_x049152_y118784_RGB",
]


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise OSError(f"Could not encode {path}")
    with path.open("xb") as handle:
        handle.write(encoded.tobytes())


def write_gray(path: Path, probability: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8)
    success, encoded = cv2.imencode(".png", data)
    if not success:
        raise OSError(f"Could not encode {path}")
    with path.open("xb") as handle:
        handle.write(encoded.tobytes())


def display_image(raw: np.ndarray, channel: int = 1) -> np.ndarray:
    gray = raw[..., channel].astype(np.float32)
    low, high = np.percentile(gray, [1, 99.8])
    if high <= low:
        high = float(gray.max())
    gray = np.clip((gray - low) / max(high - low, 1e-12), 0, 1)
    return np.repeat(gray[..., None], 3, axis=2)


def semantic_colors(labels: np.ndarray) -> np.ndarray:
    color = np.zeros((*labels.shape, 3), dtype=np.float32)
    color[labels == 1] = (0, 1, 0)
    color[labels == 2] = (1, 0, 0)
    return color


def overlay(base: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = base.copy()
    color = semantic_colors(labels)
    foreground = labels > 0
    result[foreground] = 0.45 * base[foreground] + 0.55 * color[foreground]
    return result


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


def save_model_predictions(
    run_dir: Path,
    model,
    config: dict,
    checkpoint: Path,
    samples: list[dict],
    device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    per_image = {}
    overlays = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        image_started = time.perf_counter()
        probability = predict_tile(model, sample["raw"], device)
        labels = probability.argmax(axis=0).astype(np.uint8)
        output = run_dir / "predictions" / sample_id
        for index, name in enumerate(CLASS_NAMES):
            write_gray(output / f"{name}_probability.png", probability[index])
        write_png(output / "semantic_prediction.png", np.rint(semantic_colors(labels) * 255).astype(np.uint8))
        output.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(output / "semantic_mask.tif", labels, photometric="minisblack")
        prediction_overlay = overlay(sample["display"], labels)
        write_png(output / "prediction_overlay.png", np.rint(prediction_overlay * 255).astype(np.uint8))
        stats = {
            "sample_id": sample_id,
            "source_sha256": sample["source_sha256"],
            "original_shape": list(sample["raw"].shape[:2]),
            "reconstructed_probability_shape": list(probability.shape),
            "finite_probabilities": True,
            "probabilities_sum_to_one": True,
            "alignment_verified": probability.shape[1:] == sample["raw"].shape[:2] == sample["label"].shape,
            "inference_seconds": round(time.perf_counter() - image_started, 3),
            **prediction_statistics(labels),
        }
        with (output / "prediction_statistics.json").open("x", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        per_image[sample_id] = stats
        overlays[sample_id] = prediction_overlay
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"]["name"],
        "training_completion_status": training_summary["status"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_reloaded_in_evaluation_mode": not model.training,
        "images_evaluated": len(samples),
        "patch_inference": "Non-overlapping 256×256 patches stitched at original 1024×1024 resolution; no whole-tile resizing",
        "preprocessing": "Same per-patch percentile 1/99.8 normalization and GFAP channel 1 as training",
        "per_image": per_image,
        "total_inference_seconds": round(time.perf_counter() - started, 3),
        "device": str(device),
        "peak_cuda_memory_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 3) if device.type == "cuda" else None,
        "peak_cuda_memory_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 3) if device.type == "cuda" else None,
        "formal_dice_or_iou_calculated": False,
        "limitations": config["limitations"],
    }
    with (run_dir / "prediction_statistics.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    lines = [
        f"# {config['experiment']['name']} experiment summary", "",
        "- Technical pilot only; not formal model evaluation.",
        f"- Training status: {training_summary['status']}; final manual batch loss: {training_summary['final_manual_train_loss']:.6f}.",
        f"- Reloaded checkpoint: `{checkpoint}`.",
        f"- Images evaluated: {len(samples)} at original 1024×1024 resolution.", "",
        "| Sample | Background | Soma | Process | Dominance warning |", "|---|---:|---:|---:|---|",
    ]
    for sample_id, stats in per_image.items():
        fractions = stats["predicted_pixel_fractions"]
        lines.append(
            f"| {sample_id} | {fractions['background']:.2%} | {fractions['soma']:.2%} "
            f"| {fractions['process']:.2%} | {stats['dominance_warning_95_percent']} |"
        )
    lines.extend([
        "", "Background probabilities are not reliable because no confirmed background labels exist.",
        "No Dice/IoU was calculated over incomplete annotations. ImageJ masks were not used as ground truth.",
        "V2 auxiliary positive-only output is not a calibrated GFAP probability.",
    ])
    (run_dir / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, overlays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-run", required=True)
    parser.add_argument("--v2-run", required=True)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results" / "gfap_unet_pilots" / "comparisons"))
    args = parser.parse_args()
    v1_run, v2_run = Path(args.v1_run).resolve(), Path(args.v2_run).resolve()
    if v1_run == v2_run:
        raise ValueError("V1 and V2 run directories must differ")
    device = get_device("auto")
    v1_model, v1_config, v1_checkpoint = load_checkpoint(v1_run, device)
    v2_model, v2_config, v2_checkpoint = load_checkpoint(v2_run, device)
    manual_keys = ["manifest", "channel", "patch_size", "patch_stride", "manual_train_slide_groups"]
    if any(v1_config["data"][key] != v2_config["data"][key] for key in manual_keys):
        raise ValueError("Manual inference settings differ")
    manifest = PROJECT_ROOT / v1_config["data"]["manifest"]
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    samples = []
    for sample_id in SELECTED_SAMPLES:
        row = next(row for row in rows if row["sample_id"] == sample_id and row["training_suitable"] == "true")
        image_path = PROJECT_ROOT / row["image_path"]
        label_path = PROJECT_ROOT / row["training_label_path"]
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest != row["source_sha256"]:
            raise ValueError(f"Source checksum mismatch: {sample_id}")
        raw = load_image(image_path)
        label = tifffile.imread(label_path)
        if raw.shape[:2] != (1024, 1024) or label.shape != raw.shape[:2]:
            raise ValueError(f"Unexpected source/label shape: {sample_id}")
        samples.append({
            "sample_id": sample_id, "raw": raw, "label": label,
            "source_sha256": digest, "display": display_image(raw),
        })
    v1_report, v1_overlays = save_model_predictions(v1_run, v1_model, v1_config, v1_checkpoint, samples, device)
    del v1_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    v2_report, v2_overlays = save_model_predictions(v2_run, v2_model, v2_config, v2_checkpoint, samples, device)

    comparison_dir = Path(args.output_root) / (
        datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%f_") + uuid.uuid4().hex[:6]
    )
    comparison_dir.mkdir(parents=True, exist_ok=False)
    rows_canvas = []
    for sample in samples:
        sample_id = sample["sample_id"]
        base = sample["display"]
        manual = overlay(base, sample["label"])
        panels = [base, manual, v1_overlays[sample_id], v2_overlays[sample_id]]
        row_canvas = np.zeros((1074, 4096, 3), dtype=np.uint8)
        row_canvas[50:] = np.concatenate([np.rint(panel * 255).astype(np.uint8) for panel in panels], axis=1)
        cv2.putText(row_canvas, sample_id, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        for index, heading in enumerate(("Original GFAP", "Manual dense labels (ignore unchanged)", "U-Net V1", "U-Net V2")):
            cv2.putText(row_canvas, heading, (10 + index * 1024, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
        rows_canvas.append(row_canvas)
        write_png(comparison_dir / sample_id / "comparison_overview.png", row_canvas)
    overview = np.concatenate(rows_canvas, axis=0)
    write_png(comparison_dir / "comparison_overview.png", overview)
    preview = cv2.resize(overview, (1024, overview.shape[0] // 4), interpolation=cv2.INTER_AREA)
    write_png(comparison_dir / "comparison_preview.png", preview)
    manifest_report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "v1_run": str(v1_run), "v2_run": str(v2_run),
        "samples": SELECTED_SAMPLES,
        "images_evaluated_per_model": len(samples),
        "overview_original_panel_resolution": [4096, 5370],
        "identical_regions_and_display_settings": True,
        "cellpose_comparison_included": False,
        "cellpose_note": "No verified matching Cellpose predictions were supplied for these tiles.",
        "formal_evaluation": False,
    }
    with (comparison_dir / "comparison_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest_report, handle, indent=2)
    print(json.dumps({
        "status": "completed", "v1": v1_report, "v2": v2_report,
        "comparison_dir": str(comparison_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Read-only four-model inference and qualitative GFAP visual comparison.

This script never loads annotations, performs optimization, or writes checkpoints.
V3 exports morphology probabilities only; it does not create autonomous instances.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import yaml

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE.parent
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(PROJECT_ROOT))

from annotation_workspace.tools.compare_unet_pilots import (  # noqa: E402
    display_image, load_checkpoint, predict_tile, prediction_statistics,
)
from demo_core.images import semantic_overlay  # noqa: E402
from demo_core.registry import get_model  # noqa: E402
from gfap_annotation.v3_model import create_v3_model  # noqa: E402
from gfap_annotation.v3_training import load_v3_checkpoint  # noqa: E402
from gfap_annotation.workflow import _validate_prediction, normalize_for_cellpose  # noqa: E402
from src.preprocessing import load_image, prepare_image  # noqa: E402
from src.train import get_device  # noqa: E402

TILE_DIR = PROJECT_ROOT / "annotation_workspace/raw_exports/GBM664_S5_full_export/rgb_tiles"
MANIFEST = PROJECT_ROOT / "annotation_workspace/manifests/gfap_v3_training_manifest.csv"
REGISTRY = PROJECT_ROOT / "configs/demo_models.yaml"
V3_RUN = PROJECT_ROOT / "results/gfap_v3/pilot_20261030T150553_106077_168461"
SELECTIONS = [
    ("GBM664_S5_Scan1_x034816_y119808_RGB", 6, "Weak or uneven fluorescence with sparse bright structures"),
    ("GBM664_S5_Scan1_x045056_y112640_RGB", 12, "Relatively isolated bright cell body with visible processes"),
    ("GBM664_S5_Scan1_x052224_y115712_RGB", 18, "Multiple nearby cell-like structures and neighboring processes"),
    ("GBM664_S5_Scan1_x054272_y109568_RGB", 19, "Long, thin branching processes across the tile"),
    ("GBM664_S5_Scan1_x055296_y112640_RGB", 20, "Dense or overlapping cellular structures"),
]
CLASS_NAMES = ("background", "soma", "process")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("CSV needs at least one row")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise OSError(f"Could not encode {path}")
    with path.open("xb") as handle:
        handle.write(encoded.tobytes())


def write_gray(path: Path, probability: np.ndarray) -> None:
    write_png(path, np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8))


def write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)


def write_tiff(path: Path, array: np.ndarray, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tifffile.imwrite(
        path, array, photometric="minisblack",
        metadata={"axes": "YX", "review_status": "unverified", "description": description},
    )


def validate_probability(probability: np.ndarray, classes: int, shape: tuple[int, int]) -> None:
    if probability.shape != (classes, *shape):
        raise ValueError(f"Probability shape {probability.shape} does not match {(classes, *shape)}")
    if probability.dtype != np.float32 or not np.isfinite(probability).all():
        raise ValueError("Probabilities must be finite float32")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Probability outside [0,1]")
    if not np.allclose(probability.sum(axis=0), 1.0, atol=1e-5):
        raise ValueError("Probabilities do not sum to one")


def predict_v3_tile(model, raw: np.ndarray, device: torch.device, patch_size: int = 256, batch_size: int = 4) -> np.ndarray:
    """Stitch two-class V3 morphology probabilities without using GT or embeddings."""
    height, width = raw.shape[:2]
    if height % patch_size or width % patch_size:
        raise ValueError(f"Tile shape {raw.shape} is not divisible by {patch_size}")
    patches, coordinates = [], []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patches.append(prepare_image(raw[y:y + patch_size, x:x + patch_size], channel=1))
            coordinates.append((x, y))
    probabilities = np.empty((2, height, width), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.stack(patches[start:start + batch_size]).to(device)
            outputs = model(batch)
            logits = outputs["morphology_logits"]
            if logits.shape != (batch.shape[0], 2, patch_size, patch_size) or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid or non-finite V3 morphology logits")
            batch_probability = torch.softmax(logits, dim=1).cpu().numpy()
            for offset, (x, y) in enumerate(coordinates[start:start + batch_size]):
                probabilities[:, y:y + patch_size, x:x + patch_size] = batch_probability[offset]
    validate_probability(probabilities, 2, (height, width))
    return probabilities


def instance_colors(mask: np.ndarray) -> np.ndarray:
    colors = np.zeros((*mask.shape, 3), dtype=np.float32)
    for instance_id in np.unique(mask):
        if instance_id == 0:
            continue
        rng = np.random.default_rng(int(instance_id) * 7919)
        color = rng.uniform(0.25, 1.0, size=3).astype(np.float32)
        colors[mask == instance_id] = color
    return colors


def instance_overlay(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = base.copy()
    colors = instance_colors(mask)
    foreground = mask > 0
    result[foreground] = 0.45 * base[foreground] + 0.55 * colors[foreground]
    return result


def heatmap(probability: np.ndarray) -> np.ndarray:
    gray = np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def title_panel(image: np.ndarray, title: str, width: int = 1024) -> np.ndarray:
    if image.shape[:2] != (width, width):
        image = cv2.resize(image, (width, width), interpolation=cv2.INTER_AREA)
    panel = np.full((width + 58, width, 3), 255, dtype=np.uint8)
    panel[58:] = image
    cv2.putText(panel, title, (14, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 0), 2, cv2.LINE_AA)
    return panel


def limitation_panel(width: int = 1024) -> np.ndarray:
    panel = np.full((width, width, 3), 245, dtype=np.uint8)
    lines = [
        "V3 exploratory output only",
        "",
        "No Background rejection logit",
        "No autonomous instance detection",
        "No validated cell counting",
        "No GT Soma anchors used here",
        "No annotation masks used here",
        "",
        "Heatmaps show two-class softmax",
        "probabilities, not foreground masks.",
        "Patch seams may be visible.",
    ]
    y = 110
    for index, line in enumerate(lines):
        scale = 1.05 if index == 0 else 0.86
        thickness = 2 if index == 0 else 1
        cv2.putText(panel, line, (48, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thickness, cv2.LINE_AA)
        y += 72
    return panel


def make_comparison(sample_id: str, panels: dict[str, np.ndarray], output: Path) -> None:
    ordered = [
        ("Original GFAP channel 1", panels["original"]),
        ("Cellpose instance overlay (unverified)", panels["cellpose_overlay"]),
        ("Cellpose boundaries (unverified)", panels["cellpose_boundaries"]),
        ("U-Net V1 semantic overlay", panels["v1_overlay"]),
        ("U-Net V2 semantic overlay", panels["v2_overlay"]),
        ("V3 Soma probability [0,1]", panels["v3_soma_heat"]),
        ("V3 Process probability [0,1]", panels["v3_process_heat"]),
        ("V3 output scope", limitation_panel()),
    ]
    top = np.concatenate([title_panel(image, title) for title, image in ordered[:4]], axis=1)
    bottom = np.concatenate([title_panel(image, title) for title, image in ordered[4:]], axis=1)
    header = np.full((82, top.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, sample_id, (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 0, 0), 2, cv2.LINE_AA)
    figure = np.concatenate([header, top, bottom], axis=0)
    write_png(output, figure)


def make_contact_sheet(records: list[dict], panels: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    size = 360
    title_height = 52
    headers = ["Original", "Cellpose", "U-Net V1", "U-Net V2", "V3 Soma prob.", "V3 Process prob."]
    keys = ["original", "cellpose_overlay", "v1_overlay", "v2_overlay", "v3_soma_heat", "v3_process_heat"]
    header = np.full((72, size * 6, 3), 255, dtype=np.uint8)
    for index, title in enumerate(headers):
        cv2.putText(header, title, (index * size + 12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
    rows = [header]
    for record in records:
        sample_id = record["sample_id"]
        row = np.full((size + title_height, size * 6, 3), 255, dtype=np.uint8)
        cv2.putText(row, sample_id, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
        for index, key in enumerate(keys):
            image = cv2.resize(panels[sample_id][key], (size, size), interpolation=cv2.INTER_AREA)
            row[title_height:, index * size:(index + 1) * size] = image
        rows.append(row)
    write_png(output, np.concatenate(rows, axis=0))


def probability_summary(probability: np.ndarray, names: tuple[str, ...]) -> dict:
    return {
        name: {
            "mean": float(probability[index].mean()),
            "p95": float(np.percentile(probability[index], 95)),
            "min": float(probability[index].min()),
            "max": float(probability[index].max()),
        }
        for index, name in enumerate(names)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New, non-existing run directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--candidate-file", type=Path, default=PROJECT_ROOT / "results/gfap_model_comparison/selection_20261030T_visual/candidates.json")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = get_device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    candidates = json.loads(args.candidate_file.read_text(encoding="utf-8"))["candidates"]
    candidate_by_id = {item["sample_id"]: item for item in candidates}
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        manifest_rows = {row["image_id"]: row for row in csv.DictReader(handle)}
    records, raw_images, displays, panels = [], {}, {}, {}
    for sample_id, candidate_number, reason in SELECTIONS:
        source = TILE_DIR / f"{sample_id}.tif"
        raw = load_image(source)
        if raw.shape != (1024, 1024, 3) or not np.isfinite(raw).all():
            raise ValueError(f"Unexpected source image: {source} {raw.shape}")
        digest = sha256(source)
        candidate = candidate_by_id[sample_id]
        if digest != candidate["source_sha256"]:
            raise ValueError(f"Source changed since candidate selection: {sample_id}")
        if sample_id in manifest_rows:
            raise ValueError(f"Selected image unexpectedly appears in V3 manifest: {sample_id}")
        display = np.rint(display_image(raw) * 255).astype(np.uint8)
        records.append({
            "sample_id": sample_id,
            "filename": source.name,
            "source_path": relative(source),
            "source_sha256": digest,
            "tile_x": candidate["tile_x"],
            "tile_y": candidate["tile_y"],
            "shape_hwc": list(raw.shape),
            "source_dtype": str(raw.dtype),
            "v3_provenance": "unannotated",
            "provenance_basis": "Image ID absent from gfap_v3_training_manifest.csv",
            "slide_sample_id": "GBM664_S5_Scan1",
            "independent_external_validation": False,
            "candidate_number": candidate_number,
            "visual_selection_reason": reason,
            "annotation_used_during_inference": False,
            "spatial_transform": "none; original HWC orientation and 1024x1024 coordinates retained",
        })
        raw_images[sample_id] = raw
        displays[sample_id] = display
        panels[sample_id] = {"original": display}
    write_json(output / "selected_images.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "Five visually selected unannotated tiles from one slide; qualitative comparison only",
        "images": records,
    })
    write_csv(output / "selected_images.csv", records)

    registry_specs = {name: get_model(name, REGISTRY) for name in ("cellpose", "unet_v1", "unet_v2")}
    v3_checkpoint = V3_RUN / "checkpoints/final.pt"
    v3_config_path = V3_RUN / "config_snapshot.yaml"
    v3_config = yaml.safe_load(v3_config_path.read_text(encoding="utf-8"))
    model_files = {
        "cellpose": registry_specs["cellpose"].checkpoint,
        "unet_v1": registry_specs["unet_v1"].checkpoint,
        "unet_v2": registry_specs["unet_v2"].checkpoint,
        "v3": v3_checkpoint,
    }
    for name, path in model_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    model_provenance = {}
    for name, path in model_files.items():
        model_provenance[name] = {"checkpoint_path": relative(path), "checkpoint_sha256": sha256(path)}
    model_provenance["v3"]["config_snapshot"] = relative(v3_config_path)
    model_provenance["v3"]["config_sha256"] = sha256(v3_config_path)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "read-only qualitative inference; no training and no annotation input",
        "model_registry": relative(REGISTRY),
        "model_registry_sha256": sha256(REGISTRY),
        "models": model_provenance,
        "cellpose": {
            "version": registry_specs["cellpose"].version,
            "parameters": {"normalize": False, "diameter": None, "min_size": 15, "max_size_fraction": 0.4, "flow_threshold": 0.4, "cellprob_threshold": 0.0},
            "preprocessing": "Existing normalize_for_cellpose on GFAP channel 1; model eval normalize=False",
        },
        "unet_v1_v2": {
            "label_mapping": {"0": "Background", "1": "Soma", "2": "Process"},
            "preprocessing": "GFAP channel 1; per-patch percentile 1/99.8; non-overlapping 256x256 patches; FP32; no resizing",
        },
        "v3": {
            "preprocessing": "GFAP channel 1; per-patch percentile 1/99.8; non-overlapping 256x256 patches; FP32; no resizing",
            "outputs": ["Soma probability", "Process probability"],
            "background_logit": False,
            "autonomous_instances": False,
            "gt_soma_anchors_used": False,
            "embedding_used_for_visualization": False,
        },
        "display": "GFAP channel 1; per-image percentile 1/99.8; Soma green, Process red; heatmap TURBO fixed [0,1]",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cellpose": importlib.metadata.version("cellpose"),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "tifffile": tifffile.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    write_json(output / "model_and_runtime_provenance.json", provenance)

    inference_records = {record["sample_id"]: {"sample_id": record["sample_id"], "source_sha256": record["source_sha256"], "models": {}} for record in records}
    model_runtimes = {}

    from cellpose import models
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_started = time.perf_counter()
    cellpose_model = models.CellposeModel(gpu=device.type == "cuda", pretrained_model=str(registry_specs["cellpose"].checkpoint))
    try:
        for record in records:
            sample_id = record["sample_id"]
            image_started = time.perf_counter()
            cell_input, normalization = normalize_for_cellpose(raw_images[sample_id][..., 1])
            raw_result = cellpose_model.eval(cell_input, **provenance["cellpose"]["parameters"])
            mask = _validate_prediction(raw_result[0] if isinstance(raw_result, (tuple, list)) else raw_result, raw_images[sample_id].shape[:2])
            if mask.shape != raw_images[sample_id].shape[:2]:
                raise ValueError("Cellpose spatial alignment failure")
            target = output / "predictions" / sample_id / "cellpose"
            write_tiff(target / "instance_mask.tif", mask, "Unverified Cellpose candidate instances; not ground truth")
            overlay = np.rint(instance_overlay(displays[sample_id].astype(np.float32) / 255, mask) * 255).astype(np.uint8)
            boundaries = np.rint(semantic_overlay(displays[sample_id].astype(np.float32) / 255, mask, instance=True) * 255).astype(np.uint8)
            write_png(target / "instance_overlay.png", overlay)
            write_png(target / "boundaries.png", boundaries)
            panels[sample_id]["cellpose_overlay"] = overlay
            panels[sample_id]["cellpose_boundaries"] = boundaries
            inference_records[sample_id]["models"]["cellpose"] = {
                "instance_mask": relative(target / "instance_mask.tif"),
                "instance_overlay": relative(target / "instance_overlay.png"),
                "boundaries": relative(target / "boundaries.png"),
                "nonempty_prediction": bool(mask.max() > 0),
                "normalization": normalization,
                "inference_seconds": round(time.perf_counter() - image_started, 3),
                "shape_verified": True,
                "review_status": "unverified",
                "ground_truth": False,
            }
            print(f"Cellpose {sample_id} done", flush=True)
    finally:
        del cellpose_model
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    model_runtimes["cellpose"] = {"seconds": round(time.perf_counter() - model_started, 3)}

    for model_id in ("unet_v1", "unet_v2"):
        spec = registry_specs[model_id]
        run_source = spec.checkpoint.parent.parent
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model_started = time.perf_counter()
        model, config, checkpoint = load_checkpoint(run_source, device)
        try:
            for record in records:
                sample_id = record["sample_id"]
                image_started = time.perf_counter()
                probability = predict_tile(model, raw_images[sample_id], device)
                validate_probability(probability, 3, raw_images[sample_id].shape[:2])
                labels = probability.argmax(axis=0).astype(np.uint8)
                target = output / "predictions" / sample_id / model_id
                write_tiff(target / "semantic_mask.tif", labels, "Unverified semantic prediction; Background is not reliable")
                write_npy(target / "probabilities.npy", probability)
                write_gray(target / "soma_probability.png", probability[1])
                write_gray(target / "process_probability.png", probability[2])
                overlay = np.rint(semantic_overlay(displays[sample_id].astype(np.float32) / 255, labels) * 255).astype(np.uint8)
                write_png(target / "prediction_overlay.png", overlay)
                panels[sample_id]["v1_overlay" if model_id == "unet_v1" else "v2_overlay"] = overlay
                inference_records[sample_id]["models"][model_id] = {
                    "semantic_mask": relative(target / "semantic_mask.tif"),
                    "probabilities": relative(target / "probabilities.npy"),
                    "prediction_overlay": relative(target / "prediction_overlay.png"),
                    "inference_seconds": round(time.perf_counter() - image_started, 3),
                    "shape_verified": True,
                    "finite_probabilities": True,
                    "probabilities_sum_to_one": True,
                    "probability_summary": probability_summary(probability, CLASS_NAMES),
                    **prediction_statistics(labels),
                    "review_status": "unverified",
                    "ground_truth": False,
                    "background_reliability": "not established; no confirmed Background supervision",
                }
                print(f"{model_id} {sample_id} done", flush=True)
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
        model_runtimes[model_id] = {"seconds": round(time.perf_counter() - model_started, 3)}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_started = time.perf_counter()
    v3_model = create_v3_model(
        in_channels=int(v3_config["data"]["input_channels"]),
        base_channels=int(v3_config["model"]["base_channels"]),
        embedding_dim=int(v3_config["model"]["embedding_dim"]),
    )
    load_v3_checkpoint(v3_model, v3_checkpoint, device)
    v3_model = v3_model.to(device).eval()
    try:
        for record in records:
            sample_id = record["sample_id"]
            image_started = time.perf_counter()
            probability = predict_v3_tile(v3_model, raw_images[sample_id], device)
            target = output / "predictions" / sample_id / "v3"
            write_npy(target / "morphology_probabilities.npy", probability)
            write_tiff(target / "soma_probability.tif", probability[0], "V3 Soma probability; not a foreground or instance mask")
            write_tiff(target / "process_probability.tif", probability[1], "V3 Process probability; not a foreground or instance mask")
            soma_heat, process_heat = heatmap(probability[0]), heatmap(probability[1])
            write_png(target / "soma_probability_heatmap.png", soma_heat)
            write_png(target / "process_probability_heatmap.png", process_heat)
            panels[sample_id]["v3_soma_heat"] = soma_heat
            panels[sample_id]["v3_process_heat"] = process_heat
            inference_records[sample_id]["models"]["v3"] = {
                "probabilities": relative(target / "morphology_probabilities.npy"),
                "soma_probability": relative(target / "soma_probability.tif"),
                "process_probability": relative(target / "process_probability.tif"),
                "soma_heatmap": relative(target / "soma_probability_heatmap.png"),
                "process_heatmap": relative(target / "process_probability_heatmap.png"),
                "inference_seconds": round(time.perf_counter() - image_started, 3),
                "shape_verified": True,
                "finite_probabilities": True,
                "probabilities_sum_to_one": True,
                "probability_summary": probability_summary(probability, ("soma", "process")),
                "review_status": "unverified",
                "ground_truth": False,
                "foreground_mask_generated": False,
                "autonomous_instance_mask_generated": False,
                "cell_count_generated": False,
                "background_rejection": False,
            }
            print(f"V3 {sample_id} done", flush=True)
    finally:
        del v3_model
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    model_runtimes["v3"] = {"seconds": round(time.perf_counter() - model_started, 3)}

    figures_dir = output / "figures"
    for record in records:
        sample_id = record["sample_id"]
        figure_path = figures_dir / f"{sample_id}_comparison.png"
        make_comparison(sample_id, panels[sample_id], figure_path)
        inference_records[sample_id]["comparison_figure"] = relative(figure_path)
    contact_sheet = figures_dir / "contact_sheet.png"
    make_contact_sheet(records, panels, contact_sheet)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "inference_and_figure_generation_complete_pending_visual_review",
        "scope": "Five unannotated tiles from GBM664_S5_Scan1; qualitative in-distribution visual exploration, not independent external validation or a benchmark",
        "selected_images": records,
        "inference": list(inference_records.values()),
        "model_runtimes": model_runtimes,
        "total_run_seconds": round(time.perf_counter() - started, 3),
        "contact_sheet": relative(contact_sheet),
        "all_predictions_unverified": True,
        "ground_truth_used": False,
        "training_performed": False,
        "optimizer_steps": 0,
        "limitations": [
            "No suitable ground truth is used; no accuracy ranking or validated cell counts are reported.",
            "All five tiles come from the same slide and are unannotated in the V3 manifest.",
            "V1/V2 Background predictions are unreliable because confirmed Background supervision is absent.",
            "V3 has no Background rejection and no validated autonomous instance detection; only probabilities are shown.",
            "Non-overlapping patch inference can produce visible patch-boundary artifacts.",
        ],
    }
    write_json(output / "inference_summary.json", summary)
    flat_rows = []
    for record in records:
        sample_id = record["sample_id"]
        item = inference_records[sample_id]
        row = {
            "sample_id": sample_id,
            "source_sha256": record["source_sha256"],
            "v3_provenance": record["v3_provenance"],
            "cellpose_nonempty": item["models"]["cellpose"]["nonempty_prediction"],
            "comparison_figure": item["comparison_figure"],
        }
        for model_id in ("unet_v1", "unet_v2"):
            fractions = item["models"][model_id]["predicted_pixel_fractions"]
            for name in CLASS_NAMES:
                row[f"{model_id}_{name}_fraction"] = fractions[name]
        for name in ("soma", "process"):
            row[f"v3_{name}_probability_mean"] = item["models"]["v3"]["probability_summary"][name]["mean"]
        flat_rows.append(row)
    write_csv(output / "inference_summary.csv", flat_rows)
    write_json(output / "validation_report.json", {
        "selected_image_count": len(records),
        "all_source_shapes_1024_1024_3": True,
        "all_source_hashes_match_selection": True,
        "all_selected_absent_from_v3_manifest": True,
        "all_prediction_shapes_match_sources": True,
        "all_probabilities_finite_and_normalized": True,
        "orientation_or_resizing_applied_to_inference": False,
        "annotation_or_gt_used": False,
        "training_or_checkpoint_modification": False,
        "visual_review_completed": False,
    })
    print(f"Output: {output}", flush=True)
    print(f"Total seconds: {time.perf_counter() - started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

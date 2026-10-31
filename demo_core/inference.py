"""Lazy adapters around the existing Cellpose and U-Net inference implementations."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import torch

from .images import encode_png, semantic_overlay, validate_for_model
from .registry import ModelSpec, PROJECT_ROOT
from .reports import new_run_directory, write_json
from .results import DemoResult
from .weights import cloud_inference_guard, ensure_unet_checkpoint


def _write_png_new(path: Path, image: np.ndarray) -> None:
    with path.open("xb") as handle:
        handle.write(encode_png(image))


def _model_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_live_inference(
    image: np.ndarray,
    sample_id: str,
    model: ModelSpec,
    output_root: str | Path = PROJECT_ROOT / "results/sciencepal_demo",
    device_name: str = "auto",
) -> tuple[DemoResult, Path]:
    """Load only the selected model and write into one new demo run directory."""
    if not model.live_available:
        raise RuntimeError(model.unavailable_reason or "Model is unavailable")
    validate_for_model(image, model.expected_input)
    with cloud_inference_guard():
        if model.inference_adapter == "unet":
            model = replace(model, checkpoint=ensure_unet_checkpoint(model))
        run_dir = new_run_directory(output_root)
        if model.inference_adapter == "cellpose":
            result = _run_cellpose(image, sample_id, model, run_dir)
        elif model.inference_adapter == "unet":
            result = _run_unet(image, sample_id, model, run_dir, device_name)
        elif model.inference_adapter == "v33b2":
            from .v33b2_adapter import run_v33b2_inference
            result = run_v33b2_inference(image, sample_id, model, run_dir, device_name)
        else:
            raise RuntimeError(f"No live adapter for {model.inference_adapter}")
        return result, run_dir


def _run_cellpose(image: np.ndarray, sample_id: str, model: ModelSpec, run_dir: Path) -> DemoResult:
    workspace = PROJECT_ROOT / "annotation_workspace"
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from gfap_annotation.workflow import (
        _validate_prediction, candidate_statistics, normalize_for_cellpose, save_overlay,
    )
    from cellpose import models

    gpu = torch.cuda.is_available()
    cell_index = int(model.expected_input["cell_channel"])
    nucleus_index = int(model.expected_input.get("nucleus_channel", 0))
    cell_input, cell_norm = normalize_for_cellpose(image[..., cell_index])
    cellpose_model = models.CellposeModel(gpu=gpu, pretrained_model=str(model.checkpoint))
    try:
        parameters = dict(
            normalize=False, diameter=None, min_size=15, max_size_fraction=0.4,
            flow_threshold=0.4, cellprob_threshold=0.0,
        )
        raw = cellpose_model.eval(cell_input, **parameters)
        cell_mask = _validate_prediction(raw[0] if isinstance(raw, (tuple, list)) else raw, cell_input.shape)
        nucleus_input, nucleus_norm = normalize_for_cellpose(image[..., nucleus_index])
        raw = cellpose_model.eval(nucleus_input, **parameters)
        nucleus_mask = _validate_prediction(raw[0] if isinstance(raw, (tuple, list)) else raw, nucleus_input.shape)
    finally:
        del cellpose_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mask_path = run_dir / "cell_mask.tif"
    nucleus_path = run_dir / "nucleus_mask.tif"
    tifffile.imwrite(mask_path, cell_mask, photometric="minisblack", metadata={"axes": "YX", "review_status": "unverified"})
    tifffile.imwrite(nucleus_path, nucleus_mask, photometric="minisblack", metadata={"axes": "YX", "review_status": "unverified"})
    cyx = np.moveaxis(image, -1, 0)
    maxima = [float(np.percentile(cyx[index], 99.8)) or 1.0 for index in range(cyx.shape[0])]
    overlay_path = run_dir / "overlay.png"
    save_overlay(cyx, cell_mask, nucleus_mask, cell_index, nucleus_index, {"display_max": maxima}, overlay_path)
    stats, issues, _ = candidate_statistics(sample_id, run_dir.name, cell_mask, nucleus_mask)
    statistics = {
        **{key: int(value) if key != "review_status" else value for key, value in stats.items() if key not in {"image_id", "prediction_revision"}},
        "candidate_cell_matching_issues": issues,
        "review_status": "unverified", "ground_truth": False,
        "cell_normalization": cell_norm, "nucleus_normalization": nucleus_norm,
        "parameters": parameters,
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "sample_id": sample_id,
        "mode": "live", "model_id": model.id, "model_version": model.version,
        "model_path": str(model.checkpoint), "model_sha256": _model_sha256(model.checkpoint),
        "gpu_used": gpu, "input_shape": list(image.shape),
        "review_status": "unverified", "ground_truth": False,
    }
    write_json(run_dir / "input_metadata.json", metadata)
    write_json(run_dir / "statistics.json", statistics)
    return DemoResult(
        sample_id=sample_id, model_id=model.id, model_name=model.name, mode="live",
        image_path=None, mask_path=mask_path, overlay_path=overlay_path,
        statistics=statistics, artifacts={"nucleus_mask": nucleus_path}, provenance=metadata,
    )


def _run_unet(image: np.ndarray, sample_id: str, model: ModelSpec, run_dir: Path, device_name: str) -> DemoResult:
    from annotation_workspace.tools.compare_unet_pilots import display_image, load_checkpoint, predict_tile, prediction_statistics
    from src.train import get_device

    run_source = model.checkpoint.parent.parent
    device = get_device(device_name)
    unet, config, checkpoint = load_checkpoint(run_source, device)
    try:
        probabilities = predict_tile(unet, image, device)
    finally:
        del unet
        if device.type == "cuda":
            torch.cuda.empty_cache()
    labels = probabilities.argmax(axis=0).astype(np.uint8)
    mask_path = run_dir / "semantic_mask.tif"
    tifffile.imwrite(mask_path, labels, photometric="minisblack", metadata={"axes": "YX", "review_status": "unverified"})
    base = display_image(image)
    overlay_path = run_dir / "overlay.png"
    _write_png_new(overlay_path, semantic_overlay(base, labels))
    process_probability_path = run_dir / "process_probability.png"
    _write_png_new(process_probability_path, probabilities[2])
    statistics = {
        **prediction_statistics(labels), "review_status": "unverified", "ground_truth": False,
        "finite_probabilities": bool(np.isfinite(probabilities).all()),
        "limitations": config.get("limitations", []),
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "sample_id": sample_id,
        "mode": "live", "model_id": model.id, "model_version": model.version,
        "checkpoint": str(checkpoint), "config_snapshot": str(run_source / "config_snapshot.yaml"),
        "device": str(device), "input_shape": list(image.shape),
        "preprocessing": "Existing per-patch percentile 1/99.8 normalization; GFAP channel 1; no resizing",
        "review_status": "unverified", "ground_truth": False,
    }
    write_json(run_dir / "input_metadata.json", metadata)
    write_json(run_dir / "statistics.json", statistics)
    return DemoResult(
        sample_id=sample_id, model_id=model.id, model_name=model.name, mode="live",
        image_path=None, mask_path=mask_path, overlay_path=overlay_path,
        statistics=statistics, artifacts={"process_probability": process_probability_path}, provenance=metadata,
    )

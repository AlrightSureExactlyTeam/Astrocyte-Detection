"""Thin Demo adapter for the existing, fixed V3.3-B.2 inference pipeline.

All detection, tracing, association, and width reconstruction remains in the
original implementations. This module only orchestrates them and writes new,
unverified Demo artifacts into an already-created unique run directory.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import tifffile
import yaml

from .images import encode_png
from .registry import ModelSpec, PROJECT_ROOT, resolve_path
from .reports import write_json
from .results import DemoResult

ROI_SIZE = 256
V31_PATCH_SIZE = 256
V31_STRIDE = 128
CELLPOSE_PARAMETERS = {
    "normalize": False,
    "diameter": None,
    "min_size": 15,
    "max_size_fraction": 0.4,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cellpose_checkpoint(model: ModelSpec) -> Path:
    value = model.expected_input.get("cellpose_checkpoint")
    if not value:
        raise ValueError("V3.3-B.2 registry entry lacks cellpose_checkpoint")
    return resolve_path(value)


def validate_v33b2_dependencies(model: ModelSpec) -> dict[str, Path]:
    """Resolve and verify both fixed checkpoints without loading either model."""
    v3_checkpoint = model.checkpoint
    cellpose_checkpoint = _cellpose_checkpoint(model)
    for name, path in (("V3", v3_checkpoint), ("Cellpose CPSAM", cellpose_checkpoint)):
        if not path.is_file():
            raise FileNotFoundError(f"Required {name} checkpoint not found: {path}")
    config_path = v3_checkpoint.parent.parent / "config_snapshot.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Required V3 configuration not found: {config_path}")
    for name, path, key in (
        ("V3", v3_checkpoint, "v3_checkpoint_sha256"),
        ("Cellpose CPSAM", cellpose_checkpoint, "cellpose_checkpoint_sha256"),
    ):
        expected = model.expected_input.get(key)
        if expected and _sha256(path) != expected:
            raise ValueError(f"{name} checkpoint checksum does not match the accepted pilot: {path}")
    return {
        "v3_checkpoint": v3_checkpoint,
        "cellpose_checkpoint": cellpose_checkpoint,
        "v3_config": config_path,
    }


@lru_cache(maxsize=2)
def _load_models(
    v3_checkpoint: str,
    cellpose_checkpoint: str,
    v3_config: str,
    device_name: str,
):
    """Cache only the selected B.2 dependency models across Streamlit reruns."""
    workspace = PROJECT_ROOT / "annotation_workspace"
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    import torch
    from cellpose import models
    from gfap_annotation.v3_model import create_v3_model
    from gfap_annotation.v3_training import load_v3_checkpoint
    from src.train import get_device

    device = get_device(device_name)
    config = yaml.safe_load(Path(v3_config).read_text(encoding="utf-8"))
    v3_model = create_v3_model(
        in_channels=int(config["data"]["input_channels"]),
        base_channels=int(config["model"]["base_channels"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
    )
    load_v3_checkpoint(v3_model, Path(v3_checkpoint), device)
    v3_model = v3_model.to(device).eval()
    cellpose_model = models.CellposeModel(
        gpu=device.type == "cuda", pretrained_model=cellpose_checkpoint,
    )
    return cellpose_model, v3_model, device


def _validate_input(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[-1] < 2:
        raise ValueError("V3.3-B.2 requires an HWC image with GFAP in channel 1")
    if min(image.shape[:2]) < V31_PATCH_SIZE:
        raise ValueError(
            f"V3.3-B.2 requires height and width of at least {V31_PATCH_SIZE} pixels; "
            f"got {image.shape[:2]}. No resizing is performed."
        )
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise ValueError("Input intensities must be finite numeric values")


def _run_pipeline(image: np.ndarray, cellpose_model, v3_model, device) -> dict[str, Any]:
    """Invoke the accepted pipeline stages with their original configurations."""
    from gfap_annotation.v31_inference import predict_v3_overlapping
    from gfap_annotation.v32_tracing import TraceConfig, select_soma_candidates, trace_tile
    from gfap_annotation.v33a_tracing import ThinTraceConfig, build_thin_candidates, trace_thin_tile
    from gfap_annotation.v33a_width import reconstruct_adaptive
    from gfap_annotation.v33b2_unified import UnifiedConfig, recognize_unified_process
    from gfap_annotation.v33b_association import AssociationConfig
    from gfap_annotation.workflow import _validate_prediction, normalize_for_cellpose

    _validate_input(image)
    trace_config = TraceConfig()
    thin_config = ThinTraceConfig()
    unified_config = UnifiedConfig()
    evidence_config = AssociationConfig()
    if unified_config.radius_cap != 4:
        raise RuntimeError("The accepted adaptive 4 px reconstruction is not configured")

    gfap, normalization = normalize_for_cellpose(image[..., 1])
    raw_cellpose = cellpose_model.eval(gfap, **CELLPOSE_PARAMETERS)
    cellpose = _validate_prediction(
        raw_cellpose[0] if isinstance(raw_cellpose, (tuple, list)) else raw_cellpose,
        image.shape[:2],
    )
    probability = predict_v3_overlapping(
        v3_model, image, device, V31_PATCH_SIZE, V31_STRIDE,
    )
    expected_shape = (2, *image.shape[:2])
    if probability.shape != expected_shape or probability.dtype != np.float32:
        raise RuntimeError(f"Invalid V3.1 probability output: {probability.shape} {probability.dtype}")
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=0), 1.0, atol=1e-5):
        raise RuntimeError("V3.1 probabilities are non-finite or not normalized")

    soma, records = select_soma_candidates(cellpose, gfap, probability[0], trace_config)
    thin = build_thin_candidates(gfap, soma, thin_config, method="sato")
    thin_result = trace_thin_tile(
        gfap, probability[1], soma, records, thin, ROI_SIZE, thin_config,
    )
    baseline32 = trace_tile(
        gfap, probability[1], soma, records, ROI_SIZE, "E1", trace_config,
    )
    forbidden = (thin_result["state"] > 1) | (baseline32["state"] > 1)
    approved = reconstruct_adaptive(
        gfap,
        thin["structural_candidate"],
        thin_result["assigned_centerline"],
        thin_result["instance"],
        soma,
        forbidden,
        unified_config.radius_cap,
    )
    result = recognize_unified_process(
        gfap,
        probability[1],
        thin["ridge"],
        soma,
        thin["structural_candidate"],
        thin_result["assigned_centerline"],
        approved["reconstructed_mask"],
        records,
        ROI_SIZE,
        thin_config,
        unified_config,
        evidence_config,
        thin["candidate_centerline"],
    )
    return {
        **result,
        "cellpose": cellpose,
        "soma": soma,
        "candidate_records": records,
        "probability": probability,
        "normalization": normalization,
        "configs": {
            "soma_selection": asdict(trace_config),
            "thin_tracing": asdict(thin_config),
            "unified": asdict(unified_config),
            "structural_evidence": asdict(evidence_config),
        },
    }


def _write_png(path: Path, image: np.ndarray) -> None:
    with path.open("xb") as handle:
        handle.write(encode_png(image))


def _write_tiff(path: Path, array: np.ndarray, description: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Demo artifact: {path}")
    tifffile.imwrite(
        path, array, photometric="minisblack",
        metadata={"axes": "YX", "description": description, "review_status": "unverified"},
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else ["sample_id", "status"]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_v33b2_inference(
    image: np.ndarray | str | Path,
    sample_id: str,
    model: ModelSpec,
    run_dir: Path,
    device_name: str = "auto",
) -> DemoResult:
    """Run one uploaded array or read-only image path; never modify the source."""
    if isinstance(image, (str, Path)):
        from .images import SUPPORTED_EXTENSIONS
        from src.preprocessing import load_image
        image_path = Path(image)
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Unsupported image format; use PNG, TIF, or TIFF")
        image = load_image(image_path)
    paths = validate_v33b2_dependencies(model)
    _validate_input(image)
    cellpose_model, v3_model, device = _load_models(
        str(paths["v3_checkpoint"]),
        str(paths["cellpose_checkpoint"]),
        str(paths["v3_config"]),
        device_name,
    )
    started = perf_counter()
    prediction = _run_pipeline(image, cellpose_model, v3_model, device)
    elapsed = perf_counter() - started

    from annotation_workspace.tools.compare_gfap_models import display_image
    from annotation_workspace.tools.run_v33b2_unified_pilot import integrated_overlay
    from annotation_workspace.tools.run_v33b_pilot import state_overlay

    state = prediction["state"]
    instance = prediction["instance"]
    soma = prediction["soma"]
    base = np.rint(display_image(image) * 255).astype(np.uint8)
    mask_path = run_dir / "integrated_instance_mask.tif"
    state_path = run_dir / "process_state.tif"
    geometry_path = run_dir / "unified_process_geometry.tif"
    soma_path = run_dir / "soma_candidates.tif"
    added_path = run_dir / "added_process_geometry.tif"
    overlay_path = run_dir / "final_integrated_overlay.png"
    assigned_overlay_path = run_dir / "assigned_only_overlay.png"
    units_path = run_dir / "structural_unit_records.csv"
    candidates_path = run_dir / "soma_candidate_records.csv"
    _write_tiff(mask_path, instance, "Original Cellpose Soma IDs plus ASSIGNED Process only")
    _write_tiff(state_path, state, "1 ASSIGNED; 2 AMBIGUOUS; 3 UNASSIGNED; 4 REJECTED")
    _write_tiff(geometry_path, prediction["geometry"].astype(np.uint8), "Unified adaptive 4 px Process geometry")
    _write_tiff(soma_path, soma, "Selected Cellpose Soma candidates; original IDs preserved")
    _write_tiff(added_path, prediction["added"].astype(np.uint8), "Exploratory added B.2 geometry; unverified")
    _write_png(overlay_path, integrated_overlay(base, soma, state, instance))
    _write_png(assigned_overlay_path, state_overlay(base, soma, state, instance, "assigned"))
    _write_csv(units_path, prediction["unit_records"])
    _write_csv(candidates_path, prediction["candidate_records"])

    statistics = {
        "sample_id": sample_id,
        **prediction["stats"],
        "cellpose_candidate_regions": len(prediction["candidate_records"]),
        "selected_soma_candidates": sum(bool(row["selected"]) for row in prediction["candidate_records"]),
        "adaptive_reconstruction_radius_cap_px": 4,
        "roi_size_px": ROI_SIZE,
        "inference_seconds": round(elapsed, 3),
        "review_status": "unverified",
        "ground_truth": False,
        "candidate_counts_are_biological_cell_counts": False,
        "biological_instance_ownership_validated": False,
        "unlabeled_pixels_are_background": False,
    }
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": sample_id,
        "mode": "live",
        "model_id": model.id,
        "model_version": model.version,
        "algorithm_entry": "gfap_annotation.v33b2_unified.recognize_unified_process",
        "adapter": "demo_core.v33b2_adapter.run_v33b2_inference",
        "v3_checkpoint": str(paths["v3_checkpoint"]),
        "v3_checkpoint_sha256": _sha256(paths["v3_checkpoint"]),
        "cellpose_checkpoint": str(paths["cellpose_checkpoint"]),
        "cellpose_checkpoint_sha256": _sha256(paths["cellpose_checkpoint"]),
        "device": str(device),
        "input_shape": list(image.shape),
        "gfap_channel": 1,
        "gfap_normalization": prediction["normalization"],
        "cellpose_parameters": CELLPOSE_PARAMETERS,
        "v31_patch_size": V31_PATCH_SIZE,
        "v31_stride": V31_STRIDE,
        "fixed_configurations": prediction["configs"],
        "review_status": "unverified",
        "ground_truth": False,
        "training_performed": False,
        "external_api_called": False,
    }
    write_json(run_dir / "input_metadata.json", provenance)
    write_json(run_dir / "statistics.json", statistics)
    return DemoResult(
        sample_id=sample_id,
        model_id=model.id,
        model_name=model.name,
        mode="live",
        image_path=None,
        mask_path=mask_path,
        overlay_path=overlay_path,
        statistics=statistics,
        artifacts={
            "process_state": state_path,
            "unified_process_geometry": geometry_path,
            "soma_candidates": soma_path,
            "added_process_geometry": added_path,
            "assigned_overlay": assigned_overlay_path,
            "structural_unit_records": units_path,
            "soma_candidate_records": candidates_path,
        },
        provenance=provenance,
    )

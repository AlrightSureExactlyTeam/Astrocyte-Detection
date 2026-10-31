"""Prepare clearly labeled, center-only pseudo-label datasets without image duplication."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .batch import deep_validate_prediction, prediction_metadata
from .workflow import (
    PROJECT_ROOT,
    WORKSPACE,
    read_manifest,
    read_mask,
    relative_path,
    resolve_path,
    utc_now,
    write_json_new,
    write_manifest,
)


def candidate_centers(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Convert instance regions to candidate center points, not semantic truth."""
    centers = []
    for instance_id in (int(value) for value in np.unique(mask) if value > 0):
        ys, xs = np.nonzero(mask == instance_id)
        centers.append((instance_id, int(round(float(xs.mean()))), int(round(float(ys.mean())))))
    return centers


def _link_exclusive(source: Path, destination: Path, created: list[Path]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        raise FileExistsError(f"Pseudo-label output conflict: {destination}")
    except OSError as exc:
        raise OSError(
            f"Could not hard-link {source} to {destination}. The dataset uses links to avoid "
            "duplicating large images; keep source and destination on the same volume."
        ) from exc
    created.append(destination)


def _write_centers_new(path: Path, centers: list[tuple[int, int, int]], created: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "x", "y", "status"])
        for instance_id, x, y in centers:
            writer.writerow([instance_id, x, y, "candidate_unverified"])
    created.append(path)


def _quality_flags(image_id: str) -> str:
    path = WORKSPACE / "matching_issues.csv"
    if not path.is_file():
        return "not_checked"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        issues = sorted({row["issue_type"] for row in csv.DictReader(handle) if row["image_id"] == image_id})
    return ";".join(issues) if issues else "none"


def prepare_pseudo_dataset(
    manifest_path: str | Path,
    batch: str,
    destination: str | Path = PROJECT_ROOT / "datasets" / "GFAP_pseudo_labeled",
    append: bool = False,
) -> dict:
    """Export valid automatic predictions as unverified center-point pseudo-labels.

    Invalid tiles are reported and skipped. No semantic soma/process labels are inferred.
    Existing outputs are never overwritten; appending requires an explicit flag.
    """
    destination = resolve_path(destination)
    if destination.name == "GFAP_labeled":
        raise ValueError("Pseudo-labels must never be exported into datasets/GFAP_labeled")
    rows = read_manifest(manifest_path)
    candidates = [
        row for row in rows
        if row.get("source_batch") == batch
        and row.get("annotation_status") == "predicted"
        and row.get("annotation_method") == "pretrained_cellpose"
    ]
    if not candidates:
        return {
            "prepared": False,
            "reason": "no_automatic_predictions_available",
            "batch": batch,
            "exported_count": 0,
            "invalid_tiles": [],
        }
    valid = []
    invalid = []
    for row in candidates:
        issues = deep_validate_prediction(row)
        if issues:
            invalid.append({"image_id": row["image_id"], "issues": issues})
        else:
            valid.append(row)
    dataset_manifest = destination / "manifest.csv"
    existing = []
    if dataset_manifest.exists():
        if not append:
            raise FileExistsError(f"Pseudo-label manifest exists; explicit append required: {dataset_manifest}")
        existing = read_manifest(dataset_manifest)
    existing_ids = {row["image_id"] for row in existing}
    if any(row["image_id"] in existing_ids for row in valid):
        raise ValueError("A pseudo-label image_id already exists in the destination")
    operations = []
    center_operations = []
    export_rows = []
    version_material = []
    for row in valid:
        source = resolve_path(row["original_image_path"])
        cell_source = resolve_path(row["cell_mask_path"])
        cell_mask = read_mask(cell_source)
        _, metadata = prediction_metadata(row)
        slide = row["source_slide"]
        stem = Path(row["original_filename"]).stem
        image_target = destination / "images" / slide / row["original_filename"]
        annotation_target = destination / "annotations" / slide / f"{stem}.csv"
        cell_target = destination / "cell_masks" / slide / f"{stem}.tif"
        nucleus_target = destination / "nucleus_masks" / slide / f"{stem}.tif"
        targets = [image_target, annotation_target, cell_target]
        if row["nucleus_mask_path"]:
            targets.append(nucleus_target)
        conflict = next((path for path in targets if path.exists()), None)
        if conflict:
            raise FileExistsError(f"Pseudo-label output conflict: {conflict}")
        exported = dict(row)
        exported.update({
            "source_image_path": row["original_image_path"],
            "original_image_path": relative_path(image_target),
            "cell_mask_path": relative_path(cell_target),
            "nucleus_mask_path": relative_path(nucleus_target) if row["nucleus_mask_path"] else "",
            "annotation_status": "pseudo_label_unverified",
            "annotation_method": "cellpose_center_pseudo_label",
            "review_status": "unverified",
            "annotation_points_path": relative_path(annotation_target),
            "source_prediction_path": relative_path(cell_source.parent),
            "prediction_model": metadata["model"],
            "prediction_model_version": metadata["cellpose_version"],
            "prediction_timestamp": metadata["created_utc"],
            "quality_control_flags": _quality_flags(row["image_id"]),
            "training_target": "center_points_only",
            "semantic_labels_inferred": "false",
        })
        export_rows.append(exported)
        operations.extend([(source, image_target), (cell_source, cell_target)])
        if row["nucleus_mask_path"]:
            operations.append((resolve_path(row["nucleus_mask_path"]), nucleus_target))
        center_operations.append((annotation_target, candidate_centers(cell_mask)))
        version_material.append({
            "image_id": row["image_id"], "source_sha256": row["source_sha256"],
            "prediction_revision": row["prediction_revision"],
            "prediction_timestamp": metadata["created_utc"],
        })
    if not valid:
        return {
            "prepared": False,
            "reason": "all_predictions_failed_validation",
            "batch": batch,
            "exported_count": 0,
            "invalid_tiles": invalid,
        }
    dataset_version = hashlib.sha256(
        json.dumps(version_material, sort_keys=True).encode("utf-8")
    ).hexdigest()
    for row in export_rows:
        row["dataset_version"] = dataset_version
    created = []
    try:
        for source, target in operations:
            _link_exclusive(source, target, created)
        for target, centers in center_operations:
            _write_centers_new(target, centers, created)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        version_path = destination / f"dataset_version_{stamp}_{uuid.uuid4().hex[:6]}.json"
        write_json_new(version_path, {
            "created_utc": utc_now(),
            "dataset_version": dataset_version,
            "batch": batch,
            "exported_count": len(export_rows),
            "training_target": "center_points_only",
            "review_status": "unverified",
            "ground_truth": False,
            "image_storage": "hard_links_to_immutable_source_and_prediction_files",
            "invalid_tiles": invalid,
        })
        created.append(version_path)
        # Publish the manifest last: its linked files and version metadata must exist first.
        write_manifest(dataset_manifest, existing + export_rows, preserve_history=append)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return {
        "prepared": True,
        "batch": batch,
        "dataset_path": relative_path(destination),
        "dataset_version": dataset_version,
        "exported_count": len(export_rows),
        "invalid_tiles": invalid,
        "training_target": "center_points_only",
        "review_status": "unverified",
        "semantic_labels_inferred": False,
    }

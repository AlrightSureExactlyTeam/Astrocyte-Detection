"""Batch-level deep validation and reproducible experiment reporting."""
from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .workflow import (
    PROJECT_ROOT,
    WORKSPACE,
    read_manifest,
    relative_path,
    resolve_path,
    sha256,
    utc_now,
    validate_rows,
    write_json_new,
    write_manifest,
)


def prediction_metadata(row: dict[str, str]) -> tuple[Path | None, dict | None]:
    if not row.get("cell_mask_path"):
        return None, None
    path = resolve_path(row["cell_mask_path"]).parent / "prediction.json"
    if not path.is_file():
        return path, None
    try:
        with path.open(encoding="utf-8") as handle:
            return path, json.load(handle)
    except (OSError, json.JSONDecodeError):
        return path, None


def deep_validate_prediction(row: dict[str, str]) -> list[str]:
    """Return integrity/provenance failures for one automatic prediction."""
    image_id = row.get("image_id", "")
    issues = [item["issue"] for item in validate_rows([row])]
    source = resolve_path(row["original_image_path"])
    if source.is_file() and row.get("source_sha256"):
        if sha256(source) != row["source_sha256"]:
            issues.append("source_checksum_mismatch")
    else:
        issues.append("missing_source_checksum")
    if not row.get("source_batch"):
        issues.append("missing_source_batch")
    try:
        channels = json.loads(row["channel_names"])
        cell_index = int(row["cell_channel"])
        if not 0 <= cell_index < len(channels):
            issues.append("invalid_cell_channel")
        nucleus_index = int(row["nucleus_channel"]) if row.get("nucleus_channel") else None
        if nucleus_index is not None and not 0 <= nucleus_index < len(channels):
            issues.append("invalid_nucleus_channel")
        if nucleus_index is not None and nucleus_index == cell_index:
            issues.append("cell_and_nucleus_channels_must_differ")
    except (ValueError, TypeError, json.JSONDecodeError):
        issues.append("invalid_channel_metadata")
    metadata_path, metadata = prediction_metadata(row)
    if metadata_path is None or metadata is None:
        issues.append("missing_or_corrupted_prediction_metadata")
        return sorted(set(issues))
    required = {
        "created_utc", "model", "model_type", "cellpose_version", "source_image",
        "source_sha256", "ground_truth", "cell_channel_index", "nucleus_channel_index",
        "channel_names", "parameters", "cell_normalization",
    }
    missing = sorted(required - set(metadata))
    if missing:
        issues.append(f"incomplete_prediction_metadata:{','.join(missing)}")
    if metadata.get("ground_truth") is not False or metadata.get("review_status") != "unverified":
        issues.append("automatic_prediction_incorrectly_marked_verified")
    if metadata.get("model_type") != "pretrained":
        issues.append("prediction_model_is_not_marked_pretrained")
    if metadata.get("source_sha256") != row.get("source_sha256"):
        issues.append("prediction_source_checksum_mismatch")
    if metadata.get("source_image") != row.get("original_image_path"):
        issues.append("prediction_source_path_mismatch")
    if metadata.get("source_batch") != row.get("source_batch"):
        issues.append("prediction_batch_mismatch")
    if metadata.get("cell_channel_index") != int(row["cell_channel"]):
        issues.append("prediction_cell_channel_mismatch")
    expected_nucleus = int(row["nucleus_channel"]) if row.get("nucleus_channel") else None
    if metadata.get("nucleus_channel_index") != expected_nucleus:
        issues.append("prediction_nucleus_channel_mismatch")
    return sorted(set(issues))


def deep_validate_batch(rows: list[dict[str, str]], batch: str) -> dict:
    selected = [row for row in rows if row.get("source_batch") == batch]
    predicted = [row for row in selected if row.get("annotation_status") == "predicted"]
    failures = []
    valid_ids = []
    for row in predicted:
        issues = deep_validate_prediction(row)
        if issues:
            failures.append({"image_id": row["image_id"], "issues": issues})
        else:
            valid_ids.append(row["image_id"])
    return {
        "batch": batch,
        "registered_count": len(selected),
        "predicted_count": len(predicted),
        "valid_prediction_count": len(valid_ids),
        "valid_image_ids": valid_ids,
        "failures": failures,
    }


def _new_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_if_present(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def generate_batch_report(
    batch: str,
    manifest_path: str | Path,
    processing_seconds: float,
    failed: list[dict] | None = None,
    pending: list[str] | None = None,
    skipped: list[str] | None = None,
    requested_model: dict | None = None,
    checkpoint_info: dict | None = None,
    pseudo_dataset_info: dict | None = None,
) -> tuple[dict, Path]:
    """Write a new report run; never overwrite an earlier batch report."""
    failed = failed or []
    pending = pending or []
    skipped = skipped or []
    rows = read_manifest(manifest_path)
    batch_rows = [row for row in rows if row.get("source_batch") == batch]
    validation = deep_validate_batch(batch_rows, batch)
    batch_root = WORKSPACE / "cellpose_predictions" / batch
    write_manifest(batch_root / "prediction_manifest.csv", batch_rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    run_dir = PROJECT_ROOT / "results" / "gfap_experiments" / batch / f"run_{stamp}_{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    valid_ids = set(validation["valid_image_ids"])
    valid_rows = [row for row in batch_rows if row["image_id"] in valid_ids]
    stats = [row for row in _read_csv_if_present(WORKSPACE / "statistics.csv") if row["image_id"] in valid_ids]
    models = []
    for row in valid_rows:
        _, metadata = prediction_metadata(row)
        if metadata:
            models.append({
                "model": metadata.get("model"),
                "cellpose_version": metadata.get("cellpose_version"),
                "model_type": metadata.get("model_type"),
                "prediction_timestamp": metadata.get("created_utc"),
                "gpu_requested": metadata.get("gpu_requested"),
            })
    totals = {
        "candidate_cell_count": sum(int(row["candidate_cell_count"] or 0) for row in valid_rows),
        "candidate_nucleus_count": sum(int(row["candidate_nucleus_count"] or 0) for row in valid_rows),
        "possible_one_to_one_pairs": sum(int(row.get("possible_one_to_one_pairs", 0) or 0) for row in stats),
        "matching_issue_count": sum(int(row.get("matching_issue_count", 0) or 0) for row in stats),
    }
    status = "completed" if valid_rows and not failed and not pending else (
        "completed_with_failures" if valid_rows else "no_valid_predictions"
    )
    report = {
        "created_utc": utc_now(),
        "batch": batch,
        "status": status,
        "processing_seconds": round(float(processing_seconds), 3),
        "registered_tile_count": len(batch_rows),
        "valid_prediction_count": len(valid_rows),
        "pending_tiles": pending,
        "skipped_completed_tiles": skipped,
        "failed_tiles": failed,
        "candidate_statistics": totals,
        "prediction_models": models or [requested_model] if requested_model else models,
        "dataset_validation": validation,
        "pseudo_dataset": pseudo_dataset_info or {"prepared": False},
        "checkpoint": checkpoint_info or {"training_started": False, "reason": "not_requested"},
        "evaluation_metrics": {},
        "evaluation_note": "No precision, recall, Dice, or F1 is reported without verified reference annotations.",
        "automatic_annotations_are_ground_truth": False,
    }
    validation_report = {
        "created_utc": utc_now(),
        **validation,
        "pending_tiles": pending,
        "failed_tiles": failed,
    }
    write_json_new(run_dir / "validation_report.json", validation_report)
    write_json_new(run_dir / "report.json", report)
    _new_csv(run_dir / "failed_tiles.csv", failed, ["image_id", "error"])
    _new_csv(run_dir / "data_manifest.csv", valid_rows, list(valid_rows[0].keys()) if valid_rows else ["image_id"])
    lines = [
        f"# GFAP batch report: {batch}",
        "",
        f"- Status: **{status}**",
        f"- Processing time: {report['processing_seconds']} seconds",
        f"- Registered tiles: {len(batch_rows)}",
        f"- Valid automatic predictions: {len(valid_rows)}",
        f"- Pending tiles: {len(pending)}",
        f"- Failed tiles: {len(failed)}",
        f"- Candidate cells: {totals['candidate_cell_count']}",
        f"- Candidate nuclei: {totals['candidate_nucleus_count']}",
        f"- Possible one-to-one spatial pairs: {totals['possible_one_to_one_pairs']}",
        f"- Spatial matching issues: {totals['matching_issue_count']}",
        f"- Training started: {report['checkpoint'].get('training_started', False)}",
        "",
        "All automatic annotations are unverified candidates, not ground truth.",
        "Counts are prediction statistics, not accuracy metrics.",
        report["evaluation_note"],
        "",
    ]
    if failed:
        lines.extend(["## Failed tiles", ""])
        lines.extend(f"- `{item['image_id']}`: {item['error']}" for item in failed)
        lines.append("")
    if pending:
        lines.extend(["## Pending tiles", ""])
        lines.extend(f"- `{image_id}`" for image_id in pending)
        lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report, run_dir

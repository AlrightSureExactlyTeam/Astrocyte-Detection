"""Core operations for independent, unverified GFAP candidate annotation."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import tifffile

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE.parent
REQUIRED_COLUMNS = [
    "image_id",
    "source_slide",
    "original_image_path",
    "cell_mask_path",
    "nucleus_mask_path",
    "annotation_status",
    "annotation_method",
    "review_status",
]
EXTRA_COLUMNS = [
    "original_filename",
    "original_rgb_path",
    "source_batch",
    "source_export_root",
    "tile_x",
    "tile_y",
    "tile_width",
    "tile_height",
    "channel_names",
    "cell_channel",
    "nucleus_channel",
    "pixel_size_um",
    "source_sha256",
    "prediction_revision",
    "candidate_cell_count",
    "candidate_nucleus_count",
    "reviewer",
    "review_notes",
    "source_image_path",
]
MANIFEST_COLUMNS = REQUIRED_COLUMNS + EXTRA_COLUMNS
STAT_COLUMNS = [
    "image_id",
    "prediction_revision",
    "candidate_cell_count",
    "candidate_nucleus_count",
    "possible_one_to_one_pairs",
    "cells_without_nucleus",
    "cells_with_multiple_nuclei",
    "nuclei_outside_cells",
    "nuclei_overlapping_multiple_cells",
    "matching_issue_count",
    "review_status",
]
ISSUE_COLUMNS = ["image_id", "issue_type", "cell_id", "nucleus_id", "details"]
LINK_COLUMNS = [
    "image_id",
    "cell_id",
    "nucleus_id",
    "overlap_pixels",
    "candidate_relationship",
    "review_status",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def relative_path(path: str | Path) -> str:
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")
        rows = list(reader)
    ids = [row["image_id"] for row in rows]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate image identifiers: {duplicates}")
    if any(not value for value in ids):
        raise ValueError("Every manifest row needs a non-empty image_id")
    return rows


def _atomic_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_manifest(path: str | Path, rows: list[dict[str, str]], preserve_history: bool = True) -> None:
    path = resolve_path(path)
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Refusing to write duplicate image identifiers")
    if preserve_history and path.is_file():
        history = path.parent / "metadata" / "manifest_history"
        history.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        backup = history / f"{path.stem}_{stamp}_{uuid.uuid4().hex[:6]}.csv"
        with backup.open("xb") as destination, path.open("rb") as source:
            shutil.copyfileobj(source, destination)
    columns = MANIFEST_COLUMNS + sorted(
        set().union(*(row.keys() for row in rows)) - set(MANIFEST_COLUMNS)
    ) if rows else MANIFEST_COLUMNS
    _atomic_csv(path, rows, columns)


def write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def upsert_csv(
    path: Path,
    new_rows: list[dict],
    columns: list[str],
    key: str = "image_id",
    replace_ids: Iterable[str] | None = None,
) -> None:
    existing = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            existing = list(csv.DictReader(handle))
    replaced = set(replace_ids) if replace_ids is not None else {row[key] for row in new_rows}
    retained = [row for row in existing if row.get(key) not in replaced]
    _atomic_csv(path, retained + new_rows, columns)


def _parse_description(description: str | None) -> dict:
    if not description:
        return {}
    try:
        return json.loads(description)
    except (json.JSONDecodeError, TypeError):
        return {}


def read_cyx_image(path: str | Path) -> tuple[np.ndarray, dict]:
    path = resolve_path(path)
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"No TIFF series found: {path}")
        series = tif.series[0]
        axes = series.axes
        shape = tuple(series.shape)
        description = _parse_description(series.pages[0].description)
        image = series.asarray()
    if axes != "CYX" or image.ndim != 3:
        raise ValueError(f"Expected explicit CYX quantitative TIFF, got {axes} {shape}: {path}")
    nested = _parse_description(description.get("Description"))
    return image, {"axes": axes, "shape": shape, "metadata": description, "tile": nested}


def spatial_shape(path: str | Path) -> tuple[int, int]:
    path = resolve_path(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes == "CYX":
            return int(series.shape[1]), int(series.shape[2])
        if series.axes == "YX":
            return int(series.shape[0]), int(series.shape[1])
        raise ValueError(f"Unsupported TIFF axes {series.axes}: {path}")


def read_mask(path: str | Path) -> np.ndarray:
    path = resolve_path(path)
    with tifffile.TiffFile(path) as tif:
        if len(tif.series) != 1 or tif.series[0].axes != "YX":
            raise ValueError(f"Mask must be a single 2D YX TIFF: {path}")
        mask = tif.series[0].asarray()
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"Mask must be a 2D integer label image: {path}")
    if np.any(mask < 0) or (mask.size and int(mask.max()) > np.iinfo(np.uint32).max):
        raise ValueError(f"Mask IDs must be nonnegative uint32-range integers: {path}")
    return mask


def write_mask_new(path: Path, mask: np.ndarray, image_id: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite mask: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path, mask.astype(np.uint32, copy=False), photometric="minisblack",
        metadata={"axes": "YX", "image_id": image_id, "review_status": "unverified"},
    )


def import_tiles(
    export_root: str | Path,
    tile_names: Iterable[str],
    source_slide: str,
    cell_channel_name: str,
    nucleus_channel_name: str | None,
    manifest_path: str | Path = WORKSPACE / "manifest.csv",
    source_batch: str | None = None,
) -> int:
    """Register explicitly named tiles only; never scans or processes a whole export."""
    export_root = resolve_path(export_root)
    tiles_dir = export_root / "tiles"
    rgb_dir = export_root / "rgb_tiles"
    with (export_root / "inspection.json").open(encoding="utf-8") as handle:
        inspection = json.load(handle)
    with (export_root / "tiles.csv").open(newline="", encoding="utf-8-sig") as handle:
        tile_rows = list(csv.DictReader(handle))
    by_xy = {}
    for tile_row in tile_rows:
        key = (int(tile_row["x"]), int(tile_row["y"]))
        if key in by_xy:
            raise ValueError(f"Duplicate tile coordinates in tiles.csv: {key}")
        by_xy[key] = tile_row
    existing = read_manifest(manifest_path) if Path(manifest_path).exists() else []
    existing_ids = {row["image_id"] for row in existing}
    new_rows = []
    names = list(tile_names)
    if not names:
        raise ValueError("Select at least one tile explicitly")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate tile names in import selection")
    for filename in names:
        if Path(filename).name != filename:
            raise ValueError(f"Tile selection must be a filename, not a path: {filename}")
        path = tiles_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Selected tile not found: {path}")
        image, info = read_cyx_image(path)
        channel_names = info["metadata"].get("Channel", {}).get("Name", inspection.get("channels"))
        if channel_names != inspection.get("channels"):
            raise ValueError(f"TIFF and inspection channel metadata disagree: {filename}")
        if cell_channel_name not in channel_names:
            raise ValueError(f"Unknown cell channel {cell_channel_name!r}; available={channel_names}")
        nucleus_index = None
        if nucleus_channel_name:
            if nucleus_channel_name not in channel_names:
                raise ValueError(f"Unknown nucleus channel {nucleus_channel_name!r}; available={channel_names}")
            nucleus_index = channel_names.index(nucleus_channel_name)
        tile = info["tile"]
        if not tile or "x" not in tile or "y" not in tile:
            raise ValueError(f"TIFF lacks explicit source tile coordinates: {filename}")
        xy = (int(tile["x"]), int(tile["y"]))
        if xy not in by_xy:
            raise ValueError(f"TIFF coordinates absent from tiles.csv: {filename} {xy}")
        tile_row = by_xy[xy]
        height, width = image.shape[1:]
        expected = (int(tile_row["height"]), int(tile_row["width"]))
        if (height, width) != expected:
            raise ValueError(f"TIFF/tiles.csv dimension mismatch for {filename}: {(height, width)} != {expected}")
        if tile_row["keep"].lower() != "true":
            raise ValueError(f"Selected tile is not marked keep=True: {filename}")
        safe_slide = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_slide).strip("._")
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
        image_id = f"{safe_slide}__{safe_stem}"
        if not safe_slide or not safe_stem or not re.fullmatch(r"[A-Za-z0-9_.-]+", image_id):
            raise ValueError(f"Could not create a safe image identifier for {filename}")
        if image_id in existing_ids:
            raise ValueError(f"Duplicate image identifier: {image_id}")
        rgb_path = rgb_dir / f"{path.stem}_RGB.tif"
        row = {name: "" for name in MANIFEST_COLUMNS}
        row.update({
            "image_id": image_id,
            "source_slide": source_slide,
            "original_image_path": relative_path(path),
            "cell_mask_path": "",
            "nucleus_mask_path": "",
            "annotation_status": "imported",
            "annotation_method": "none",
            "review_status": "unverified",
            "original_filename": filename,
            "original_rgb_path": relative_path(rgb_path) if rgb_path.is_file() else "",
            "source_batch": source_batch or export_root.name,
            "source_export_root": relative_path(export_root),
            "tile_x": str(xy[0]),
            "tile_y": str(xy[1]),
            "tile_width": str(width),
            "tile_height": str(height),
            "channel_names": json.dumps(channel_names, ensure_ascii=False),
            "cell_channel": str(channel_names.index(cell_channel_name)),
            "nucleus_channel": "" if nucleus_index is None else str(nucleus_index),
            "pixel_size_um": str(info["metadata"].get("PhysicalSizeX", inspection.get("pixel_size_um", ""))),
            "source_sha256": sha256(path),
        })
        new_rows.append(row)
        existing_ids.add(image_id)
    write_manifest(manifest_path, existing + new_rows)
    return len(new_rows)


def normalize_for_cellpose(channel: np.ndarray) -> tuple[np.ndarray, dict]:
    """Match the existing baseline's documented percentile preprocessing independently."""
    values = np.asarray(channel)
    if values.ndim != 2 or values.size == 0 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"Expected a nonempty 2D numeric channel, got {values.shape} {values.dtype}")
    values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Image channel contains NaN or infinite values")
    low, high = np.percentile(values, [1, 99.8])
    if high <= low:
        high = float(values.max())
    scaled = np.zeros_like(values) if high <= low else np.clip((values - low) / (high - low), 0, 1)
    return scaled.astype(np.float32, copy=False), {
        "method": "percentile_clip", "low_percentile": 1.0,
        "high_percentile": 99.8, "low": float(low), "high": float(high),
        "fallback_to_max_when_sparse": True,
    }


def _next_revision(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        candidate = parent / f"{prefix}_{index:03d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"No available revision directory in {parent}")


def _validate_prediction(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape != shape:
        raise ValueError(f"Cellpose mask shape {mask.shape} does not match image {shape}")
    if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise ValueError("Cellpose returned invalid instance IDs")
    if mask.size and int(mask.max()) > np.iinfo(np.uint32).max:
        raise ValueError("Cellpose instance ID exceeds uint32")
    return mask.astype(np.uint32, copy=False)


def _boundaries(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return boundary


def save_overlay(
    image: np.ndarray,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray | None,
    cell_index: int,
    nucleus_index: int | None,
    display_settings: dict,
    path: Path,
) -> None:
    height, width = cell_mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    maxima = display_settings.get("display_max", [255.0] * image.shape[0])
    minima = display_settings.get("display_min", [0.0] * image.shape[0])

    def scaled(index: int) -> np.ndarray:
        low = float(minima[index]) if index < len(minima) else 0.0
        high = float(maxima[index]) if index < len(maxima) else 255.0
        if high <= low:
            high = float(image[index].max()) or 1.0
        return np.clip((image[index].astype(np.float32) - low) / (high - low), 0, 1)

    if nucleus_index is not None:
        rgb[..., 2] = np.round(scaled(nucleus_index) * 255).astype(np.uint8)
    cell_display = scaled(cell_index)
    rgb[..., 0] = np.round(cell_display * 255).astype(np.uint8)
    rgb[..., 1] = np.round(cell_display * 128).astype(np.uint8)
    rgb[_boundaries(cell_mask)] = (0, 255, 0)
    if nucleus_mask is not None:
        rgb[_boundaries(nucleus_mask)] = (255, 0, 255)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (8, 8), (355, 68), (0, 0, 0), thickness=-1)
    cv2.putText(bgr, "Green: candidate cell regions (unverified)", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(bgr, "Magenta: candidate nuclei (unverified)", (16, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 255), 1, cv2.LINE_AA)
    success, encoded = cv2.imencode(".png", bgr)
    if not success:
        raise OSError("Could not encode overlay")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded.tobytes())


def candidate_statistics(
    image_id: str,
    revision: str,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray | None,
) -> tuple[dict, list[dict], list[dict]]:
    cell_ids = [int(v) for v in np.unique(cell_mask) if v > 0]
    nucleus_ids = [] if nucleus_mask is None else [int(v) for v in np.unique(nucleus_mask) if v > 0]
    links = []
    issues = []
    cell_to_nuclei = defaultdict(list)
    nucleus_to_cells = defaultdict(list)
    for nucleus_id in nucleus_ids:
        values, counts = np.unique(cell_mask[nucleus_mask == nucleus_id], return_counts=True)
        overlaps = [(int(cell_id), int(count)) for cell_id, count in zip(values, counts) if cell_id > 0]
        for cell_id, overlap in overlaps:
            cell_to_nuclei[cell_id].append(nucleus_id)
            nucleus_to_cells[nucleus_id].append(cell_id)
            relationship = "unverified_spatial_overlap" if len(overlaps) == 1 else "ambiguous_multiple_cells"
            links.append({
                "image_id": image_id, "cell_id": str(cell_id), "nucleus_id": str(nucleus_id),
                "overlap_pixels": str(overlap), "candidate_relationship": relationship,
                "review_status": "unverified",
            })
        if not overlaps:
            issues.append({
                "image_id": image_id, "issue_type": "nucleus_outside_cell",
                "cell_id": "", "nucleus_id": str(nucleus_id),
                "details": "Candidate nucleus has no spatial overlap with a candidate cell region",
            })
        elif len(overlaps) > 1:
            issues.append({
                "image_id": image_id, "issue_type": "nucleus_overlaps_multiple_cells",
                "cell_id": "", "nucleus_id": str(nucleus_id),
                "details": f"Candidate nucleus overlaps {len(overlaps)} candidate cell regions",
            })
    for cell_id in cell_ids:
        count = len(cell_to_nuclei[cell_id])
        if count == 0:
            issues.append({
                "image_id": image_id, "issue_type": "cell_without_nucleus",
                "cell_id": str(cell_id), "nucleus_id": "",
                "details": "Candidate cell region has no overlapping candidate nucleus",
            })
        elif count > 1:
            issues.append({
                "image_id": image_id, "issue_type": "cell_with_multiple_nuclei",
                "cell_id": str(cell_id), "nucleus_id": "",
                "details": f"Candidate cell region overlaps {count} candidate nuclei: {cell_to_nuclei[cell_id]}",
            })
    one_to_one = sum(
        1 for cell_id, nucleus_list in cell_to_nuclei.items()
        if len(nucleus_list) == 1 and len(nucleus_to_cells[nucleus_list[0]]) == 1
    )
    stats = {
        "image_id": image_id,
        "prediction_revision": revision,
        "candidate_cell_count": str(len(cell_ids)),
        "candidate_nucleus_count": str(len(nucleus_ids)),
        "possible_one_to_one_pairs": str(one_to_one),
        "cells_without_nucleus": str(sum(not cell_to_nuclei[cell_id] for cell_id in cell_ids)),
        "cells_with_multiple_nuclei": str(sum(len(cell_to_nuclei[cell_id]) > 1 for cell_id in cell_ids)),
        "nuclei_outside_cells": str(sum(not nucleus_to_cells[nucleus_id] for nucleus_id in nucleus_ids)),
        "nuclei_overlapping_multiple_cells": str(sum(len(nucleus_to_cells[nucleus_id]) > 1 for nucleus_id in nucleus_ids)),
        "matching_issue_count": str(len(issues)),
        "review_status": "unverified",
    }
    return stats, issues, links


def run_predictions(
    manifest_path: str | Path,
    image_ids: Iterable[str],
    model_name: str = "cpsam",
    gpu: bool = False,
    cell_diameter: float | None = None,
    nucleus_diameter: float | None = None,
    min_size: int = 15,
    max_size_fraction: float = 0.4,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    model_factory: Callable | None = None,
    batch: str | None = None,
) -> list[dict]:
    selected = list(image_ids)
    if not selected:
        raise ValueError("Prediction requires explicit image_id selection; no whole-dataset mode is provided")
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate image_id in prediction selection")
    rows = read_manifest(manifest_path)
    by_id = {row["image_id"]: row for row in rows}
    missing = [image_id for image_id in selected if image_id not in by_id]
    if missing:
        raise ValueError(f"Unknown image_id values: {missing}")
    for image_id in selected:
        row = by_id[image_id]
        if row["review_status"] == "verified" or row["annotation_status"].startswith("manual"):
            raise ValueError(f"Refusing to replace manual/verified annotations with automatic predictions: {image_id}")
        source = resolve_path(row["original_image_path"])
        if sha256(source) != row["source_sha256"]:
            raise ValueError(f"Source image checksum changed since import: {image_id}")
    if model_factory is None:
        from cellpose import models
        if gpu:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("GPU was requested but CUDA is unavailable")
        model = models.CellposeModel(gpu=gpu, pretrained_model=model_name)
    else:
        model = model_factory(model_name=model_name, gpu=gpu)
    outputs = []
    prediction_root = WORKSPACE / "cellpose_predictions"
    if batch:
        prediction_root = prediction_root / batch
    for image_id in selected:
        row = by_id[image_id]
        source = resolve_path(row["original_image_path"])
        image, _ = read_cyx_image(source)
        cell_index = int(row["cell_channel"])
        nucleus_index = int(row["nucleus_channel"]) if row["nucleus_channel"] else None
        revision_dir = _next_revision(prediction_root / image_id, "pred")
        revision = revision_dir.name
        cell_input, cell_normalization = normalize_for_cellpose(image[cell_index])
        result = model.eval(
            cell_input, normalize=False, diameter=cell_diameter, min_size=min_size,
            max_size_fraction=max_size_fraction, flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        cell_raw = result[0] if isinstance(result, (tuple, list)) else result
        cell_mask = _validate_prediction(cell_raw, cell_input.shape)
        cell_path = revision_dir / "cell_mask.tif"
        write_mask_new(cell_path, cell_mask, image_id)
        nucleus_mask = None
        nucleus_path = None
        nucleus_normalization = None
        if nucleus_index is not None:
            nucleus_input, nucleus_normalization = normalize_for_cellpose(image[nucleus_index])
            result = model.eval(
                nucleus_input, normalize=False, diameter=nucleus_diameter, min_size=min_size,
                max_size_fraction=max_size_fraction, flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
            )
            nucleus_raw = result[0] if isinstance(result, (tuple, list)) else result
            nucleus_mask = _validate_prediction(nucleus_raw, nucleus_input.shape)
            nucleus_path = revision_dir / "nucleus_mask.tif"
            write_mask_new(nucleus_path, nucleus_mask, image_id)
        export_root = resolve_path(row["source_export_root"])
        with (export_root / "display_settings.json").open(encoding="utf-8") as handle:
            display_settings = json.load(handle)
        overlay_root = prediction_root / "overlays" if batch else WORKSPACE / "previews"
        overlay_path = overlay_root / image_id / f"{revision}_overlay.png"
        save_overlay(
            image, cell_mask, nucleus_mask, cell_index, nucleus_index,
            display_settings, overlay_path,
        )
        metadata = {
            "image_id": image_id,
            "created_utc": utc_now(),
            "review_status": "unverified",
            "ground_truth": False,
            "source_batch": batch or row.get("source_batch", ""),
            "source_image": row["original_image_path"],
            "source_sha256": row["source_sha256"],
            "model": model_name,
            "model_type": "pretrained",
            "gpu_requested": gpu,
            "cell_channel_index": cell_index,
            "nucleus_channel_index": nucleus_index,
            "channel_names": json.loads(row["channel_names"]),
            "cell_normalization": cell_normalization,
            "nucleus_normalization": nucleus_normalization,
            "parameters": {
                "cell_diameter": cell_diameter, "nucleus_diameter": nucleus_diameter,
                "min_size": min_size, "max_size_fraction": max_size_fraction,
                "flow_threshold": flow_threshold, "cellprob_threshold": cellprob_threshold,
                "cellpose_normalize": False,
            },
        }
        try:
            import cellpose
            metadata["cellpose_version"] = cellpose.version
        except ImportError:
            metadata["cellpose_version"] = "unknown_test_double"
        write_json_new(revision_dir / "prediction.json", metadata)
        stats, issues, links = candidate_statistics(image_id, revision, cell_mask, nucleus_mask)
        upsert_csv(WORKSPACE / "statistics.csv", [stats], STAT_COLUMNS)
        upsert_csv(WORKSPACE / "matching_issues.csv", issues, ISSUE_COLUMNS, replace_ids=[image_id])
        upsert_csv(WORKSPACE / "candidate_links.csv", links, LINK_COLUMNS, replace_ids=[image_id])
        row.update({
            "cell_mask_path": relative_path(cell_path),
            "nucleus_mask_path": relative_path(nucleus_path) if nucleus_path else "",
            "annotation_status": "predicted",
            "annotation_method": "pretrained_cellpose",
            "review_status": "unverified",
            "source_batch": batch or row.get("source_batch", ""),
            "prediction_revision": revision,
            "candidate_cell_count": stats["candidate_cell_count"],
            "candidate_nucleus_count": stats["candidate_nucleus_count"],
        })
        write_manifest(manifest_path, rows)
        outputs.append({"stats": stats, "overlay": str(overlay_path), "revision": revision})
    return outputs


def validate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    issues = []
    ids = [row.get("image_id", "") for row in rows]
    for image_id in sorted({value for value in ids if ids.count(value) > 1}):
        issues.append({"image_id": image_id, "issue": "duplicate_image_id"})
    for row in rows:
        image_id = row.get("image_id", "")
        source = resolve_path(row["original_image_path"])
        if not source.is_file():
            issues.append({"image_id": image_id, "issue": f"missing_image:{source}"})
            continue
        try:
            image_hw = spatial_shape(source)
        except Exception as exc:
            issues.append({"image_id": image_id, "issue": f"invalid_image:{exc}"})
            continue
        if row.get("original_filename") and source.name != row["original_filename"]:
            issues.append({"image_id": image_id, "issue": "original_filename_mismatch"})
        for layer in ("cell", "nucleus"):
            value = row.get(f"{layer}_mask_path", "")
            required = row.get("annotation_status") != "imported" and (
                layer == "cell" or bool(row.get("nucleus_channel"))
            )
            if not value:
                if required:
                    issues.append({"image_id": image_id, "issue": f"missing_{layer}_annotation"})
                continue
            path = resolve_path(value)
            if not path.is_file():
                issues.append({"image_id": image_id, "issue": f"missing_{layer}_mask:{path}"})
                continue
            try:
                mask = read_mask(path)
                if mask.shape != image_hw:
                    issues.append({
                        "image_id": image_id,
                        "issue": f"{layer}_mask_dimension_mismatch:{mask.shape}!={image_hw}",
                    })
            except Exception as exc:
                issues.append({"image_id": image_id, "issue": f"invalid_{layer}_mask:{exc}"})
        if row.get("review_status") == "verified" and row.get("annotation_status") != "verified":
            issues.append({"image_id": image_id, "issue": "verified_review_without_verified_annotation"})
    return issues


def validation_report(manifest_path: str | Path, only_verified: bool = False) -> dict:
    rows = read_manifest(manifest_path)
    if only_verified:
        rows = [row for row in rows if row["review_status"] == "verified" and row["annotation_status"] == "verified"]
    issues = validate_rows(rows)
    return {
        "created_utc": utc_now(),
        "manifest": relative_path(resolve_path(manifest_path)),
        "row_count": len(rows),
        "verified_row_count": sum(row["review_status"] == "verified" for row in rows),
        "issue_count": len(issues),
        "issues": issues,
    }


def prepare_manual(manifest_path: str | Path, image_ids: Iterable[str]) -> list[Path]:
    rows = read_manifest(manifest_path)
    by_id = {row["image_id"]: row for row in rows}
    prepared = []
    for image_id in image_ids:
        row = by_id[image_id]
        if not row["cell_mask_path"]:
            raise ValueError(f"No automatic or existing cell mask to edit: {image_id}")
        revision = _next_revision(WORKSPACE / "manual_annotations" / image_id, "draft")
        source = resolve_path(row["original_image_path"])
        for layer in ("cell", "nucleus"):
            mask_value = row[f"{layer}_mask_path"]
            if not mask_value:
                continue
            layer_dir = revision / layer
            layer_dir.mkdir(parents=True)
            shutil.copy2(source, layer_dir / row["original_filename"])
            shutil.copy2(resolve_path(mask_value), layer_dir / f"{Path(row['original_filename']).stem}_masks.tif")
        write_json_new(revision / "session.json", {
            "image_id": image_id,
            "created_utc": utc_now(),
            "source_image": row["original_image_path"],
            "source_cell_mask": row["cell_mask_path"],
            "source_nucleus_mask": row["nucleus_mask_path"],
            "review_status": "unverified",
            "instructions": "Open the working TIFF in Cellpose GUI. Edit only this draft; original exports are untouched.",
        })
        prepared.append(revision)
    return prepared


def finalize_manual(manifest_path: str | Path, image_id: str, draft_revision: str) -> str:
    rows = read_manifest(manifest_path)
    by_id = {row["image_id"]: row for row in rows}
    row = by_id[image_id]
    draft = WORKSPACE / "manual_annotations" / image_id / draft_revision
    if not draft.is_dir() or not draft.name.startswith("draft_"):
        raise ValueError(f"Invalid manual draft: {draft}")
    frozen = _next_revision(WORKSPACE / "manual_annotations" / image_id, "frozen")
    source_hw = spatial_shape(row["original_image_path"])
    updates = {}
    for layer in ("cell", "nucleus"):
        candidates = list((draft / layer).glob("*_masks.tif"))
        if not candidates:
            if layer == "cell" or row["nucleus_channel"]:
                raise ValueError(f"Missing edited {layer} mask in {draft}")
            continue
        if len(candidates) != 1:
            raise ValueError(f"Expected one edited {layer} mask, found {len(candidates)}")
        mask = read_mask(candidates[0])
        if mask.shape != source_hw:
            raise ValueError(f"Edited {layer} mask dimensions do not match source")
        destination = frozen / f"{layer}_mask.tif"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], destination)
        updates[f"{layer}_mask_path"] = relative_path(destination)
    row.update(updates)
    row.update({
        "annotation_status": "manual_in_progress",
        "annotation_method": "cellpose_assisted_manual",
        "review_status": "unverified",
        "prediction_revision": frozen.name,
    })
    write_manifest(manifest_path, rows)
    return frozen.name


def review_annotation(
    manifest_path: str | Path,
    image_id: str,
    status: str,
    reviewer: str = "",
    notes: str = "",
) -> None:
    if status not in {"needs_review", "verified"}:
        raise ValueError("Review status must be needs_review or verified")
    rows = read_manifest(manifest_path)
    by_id = {row["image_id"]: row for row in rows}
    row = by_id[image_id]
    if status == "needs_review":
        row["review_status"] = status
        row["review_notes"] = notes
    else:
        if not reviewer.strip():
            raise ValueError("Verified export requires an explicit reviewer")
        row_issues = validate_rows([row])
        if row_issues:
            raise ValueError(f"Cannot verify invalid annotation: {row_issues}")
        verified_dir = _next_revision(WORKSPACE / "verified_masks" / image_id, "review")
        for layer in ("cell", "nucleus"):
            value = row[f"{layer}_mask_path"]
            if value:
                shutil.copy2(resolve_path(value), verified_dir / f"{layer}_mask.tif")
                row[f"{layer}_mask_path"] = relative_path(verified_dir / f"{layer}_mask.tif")
        row.update({
            "annotation_status": "verified",
            "review_status": "verified",
            "reviewer": reviewer,
            "review_notes": notes,
            "prediction_revision": verified_dir.name,
        })
    write_manifest(manifest_path, rows)


def _copy_exclusive(source: Path, destination: Path, created: list[Path]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    created.append(destination)
    shutil.copystat(source, destination)


def export_verified(
    manifest_path: str | Path,
    destination: str | Path = PROJECT_ROOT / "datasets" / "GFAP_labeled",
    append: bool = False,
) -> int:
    rows = read_manifest(manifest_path)
    verified = [row for row in rows if row["review_status"] == "verified" and row["annotation_status"] == "verified"]
    if not verified:
        return 0
    issues = validate_rows(verified)
    if issues:
        raise ValueError(f"Verified annotations failed validation: {issues}")
    destination = resolve_path(destination)
    dataset_manifest = destination / "manifest.csv"
    existing = []
    if dataset_manifest.exists():
        if not append:
            raise FileExistsError(f"Dataset manifest exists; explicit append is required: {dataset_manifest}")
        existing = read_manifest(dataset_manifest)
    existing_ids = {row["image_id"] for row in existing}
    if any(row["image_id"] in existing_ids for row in verified):
        raise ValueError("A verified image_id already exists in the destination dataset")
    export_rows = []
    operations = []
    for row in verified:
        slide = row["source_slide"]
        stem = Path(row["original_filename"]).stem
        image_target = destination / "images" / slide / row["original_filename"]
        cell_target = destination / "cell_masks" / slide / f"{stem}_cell.tif"
        nucleus_target = destination / "nucleus_masks" / slide / f"{stem}_nucleus.tif"
        targets = [image_target, cell_target] + ([nucleus_target] if row["nucleus_mask_path"] else [])
        conflict = next((path for path in targets if path.exists()), None)
        if conflict:
            raise FileExistsError(f"Export naming conflict: {conflict}")
        exported = dict(row)
        exported["source_image_path"] = row["original_image_path"]
        exported["original_image_path"] = relative_path(image_target)
        exported["cell_mask_path"] = relative_path(cell_target)
        exported["nucleus_mask_path"] = relative_path(nucleus_target) if row["nucleus_mask_path"] else ""
        export_rows.append(exported)
        operations.extend([
            (resolve_path(row["original_image_path"]), image_target),
            (resolve_path(row["cell_mask_path"]), cell_target),
        ])
        if row["nucleus_mask_path"]:
            operations.append((resolve_path(row["nucleus_mask_path"]), nucleus_target))
    created = []
    try:
        for source, target in operations:
            _copy_exclusive(source, target, created)
        write_manifest(dataset_manifest, existing + export_rows, preserve_history=append)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return len(verified)

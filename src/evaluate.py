"""Evaluate astrocyte centers, soma/process segmentation, and cell instances.

# REDESIGN: This module evaluates predictions; expert adjudication is separate.
# REDESIGN: Unlabeled FFPE slides are NOT ground truth and must not be scored.

Input per image (predictions/annotations dictionaries, keyed by identical image ID):
    points: float array [N,2] in (x,y), optional when instance_mask is present
    semantic_mask: int array [H,W], 0=background, 1=soma, 2=process
    instance_mask: int array [H,W], 0=background, positive ID=one cell
    valid_mask: bool array [H,W], optional GT annotation coverage
    metadata: optional dict (slide_id, pixel_size_um, level, etc.)
Predictions may contain 'centers' (Cellpose [id,x,y] or Nx2) instead of 'points'.

# FIX: Missing labels are never interpreted as zero-cell ground truth.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize


def _points(value, name: str) -> np.ndarray:
    """# FIX: Normalize valid empty points to (0,2); reject missing/malformed input."""
    if value is None:
        raise ValueError(f"{name} is missing (not an empty annotation)")
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        if arr.ndim > 2 or (arr.ndim == 2 and arr.shape[1] not in (0, 2)):
            raise ValueError(f"{name} must have shape (N,2)")
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite (N,2) array of (x,y) coordinates")
    return arr


def _nonnegative_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _scores(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """# FIX: Both empty = perfect; one empty = zero precision/recall/F1."""
    if tp + fp + fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def match_points(pred_points: np.ndarray, gt_points: np.ndarray, max_distance: float):
    """Match (x,y) centers one-to-one; matches are (pred_index, gt_index, distance)."""
    # IMPLEMENTED [Liu]: Maximize eligible matches first, then minimize distance.
    pred = _points(pred_points, "pred_points")
    gt = _points(gt_points, "gt_points")
    if not np.isfinite(max_distance) or max_distance < 0:
        raise ValueError("max_distance must be finite and nonnegative")
    n, m = len(pred), len(gt)
    if not n or not m:
        return {"tp": 0, "fp": n, "fn": m, "matches": []}
    dist = cdist(pred, gt)
    # FIX: A dummy column per prediction permits unmatched points. A cost of
    # (min(n,m)+1)*(max_distance+1) ensures one more valid match always wins.
    penalty = (min(n, m) + 1) * (float(max_distance) + 1.0)
    cost = np.full((n, m + n), penalty, dtype=np.float64)
    valid = dist <= max_distance
    cost[:, :m] = np.where(valid, dist, 2 * penalty)
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(i), int(j), float(dist[i, j]))
               for i, j in zip(rows, cols) if j < m and valid[i, j]]
    matches.sort(key=lambda x: x[0])
    tp = len(matches)
    return {"tp": tp, "fp": n - tp, "fn": m - tp, "matches": matches}


def detection_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Precision, recall, F1; both-empty images score 1, missing labels error."""
    return _scores(*(_nonnegative_int(v, k) for k, v in (("tp", tp), ("fp", fp), ("fn", fn))))


def counting_metrics(pred_counts, gt_counts) -> Dict[str, float]:
    """Return MAE and RMSE for aligned, nonempty lists of image counts."""
    # IMPLEMENTED [Liu]: Validate lengths and count integrality.
    p = np.asarray(pred_counts)
    g = np.asarray(gt_counts)
    if p.ndim != 1 or g.ndim != 1 or not len(p) or len(p) != len(g):
        raise ValueError("Count lists must be nonempty, one-dimensional, and equal length")
    # IMPLEMENTED [Liu]: Check valid integer cell counts.
    for counts, name in ((p, "pred_counts"), (g, "gt_counts")):
        try:
            values = counts.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric counts") from exc

        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains invalid values")

        if np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError(f"{name} must contain nonnegative integer counts")

    # IMPLEMENTED [Liu]: Calculate counting errors.
    errors = p.astype(np.float64) - g.astype(np.float64)

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    return {
        "mae": float(mae),
        "rmse": float(rmse)
    }

def export_report(metrics: dict, output_path: str):
    """Export evaluation metrics to a JSON or CSV file."""

    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

    elif path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])

            for key, value in metrics.items():
                writer.writerow([key, value])

    else:
        raise ValueError("Output must be .json or .csv")

    return str(path)
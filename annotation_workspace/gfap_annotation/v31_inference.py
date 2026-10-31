"""V3.1 inference-only utilities: overlapping windows, seam metrics, candidate crops.

No training, labels, GT anchors, or autonomous instance-assignment logic are used.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from src.preprocessing import prepare_image


def axis_window_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size or patch_size <= 0 or stride <= 0:
        raise ValueError("Require length >= patch_size > 0 and stride > 0")
    positions = list(range(0, length - patch_size + 1, stride))
    if positions[-1] != length - patch_size:
        positions.append(length - patch_size)
    return positions


def sliding_window_coordinates(shape: tuple[int, int], patch_size: int, stride: int) -> list[tuple[int, int]]:
    height, width = shape
    return [
        (x, y)
        for y in axis_window_positions(height, patch_size, stride)
        for x in axis_window_positions(width, patch_size, stride)
    ]


def smooth_blend_window(patch_size: int) -> np.ndarray:
    """Positive 2-D Hann window; endpoints remain nonzero at tile borders."""
    one_dimensional = np.hanning(patch_size + 2)[1:-1].astype(np.float64)
    one_dimensional /= one_dimensional.max()
    window = np.outer(one_dimensional, one_dimensional)
    if window.shape != (patch_size, patch_size) or not np.isfinite(window).all() or window.min() <= 0:
        raise RuntimeError("Invalid smooth blending window")
    return window


def predict_v3_overlapping(
    model,
    raw: np.ndarray,
    device: torch.device,
    patch_size: int = 256,
    stride: int = 128,
    batch_size: int = 4,
) -> np.ndarray:
    """Blend two-class V3 probabilities over overlapping windows at original resolution."""
    height, width = raw.shape[:2]
    coordinates = sliding_window_coordinates((height, width), patch_size, stride)
    window = smooth_blend_window(patch_size)
    accumulated = np.zeros((2, height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(coordinates), batch_size):
            batch_coordinates = coordinates[start:start + batch_size]
            patches = [
                prepare_image(raw[y:y + patch_size, x:x + patch_size], channel=1)
                for x, y in batch_coordinates
            ]
            batch = torch.stack(patches).to(device)
            logits = model(batch)["morphology_logits"]
            expected = (len(batch_coordinates), 2, patch_size, patch_size)
            if logits.shape != expected or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid or non-finite V3 morphology logits")
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            for offset, (x, y) in enumerate(batch_coordinates):
                target = (slice(y, y + patch_size), slice(x, x + patch_size))
                accumulated[(slice(None), *target)] += probabilities[offset] * window
                weights[target] += window
    if not np.isfinite(weights).all() or weights.min() <= 0:
        raise RuntimeError("Sliding-window blending left uncovered pixels")
    probability = (accumulated / weights).astype(np.float32)
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise RuntimeError("Invalid blended probabilities")
    if not np.allclose(probability.sum(axis=0), 1.0, atol=1e-5):
        raise RuntimeError("Blended probabilities do not sum to one")
    return probability


def patch_seam_metrics(probability: np.ndarray, patch_size: int = 256) -> dict:
    """Compare adjacent-pixel differences at original patch seams vs elsewhere.

    This is a spatial-continuity diagnostic, not a segmentation accuracy metric.
    """
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("Expected one finite 2-D probability map")
    height, width = probability.shape
    vertical_seams = set(range(patch_size, width, patch_size))
    horizontal_seams = set(range(patch_size, height, patch_size))
    vertical = np.abs(np.diff(probability, axis=1))
    horizontal = np.abs(np.diff(probability, axis=0))
    vertical_mask = np.array([x in vertical_seams for x in range(1, width)], dtype=bool)
    horizontal_mask = np.array([y in horizontal_seams for y in range(1, height)], dtype=bool)
    seam_values = np.concatenate([vertical[:, vertical_mask].ravel(), horizontal[horizontal_mask, :].ravel()])
    internal_values = np.concatenate([vertical[:, ~vertical_mask].ravel(), horizontal[~horizontal_mask, :].ravel()])
    seam_mad = float(seam_values.mean())
    internal_mad = float(internal_values.mean())
    return {
        "seam_mean_absolute_difference": seam_mad,
        "nonseam_mean_absolute_difference": internal_mad,
        "seam_to_nonseam_mad_ratio": seam_mad / internal_mad if internal_mad > 0 else 0.0,
        "seam_p95_absolute_difference": float(np.percentile(seam_values, 95)),
        "nonseam_p95_absolute_difference": float(np.percentile(internal_values, 95)),
        "seam_pair_count": int(seam_values.size),
        "nonseam_pair_count": int(internal_values.size),
        "metric_scope": "Continuity at original 256-pixel grid seams; not accuracy",
    }


def candidate_crop(mask: np.ndarray, instance_id: int, margin: int = 96) -> dict:
    """Extract one Cellpose candidate and its clipped full-image crop coordinates."""
    if mask.ndim != 2 or instance_id <= 0 or margin < 0:
        raise ValueError("Invalid candidate crop arguments")
    ys, xs = np.where(mask == instance_id)
    if not len(xs):
        raise ValueError(f"Candidate ID {instance_id} is absent")
    x0, x1 = max(0, int(xs.min()) - margin), min(mask.shape[1], int(xs.max()) + margin + 1)
    y0, y1 = max(0, int(ys.min()) - margin), min(mask.shape[0], int(ys.max()) + margin + 1)
    return {
        "instance_id": int(instance_id),
        "bbox_x0": int(xs.min()), "bbox_y0": int(ys.min()),
        "bbox_x1": int(xs.max()) + 1, "bbox_y1": int(ys.max()) + 1,
        "crop_x0": x0, "crop_y0": y0, "crop_x1": x1, "crop_y1": y1,
        "candidate_pixels": int(len(xs)),
        "mask": mask[y0:y1, x0:x1] == instance_id,
    }

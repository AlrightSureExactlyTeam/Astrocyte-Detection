"""Postprocess center probabilities, semantic logits, and instance masks.

Coordinate convention: ``(x,y)`` with x=column and y=row.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def to_probability(heatmap_logits: np.ndarray) -> np.ndarray:
    """Convert center logits ``(H,W)`` or ``(1,H,W)`` into probabilities."""
    if heatmap_logits is None:
        raise ValueError("Heatmap logits must not be None")
    logits = np.asarray(heatmap_logits, dtype=np.float32)
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 2 or logits.size == 0:
        raise ValueError(f"Expected non-empty (H,W) or (1,H,W), got {logits.shape}")
    if not np.all(np.isfinite(logits)):
        raise ValueError("Heatmap logits contain NaN or infinity")
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def _validate_probability_map(probability_map: np.ndarray) -> np.ndarray:
    if probability_map is None:
        raise ValueError("probability_map must not be None")
    probability = np.asarray(probability_map, dtype=np.float32)
    if probability.ndim != 2 or probability.size == 0:
        raise ValueError(f"probability_map must be a non-empty (H,W) array, got {probability.shape}")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError("probability_map must contain finite values in [0,1]")
    return probability


def extract_centers(
    probability_map: np.ndarray,
    threshold: float = 0.5,
    min_distance: int = 5,
) -> np.ndarray:
    """Return deterministic local-maximum centers as an ``(N,2)`` array."""
    probability = _validate_probability_map(probability_map)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    if not isinstance(min_distance, (int, np.integer)) or min_distance <= 0:
        raise ValueError("min_distance must be a positive integer")

    # IMPLEMENTED [Liu]: decode only center probabilities, including border peaks.
    coordinates = peak_local_max(
        probability,
        min_distance=int(min_distance),
        threshold_abs=float(threshold),
        exclude_border=False,
    )
    if len(coordinates) == 0:
        return np.empty((0, 2), dtype=np.float64)
    # Resolve equal-confidence output ordering deterministically by (y,x).
    coordinates = coordinates[np.lexsort((coordinates[:, 1], coordinates[:, 0]))]
    return coordinates[:, [1, 0]].astype(np.float64)


def decode_semantic_mask(
    semantic_logits: np.ndarray,
    confidence_threshold: float | None = None,
    return_confidence: bool = False,
):
    """Decode ``[3,H,W]`` logits into labels 0=background, 1=soma, 2=process.

    Pixels below ``confidence_threshold`` are marked background/uncertain (0).
    Set ``return_confidence=True`` to also receive softmax probabilities.
    """
    logits = np.asarray(semantic_logits, dtype=np.float32)
    if logits.ndim != 3 or logits.shape[0] != 3 or logits.size == 0:
        raise ValueError(f"semantic_logits must have shape [3,H,W], got {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError("semantic_logits contain NaN or infinity")
    if confidence_threshold is not None and (
        not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1
    ):
        raise ValueError("confidence_threshold must be in [0,1]")

    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(np.clip(shifted, -80.0, 80.0))
    probability = exp / exp.sum(axis=0, keepdims=True)
    confidence = probability.max(axis=0)
    mask = probability.argmax(axis=0).astype(np.uint8)
    if confidence_threshold is not None:
        mask[confidence < confidence_threshold] = 0
    if return_confidence:
        return mask, probability
    return mask


def separate_instances(
    semantic_mask: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Assign foreground pixels to center markers using a watershed.

    Returns ``(instance_mask, assigned_centers)``. Centers outside foreground,
    outside the image, or duplicated on an existing marker remain unassigned;
    consequently the instance count is not blindly equated with raw peak count.
    Overlapping-process ownership is an approximation and requires expert review.
    """
    mask = np.asarray(semantic_mask)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("semantic_mask must be a non-empty [H,W] array")
    if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0) or np.any(mask > 2):
        raise ValueError("semantic_mask must use integer labels 0, 1, and 2")
    points = np.asarray(centers, dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("centers must be a finite (N,2) array")

    height, width = mask.shape
    foreground = mask > 0
    markers = np.zeros((height, width), dtype=np.int32)
    assigned = []
    for x, y in points:
        col, row = int(round(x)), int(round(y))
        if not (0 <= col < width and 0 <= row < height):
            continue
        if not foreground[row, col] or markers[row, col] != 0:
            continue
        marker_id = len(assigned) + 1
        markers[row, col] = marker_id
        assigned.append((float(x), float(y)))

    if not assigned:
        return (
            np.zeros((height, width), dtype=np.uint32),
            np.empty((0, 2), dtype=np.float64),
        )
    distance = ndi.distance_transform_edt(foreground)
    instances = watershed(-distance, markers, mask=foreground).astype(np.uint32)
    return instances, np.asarray(assigned, dtype=np.float64).reshape(-1, 2)


def count_centers(centers: np.ndarray) -> int:
    """Count detections represented by an ``(N,2)`` array."""
    if centers is None:
        raise ValueError("centers must not be None")
    points = np.asarray(centers)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected (N, 2), got {points.shape}")
    if not np.issubdtype(points.dtype, np.number) or not np.all(np.isfinite(points)):
        raise ValueError("centers must contain finite numeric coordinates")
    return int(points.shape[0])


def centers_to_global_coordinates(
    centers: np.ndarray,
    tile_offset: tuple[int, int] = (0, 0),
    scale_xy: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Convert tile-local points into a common global coordinate system."""
    if centers is None:
        raise ValueError("centers must not be None")
    points = np.asarray(centers)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected (N, 2), got {points.shape}")
    if not np.issubdtype(points.dtype, np.number) or not np.all(np.isfinite(points)):
        raise ValueError("centers must contain finite numeric coordinates")
    offset = np.asarray(tile_offset, dtype=np.float64)
    scale = np.asarray(scale_xy, dtype=np.float64)
    if offset.shape != (2,) or scale.shape != (2,):
        raise ValueError("tile_offset and scale_xy must each contain (x,y) pairs")
    if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("offset must be finite and scale must be positive and finite")
    return (points.astype(np.float64) + offset) * scale

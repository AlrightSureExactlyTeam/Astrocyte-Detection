"""Postprocess astrocyte morphology, center points and instance masks.

Owner: Zhang (peak extraction / instance processing), Liu (evaluation integration).
REDESIGN: semantic labels 0 background / 1 soma / 2 visible processes.
Coordinate convention: (x, y), with x=column and y=row.
"""

import numpy as np


def to_probability(heatmap_logits: np.ndarray) -> np.ndarray:
    """Convert center logits (H,W) or (1,H,W) into probabilities."""
    # FIX: Reject missing, empty and non-finite logits before sigmoid.
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


def extract_centers(
    probability_map: np.ndarray,
    threshold: float = 0.5,
    min_distance: int = 5,
) -> np.ndarray:
    """Return N x 2 center coordinates as (x, y) in image pixels."""
    # TODO [Zhang]: implement peak detection (e.g. skimage.feature.peak_local_max).
    # TODO [Zhang]: handle adjacent peaks, borders and ties deterministically.
    # REDESIGN TODO [Zhang]: use center_logits sigmoid output ONLY; do not
    # apply to_probability() again to an already converted probability map.
    # EVALUATION [Liu]: tune threshold/min_distance with tune_center_parameters()
    # on a held-out validation set; never tune using the test set.
    raise NotImplementedError("Peak extraction is not implemented yet")


# REDESIGN TODO [Zhang]: implement decode_semantic_mask(semantic_logits)
# with softmax over the 3 semantic classes, validation of [3,H,W] input,
# optional confidence/uncertain handling, and explicit label IDs 0/1/2.
# REDESIGN TODO [Zhang]: implement separate_instances(semantic_mask,
# centers, ...), associating visible soma/process pixels to distinct cell IDs;
# retain unassigned/ambiguous crossings instead of forcing ownership.
# REVIEW [Zhu/non-code]: review ambiguous crossing-process assignments on
# microscopy images; document cases where ownership cannot be established.
# REDESIGN TODO [Zhang]: output instance_mask [H,W] with 0 as background;
# validate one-to-one association of instance IDs and cell records. A center
# alone does not determine the ownership of overlapping processes.


def count_centers(centers: np.ndarray) -> int:
    """Count detections represented by an (N,2) array."""
    # FIX: (0,2) means a valid zero-detection result; None and (0,) do not.
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
    """Convert tile-local (x,y) points into a common global coordinate system.

    global_x = (local_x + offset_x) * scale_x, similarly for y.
    The offset is measured in the SAME level/pixel grid as local coordinates.
    For resized inference, convert to the tile's native grid before calling.
    Returns float64 (N,2), preserving an empty (0,2) array.
    """
    # IMPLEMENTED [Liu]: Explicit coordinate conversion for evaluation.
    # FIX [Liu]: Validate offsets/scales to avoid silently mismatched metrics.
    if centers is None:
        raise ValueError("centers must not be None")
    points = np.asarray(centers)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected centers (N,2), got {points.shape}")
    if not np.issubdtype(points.dtype, np.number) or not np.all(np.isfinite(points)):
        raise ValueError("centers must contain finite numeric coordinates")
    offset = np.asarray(tile_offset, dtype=np.float64)
    scale = np.asarray(scale_xy, dtype=np.float64)
    if offset.shape != (2,) or scale.shape != (2,):
        raise ValueError("tile_offset and scale_xy must each contain (x,y) pairs")
    if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("offset must be finite and scale must be positive and finite")
    return (points.astype(np.float64) + offset) * scale


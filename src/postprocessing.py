"""Convert U-Net heatmaps into astrocyte center points.

Owner: Zhang (implementation), Liu (evaluation integration).
Coordinate convention: (x, y), with x=column and y=row.
"""
import numpy as np


def to_probability(heatmap"""Postprocess astrocyte morphology, center points and instance masks.

Owner: Zhang (implementation), Liu (evaluation integration).
REDESIGN: semantic labels 0 background / 1 soma / 2 visible processes.
Coordinate convention: (x, y), with x=column and y=row.
"""
import numpy as np


def to_probability(heatmap_logits: np.ndarray) -> np.ndarray:
    """Convert logits into a 2D probability map via sigmoid.

    Accepts shape (H, W) or (1, H, W); rejects batch/multiclass maps.
    """
    # FIX: Reject absent maps explicitly; do not confuse missing input with no detections.
    if heatmap_logits is None:
        raise ValueError("Heatmap logits must not be None")
    logits = np.asarray(heatmap_logits, dtype=np.float32)
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 2:
        raise ValueError(f"Expected (H,W) or (1,H,W), got {logits.shape}")
    # FIX: Shape checks alone allow (0,0) maps and NaN/Inf logits.
    if logits.size == 0 or 0 in logits.shape:
        raise ValueError("Heatmap must contain pixels")
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
    # TODO [Liu]: tune threshold/min_distance using validation set only.
    # TODO [Liu]: confirm coordinates align after any image resizing/tiling.
    # REDESIGN TODO [Zhang]: use center_logits sigmoid output ONLY; do not
    # apply to_probability() again to an already converted probability map.
    raise NotImplementedError("Peak extraction is not implemented yet")


# REDESIGN TODO [Zhang]: implement decode_semantic_mask(semantic_logits)
# with softmax over the 3 semantic classes, validation of [3,H,W] input,
# optional confidence/uncertain handling, and explicit label IDs 0/1/2.
# REDESIGN TODO [Zhang + Zhu]: implement separate_instances(semantic_mask,
# centers, ...), associating visible soma/process pixels to distinct cell IDs;
# retain unassigned/ambiguous crossings instead of forcing ownership.
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
    # FIX: Count only finite numeric coordinate pairs, not malformed rows.
    if not np.issubdtype(points.dtype, np.number) or not np.all(np.isfinite(points)):
        raise ValueError("centers must contain finite numeric coordinates")
    return int(points.shape[0])


def merge_tile_centers(global_centers: np.ndarray, radius: float) -> np.ndarray:
    """Merge duplicate detections from overlapping whole-slide tiles."""
    # TODO [Zhang]: convert tile-local centers to global coordinates first.
    # TODO [Zhang]: deduplicate centers from tile overlap using radius in pixels.
    # TODO [Zhu]: check pixel size so deduplication radius is meaningful.
    # REDESIGN TODO [Zhang]: point deduplication is not instance-mask merging;
    # implement merge_tile_instances() using overlap, global offsets, resolution
    # level, and stable IDs; avoid merging adjacent distinct cells.
    raise NotImplementedError("Tile deduplication is not implemented yet")
_logits: np.ndarray) -> np.ndarray:
    """Convert logits into a 2D probability map via sigmoid.

    Accepts shape (H, W) or (1, H, W); rejects batch/multiclass maps.
    """
    logits = np.asarray(heatmap_logits, dtype=np.float32)
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 2:
        raise ValueError(f"Expected (H,W) or (1,H,W), got {logits.shape}")
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def extract_centers(
    probability_map: np.ndarray,
    threshold: float = 0.5,
    min_distance: int = 5,
) -> np.ndarray:
    """Return N x 2 center coordinates as (x, y) in image pixels."""
    # TODO [Zhang]: implement peak detection (e.g. skimage.feature.peak_local_max).
    # TODO [Zhang]: handle adjacent peaks, borders and ties deterministically.
    # TODO [Liu]: tune threshold/min_distance using validation set only.
    # TODO [Liu]: confirm coordinates align after any image resizing/tiling.
    raise NotImplementedError("Peak extraction is not implemented yet")


def count_centers(centers: np.ndarray) -> int:
    """Count detections represented by an (N,2) array."""
    points = np.asarray(centers)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected (N, 2), got {points.shape}")
    return int(points.shape[0])


def merge_tile_centers(global_centers: np.ndarray, radius: float) -> np.ndarray:
    """Merge duplicate detections from overlapping whole-slide tiles."""
    # TODO [Zhang]: convert tile-local centers to global coordinates first.
    # TODO [Zhang]: deduplicate centers from tile overlap using radius in pixels.
    # TODO [Zhu]: check pixel size so deduplication radius is meaningful.
    raise NotImplementedError("Tile deduplication is not implemented yet")

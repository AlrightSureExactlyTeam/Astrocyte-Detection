"""Convert U-Net heatmaps into astrocyte center points.

Owner: Zhang (implementation), Liu (evaluation integration).
Coordinate convention: (x, y), with x=column and y=row.
"""
import numpy as np


def to_probability(heatmap_logits: np.ndarray) -> np.ndarray:
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

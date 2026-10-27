"""Image loading and preprocessing for small microscopy images and QPTIFF tiles."""

from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np
import tifffile
import torch


def load_image(path: str | Path) -> np.ndarray:
    """Read a small PNG/TIF/TIFF. QPTIFF must use iter_qptiff_tiles instead."""
    # FIX: Reject missing/empty paths before Path conversion; avoid ambiguous directory errors.
    if path is None or not str(path).strip():
        raise ValueError("Image path must not be empty")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".qptiff":
        raise ValueError("Use iter_qptiff_tiles() for QPTIFF; do not load the whole slide.")
    if suffix == ".png":
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"Cannot read PNG: {path}")
        if image.ndim == 3:
            # FIX: Explicitly support only OpenCV BGR/BGRA channel layouts.
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            else:
                raise ValueError(f"Unsupported PNG channel count: {image.shape[2]}")
        return image
    if suffix in {".tif", ".tiff"}:
        return tifffile.imread(str(path))
    raise ValueError(f"Unsupported image extension: {suffix}")


def prepare_image(image: np.ndarray, channel: int | None = None) -> torch.Tensor:
    """Convert a 2D grayscale or HWC RGB image to float32 CHW tensor in [0,1].

    channel selects a channel in HWC input; if omitted, RGB is converted to gray.
    Fluorescence channels should be selected based on microscopy metadata.
    REDESIGN: preserve fine visible processes; per-tile contrast normalization
    can hide intensity differences relevant to later slide-quality assessment.
    """
    # FIX: A missing image must not be interpreted as a zero-cell detection.
    if image is None:
        raise ValueError("Image must not be None")
    image = np.asarray(image)
    # FIX: Reject empty dimensions before percentile calculation.
    if image.size == 0 or any(dim == 0 for dim in image.shape):
        raise ValueError(f"Image is empty: shape={image.shape}")
    if image.ndim == 3:
        if image.shape[-1] not in (3, 4):
            raise ValueError("Expected HWC RGB/RGBA image; inspect axes for multichannel TIFF.")
        if channel is not None:
            # FIX: Disallow negative/out-of-range channel indexing.
            if not isinstance(channel, (int, np.integer)) or not 0 <= channel < image.shape[-1]:
                raise ValueError(f"Invalid channel {channel!r} for {image.shape[-1]} channels")
            image = image[..., channel]
        else:
            image = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
    elif image.ndim != 2:
        raise ValueError(f"Expected 2D or HWC image; got shape {image.shape}")
    elif channel is not None:
        # FIX: A channel selector on a grayscale image is a configuration error.
        raise ValueError("channel must be None for a 2D grayscale image")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"Unsupported dtype: {image.dtype}")
    # FIX: Non-finite intensities would poison normalization and model predictions.
    if not np.all(np.isfinite(image)):
        raise ValueError("Image contains NaN or infinite intensities")
    image = image.astype(np.float32)
    if not np.all(np.isfinite(image)):
        raise ValueError("Image intensities exceed float32 finite range")
    low, high = np.percentile(image, [1, 99.8])
    if high <= low:
        scaled = np.zeros_like(image, dtype=np.float32)
    else:
        scaled = np.clip((image - low) / (high - low), 0, 1)
    return torch.from_numpy(np.ascontiguousarray(scaled[None, ...]))


def iter_qptiff_tiles(
    path: str | Path, tile_size: int = 512, level: int = 0
) -> Iterator[Tuple[np.ndarray, int, int]]:
    """Yield (tile, x, y) using tifffile's Zarr-backed lazy array.

    Requires `zarr` and an uncompressed/compatible tiled QPTIFF. Level/series
    and channel axes differ by scanner: inspect them before choosing a level.
    Coordinates are at the selected resolution level, not automatically level 0.
    REDESIGN: current implementation has NO overlap and accepts YX only;
    instance stitching requires an explicit overlap/edge policy and scale map.
    """
    # REDESIGN TODO [Zhang]: add configurable overlap/stride and stable tile IDs
    # for instance-mask stitching; document pixel scale and source series.
    # REDESIGN TODO [Zhang + Zhu]: select correct QPTIFF channel/series/level
    # from scanner metadata; never assume the first series is the target stain.
    # FIX: Validate tiling arguments; negative tile_size could silently yield no tiles.
    if path is None or not str(path).strip():
        raise ValueError("Slide path must not be empty")
    if not isinstance(tile_size, (int, np.integer)) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    if not isinstance(level, (int, np.integer)) or level < 0:
        raise ValueError("level must be a non-negative integer")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Slide file not found: {path}")
    import zarr

    with tifffile.TiffFile(str(path)) as tif:
        # FIX: Some TIFF files expose no readable image series.
        if not tif.series:
            raise ValueError("No image series found in slide")
        series = tif.series[0]
        if level >= len(series.levels):
            raise ValueError(f"Level {level} unavailable; {len(series.levels)} levels")
        selected = series.levels[level]
        if selected.axes not in ("YX",):
            raise ValueError(
                f"QPTIFF axes {selected.axes!r} not supported by this basic tiler. "
                "Select the desired channel/series before tiling."
            )
        store = selected.aszarr()
        try:
            arr = zarr.open(store, mode="r")
            height, width = arr.shape
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    tile = np.asarray(arr[y:min(y + tile_size, height),
                                          x:min(x + tile_size, width)])
                    yield tile, x, y
        finally:
            store.close()


if __name__ == "__main__":
    sample = np.random.default_rng(0).integers(0, 65535, (256, 256), dtype=np.uint16)
    tensor = prepare_image(sample)
    print("Prepared image:", tuple(tensor.shape), tensor.dtype)

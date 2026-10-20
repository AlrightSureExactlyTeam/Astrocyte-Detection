"""Image loading and preprocessing for small microscopy images and QPTIFF tiles."""

from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np
import tifffile
import torch


def load_image(path: str | Path) -> np.ndarray:
    """Read a small PNG/TIF/TIFF. QPTIFF must use iter_qptiff_tiles instead."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".qptiff":
        raise ValueError("Use iter_qptiff_tiles() for QPTIFF; do not load the whole slide.")
    if suffix == ".png":
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"Cannot read PNG: {path}")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return image
    if suffix in {".tif", ".tiff"}:
        return tifffile.imread(str(path))
    raise ValueError(f"Unsupported image extension: {suffix}")


def prepare_image(image: np.ndarray, channel: int | None = None) -> torch.Tensor:
    """Convert a 2D grayscale or HWC RGB image to float32 CHW tensor in [0,1].

    channel selects a channel in HWC input; if omitted, RGB is converted to gray.
    Fluorescence channels should be selected based on microscopy metadata.
    """
    image = np.asarray(image)
    if image.ndim == 3:
        if image.shape[-1] not in (3, 4):
            raise ValueError("Expected HWC RGB/RGBA image; inspect axes for multichannel TIFF.")
        if channel is not None:
            image = image[..., channel]
        else:
            image = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
    elif image.ndim != 2:
        raise ValueError(f"Expected 2D or HWC image; got shape {image.shape}")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"Unsupported dtype: {image.dtype}")
    image = image.astype(np.float32)
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
    """
    import zarr

    with tifffile.TiffFile(str(path)) as tif:
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

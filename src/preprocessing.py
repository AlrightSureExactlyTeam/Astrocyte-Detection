"""Image loading and preprocessing for small microscopy images and QPTIFF tiles.

Implemented (Zhang, 2026-10-28):
- load_image: PNG/TIF/TIFF loading with strict path and channel-layout validation.
- prepare_image: 2D/HWC -> float32 CHW tensor in [0, 1] with percentile scaling.
- iter_qptiff_tiles: lazy tiled iteration over large TIFF/QPTIFF slides with
  configurable overlap, stable tile IDs, channel/series/level selection and
  optional per-tile metadata for later instance stitching.

Overlap note: tiles MAY overlap when overlap > 0; deduplication of detections
in the overlap band is the caller's responsibility (see predict.py).
"""

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
    # FIX: Validate the extension before existence so unsupported formats
    # (e.g. .qptiff/.jpg) get a clear message even when the file is missing.
    suffix = path.suffix.lower()
    if suffix == ".qptiff":
        raise ValueError("Use iter_qptiff_tiles() for QPTIFF; do not load the whole slide.")
    if suffix not in {".png", ".tif", ".tiff"}:
        raise ValueError(f"Unsupported image extension: {suffix}")
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
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
        return _load_tiff(path)
    raise ValueError(f"Unsupported image extension: {suffix}")


def _load_tiff(path: Path) -> np.ndarray:
    """Load a TIFF using axis metadata instead of shape guessing.

    FIX (review A1/A5): decide the axis layout from tifffile's `series.axes`,
    not from `2 <= shape[0] <= 8` heuristics:
    - Z/T stacks are rejected (mixing z-planes/timepoints into RGB silently
      corrupts fluorescence data); select a single plane first.
    - (1,H,W) single-channel exports (Fiji/ImageJ) are squeezed to 2D.
    - Planar multi-channel TIFFs (C/Q axis first) are transposed to HWC so
      prepare_image(channel=...) / --channel work.
    - HWC sample layouts (YXS, last axis 3/4) are returned unchanged.

    KNOWN LIMITATION: a 3-plane stack saved WITHOUT axis metadata is
    indistinguishable from planar 3-channel data (tifffile labels both QYX),
    so it will be transposed to (H,W,3). There is no shape-based way to tell
    the two apart. When exporting z-stacks, write axes metadata (e.g.
    metadata={"axes": "ZYX"}), or extract the single plane before loading.
    """
    with tifffile.TiffFile(str(path)) as tif:
        if not tif.series:
            raise ValueError("No image series found in TIFF")
        series = tif.series[0]
        axes = series.axes
        image = series.asarray()
    if image.ndim > 3:
        raise ValueError(
            f"TIFF axes {axes!r} with shape {image.shape} are not a single 2D "
            "image or plane; select series/plane before loading."
        )
    if image.ndim == 3:
        if "Z" in axes or "T" in axes:
            raise ValueError(
                f"TIFF has a stack axis ({axes!r}); select a single plane first."
            )
        if axes.endswith("S") or image.shape[-1] in (3, 4):
            pass  # HWC samples (RGB/RGBA); returned unchanged.
        elif image.shape[0] == 1:
            image = image[0]  # (1,H,W) single-channel export -> (H,W)
        else:
            # Planar (C,H,W); without metadata tifffile labels the axis Q.
            image = np.moveaxis(image, 0, -1)
    return image


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
        n_channels = image.shape[-1]
        if channel is not None:
            # FIX (review A2): an explicit channel= works for ANY channel count
            # (e.g. 2-channel GFAP+DAPI), not only RGB/RGBA.
            if not isinstance(channel, (int, np.integer)) or not 0 <= channel < n_channels:
                raise ValueError(f"Invalid channel {channel!r} for {n_channels} channels")
            image = image[..., channel]
        elif n_channels == 1:
            image = image[..., 0]
        elif n_channels in (3, 4):
            image = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(
                f"Image has {n_channels} channels; pass channel= explicitly "
                "based on microscopy metadata."
            )
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
        # FIX (review A4): when foreground covers < 0.2% of pixels the 99.8th
        # percentile equals the background level; fall back to the image max
        # so sparse fluorescence is not silently flattened to all zeros.
        high = float(image.max())
    if high <= low:
        scaled = np.zeros_like(image, dtype=np.float32)
    else:
        # FIX: np.percentile returns float64; cast back so the promised
        # float32 output contract holds (float64 would double model memory).
        scaled = np.clip((image - low) / (high - low), 0, 1).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(scaled[None, ...]))


def _select_plane_index(axes: str, shape: tuple, channel: int | None) -> tuple[int | None, int, int]:
    """Resolve (channel_axis_index, y_axis, x_axis) for a TIFF level.

    Returns channel_axis_index=None for plain YX data. Raises when a
    multi-channel slide is requested without an explicit channel choice:
    never assume the first series/channel is the target stain.
    """
    if axes == "YX":
        if channel is not None:
            raise ValueError("channel must be None for a single-channel (YX) slide")
        return None, axes.index("Y"), axes.index("X")
    if "Y" not in axes or "X" not in axes:
        raise ValueError(
            f"Slide axes {axes!r} not supported by this tiler. "
            "Expected YX or a single-channel-axis layout such as CYX/QYX/YXS."
        )
    # IMPLEMENTED [Zhang]: tifffile labels the non-spatial axis C (channel),
    # Q (unknown) or S (samples) depending on metadata; accept exactly one
    # extra axis as the channel axis and require an explicit channel choice.
    extra = [i for i, a in enumerate(axes) if a not in "YX"]
    if len(extra) != 1:
        raise ValueError(
            f"Slide axes {axes!r} not supported by this tiler: "
            "expected exactly one non-spatial (channel) axis."
        )
    c_axis = extra[0]
    n_channels = shape[c_axis]
    if channel is None:
        if n_channels == 1:
            channel = 0
        else:
            raise ValueError(
                f"Slide has {n_channels} channels; pass channel= explicitly "
                "based on scanner metadata (e.g. the GFAP channel)."
            )
    if not isinstance(channel, (int, np.integer)) or not 0 <= channel < n_channels:
        raise ValueError(f"Invalid channel {channel!r} for {n_channels} channels")
    return c_axis, axes.index("Y"), axes.index("X")


def iter_qptiff_tiles(
    path: str | Path,
    tile_size: int = 512,
    level: int = 0,
    overlap: int = 0,
    channel: int | None = None,
    series_index: int = 0,
    with_meta: bool = False,
) -> Iterator:
    """Yield tiles from a large TIFF/QPTIFF using tifffile's Zarr-backed lazy array.

    Default output is (tile, x, y) with x/y in the SELECTED level's pixel grid
    (backward compatible). With with_meta=True, yields a dict per tile:
        tile, x, y, tile_id, row, col, level, series_index, channel,
        scale_xy (level-0 pixels per selected-level pixel), source
    tile_id is stable for a given (path, series_index, level, tile_size,
    overlap, channel) and can be used for instance-mask stitching.

    Requires `zarr` and an uncompressed/compatible tiled slide. Overlap must
    satisfy 0 <= overlap < tile_size; overlapping detections are NOT deduped
    here. Coordinates are at the selected resolution level, not level 0.
    """
    # FIX: Validate tiling arguments; negative tile_size could silently yield no tiles.
    if path is None or not str(path).strip():
        raise ValueError("Slide path must not be empty")
    if not isinstance(tile_size, (int, np.integer)) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    if not isinstance(level, (int, np.integer)) or level < 0:
        raise ValueError("level must be a non-negative integer")
    if not isinstance(overlap, (int, np.integer)) or not 0 <= overlap < tile_size:
        raise ValueError("overlap must be an integer with 0 <= overlap < tile_size")
    if not isinstance(series_index, (int, np.integer)) or series_index < 0:
        raise ValueError("series_index must be a non-negative integer")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Slide file not found: {path}")
    import zarr

    with tifffile.TiffFile(str(path)) as tif:
        # FIX: Some TIFF files expose no readable image series.
        if not tif.series:
            raise ValueError("No image series found in slide")
        if series_index >= len(tif.series):
            raise ValueError(
                f"Series {series_index} unavailable; {len(tif.series)} series present"
            )
        series = tif.series[series_index]
        if level >= len(series.levels):
            raise ValueError(f"Level {level} unavailable; {len(series.levels)} levels")
        selected = series.levels[level]
        axes, shape = selected.axes, tuple(selected.shape)
        c_axis, y_axis, x_axis = _select_plane_index(axes, shape, channel)
        height, width = shape[y_axis], shape[x_axis]

        # IMPLEMENTED [Zhang]: pixel scale of the selected level relative to level 0,
        # so callers can map tile coordinates back to full resolution.
        base = series.levels[0]
        scale_y = base.shape[base.axes.index("Y")] / height
        scale_x = base.shape[base.axes.index("X")] / width

        store = selected.aszarr()
        try:
            arr = zarr.open(store, mode="r")
            stride = tile_size - overlap
            row = 0
            for y in range(0, height, stride):
                col = 0
                for x in range(0, width, stride):
                    index = [slice(None)] * arr.ndim
                    if c_axis is not None:
                        index[c_axis] = channel if channel is not None else 0
                    index[y_axis] = slice(y, min(y + tile_size, height))
                    index[x_axis] = slice(x, min(x + tile_size, width))
                    tile = np.asarray(arr[tuple(index)])
                    if tile.ndim > 2:
                        tile = np.squeeze(tile)
                    if not with_meta:
                        yield tile, x, y
                    else:
                        # IMPLEMENTED [Zhang]: stable tile IDs + provenance metadata
                        # for instance-mask stitching and global-coordinate mapping.
                        yield {
                            "tile": tile,
                            "x": x,
                            "y": y,
                            "tile_id": f"r{row:05d}_c{col:05d}",
                            "row": row,
                            "col": col,
                            "level": int(level),
                            "series_index": int(series_index),
                            "channel": None if c_axis is None else int(channel if channel is not None else 0),
                            "scale_xy": (float(scale_x), float(scale_y)),
                            "source": str(path),
                        }
                    col += 1
                row += 1
        finally:
            store.close()


if __name__ == "__main__":
    sample = np.random.default_rng(0).integers(0, 65535, (256, 256), dtype=np.uint16)
    tensor = prepare_image(sample)
    print("Prepared image:", tuple(tensor.shape), tensor.dtype)

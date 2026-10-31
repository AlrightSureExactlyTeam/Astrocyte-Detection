"""Strict image decoding and lightweight, original-resolution visualization."""
from __future__ import annotations

import colorsys
import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tifffile

from src.preprocessing import load_image

SUPPORTED_EXTENSIONS = {".png", ".tif", ".tiff"}
LABEL_COLORS = {
    1: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    2: np.array([1.0, 0.1, 0.1], dtype=np.float32),
}


def read_upload(filename: str, content: bytes) -> tuple[np.ndarray, dict]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format {suffix or '(none)'}; use PNG, TIF, or TIFF")
    if not content:
        raise ValueError("Uploaded image is empty")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        image = load_image(temporary_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not validate {Path(filename).name}: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError(f"Expected a nonempty 2D or HWC image, got {image.shape}")
    if image.ndim == 3 and image.shape[-1] not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported HWC channel count: {image.shape[-1]}")
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise ValueError("Image intensities must be finite numeric values")
    metadata = {
        "filename": Path(filename).name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "channels": 1 if image.ndim == 2 else int(image.shape[-1]),
        "resized": False,
        "content_modified": False,
    }
    return image, metadata


def validate_for_model(image: np.ndarray, expected_input: dict) -> None:
    channels = 1 if image.ndim == 2 else image.shape[-1]
    minimum = int(expected_input.get("min_channels", 1))
    if channels < minimum:
        raise ValueError(f"This model requires at least {minimum} channels; the image has {channels}")
    divisor = int(expected_input.get("spatial_divisor", 1))
    if divisor > 1 and (image.shape[0] % divisor or image.shape[1] % divisor):
        raise ValueError(
            f"Image height and width must be divisible by {divisor} for patch inference; "
            f"got {image.shape[:2]}. No resizing is performed."
        )


def display_image(image: np.ndarray, channel: int | None = 1) -> np.ndarray:
    """Return an RGB float display without changing the inference image."""
    if image.ndim == 2:
        gray = image
    else:
        index = 0 if channel is None or channel >= image.shape[-1] else channel
        gray = image[..., index]
    gray = gray.astype(np.float32)
    low, high = np.percentile(gray, [1, 99.8])
    if high <= low:
        high = float(gray.max())
    scaled = np.zeros_like(gray) if high <= low else np.clip((gray - low) / (high - low), 0, 1)
    return np.repeat(scaled[..., None], 3, axis=2)


def read_mask(path: str | Path) -> np.ndarray:
    mask = tifffile.imread(path)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise ValueError(f"Expected a nonnegative 2D integer mask: {path}")
    return mask


def semantic_overlay(base: np.ndarray, mask: np.ndarray, instance: bool = False) -> np.ndarray:
    if base.shape[:2] != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image {base.shape[:2]}")
    result = base.astype(np.float32).copy()
    if instance:
        boundaries = np.zeros(mask.shape, dtype=bool)
        boundaries[1:] |= mask[1:] != mask[:-1]
        boundaries[:-1] |= mask[:-1] != mask[1:]
        boundaries[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        boundaries[:, :-1] |= mask[:, :-1] != mask[:, 1:]
        result[boundaries & (mask > 0)] = (0.0, 1.0, 0.0)
        return result
    for label, color in LABEL_COLORS.items():
        selected = mask == label
        result[selected] = 0.45 * result[selected] + 0.55 * color
    return result


def instance_display(mask: np.ndarray) -> np.ndarray:
    """Bright deterministic categorical colors; original integer IDs are unchanged."""
    array = np.asarray(mask)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
        raise ValueError("Expected a nonnegative 2D integer instance mask")
    result = np.zeros((*array.shape, 3), dtype=np.float32)
    for instance_id in [int(value) for value in np.unique(array) if value > 0]:
        color = colorsys.hsv_to_rgb((instance_id * 0.61803398875) % 1.0, 0.78, 1.0)
        result[array == instance_id] = color
    return result


def association_state_display(state: np.ndarray) -> np.ndarray:
    """Display B.2 states without assigning ownership or changing status labels."""
    array = np.asarray(state)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Expected a 2D integer association-state mask")
    result = np.zeros((*array.shape, 3), dtype=np.float32)
    colors = {
        1: (0.0, 0.85, 1.0),
        2: (1.0, 0.85, 0.0),
        3: (1.0, 0.0, 0.75),
        4: (1.0, 0.10, 0.10),
    }
    for label, color in colors.items():
        result[array == label] = color
    return result


def mask_display(mask: np.ndarray, instance: bool = False) -> np.ndarray:
    if instance:
        return instance_display(mask)
    result = np.zeros((*mask.shape, 3), dtype=np.float32)
    for label, color in LABEL_COLORS.items():
        result[mask == label] = color
    return result


def encode_png(image: np.ndarray) -> bytes:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", array)
    if not success:
        raise OSError("Could not encode PNG")
    return encoded.tobytes()

"""Inference-only Gaussian blur ablation helpers for GFAP and V3 probabilities."""
from __future__ import annotations

import cv2
import numpy as np


def validate_sigma(sigma: float) -> float:
    value = float(sigma)
    if value < 0 or not np.isfinite(value):
        raise ValueError("Gaussian sigma must be finite and nonnegative")
    return value


def blur_gfap_input(raw: np.ndarray, sigma: float, channel: int = 1) -> np.ndarray:
    """Blur only GFAP channel 1 before model-specific preprocessing; no resizing."""
    sigma = validate_sigma(sigma)
    if raw.ndim != 3 or not 0 <= channel < raw.shape[-1]:
        raise ValueError("Expected HWC input with a valid GFAP channel")
    result = raw.copy()
    if sigma > 0:
        result[..., channel] = cv2.GaussianBlur(raw[..., channel], (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return result


def smooth_probability_maps(probability: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth continuous class probabilities and renormalize; never smooth IDs/masks."""
    sigma = validate_sigma(sigma)
    if probability.ndim != 3 or probability.shape[0] < 2 or probability.dtype != np.float32:
        raise ValueError("Expected float32 CHW probabilities")
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError("Invalid input probabilities")
    result = probability.copy()
    if sigma > 0:
        for index in range(probability.shape[0]):
            result[index] = cv2.GaussianBlur(probability[index], (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    denominator = result.sum(axis=0, keepdims=True)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise RuntimeError("Probability renormalization failed")
    result = (result / denominator).astype(np.float32)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=0), 1, atol=1e-5):
        raise RuntimeError("Smoothed probabilities are not normalized")
    return result


def probability_smoothness(probability: np.ndarray) -> dict:
    """Lower values indicate smoother maps, not necessarily better segmentation."""
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("Expected one finite 2-D probability map")
    dx = np.abs(np.diff(probability, axis=1))
    dy = np.abs(np.diff(probability, axis=0))
    laplacian = cv2.Laplacian(probability, cv2.CV_32F, ksize=3)
    return {
        "adjacent_gradient_mad": float((dx.sum() + dy.sum()) / (dx.size + dy.size)),
        "laplacian_mean_absolute": float(np.abs(laplacian).mean()),
    }


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    remaining = mask.astype(np.uint8).copy()
    skeleton = np.zeros_like(remaining)
    while remaining.any():
        opened = cv2.morphologyEx(remaining, cv2.MORPH_OPEN, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(remaining, opened))
        remaining = cv2.erode(remaining, kernel)
    return skeleton.astype(bool)


def fine_process_proxy(baseline_process: np.ndarray, threshold: float = 0.7, maximum_half_width: float = 3.0) -> np.ndarray:
    """Fixed exploratory thin-signal proxy from the unblurred V3 Process baseline.

    This is not a biological label or a validated thin-process segmentation.
    """
    if baseline_process.ndim != 2 or not np.isfinite(baseline_process).all():
        raise ValueError("Expected a finite baseline Process map")
    support = (baseline_process >= threshold).astype(np.uint8)
    if not support.any():
        return np.zeros(support.shape, dtype=bool)
    distance = cv2.distanceTransform(support, cv2.DIST_L2, 5)
    proxy = _morphological_skeleton(support) & (distance <= maximum_half_width)
    return proxy


def fine_signal_retention(baseline_process: np.ndarray, variant_process: np.ndarray, proxy: np.ndarray) -> dict:
    if baseline_process.shape != variant_process.shape or proxy.shape != baseline_process.shape:
        raise ValueError("Fine-signal inputs must share coordinates")
    count = int(proxy.sum())
    if count == 0:
        return {"proxy_pixels": 0, "baseline_mean": None, "variant_mean": None, "retention_ratio": None}
    baseline_mean = float(baseline_process[proxy].mean())
    variant_mean = float(variant_process[proxy].mean())
    return {
        "proxy_pixels": count,
        "baseline_mean": baseline_mean,
        "variant_mean": variant_mean,
        "retention_ratio": variant_mean / baseline_mean if baseline_mean > 0 else None,
    }

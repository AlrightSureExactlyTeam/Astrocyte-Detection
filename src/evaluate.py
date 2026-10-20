"""Evaluate center-point detection and cell counting for U-Net and Cellpose.
Owner: Liu (metrics), Zhu (error review) | Inputs: predicted and ground-truth Nx2 points.
"""
from typing import Dict
import numpy as np


def match_points(pred_points: np.ndarray, gt_points: np.ndarray, max_distance: float):
    """One-to-one match predictions to GT within max_distance (same coordinate units)."""
    # TODO [Liu]: compute pairwise distances; use one-to-one assignment.
    # TODO [Liu]: reject matches beyond threshold; handle empty arrays.
    # TODO [Liu + Zhu]: choose matching distance using pixel size / annotation convention.
    # Expected return: {'tp': int, 'fp': int, 'fn': int, 'matches': list}
    raise NotImplementedError("TODO: one-to-one point matching")


def detection_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Return precision, recall, and F1 with documented zero-denominator handling."""
    # TODO [Liu]: calculate precision, recall, F1.
    raise NotImplementedError("TODO: detection metrics")


def counting_metrics(pred_counts, gt_counts) -> Dict[str, float]:
    """Return MAE and RMSE over per-image cell counts."""
    # TODO [Liu]: check array lengths, compute MAE and RMSE.
    raise NotImplementedError("TODO: counting metrics")


def evaluate_dataset(predictions: dict, annotations: dict, max_distance: float):
    """Compare predictions and GT for each test image and aggregate results."""
    # TODO [Liu]: evaluate each image, sum TP/FP/FN, calculate counting errors.
    # TODO [Liu]: compare U-Net and Cellpose on the SAME held-out test images.
    # TODO [Zhu]: review representative false positives and false negatives.
    raise NotImplementedError("TODO: dataset evaluation")


def export_report(metrics: dict, output_path: str):
    """Save machine-readable metrics for the final report."""
    # TODO [Liu]: export summary as JSON/CSV with experiment settings.
    raise NotImplementedError("TODO: evaluation report")

"""Predict visible astrocyte morphology, instance masks, centers and counts.
Owner: Zhang | Input: image + checkpoint | Output: masks, centers, count, overlay.
REDESIGN: prediction contract includes semantic and center outputs; instance
assignment requires separate postprocessing, not merely peak extraction.
"""
from pathlib import Path
import numpy as np
import torch


def load_trained_model(checkpoint_path: str, config: dict, device: str):
    """Return a trained model in eval mode."""
    # TODO [Zhang]: build architecture using model.py with matching channels.
    # TODO [Zhang]: load checkpoint safely, set model.eval(), move to device.
    # REDESIGN TODO [Zhang]: check checkpoint schema/output heads and class IDs;
    # reject legacy single-heatmap weights rather than silently loading them.
    raise NotImplementedError("TODO: load trained model")


@torch.no_grad()
def predict_heatmap(model, image_tensor: torch.Tensor, device: str) -> np.ndarray:
    """REDESIGN: Return {'semantic_prob': [3,H,W], 'center_prob': [H,W]}."""
    # TODO [Zhang]: add batch dimension, run model, apply sigmoid if logits.
    # REDESIGN TODO [Zhang]: apply softmax ONLY to semantic_logits, sigmoid ONLY
    # to center_logits; convert both to CPU NumPy and preserve image coordinates.
    # TODO [Zhang]: convert to CPU NumPy and preserve image coordinates.
    raise NotImplementedError("TODO: predict heatmap")


def heatmap_to_points(heatmap: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    """Return N x 2 array of (x,y) centers from local maxima."""
    # TODO [Zhang]: use skimage.feature.peak_local_max or equivalent.
    # REDESIGN TODO [Zhang]: delegate to postprocessing.extract_centers()
    # to prevent inconsistent thresholds and peak handling.
    # TODO [Zhang + Zhu]: calibrate threshold and distance on validation data only.
    raise NotImplementedError("TODO: decode heatmap")


def predict_image(image_path: str, checkpoint_path: str, config: dict):
    """Run the complete single-image pipeline and return predictions."""
    # TODO [Zhang]: call preprocessing.py, model inference, and heatmap_to_points.
    # REDESIGN TODO [Zhang]: return semantic_mask [H,W], instance_mask [H,W],
    # points [N,2], count int, center_prob [H,W], semantic_prob [3,H,W],
    # plus provenance (image size, scale, tile offsets, uncertainty flags).
    # REDESIGN TODO [Zhang]: separate instances using soma/process evidence;
    # do not equate number of raw peaks with number of valid cell instances.
    raise NotImplementedError("TODO: single-image prediction")


def predict_whole_slide(slide_path: str, checkpoint_path: str, config: dict):
    """Predict QPTIFF in tiles and merge coordinates into whole-slide space."""
    # TODO [Zhang]: iterate tiles with slide reader; keep x/y offsets and resolution level.
    # TODO [Zhang]: merge overlapping detections and avoid double counting.
    # REDESIGN TODO [Zhang]: stitch semantic and instance masks using overlap,
    # tile offsets, scale/level metadata, and stable global instance IDs;
    # stream results without materializing the entire high-resolution slide.
    raise NotImplementedError("TODO: whole-slide prediction")


def save_predictions(points: np.ndarray, image, output_dir: str):
    """REDESIGN: Export instance/semantic masks, per-cell CSV, count, overlay."""
    # TODO [Zhang]: export (x,y) coordinates and an overlay image.
    # REDESIGN TODO [Zhang]: include instance ID, visible soma/process area,
    # uncertainty status, and slide-level coordinate metadata; save label masks
    # losslessly (not normalized uint8 when instance IDs exceed 255).
    # REDESIGN TODO [Zhang]: for large slides save scalable masks/tiles and
    # thumbnail overlays rather than full-resolution PNGs.
    raise NotImplementedError("TODO: save prediction outputs")


def main():
    # TODO [Zhang]: parse command-line arguments and call predict_image / predict_whole_slide.
    raise NotImplementedError("TODO: prediction CLI")


if __name__ == "__main__":
    main()

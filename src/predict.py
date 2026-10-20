"""Run the trained U-Net and export astrocyte centers and counts.
Owner: Zhang | Input: image + checkpoint | Output: center coordinates, count, overlay/CSV.
"""
from pathlib import Path
import numpy as np
import torch


def load_trained_model(checkpoint_path: str, config: dict, device: str):
    """Return a trained model in eval mode."""
    # TODO [Zhang]: build architecture using model.py with matching channels.
    # TODO [Zhang]: load checkpoint safely, set model.eval(), move to device.
    raise NotImplementedError("TODO: load trained model")


@torch.no_grad()
def predict_heatmap(model, image_tensor: torch.Tensor, device: str) -> np.ndarray:
    """Return a [H,W] probability heatmap for one input image."""
    # TODO [Zhang]: add batch dimension, run model, apply sigmoid if logits.
    # TODO [Zhang]: convert to CPU NumPy and preserve image coordinates.
    raise NotImplementedError("TODO: predict heatmap")


def heatmap_to_points(heatmap: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    """Return N x 2 array of (x,y) centers from local maxima."""
    # TODO [Zhang]: use skimage.feature.peak_local_max or equivalent.
    # TODO [Zhang + Zhu]: calibrate threshold and distance on validation data only.
    raise NotImplementedError("TODO: decode heatmap")


def predict_image(image_path: str, checkpoint_path: str, config: dict):
    """Run the complete single-image pipeline and return predictions."""
    # TODO [Zhang]: call preprocessing.py, model inference, and heatmap_to_points.
    # TODO [Zhang]: return {'points': Nx2, 'count': int, 'heatmap': HxW}.
    raise NotImplementedError("TODO: single-image prediction")


def predict_whole_slide(slide_path: str, checkpoint_path: str, config: dict):
    """Predict QPTIFF in tiles and merge coordinates into whole-slide space."""
    # TODO [Zhang]: iterate tiles with slide reader; keep x/y offsets and resolution level.
    # TODO [Zhang]: merge overlapping detections and avoid double counting.
    raise NotImplementedError("TODO: whole-slide prediction")


def save_predictions(points: np.ndarray, image, output_dir: str):
    """Write CSV, annotated image, and cell count."""
    # TODO [Zhang]: export (x,y) coordinates and an overlay image.
    raise NotImplementedError("TODO: save prediction outputs")


def main():
    # TODO [Zhang]: parse command-line arguments and call predict_image / predict_whole_slide.
    raise NotImplementedError("TODO: prediction CLI")


if __name__ == "__main__":
    main()

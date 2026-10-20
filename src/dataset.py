"""BBBC042 dataset interface.
Owner: Zhang | Input: image files + point annotations | Output: image tensor + heatmap.
All incomplete implementation is marked with # TODO.
"""
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AstrocyteDataset(Dataset):
    """One sample returns (image, target_heatmap, metadata)."""

    def __init__(self, image_dir: str, annotation_dir: str, image_size=(512, 512), transform=None):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.image_size = image_size
        self.transform = transform
        self.samples = []
        # TODO [Zhang]: inspect BBBC042 filenames and annotation formats.
        # TODO [Zhang]: pair each image with its point-annotation file.
        # TODO [Zhang]: split by source image/slide before extracting tiles to prevent leakage.
        raise NotImplementedError("TODO: build image/annotation sample index")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # TODO [Zhang]: load image via preprocessing.py.
        # TODO [Zhang]: read points as (x, y), convert to patch coordinates if needed.
        # TODO [Zhang]: apply identical geometric transforms to image and points.
        # TODO [Zhang]: generate Gaussian center heatmap with shape [1,H,W].
        # TODO [Zhang]: return float32 image tensor [C,H,W], heatmap tensor [1,H,W], metadata.
        raise NotImplementedError("TODO: load and transform one sample")


def read_point_annotations(annotation_path: str) -> np.ndarray:
    """Return N x 2 array of cell centers (x, y), in pixel coordinates."""
    # TODO [Zhang]: implement actual BBBC042 annotation parsing after inspecting sample files.
    raise NotImplementedError("TODO: parse annotation file")


def points_to_heatmap(points: np.ndarray, height: int, width: int, sigma: float = 3.0) -> np.ndarray:
    """Create [H,W] heatmap from N x 2 centers, with Gaussian peaks."""
    # TODO [Zhang]: draw Gaussian peaks, clip to image bounds, handle zero cells.
    # TODO [Liu + Zhang]: agree on sigma and coordinate convention with model training.
    raise NotImplementedError("TODO: create target heatmap")


def build_dataloaders(config: dict):
    """Return training and validation DataLoaders."""
    # TODO [Zhang]: construct independent train/validation datasets.
    # TODO [Zhang]: set batch size, workers, shuffling, and reproducible random seed.
    raise NotImplementedError("TODO: build DataLoaders")

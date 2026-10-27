"""Astrocyte dataset interface for visible morphology + center supervision.
Owner: Zhang | Input: images, masks and optional point annotations.
REDESIGN: BBBC042 point-only labels cannot supervise soma/process segmentation.
All incomplete implementation remains marked with # TODO.
"""
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AstrocyteDataset(Dataset):
    """REDESIGN: Return (image, targets, metadata).

    targets contract (not implemented yet):
      semantic_mask: [H,W] int64, 0 background / 1 soma / 2 visible process;
      instance_mask: [H,W] int32/int64, 0 background / 1..N instance IDs;
      center_heatmap: [1,H,W] float32, values [0,1];
      valid_mask: [H,W] bool, false for unlabeled/ambiguous pixels.
    Instance IDs must be unique within each image, not across all slides.
    """

    def __init__(self, image_dir: str, annotation_dir: str, image_size=(512, 512), transform=None):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.image_size = image_size
        self.transform = transform
        self.samples = []
        # TODO [Zhang]: inspect BBBC042 filenames and annotation formats.
        # TODO [Zhang]: pair each image with its point-annotation file.
        # REDESIGN TODO [Zhang + Zhu]: inspect availability of expert soma/process
        # semantic masks and per-cell instance masks; point-only samples must not
        # be treated as fully labeled background in segmentation training.
        # REDESIGN TODO [Zhang]: index masks and annotation provenance per sample;
        # distinguish missing annotation from a verified empty-cell image.
        # TODO [Zhang]: split by source image/slide before extracting tiles to prevent leakage.
        raise NotImplementedError("TODO: build image/annotation sample index")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # TODO [Zhang]: load image via preprocessing.py.
        # TODO [Zhang]: read points as (x, y), convert to patch coordinates if needed.
        # TODO [Zhang]: apply identical geometric transforms to image and points.
        # REDESIGN TODO [Zhang]: transform semantic/instance/valid masks with
        # nearest-neighbor interpolation; do not alter discrete instance IDs.
        # REDESIGN TODO [Zhang + Zhu]: define ambiguous crossing processes and
        # partially visible cells as ignore/uncertain regions in valid_mask.
        # TODO [Zhang]: generate Gaussian center heatmap with shape [1,H,W].
        # REDESIGN TODO [Zhang]: return image [C,H,W] float32, targets dict
        # (semantic_mask, instance_mask, center_heatmap, valid_mask), metadata
        # (slide_id, tile origin, level, scale, annotation availability).
        raise NotImplementedError("TODO: load and transform one sample")


def read_point_annotations(annotation_path: str) -> np.ndarray:
    """Return N x 2 array of cell centers (x, y), in pixel coordinates."""
    # TODO [Zhang]: implement actual BBBC042 annotation parsing after inspecting sample files.
    raise NotImplementedError("TODO: parse annotation file")


# REDESIGN TODO [Zhang]: add read_semantic_mask() and read_instance_mask()
# only after checking actual annotation files; verify ID-to-cell ownership and
# semantic/instance consistency. Do not synthesize process labels from points.


def points_to_heatmap(points: np.ndarray, height: int, width: int, sigma: float = 3.0) -> np.ndarray:
    """Create [H,W] heatmap from N x 2 centers, with Gaussian peaks."""
    # TODO [Zhang]: draw Gaussian peaks, clip to image bounds, handle zero cells.
    # TODO [Liu + Zhang]: agree on sigma and coordinate convention with model training.
    raise NotImplementedError("TODO: create target heatmap")


def build_dataloaders(config: dict):
    """Return training and validation DataLoaders."""
    # TODO [Zhang]: construct independent train/validation datasets.
    # REDESIGN TODO [Zhang]: split by slide/patient/source before tiling; ensure
    # batches can distinguish point-only and fully annotated samples.
    # TODO [Zhang]: set batch size, workers, shuffling, and reproducible random seed.
    raise NotImplementedError("TODO: build DataLoaders")

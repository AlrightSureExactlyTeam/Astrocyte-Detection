"""BBBC042 point-supervised dataset and center-heatmap generation.

Point-only annotations supervise the center head. They are never converted into
synthetic soma/process labels; semantic supervision requires real mask files.
"""

from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

if __package__:
    from .preprocessing import load_image, prepare_image
else:
    from preprocessing import load_image, prepare_image

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff"}


def _validate_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        raise ValueError("points must be a finite (N,2) array of (x,y) coordinates")
    return arr


def _index_by_stem(folder: Path, extensions: set[str]) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")
    files = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    index: dict[str, Path] = {}
    for path in files:
        if path.stem in index:
            raise ValueError(f"Duplicate filename stem {path.stem!r} in {folder}")
        index[path.stem] = path
    return index


class AstrocyteDataset(Dataset):
    """Return ``(image, targets, metadata)`` for one resized source image.

    ``image_size`` is ``(height, width)``. Point coordinates are resized with
    the image. Optional semantic/instance mask directories use matching stems;
    no mask content is inferred from point annotations.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        image_size=(512, 512),
        transform=None,
        sigma: float = 3.0,
        channel: int | None = None,
        semantic_mask_dir: str | None = None,
        instance_mask_dir: str | None = None,
    ):
        if (
            not isinstance(image_size, (tuple, list))
            or len(image_size) != 2
            or any(not isinstance(v, (int, np.integer)) or v <= 0 for v in image_size)
        ):
            raise ValueError("image_size must be a positive (height,width) pair")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")

        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.transform = transform
        self.sigma = float(sigma)
        self.channel = channel

        images = _index_by_stem(self.image_dir, IMAGE_EXTENSIONS)
        annotations = _index_by_stem(self.annotation_dir, {".txt", ".csv"})
        missing_annotations = sorted(images.keys() - annotations.keys())
        missing_images = sorted(annotations.keys() - images.keys())
        if missing_annotations or missing_images:
            raise ValueError(
                f"Image/annotation mismatch: missing annotations={missing_annotations[:10]}, "
                f"missing images={missing_images[:10]}"
            )
        if not images:
            raise ValueError("No supported image/annotation pairs found")

        semantic_masks = (
            _index_by_stem(Path(semantic_mask_dir), IMAGE_EXTENSIONS)
            if semantic_mask_dir is not None else {}
        )
        instance_masks = (
            _index_by_stem(Path(instance_mask_dir), IMAGE_EXTENSIONS)
            if instance_mask_dir is not None else {}
        )

        # IMPLEMENTED [Liu]: index complete samples and annotation provenance.
        self.samples = [
            {
                "sample_id": stem,
                "image_path": images[stem],
                "annotation_path": annotations[stem],
                "semantic_mask_path": semantic_masks.get(stem),
                "instance_mask_path": instance_masks.get(stem),
            }
            for stem in sorted(images)
        ]

    def __len__(self):
        return len(self.samples)

    def _load_mask(self, path: Path | None, original_hw: tuple[int, int]) -> np.ndarray | None:
        if path is None:
            return None
        mask = load_image(path)
        mask = np.asarray(mask)
        if mask.ndim != 2 or mask.shape != original_hw:
            raise ValueError(
                f"Mask {path} must be 2D with shape {original_hw}, got {mask.shape}"
            )
        if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
            raise ValueError(f"Mask {path} must contain non-negative integer labels")
        height, width = self.image_size
        return cv2.resize(
            mask.astype(np.int64), (width, height), interpolation=cv2.INTER_NEAREST
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        raw_image = load_image(sample["image_path"])
        original_height, original_width = raw_image.shape[:2]
        points = read_point_annotations(sample["annotation_path"])

        # IMPLEMENTED [Liu]: apply identical resize geometry to image and points.
        image = prepare_image(raw_image, channel=self.channel).unsqueeze(0)
        height, width = self.image_size
        image = F.interpolate(
            image, size=self.image_size, mode="bilinear", align_corners=False
        )[0]
        scaled_points = points.copy()
        if len(scaled_points):
            scaled_points[:, 0] *= width / original_width
            scaled_points[:, 1] *= height / original_height
        if self.transform is not None:
            image, scaled_points = self.transform(image, scaled_points)
            scaled_points = _validate_points(scaled_points)

        semantic_mask = self._load_mask(
            sample["semantic_mask_path"], (original_height, original_width)
        )
        instance_mask = self._load_mask(
            sample["instance_mask_path"], (original_height, original_width)
        )
        if semantic_mask is not None and np.any(semantic_mask > 2):
            raise ValueError("semantic masks must use labels 0=background, 1=soma, 2=process")

        heatmap = points_to_heatmap(scaled_points, height, width, self.sigma)
        targets = {
            "semantic_mask": torch.from_numpy(
                np.zeros((height, width), dtype=np.int64)
                if semantic_mask is None else semantic_mask.astype(np.int64)
            ),
            "instance_mask": torch.from_numpy(
                np.zeros((height, width), dtype=np.int64)
                if instance_mask is None else instance_mask.astype(np.int64)
            ),
            "center_heatmap": torch.from_numpy(heatmap[None].astype(np.float32)),
            # Center supervision is available for every parsed point sample.
            "valid_mask": torch.ones((height, width), dtype=torch.bool),
            # Point-only samples must not train the semantic head as background.
            "semantic_valid_mask": torch.full(
                (height, width), semantic_mask is not None, dtype=torch.bool
            ),
        }
        metadata = {
            "sample_id": sample["sample_id"],
            "image_path": str(sample["image_path"]),
            "annotation_path": str(sample["annotation_path"]),
            "original_size": (original_height, original_width),
            "image_size": self.image_size,
            "scale_xy": (width / original_width, height / original_height),
            "tile_offset": (0, 0),
            "level": 0,
            "annotation_available": True,
            "semantic_available": semantic_mask is not None,
            "instance_available": instance_mask is not None,
        }
        return image, targets, metadata


def read_point_annotations(annotation_path: str | Path) -> np.ndarray:
    """Read BBBC042 boxes or simple ``x,y`` rows and return center points.

    BBBC042 rows are ``Cell ... x1 y1 x2 y2 ...``; the center is the box midpoint.
    Empty files represent an explicitly empty annotation, not a missing label.
    """
    path = Path(annotation_path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    points = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                if line.startswith("Cell"):
                    fields = line.split()
                    if len(fields) < 8:
                        raise ValueError("BBBC042 row has fewer than 8 fields")
                    x1, y1, x2, y2 = map(float, fields[4:8])
                    if x2 < x1 or y2 < y1:
                        raise ValueError("BBBC042 box corners are reversed")
                    points.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
                else:
                    if line.lower().replace(" ", "").startswith(("x,y", "cell_id,x,y")):
                        continue
                    fields = line.replace(",", " ").split()
                    if len(fields) == 4 and fields[0].isdigit():
                        # Optional cell_id,x,y,status CSV format.
                        fields = fields[1:3]
                    if len(fields) < 2:
                        raise ValueError("Expected x,y coordinates")
                    points.append((float(fields[0]), float(fields[1])))
            except ValueError as exc:
                raise ValueError(f"Invalid annotation at {path}:{line_number}: {exc}") from exc
    return _validate_points(np.asarray(points, dtype=np.float64).reshape(-1, 2))


def points_to_heatmap(
    points: np.ndarray, height: int, width: int, sigma: float = 3.0
) -> np.ndarray:
    """Create an ``[H,W]`` heatmap with bounded Gaussian peaks."""
    if not isinstance(height, (int, np.integer)) or not isinstance(width, (int, np.integer)):
        raise ValueError("height and width must be integers")
    if height <= 0 or width <= 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("height/width and sigma must be positive")
    points = _validate_points(points)
    heatmap = np.zeros((height, width), dtype=np.float32)
    radius = max(1, int(np.ceil(3.0 * float(sigma))))
    for x, y in points:
        if not (0 <= x < width and 0 <= y < height):
            continue
        x0, x1 = max(0, int(np.floor(x)) - radius), min(width, int(np.floor(x)) + radius + 1)
        y0, y1 = max(0, int(np.floor(y)) - radius), min(height, int(np.floor(y)) + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        peak = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * float(sigma) ** 2))
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], peak)
    return heatmap


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloaders(config: dict):
    """Build deterministic train/validation loaders split by source image."""
    data_cfg = config["data"]
    train_cfg = config["training"]
    if int(data_cfg.get("input_channels", 1)) != 1:
        raise ValueError("The current dataset pipeline supports one input channel")

    dataset = AstrocyteDataset(
        data_cfg["images_dir"],
        data_cfg["annotations_dir"],
        image_size=tuple(data_cfg.get("image_size", (512, 512))),
        sigma=float(config.get("heatmap", {}).get("sigma_pixels", 3.0)),
        channel=data_cfg.get("channel"),
        semantic_mask_dir=data_cfg.get("semantic_mask_dir"),
        instance_mask_dir=data_cfg.get("instance_mask_dir"),
    )
    train_fraction = float(data_cfg.get("train_fraction", 0.7))
    val_fraction = float(data_cfg.get("val_fraction", 0.15))
    test_fraction = float(data_cfg.get("test_fraction", 0.15))
    fractions = np.asarray([train_fraction, val_fraction, test_fraction])
    if not np.isfinite(fractions).all() or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train/val/test fractions must be nonnegative and sum to 1")

    indices = list(range(len(dataset)))
    random.Random(int(config.get("project", {}).get("seed", 42))).shuffle(indices)
    n_train = int(len(indices) * train_fraction)
    n_val = int(len(indices) * val_fraction)
    if n_train == 0 or n_val == 0:
        raise ValueError("Dataset split must contain at least one training and one validation image")
    train_subset = Subset(dataset, indices[:n_train])
    val_subset = Subset(dataset, indices[n_train:n_train + n_val])

    seed = int(config.get("project", {}).get("seed", 42))
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": int(train_cfg.get("batch_size", 8)),
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "worker_init_fn": _seed_worker,
        "pin_memory": bool(torch.cuda.is_available()),
    }
    train_loader = DataLoader(train_subset, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_subset, shuffle=False, **common)
    return train_loader, val_loader

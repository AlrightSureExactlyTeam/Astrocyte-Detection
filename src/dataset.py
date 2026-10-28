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
        self.supervision_summary = {
            "sample_count": len(self.samples),
            "semantic_supervised_samples": sum(
                sample["semantic_mask_path"] is not None for sample in self.samples
            ),
            "instance_supervised_samples": sum(
                sample["instance_mask_path"] is not None for sample in self.samples
            ),
        }

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
        # FIX [Liu]: reject invalid source coordinates before resizing can hide them.
        out_of_bounds = (
            (points[:, 0] < 0) | (points[:, 0] >= original_width)
            | (points[:, 1] < 0) | (points[:, 1] >= original_height)
        )
        if np.any(out_of_bounds):
            examples = points[out_of_bounds][:5].tolist()
            raise ValueError(
                f"Out-of-bounds annotations in sample {sample['sample_id']!r}: "
                f"image size (width={original_width}, height={original_height}), "
                f"point examples={examples}"
            )

        # IMPLEMENTED [Liu]: apply identical resize geometry to image and points.
        image = prepare_image(raw_image, channel=self.channel).unsqueeze(0)
        height, width = self.image_size
        resize_scale_xy = (width / original_width, height / original_height)
        image = F.interpolate(
            image, size=self.image_size, mode="bilinear", align_corners=False
        )[0]
        scaled_points = points.copy()
        if len(scaled_points):
            scaled_points[:, 0] *= resize_scale_xy[0]
            scaled_points[:, 1] *= resize_scale_xy[1]
        semantic_mask = self._load_mask(
            sample["semantic_mask_path"], (original_height, original_width)
        )
        instance_mask = self._load_mask(
            sample["instance_mask_path"], (original_height, original_width)
        )

        if self.transform is not None:
            has_masks = semantic_mask is not None or instance_mask is not None
            if has_masks:
                # FIX [Liu]: legacy two-output transforms cannot keep mask labels aligned.
                transform_with_masks = getattr(self.transform, "transform_with_masks", None)
                if not callable(transform_with_masks):
                    raise ValueError(
                        "Transforms on mask-bearing samples must implement "
                        "transform_with_masks(image, points, semantic_mask, instance_mask)"
                    )
                transformed = transform_with_masks(
                    image, scaled_points, semantic_mask, instance_mask
                )
                if not isinstance(transformed, (tuple, list)) or len(transformed) != 4:
                    raise ValueError("transform_with_masks must return four outputs")
                image, scaled_points, semantic_mask, instance_mask = transformed
            else:
                transformed = self.transform(image, scaled_points)
                if not isinstance(transformed, (tuple, list)) or len(transformed) != 2:
                    raise ValueError("transform must return (image, points)")
                image, scaled_points = transformed
            scaled_points = _validate_points(scaled_points)

        if not isinstance(image, torch.Tensor) or image.ndim != 3:
            raise ValueError("Transformed image must be a [C,H,W] torch.Tensor")
        height, width = int(image.shape[-2]), int(image.shape[-1])
        if height <= 0 or width <= 0:
            raise ValueError("Transformed image must have non-empty spatial dimensions")
        for name, mask in (("semantic_mask", semantic_mask), ("instance_mask", instance_mask)):
            if mask is None:
                continue
            mask = np.asarray(mask)
            if (
                mask.shape != (height, width)
                or not np.issubdtype(mask.dtype, np.integer)
                or np.any(mask < 0)
            ):
                raise ValueError(
                    f"Transformed {name} must be a non-negative integer [{height},{width}] array"
                )
            if name == "semantic_mask":
                semantic_mask = mask
            else:
                instance_mask = mask
        if semantic_mask is not None and np.any(semantic_mask > 2):
            raise ValueError("semantic masks must use labels 0=background, 1=soma, 2=process")

        # FIX [Liu]: reject transformed points outside the final crop/canvas.
        out_of_bounds = (
            (scaled_points[:, 0] < 0) | (scaled_points[:, 0] >= width)
            | (scaled_points[:, 1] < 0) | (scaled_points[:, 1] >= height)
        )
        if np.any(out_of_bounds):
            raise ValueError(
                f"Transform produced out-of-bounds points in sample {sample['sample_id']!r}: "
                f"{scaled_points[out_of_bounds][:5].tolist()}"
            )
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
            "image_size": (height, width),
            "scale_xy": resize_scale_xy,
            "tile_offset": (0, 0),
            "level": 0,
            "annotation_available": True,
            "annotation_empty": len(points) == 0,
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


def _group_split(dataset, group_map, fractions, seed):
    """Split explicit source groups without inferring identities from filenames."""
    if not isinstance(group_map, dict):
        raise ValueError("data.groups must map every sample_id to a reliable source group")
    sample_ids = [sample["sample_id"] for sample in dataset.samples]
    missing = sorted(set(sample_ids) - set(group_map))
    if missing:
        raise ValueError(f"data.groups is missing sample IDs: {missing[:10]}")
    grouped_indices = {}
    for index, sample_id in enumerate(sample_ids):
        group_id = group_map[sample_id]
        if not isinstance(group_id, (str, int, np.integer)) or not str(group_id).strip():
            raise ValueError("Group IDs must be non-empty strings or integers")
        grouped_indices.setdefault(group_id, []).append(index)
    if len(grouped_indices) < 2:
        raise ValueError("Group splitting needs at least two source groups")

    groups = sorted(grouped_indices, key=str)
    random.Random(seed).shuffle(groups)
    buckets = [[], [], []]
    counts = [0, 0, 0]
    total = len(dataset)
    # FIX [Liu]: assign whole groups to the largest remaining sample deficit.
    tie_priority = (2, 0, 1)  # validation wins a tie, then test, then training.
    for group_id in groups:
        deficits = [fractions[i] * total - counts[i] for i in range(3)]
        partition = max(range(3), key=lambda i: (deficits[i], -tie_priority[i]))
        buckets[partition].extend(grouped_indices[group_id])
        counts[partition] += len(grouped_indices[group_id])
    return buckets, grouped_indices


def build_dataloaders(config: dict):
    """Build reproducible loaders, optionally split by explicit source groups."""
    data_cfg = config["data"]
    train_cfg = config["training"]
    input_channels = data_cfg.get("input_channels", 1)
    if isinstance(input_channels, (bool, np.bool_)) or not isinstance(
        input_channels, (int, np.integer)
    ) or input_channels != 1:
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
    fractions = np.asarray([
        float(data_cfg.get("train_fraction", 0.7)),
        float(data_cfg.get("val_fraction", 0.15)),
        float(data_cfg.get("test_fraction", 0.15)),
    ])
    if not np.isfinite(fractions).all() or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train/val/test fractions must be nonnegative and sum to 1")
    seed = config.get("project", {}).get("seed", 42)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("project.seed must be a non-negative integer")

    group_map = data_cfg.get("groups")
    if group_map is not None:
        buckets, grouped_indices = _group_split(dataset, group_map, fractions, seed)
        train_indices, val_indices = buckets[0], buckets[1]
    else:
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        n_train = int(len(indices) * fractions[0])
        n_val = int(len(indices) * fractions[1])
        buckets = [indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]]
        grouped_indices = None
        train_indices, val_indices = buckets[0], buckets[1]
    if not train_indices or not val_indices:
        raise ValueError("Dataset split must contain at least one training and one validation image")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 0)
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
        batch_size, (int, np.integer)
    ) or batch_size <= 0:
        raise ValueError("training.batch_size must be a positive integer")
    if isinstance(num_workers, (bool, np.bool_)) or not isinstance(
        num_workers, (int, np.integer)
    ) or num_workers < 0:
        raise ValueError("training.num_workers must be a non-negative integer")
    common = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "worker_init_fn": _seed_worker,
        "pin_memory": bool(torch.cuda.is_available()),
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_subset, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_subset, shuffle=False, **common)
    summary = {
        "train_samples": len(buckets[0]),
        "validation_samples": len(buckets[1]),
        "test_samples": len(buckets[2]),
        "grouped_split": group_map is not None,
        **dataset.supervision_summary,
    }
    if grouped_indices is not None:
        summary["source_group_count"] = len(grouped_indices)
        summary["train_groups"] = len({
            group_map[dataset.samples[index]["sample_id"]] for index in buckets[0]
        })
        summary["validation_groups"] = len({
            group_map[dataset.samples[index]["sample_id"]] for index in buckets[1]
        })
    train_loader.split_summary = summary
    val_loader.split_summary = summary.copy()
    if summary["semantic_supervised_samples"] == 0:
        import warnings
        warnings.warn(
            "No semantic masks are available; training will supervise only center heatmaps.",
            UserWarning,
            stacklevel=2,
        )
    return train_loader, val_loader

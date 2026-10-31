"""Original-resolution 256×256 patch datasets for manual and weak GFAP pilots."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from .unet_smoke import MANUAL_IGNORE, MANUAL_LABEL_VALUES, ManualDenseLabelDataset
from .workflow import PROJECT_ROOT
from src.preprocessing import load_image, prepare_image

VALID_STATUSES = {"valid", "valid_with_warnings"}


def _patch_origins(length: int, patch: int, stride: int) -> list[int]:
    if length < patch:
        raise ValueError(f"Image extent {length} is smaller than patch {patch}")
    origins = list(range(0, length - patch + 1, stride))
    if origins[-1] != length - patch:
        origins.append(length - patch)
    return origins


def _resolve(workspace: Path, relative: str) -> Path:
    path = (workspace / relative).resolve()
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Dataset path escapes workspace: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _check_source(row: dict, image_path: Path) -> None:
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if row["source_sha256"] and digest != row["source_sha256"]:
        raise ValueError(f"Source checksum mismatch: {row['sample_id']}")


class ManualPatchDataset(Dataset):
    """Non-resized dense-label patches containing at least one valid manual pixel."""

    def __init__(
        self,
        manifest_path: str | Path,
        workspace_root: str | Path = PROJECT_ROOT,
        patch_size: int = 256,
        stride: int = 256,
        channel: int = 1,
        slide_groups: set[str] | None = None,
    ) -> None:
        self.patch_size = self._positive_int(patch_size, "patch_size")
        self.stride = self._positive_int(stride, "stride")
        self.channel = channel
        base = ManualDenseLabelDataset(manifest_path, workspace_root)
        self.workspace_root = Path(workspace_root).resolve()
        self.records = [
            row for row in base.records
            if slide_groups is None or row["slide_sample_id"] in slide_groups
        ]
        if not self.records:
            raise ValueError("No manual samples in requested slide groups")
        self.tiles = []
        self.patches = []
        for row in self.records:
            image_path = _resolve(self.workspace_root, row["image_path"])
            label_path = _resolve(self.workspace_root, row["training_label_path"])
            _check_source(row, image_path)
            image = load_image(image_path)
            label = tifffile.imread(label_path)
            if label.ndim != 2 or label.shape != image.shape[:2]:
                raise ValueError(f"Image/label shape mismatch: {row['sample_id']}")
            values = {int(value) for value in np.unique(label)}
            if not values.issubset(MANUAL_LABEL_VALUES):
                raise ValueError(f"Unexpected manual labels: {row['sample_id']}")
            tile_index = len(self.tiles)
            self.tiles.append((image, label))
            height, width = label.shape
            for y in _patch_origins(height, self.patch_size, self.stride):
                for x in _patch_origins(width, self.patch_size, self.stride):
                    patch_label = label[y:y + self.patch_size, x:x + self.patch_size]
                    if np.any(patch_label != MANUAL_IGNORE):
                        self.patches.append((tile_index, x, y))
        if not self.patches:
            raise ValueError("No manual patches contain valid supervision")

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @property
    def slide_groups(self) -> set[str]:
        return {row["slide_sample_id"] for row in self.records}

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int):
        tile_index, x, y = self.patches[index]
        row = self.records[tile_index]
        image, label = self.tiles[tile_index]
        size = self.patch_size
        image_patch = image[y:y + size, x:x + size]
        label_patch = label[y:y + size, x:x + size]
        image_tensor = prepare_image(image_patch, channel=self.channel)
        labels = torch.from_numpy(label_patch.astype(np.int64, copy=False))
        return image_tensor, {
            "semantic_mask": labels,
            "semantic_valid_mask": labels != MANUAL_IGNORE,
        }, {
            "sample_id": row["sample_id"], "slide_sample_id": row["slide_sample_id"],
            "patch_x": x, "patch_y": y, "patch_size": size,
            "annotation_source": "manual", "annotation_type": "dense",
        }


class ImageJWeakPatchDataset(Dataset):
    """Positive-only ImageJ candidate patches; zero pixels are unknown, not background."""

    def __init__(
        self,
        manifest_path: str | Path,
        workspace_root: str | Path = PROJECT_ROOT,
        patch_size: int = 256,
        stride: int = 256,
        channel: int = 1,
        slide_groups: set[str] | None = None,
    ) -> None:
        self.patch_size = ManualPatchDataset._positive_int(patch_size, "patch_size")
        self.stride = ManualPatchDataset._positive_int(stride, "stride")
        self.channel = channel
        self.workspace_root = Path(workspace_root).resolve()
        with Path(manifest_path).open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.records = [
            row for row in rows
            if row["annotation_source"] == "imagej_pseudo"
            and row["annotation_type"] == "pseudo"
            and row["validation_status"] in VALID_STATUSES
            and (slide_groups is None or row["slide_sample_id"] in slide_groups)
        ]
        if not self.records:
            raise ValueError("No valid ImageJ samples in requested slide groups")
        self.tiles = []
        self.patches = []
        for row in self.records:
            image_path = _resolve(self.workspace_root, row["image_path"])
            mask_path = _resolve(self.workspace_root, row["annotation_path"])
            _check_source(row, image_path)
            image = load_image(image_path)
            mask = tifffile.imread(mask_path)
            if mask.ndim != 2 or mask.shape != image.shape[:2]:
                raise ValueError(f"ImageJ image/mask shape mismatch: {row['sample_id']}")
            if not {int(value) for value in np.unique(mask)}.issubset({0, 255}):
                raise ValueError(f"ImageJ mask must be binary: {row['sample_id']}")
            tile_index = len(self.tiles)
            self.tiles.append((image, mask))
            height, width = mask.shape
            for y in _patch_origins(height, self.patch_size, self.stride):
                for x in _patch_origins(width, self.patch_size, self.stride):
                    if np.any(mask[y:y + self.patch_size, x:x + self.patch_size] == 255):
                        self.patches.append((tile_index, x, y))
        if not self.patches:
            raise ValueError("No ImageJ patches contain positive candidates")

    @property
    def slide_groups(self) -> set[str]:
        return {row["slide_sample_id"] for row in self.records}

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int):
        tile_index, x, y = self.patches[index]
        row = self.records[tile_index]
        image, mask = self.tiles[tile_index]
        size = self.patch_size
        image_patch = image[y:y + size, x:x + size]
        candidate = mask[y:y + size, x:x + size] == 255
        image_tensor = prepare_image(image_patch, channel=self.channel)
        positive = torch.from_numpy(candidate.astype(np.float32))[None]
        return image_tensor, {
            # Zeros are unlabeled/unknown and are always excluded by weak_valid_mask.
            "weak_foreground": positive,
            "weak_valid_mask": positive.bool(),
        }, {
            "sample_id": row["sample_id"], "slide_sample_id": row["slide_sample_id"],
            "patch_x": x, "patch_y": y, "patch_size": size,
            "annotation_source": "imagej_pseudo", "annotation_type": "pseudo",
            "formal_evaluation_ground_truth": False,
        }

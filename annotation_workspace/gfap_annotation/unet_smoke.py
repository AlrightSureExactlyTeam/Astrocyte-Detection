"""Isolated ignore-aware data and loss utilities for the first U-Net smoke test.

This module does not alter the project dataset, training loop, or Cellpose
baseline. Manual label 255 remains the cross-entropy ignore index; it is never
reinterpreted as confirmed background.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .workflow import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_image, prepare_image  # noqa: E402

MANUAL_IGNORE = 255
MANUAL_LABEL_VALUES = {0, 1, 2, MANUAL_IGNORE}


class ManualDenseLabelDataset(Dataset):
    """Load only validated manual dense labels from a dataset manifest.

    ImageJ pseudo-labels, point/mixed annotations, duplicates, and other rows
    not explicitly marked training-suitable are excluded by construction.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        workspace_root: str | Path = PROJECT_ROOT,
        image_size: tuple[int, int] = (256, 256),
        channel: int = 1,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        if (
            not isinstance(image_size, (tuple, list)) or len(image_size) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in image_size)
        ):
            raise ValueError("image_size must be a positive (height, width) pair")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.channel = channel
        with self.manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.records = [
            row for row in rows
            if row["annotation_source"] == "manual"
            and row["annotation_type"] == "dense"
            and row["training_suitable"] == "true"
        ]
        if not self.records:
            raise ValueError("No validated manual dense-label samples found")
        sample_ids = [row["sample_id"] for row in self.records]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Duplicate selected manual sample IDs")

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, relative: str) -> Path:
        path = (self.workspace_root / relative).resolve()
        try:
            path.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(f"Dataset path escapes workspace: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def __getitem__(self, index: int):
        row = self.records[index]
        image_path = self._resolve(row["image_path"])
        label_path = self._resolve(row["training_label_path"])
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if row["source_sha256"] and digest != row["source_sha256"]:
            raise ValueError(f"Source checksum mismatch: {row['sample_id']}")

        raw_image = load_image(image_path)
        raw_label = tifffile.imread(label_path)
        if raw_label.ndim != 2 or raw_label.shape != raw_image.shape[:2]:
            raise ValueError(f"Image/label shape mismatch: {row['sample_id']}")
        if not np.issubdtype(raw_label.dtype, np.integer):
            raise ValueError(f"Manual label is not integer typed: {row['sample_id']}")
        values = {int(value) for value in np.unique(raw_label)}
        if not values.issubset(MANUAL_LABEL_VALUES):
            raise ValueError(f"Unexpected manual label values {sorted(values)}: {row['sample_id']}")
        if not np.any(raw_label != MANUAL_IGNORE):
            raise ValueError(f"Sample has no valid manual pixels: {row['sample_id']}")

        image = prepare_image(raw_image, channel=self.channel).unsqueeze(0)
        image = F.interpolate(
            image, size=self.image_size, mode="bilinear", align_corners=False
        )[0]
        label = cv2.resize(
            raw_label.astype(np.uint8), (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        if not torch.isfinite(image).all() or torch.any(image < 0) or torch.any(image > 1):
            raise ValueError(f"Normalized image outside finite [0,1]: {row['sample_id']}")
        semantic_mask = torch.from_numpy(label.astype(np.int64, copy=False))
        valid_mask = semantic_mask != MANUAL_IGNORE
        metadata = {
            "sample_id": row["sample_id"],
            "slide_sample_id": row["slide_sample_id"],
            "annotation_source": row["annotation_source"],
            "annotation_type": row["annotation_type"],
            "image_path": str(image_path),
            "label_path": str(label_path),
            "source_sha256": digest,
            "original_height": raw_image.shape[0],
            "original_width": raw_image.shape[1],
            "normalization": "per_tile_percentile_1_99_8_clipped_0_1",
            "channel": self.channel,
            "original_label_values": sorted(values),
        }
        return image, {
            "semantic_mask": semantic_mask,
            "semantic_valid_mask": valid_mask,
        }, metadata


def masked_manual_segmentation_loss(outputs: dict, targets: dict) -> dict:
    """Semantic CE over manual pixels only, with native label-255 ignoring."""
    logits = outputs["semantic_logits"]
    labels = targets["semantic_mask"]
    valid = targets["semantic_valid_mask"].bool()
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError("semantic_logits must be [B,3,H,W]")
    if labels.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("semantic_mask must be [B,H,W] and match logits")
    if labels.dtype != torch.int64:
        raise ValueError("semantic_mask must be int64")
    if valid.shape != labels.shape or not torch.equal(valid, labels != MANUAL_IGNORE):
        raise ValueError("semantic_valid_mask must exactly identify labels other than 255")
    if valid.any():
        valid_labels = labels[valid]
        if torch.any(valid_labels < 0) or torch.any(valid_labels > 2):
            raise ValueError("Valid manual labels must be 0, 1, or 2")
    if not valid.any():
        return {
            "loss": logits.sum() * 0.0,
            "per_pixel_loss": torch.zeros_like(labels, dtype=logits.dtype),
            "valid_mask": valid,
        }
    per_pixel = F.cross_entropy(
        logits, labels, ignore_index=MANUAL_IGNORE, reduction="none"
    )
    loss = per_pixel[valid].mean()
    return {"loss": loss, "per_pixel_loss": per_pixel, "valid_mask": valid}

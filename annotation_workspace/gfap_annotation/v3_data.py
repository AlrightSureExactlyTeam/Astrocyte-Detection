"""Partially supervised, Soma-anchored instance crops for the isolated V3 pilot.

Only confirmed component pixels supervise V3. Source semantic Background (0),
unannotated pixels, Uncertain regions, and cross-cell conflicts never become
negative examples. Association targets exist only for complete Soma/Process
pairs with a visible in-crop Soma anchor.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from .workflow import PROJECT_ROOT
from src.preprocessing import load_image, prepare_image

MORPHOLOGY_IGNORE = 255
COMPLETE_CELL_STATUS = "soma_and_process_present"


def _read_binary(path: Path, shape: tuple[int, int]) -> np.ndarray:
    array = tifffile.imread(path)
    if array.shape != shape or array.ndim != 2:
        raise ValueError(f"Component shape mismatch: {path}: {array.shape} != {shape}")
    if not {int(value) for value in np.unique(array)}.issubset({0, 1, 255}):
        raise ValueError(f"Component mask is not binary 0/1 or 0/255: {path}")
    return array > 0


def reconstruct_positive_supervision(
    sample_dir: str | Path,
    metadata: dict,
    expected_shape: tuple[int, int] | None = None,
) -> dict:
    """Rebuild positive-only targets from preserved per-cell components.

    This deliberately does not use source semantic 0 as Background. It also
    recovers confirmed components in samples whose visible Background layer
    caused the converter to mark semantic pixels unknown.
    """
    sample_dir = Path(sample_dir)
    shape = expected_shape or (int(metadata["original_height"]), int(metadata["original_width"]))
    soma_owner = np.zeros(shape, dtype=np.int32)
    process_owner = np.zeros(shape, dtype=np.int32)
    cross_cell_conflict = np.zeros(shape, dtype=bool)
    complete_ids: list[int] = []
    morphology_ids: list[int] = []
    association_excluded_ids: list[int] = []

    for cell in metadata["cells"]:
        cell_id = int(cell["cell_id"])
        component_paths = cell.get("component_mask_paths", {})
        present_kinds = [kind for kind in ("soma", "process") if component_paths.get(kind)]
        if cell.get("annotation_status") == COMPLETE_CELL_STATUS:
            if set(present_kinds) != {"soma", "process"} or cell_id <= 0:
                raise ValueError(f"Invalid complete cell {cell_id} in {sample_dir}")
            complete_ids.append(cell_id)
        elif present_kinds:
            association_excluded_ids.append(cell_id)
        if present_kinds:
            morphology_ids.append(cell_id)
        for kind, owner in (("soma", soma_owner), ("process", process_owner)):
            relative = component_paths.get(kind)
            if not relative:
                continue
            mask = _read_binary(sample_dir / relative, shape)
            other_owner = mask & (owner > 0) & (owner != cell_id)
            cross_cell_conflict |= other_owner
            owner[mask & (owner == 0)] = cell_id

    different_type_conflict = (
        (soma_owner > 0) & (process_owner > 0) & (soma_owner != process_owner)
    )
    cross_cell_conflict |= different_type_conflict
    uncertain = _read_binary(sample_dir / "uncertain_mask.tif", shape)
    valid_positive = ~cross_cell_conflict & ~uncertain
    soma_valid = (soma_owner > 0) & valid_positive
    process_valid = (process_owner > 0) & valid_positive

    # Internal two-class mapping: 0=Soma, 1=Process, 255=unknown/ignored.
    morphology = np.full(shape, MORPHOLOGY_IGNORE, dtype=np.uint8)
    morphology[process_valid] = 1
    morphology[soma_valid] = 0  # same-cell Soma priority; cross-cell was removed
    association = np.zeros(shape, dtype=np.int32)
    if complete_ids:
        complete = np.isin(soma_owner, complete_ids)
        association[soma_valid & complete] = soma_owner[soma_valid & complete]
        complete = np.isin(process_owner, complete_ids)
        association[process_valid & complete] = process_owner[process_valid & complete]

    return {
        "morphology": morphology,
        "association": association,
        "uncertain": uncertain,
        "cross_cell_conflict": cross_cell_conflict,
        "soma_owner": soma_owner,
        "process_owner": process_owner,
        "complete_cell_ids": sorted(complete_ids),
        "morphology_cell_ids": sorted(set(morphology_ids)),
        "association_excluded_cell_ids": sorted(set(association_excluded_ids)),
        "diagnostics": {
            "soma_pixels": int(np.count_nonzero(morphology == 0)),
            "process_pixels": int(np.count_nonzero(morphology == 1)),
            "unknown_pixels": int(np.count_nonzero(morphology == MORPHOLOGY_IGNORE)),
            "cross_cell_conflict_pixels": int(np.count_nonzero(cross_cell_conflict)),
            "uncertain_pixels": int(np.count_nonzero(uncertain)),
            "association_pixels": int(np.count_nonzero(association > 0)),
        },
    }


def validate_against_converted_masks(sample_dir: str | Path, reconstructed: dict) -> dict:
    """Require agreement wherever converter semantic labels are confirmed positive."""
    sample_dir = Path(sample_dir)
    semantic = tifffile.imread(sample_dir / "semantic_mask.tif")
    instance = tifffile.imread(sample_dir / "instance_mask.tif")
    morphology = reconstructed["morphology"]
    association = reconstructed["association"]
    if semantic.shape != morphology.shape or instance.shape != morphology.shape:
        raise ValueError(f"Converted mask shape mismatch: {sample_dir}")
    if not {int(v) for v in np.unique(semantic)}.issubset({0, 1, 2, 255}):
        raise ValueError(f"Unexpected semantic values: {sample_dir}")
    expected = np.full(morphology.shape, MORPHOLOGY_IGNORE, dtype=np.uint8)
    expected[morphology == 0] = 1
    expected[morphology == 1] = 2
    confirmed = (semantic == 1) | (semantic == 2)
    mismatch = confirmed & (semantic != expected)
    if mismatch.any():
        raise ValueError(f"Component/semantic disagreement in {sample_dir}: {int(mismatch.sum())} pixels")
    complete_association = confirmed & (association > 0)
    instance_mismatch = complete_association & (instance != association.astype(instance.dtype))
    if instance_mismatch.any():
        raise ValueError(f"Component/instance disagreement in {sample_dir}: {int(instance_mismatch.sum())} pixels")
    return {
        "converter_confirmed_positive_pixels": int(confirmed.sum()),
        "component_positive_pixels_recovered_from_semantic_unknown": int(
            np.count_nonzero((semantic == 255) & (morphology != MORPHOLOGY_IGNORE))
        ),
        "semantic_zero_pixels_ignored_not_background_supervision": int(np.count_nonzero(semantic == 0)),
        "semantic_positive_mismatch_pixels": 0,
        "instance_positive_mismatch_pixels": 0,
    }


def load_v3_manifest(manifest_path: str | Path, split: str | None = None) -> list[dict]:
    with Path(manifest_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if row["included"].lower() == "true" and (split is None or row["split"] == split)
    ]
    if split is not None and not selected:
        raise ValueError(f"No included V3 rows for split={split!r}")
    image_ids = [row["image_id"] for row in selected]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate source images in V3 selection")
    return selected


class SomaAnchoredCropDataset(Dataset):
    """Original-resolution 256×256 crops centered on complete-cell Soma anchors."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        workspace_root: str | Path = PROJECT_ROOT,
        patch_size: int = 256,
        channel: int = 1,
        seed: int = 42,
        augment: bool = False,
        jitter: int = 32,
        max_crops: int | None = None,
    ) -> None:
        if patch_size <= 0 or jitter < 0:
            raise ValueError("patch_size must be positive and jitter nonnegative")
        self.workspace_root = Path(workspace_root).resolve()
        self.patch_size = int(patch_size)
        self.channel = channel
        self.seed = int(seed)
        self.augment = bool(augment)
        self.jitter = int(jitter)
        self.epoch = 0
        self.records = load_v3_manifest(manifest_path, split)
        self.tiles: list[dict] = []
        self.crops: list[dict] = []
        for row in self.records:
            sample_dir = self.workspace_root / row["sample_dir"]
            image_path = self.workspace_root / row["source_image"]
            image = load_image(image_path)
            if image.ndim != 3:
                raise ValueError(f"Expected HWC RGB source image: {image_path}")
            with (sample_dir / "metadata.json").open(encoding="utf-8") as handle:
                import json
                metadata = json.load(handle)
            supervision = reconstruct_positive_supervision(sample_dir, metadata, image.shape[:2])
            validate_against_converted_masks(sample_dir, supervision)
            tile_index = len(self.tiles)
            self.tiles.append({"row": row, "image": image, **supervision})
            association = supervision["association"]
            morphology = supervision["morphology"]
            for cell_id in supervision["complete_cell_ids"]:
                anchor = (association == cell_id) & (morphology == 0)
                process = (association == cell_id) & (morphology == 1)
                if not anchor.any() or not process.any():
                    continue
                ys, xs = np.nonzero(anchor)
                self.crops.append({
                    "tile_index": tile_index,
                    "cell_id": int(cell_id),
                    "center_x": float(xs.mean()),
                    "center_y": float(ys.mean()),
                    "full_soma_pixels": int(anchor.sum()),
                    "full_process_pixels": int(process.sum()),
                })
        if not self.crops:
            raise ValueError(f"No complete Soma/Process crop anchors for split={split}")
        self.crops.sort(key=lambda c: (self.records[c["tile_index"]]["image_id"], c["cell_id"]))
        if max_crops is not None:
            if max_crops <= 0:
                raise ValueError("max_crops must be positive")
            self.crops = self.crops[:max_crops]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def source_images(self) -> set[str]:
        return {row["image_id"] for row in self.records}

    @property
    def crop_source_images(self) -> set[str]:
        return {self.records[crop["tile_index"]]["image_id"] for crop in self.crops}

    def __len__(self) -> int:
        return len(self.crops)

    def _crop_origin(self, crop: dict, rng: np.random.Generator) -> tuple[int, int]:
        height, width = self.tiles[crop["tile_index"]]["morphology"].shape
        if height < self.patch_size or width < self.patch_size:
            raise ValueError("Source tile is smaller than requested crop")
        dx, dy = (0, 0) if not self.augment else (
            int(rng.integers(-self.jitter, self.jitter + 1)),
            int(rng.integers(-self.jitter, self.jitter + 1)),
        )
        x = int(round(crop["center_x"] - self.patch_size / 2 + dx))
        y = int(round(crop["center_y"] - self.patch_size / 2 + dy))
        return np.clip(x, 0, width - self.patch_size), np.clip(y, 0, height - self.patch_size)

    def __getitem__(self, index: int):
        crop = self.crops[index]
        tile = self.tiles[crop["tile_index"]]
        rng = np.random.default_rng(self.seed + self.epoch * 100003 + index * 1009)
        x, y = self._crop_origin(crop, rng)
        size = self.patch_size
        slices = np.s_[y:y + size, x:x + size]
        image = tile["image"][slices]
        morphology = tile["morphology"][slices].copy()
        association = tile["association"][slices].copy()
        uncertain = tile["uncertain"][slices].copy()
        conflict = tile["cross_cell_conflict"][slices].copy()
        flip_h = self.augment and bool(rng.integers(0, 2))
        flip_v = self.augment and bool(rng.integers(0, 2))
        rotate_k = int(rng.integers(0, 4)) if self.augment else 0
        if flip_h:
            image, morphology, association = image[:, ::-1], morphology[:, ::-1], association[:, ::-1]
            uncertain, conflict = uncertain[:, ::-1], conflict[:, ::-1]
        if flip_v:
            image, morphology, association = image[::-1], morphology[::-1], association[::-1]
            uncertain, conflict = uncertain[::-1], conflict[::-1]
        if rotate_k:
            image = np.rot90(image, rotate_k)
            morphology = np.rot90(morphology, rotate_k)
            association = np.rot90(association, rotate_k)
            uncertain = np.rot90(uncertain, rotate_k)
            conflict = np.rot90(conflict, rotate_k)
        image_tensor = prepare_image(np.ascontiguousarray(image), channel=self.channel)

        association_ids = []
        truncated_ids = []
        retention = {}
        for cell_id in tile["complete_cell_ids"]:
            soma = (association == cell_id) & (morphology == 0)
            process = (association == cell_id) & (morphology == 1)
            if soma.any() and process.any():
                association_ids.append(int(cell_id))
                full_soma = int(np.count_nonzero((tile["association"] == cell_id) & (tile["morphology"] == 0)))
                full_process = int(np.count_nonzero((tile["association"] == cell_id) & (tile["morphology"] == 1)))
                retention[str(cell_id)] = {
                    "soma": float(soma.sum() / full_soma),
                    "process": float(process.sum() / full_process),
                }
                if soma.sum() < full_soma or process.sum() < full_process:
                    truncated_ids.append(int(cell_id))
            else:
                # No in-crop Soma anchor: remove this cell from association only.
                association[(association == cell_id)] = 0
        targets = {
            "morphology_mask": torch.from_numpy(np.ascontiguousarray(morphology)).long(),
            "morphology_valid_mask": torch.from_numpy(np.ascontiguousarray(morphology != MORPHOLOGY_IGNORE)),
            "association_mask": torch.from_numpy(np.ascontiguousarray(association)).long(),
            "uncertain_mask": torch.from_numpy(np.ascontiguousarray(uncertain)),
            "conflict_mask": torch.from_numpy(np.ascontiguousarray(conflict)),
        }
        metadata = {
            "sample_id": tile["row"]["sample_id"],
            "image_id": tile["row"]["image_id"],
            "slide_sample_id": tile["row"]["slide_sample_id"],
            "split": tile["row"]["split"],
            "center_cell_id": crop["cell_id"],
            "patch_x": int(x), "patch_y": int(y), "patch_size": size,
            "flip_h": flip_h, "flip_v": flip_v, "rotate_k": rotate_k,
            "association_cell_ids": association_ids,
            "truncated_association_cell_ids": truncated_ids,
            "association_retention": retention,
            "background_supervision": False,
            "free_instance_detection_target": False,
        }
        return image_tensor, targets, metadata

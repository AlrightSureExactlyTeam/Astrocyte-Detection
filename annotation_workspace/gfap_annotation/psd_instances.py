"""Conservative layered-PSD to partial instance-dataset conversion.

No source PSD is modified. Unknown is never converted to confirmed background.
This module performs conversion only; it contains no training or model code.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

UNKNOWN_INSTANCE = np.iinfo(np.uint32).max
CELL_PATTERN = re.compile(r"^cell(\d+)(soma|process|uncertain)$")
BACKGROUND_NAMES = {"background", "confirmedbackground", "背景"}
ORIGINAL_NAMES = {"original", "originalimage"}
ORIENTATION_TRANSFORMS = (
    "identity", "flip_vertical", "flip_horizontal", "rotate_180",
    "transpose", "rotate_90_ccw", "rotate_90_cw", "anti_transpose",
)


class AlignmentError(ValueError):
    """Raised when a unique exact PSD-to-external-image orientation is unavailable."""


def apply_orientation_transform(array: np.ndarray, transform: str) -> np.ndarray:
    """Apply a lossless dihedral array operation to the two spatial axes."""
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError(f"Expected at least two dimensions, got {array.shape}")
    if transform == "identity":
        result = array
    elif transform == "flip_vertical":
        result = np.flipud(array)
    elif transform == "flip_horizontal":
        result = np.fliplr(array)
    elif transform == "rotate_180":
        result = np.rot90(array, 2, axes=(0, 1))
    elif transform == "transpose":
        result = np.swapaxes(array, 0, 1)
    elif transform == "rotate_90_ccw":
        result = np.rot90(array, 1, axes=(0, 1))
    elif transform == "rotate_90_cw":
        result = np.rot90(array, -1, axes=(0, 1))
    elif transform == "anti_transpose":
        result = np.rot90(np.swapaxes(array, 0, 1), 2, axes=(0, 1))
    else:
        raise ValueError(f"Unknown orientation transform: {transform}")
    return np.ascontiguousarray(result)


def _rgb_view(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] in {3, 4}:
        return image[..., :3]
    raise ValueError(f"Expected a 2D or HWC RGB/RGBA image, got {image.shape}")


def detect_orientation_transform(embedded: np.ndarray, external: np.ndarray) -> tuple[str, dict[str, Any]]:
    """Accept only one unique, exact RGB pixel match; never resize or register arbitrarily."""
    embedded = _rgb_view(embedded)
    external = _rgb_view(external)
    matches = []
    for name in ORIENTATION_TRANSFORMS:
        candidate = apply_orientation_transform(embedded, name)
        if candidate.shape == external.shape and np.array_equal(candidate, external):
            matches.append(name)
    evidence = {
        "method": "unique_exact_rgb_pixel_match",
        "candidate_transforms": list(ORIENTATION_TRANSFORMS),
        "exact_matches": matches,
        "embedded_shape_hwc": list(embedded.shape),
        "external_shape_hwc": list(external.shape),
        "image_resampling_applied": False,
    }
    if len(matches) != 1:
        raise AlignmentError(
            f"Reliable PSD/image orientation requires exactly one exact transform; found {matches or 'none'}"
        )
    evidence["accepted_transform"] = matches[0]
    evidence["accepted_transform_exact_pixel_match"] = True
    return matches[0], evidence


def normalized_name(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", name.strip()).casefold()


def parse_layer_name(name: str) -> dict[str, Any]:
    normalized = normalized_name(name)
    match = CELL_PATTERN.fullmatch(normalized)
    if match:
        return {"role": match.group(2), "cell_id": int(match.group(1))}
    if normalized in BACKGROUND_NAMES:
        return {"role": "background", "cell_id": None}
    if normalized in ORIGINAL_NAMES:
        return {"role": "original", "cell_id": None}
    if normalized.startswith("cell"):
        return {"role": "unrecognized_cell_layer", "cell_id": None}
    return {"role": "other", "cell_id": None}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _alpha_canvas(layer: Any, width: int, height: int) -> tuple[np.ndarray, list[str]]:
    """Extract layer alpha, then place it at the PSD layer offset without resampling."""
    warnings = []
    array = layer.numpy()
    alpha = None
    if array is not None and array.ndim == 3 and array.shape[-1] >= 4:
        alpha = array[..., 3]
        if np.issubdtype(alpha.dtype, np.floating) and alpha.max(initial=0) <= 1:
            alpha = alpha * 255
        alpha = np.rint(np.clip(alpha, 0, 255)).astype(np.uint8)
    else:
        image = layer.topil()
        if image is None:
            return np.zeros((height, width), dtype=np.uint8), ["missing_pixel_and_alpha_data"]
        if "A" not in image.getbands():
            warnings.append("alpha_absent_used_rgba_conversion_transparency")
        alpha = np.asarray(image.convert("RGBA"))[..., 3]
    left, top, right, bottom = (int(value) for value in layer.bbox)
    expected = (bottom - top, right - left)
    if alpha.shape != expected:
        raise ValueError(f"Layer alpha shape {alpha.shape} does not match bbox {expected}: {layer.name}")
    canvas = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = max(0, left), max(0, top), min(width, right), min(height, bottom)
    if x0 < x1 and y0 < y1:
        canvas[y0:y1, x0:x1] = alpha[y0 - top:y1 - top, x0 - left:x1 - left]
    if (left, top, right, bottom) != (0, 0, width, height):
        warnings.append(f"layer_bbox_offset:{left},{top},{right},{bottom}")
    if left < 0 or top < 0 or right > width or bottom > height:
        warnings.append("layer_clipped_to_psd_canvas")
    return canvas, warnings


def inspect_psd_layers(psd: Any) -> tuple[list[dict], list[str]]:
    width, height = (int(value) for value in psd.size)
    records = []
    warnings = []
    names = Counter(layer.name.strip().casefold() for layer in psd)
    duplicates = sorted(name for name, count in names.items() if count > 1)
    if duplicates:
        warnings.append(f"duplicate_layer_names:{duplicates}")
    for index, layer in enumerate(psd):
        parsed = parse_layer_name(layer.name)
        record = {
            "layer_index": index, "layer_name": layer.name, **parsed,
            "visible": bool(layer.visible), "layer_kind": str(layer.kind),
            "supported_pixel_layer": layer.kind == "pixel",
            "alpha_pixels": 0, "alpha_canvas": None, "warnings": [],
        }
        if not record["visible"]:
            record["warnings"].append("hidden_layer_excluded_from_labels")
        elif not record["supported_pixel_layer"]:
            record["warnings"].append("unsupported_layer_format_excluded")
        elif record["role"] not in {"other", "original"}:
            try:
                alpha, alpha_warnings = _alpha_canvas(layer, width, height)
                record["alpha_canvas"] = alpha
                record["alpha_pixels"] = int(np.count_nonzero(alpha))
                record["warnings"].extend(alpha_warnings)
                if not record["alpha_pixels"]:
                    record["warnings"].append("empty_layer")
            except Exception as exc:
                record["supported_pixel_layer"] = False
                record["warnings"].append(f"alpha_extraction_failed:{exc}")
        records.append(record)
    for record in records:
        for warning in record["warnings"]:
            if warning != "layer_bbox_offset:0,0,1024,1024":
                warnings.append(f"{record['layer_name']}:{warning}")
    return records, warnings


def _json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_mask(path: Path, mask: np.ndarray, metadata: dict) -> None:
    tifffile.imwrite(path, mask, photometric="minisblack", metadata=metadata)


def _display_original(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = image[..., :3].astype(np.float32)
    if image.max(initial=0) > 255:
        low, high = np.percentile(image, [1, 99.8])
        image = (image - low) / max(high - low, 1e-6) * 255
    return np.clip(image, 0, 255).astype(np.uint8)


def _qc_visualization(path: Path, original: np.ndarray, semantic: np.ndarray, instance: np.ndarray, uncertain: np.ndarray, conflicts: np.ndarray) -> None:
    height, width = semantic.shape
    original = _display_original(original)
    instance_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for cell_id in [int(value) for value in np.unique(instance) if value not in {0, UNKNOWN_INSTANCE}]:
        color = np.array([(53 * cell_id) % 205 + 50, (97 * cell_id) % 205 + 50, (149 * cell_id) % 205 + 50], dtype=np.uint8)
        instance_rgb[instance == cell_id] = color
    semantic_rgb = np.zeros_like(original)
    semantic_rgb[semantic == 0] = (70, 70, 70)
    semantic_rgb[semantic == 1] = (0, 230, 70)
    semantic_rgb[semantic == 2] = (240, 50, 50)
    uncertain_rgb = np.zeros_like(original)
    uncertain_rgb[uncertain > 0] = (240, 220, 0)
    uncertain_rgb[conflicts] = (255, 0, 255)
    overlays = []
    for color in (instance_rgb, semantic_rgb, uncertain_rgb):
        overlay = original.copy()
        selected = np.any(color > 0, axis=2)
        overlay[selected] = (0.35 * overlay[selected] + 0.65 * color[selected]).astype(np.uint8)
        overlays.append(overlay)
    panels = [cv2.resize(panel, (512, 512), interpolation=cv2.INTER_NEAREST) for panel in [original, *overlays]]
    canvas = np.zeros((570, 2048, 3), dtype=np.uint8)
    canvas[58:] = np.concatenate(panels, axis=1)
    for index, heading in enumerate(("Original", "Cell instances", "Soma green / Process red", "Uncertain yellow / Conflict magenta")):
        cv2.putText(canvas, heading, (10 + index * 512, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    success, encoded = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not success:
        raise OSError("Could not encode PSD QC visualization")
    with path.open("xb") as handle:
        handle.write(encoded.tobytes())


def convert_psd(psd: Any, source_psd: str | Path, output_dir: str | Path, original_image: np.ndarray, original_image_path: str | Path | None = None) -> dict:
    """Convert one in-memory layered PSD into a new, non-overwriting dataset directory."""
    source_psd = Path(source_psd)
    output_dir = Path(output_dir)
    width, height = (int(value) for value in psd.size)
    original_image = np.asarray(original_image)
    if original_image.shape[:2] != (height, width):
        raise ValueError(f"Original image shape {original_image.shape[:2]} does not match PSD {(height, width)}")
    records, warnings = inspect_psd_layers(psd)
    output_dir.mkdir(parents=True, exist_ok=False)
    components_dir = output_dir / "components"
    components_dir.mkdir()

    semantic = np.full((height, width), 255, dtype=np.uint8)
    instance = np.full((height, width), UNKNOWN_INSTANCE, dtype=np.uint32)
    uncertain = np.zeros((height, width), dtype=np.uint8)
    soma_votes = np.zeros((height, width), dtype=np.uint16)
    process_votes = np.zeros_like(soma_votes)
    background = np.zeros((height, width), dtype=bool)
    owner_count = np.zeros((height, width), dtype=np.uint16)
    owner_id = np.zeros((height, width), dtype=np.uint32)
    multiple_owners = np.zeros((height, width), dtype=bool)

    for record in records:
        if record["role"] == "background" and record["alpha_canvas"] is not None and record["visible"] and record["supported_pixel_layer"]:
            background |= record["alpha_canvas"] > 0
    semantic[background] = 0
    instance[background] = 0

    grouped: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record["cell_id"] is not None:
            grouped[record["cell_id"]][record["role"]].append(record)
    cells = []
    component_masks: dict[tuple[int, str], np.ndarray] = {}
    for cell_id in sorted(grouped):
        roles = grouped[cell_id]
        cell_record = {
            "cell_id": cell_id, "instance_id": cell_id,
            "soma_layer_names": [item["layer_name"] for item in roles.get("soma", [])],
            "process_layer_names": [item["layer_name"] for item in roles.get("process", [])],
            "uncertain_layer_names": [item["layer_name"] for item in roles.get("uncertain", [])],
            "component_mask_paths": {"soma": None, "process": None},
            "annotation_status": "missing_visible_components", "warnings": [],
        }
        for role in ("soma", "process"):
            layers = [item for item in roles.get(role, []) if item["visible"] and item["supported_pixel_layer"] and item["alpha_canvas"] is not None]
            if not layers:
                cell_record["warnings"].append(f"missing_visible_{role}_annotation_not_inferred")
                continue
            combined = np.zeros((height, width), dtype=np.uint8)
            for layer in layers:
                combined[layer["alpha_canvas"] > 0] = 255
            component_masks[(cell_id, role)] = combined
            relative = Path("components") / f"cell{cell_id:03d}_{role}.tif"
            _write_mask(output_dir / relative, combined, {"axes": "YX", "cell_id": cell_id, "component": role, "review_status": "unverified"})
            cell_record["component_mask_paths"][role] = relative.as_posix()
            cell_record[f"{role}_pixels"] = int(np.count_nonzero(combined))
            if len(layers) > 1:
                cell_record["warnings"].append(f"multiple_{role}_layers_combined_by_union_review_required")
            votes = soma_votes if role == "soma" else process_votes
            selected = combined > 0
            votes[selected] += 1
            owner_count[selected] += 1
            first = owner_id == 0
            owner_id[first & selected] = cell_id
            multiple_owners |= selected & (owner_id != cell_id)
        for layer in [item for item in roles.get("uncertain", []) if item["visible"] and item["supported_pixel_layer"] and item["alpha_canvas"] is not None]:
            uncertain[layer["alpha_canvas"] > 0] = 255
        if component_masks.get((cell_id, "soma")) is not None and component_masks.get((cell_id, "process")) is not None:
            cell_record["annotation_status"] = "soma_and_process_present"
        elif any(key[0] == cell_id for key in component_masks):
            cell_record["annotation_status"] = "partial_cell_components_not_inferred"
        if roles.get("uncertain") and not roles.get("soma") and not roles.get("process"):
            cell_record["annotation_status"] = "uncertain_only_excluded_from_positive_labels"
        cells.append(cell_record)

    confirmed = (soma_votes > 0) | (process_votes > 0)
    # Any overlap between different cell IDs remains unknown. Within one cell,
    # Process intentionally includes/surrounds Soma, so Soma takes semantic
    # priority while both raw components and the shared instance are preserved.
    semantic_conflict = multiple_owners | (background & confirmed)
    instance_conflict = semantic_conflict.copy()
    soma_priority = (soma_votes > 0) & (process_votes > 0) & ~semantic_conflict
    semantic[(soma_votes > 0) & ~semantic_conflict] = 1
    semantic[(process_votes > 0) & (soma_votes == 0) & ~semantic_conflict] = 2
    valid_instance = confirmed & (owner_count > 0) & ~instance_conflict
    instance[valid_instance] = owner_id[valid_instance]
    semantic[semantic_conflict] = 255
    instance[instance_conflict] = UNKNOWN_INSTANCE
    uncertain_conflict = (uncertain > 0) & (confirmed | background)
    same_cell_overlaps_resolved = int(np.count_nonzero(soma_priority & (uncertain == 0)))
    unresolved_cross_cell_conflicts = int(np.count_nonzero(multiple_owners))
    semantic[uncertain > 0] = 255
    instance[uncertain > 0] = UNKNOWN_INSTANCE
    conflicts = semantic_conflict | instance_conflict | uncertain_conflict

    conflict_records = []
    items = sorted(component_masks.items())
    for index, ((cell_a, role_a), mask_a) in enumerate(items):
        for (cell_b, role_b), mask_b in items[index + 1:]:
            overlap = int(np.count_nonzero((mask_a > 0) & (mask_b > 0)))
            if overlap:
                conflict_records.append({"cell_a": cell_a, "component_a": role_a, "cell_b": cell_b, "component_b": role_b, "overlap_pixels": overlap})
    if conflict_records:
        warnings.append(f"component_overlaps_recorded_with_soma_priority_policy:{len(conflict_records)}")
    if np.any(uncertain_conflict):
        warnings.append("uncertain_overlaps_confirmed_or_background_marked_unknown")
    if len(cells) == 0:
        warnings.append("no_parseable_cell_layers")
    if not np.any(background):
        warnings.append("no_explicit_confirmed_background_all_unannotated_pixels_ignore")

    _write_mask(output_dir / "semantic_mask.tif", semantic, {"axes": "YX", "background": 0, "soma": 1, "process": 2, "ignore_unknown": 255})
    _write_mask(output_dir / "instance_mask.tif", instance, {"axes": "YX", "confirmed_background": 0, "unknown_uncertain_conflict": int(UNKNOWN_INSTANCE)})
    _write_mask(output_dir / "uncertain_mask.tif", uncertain, {"axes": "YX", "uncertain": 255})
    _write_mask(output_dir / "background_mask.tif", background.astype(np.uint8) * 255, {"axes": "YX", "explicit_confirmed_background": 255})
    _write_mask(output_dir / "conflict_mask.tif", conflicts.astype(np.uint8) * 255, {"axes": "YX", "conflict": 255})
    _qc_visualization(output_dir / "qc_overlay.png", original_image, semantic, instance, uncertain, conflicts)
    _json(output_dir / "conflicts.json", conflict_records)

    layer_metadata = [{key: value for key, value in record.items() if key != "alpha_canvas"} for record in records]
    metadata = {
        "source_psd_filename": source_psd.name, "source_psd_sha256": _sha256(source_psd),
        "original_image_filename": Path(original_image_path).name if original_image_path else None,
        "original_width": width, "original_height": height, "coordinate_system": "PSD canvas pixels; layer bbox offsets applied without resampling",
        "annotation_status": "partial_manual_unverified", "review_status": "unverified",
        "background_policy": "0 only under visible explicitly named background alpha; all other unannotated pixels remain unknown",
        "semantic_label_mapping": {"confirmed_background": 0, "soma": 1, "process": 2, "ignore_unknown_uncertain_conflict": 255},
        "instance_label_mapping": {"confirmed_background": 0, "cell_ids": "positive local integer IDs", "unknown_uncertain_conflict": int(UNKNOWN_INSTANCE)},
        "conflict_policy": "Same-cell Soma/Process overlaps use Soma semantic priority and preserve the shared instance ID. Any overlap between different cell IDs becomes semantic and instance unknown. Background/confirmed-cell conflicts become unknown; raw component masks preserve all source annotations.",
        "cells": cells, "annotated_cell_count": len({cell_id for cell_id, _ in component_masks}),
        "uncertain_candidate_count": len({record["cell_id"] for record in records if record["role"] == "uncertain" and record["alpha_canvas"] is not None and record["alpha_pixels"] > 0}),
        "semantic_pixel_counts": {str(value): int(np.count_nonzero(semantic == value)) for value in (0, 1, 2, 255)},
        "same_cell_overlap_pixels_resolved_soma_priority": same_cell_overlaps_resolved,
        "unresolved_cross_cell_conflict_pixels": unresolved_cross_cell_conflicts,
        "conflict_pixels": int(np.count_nonzero(conflicts)), "uncertain_pixels": int(np.count_nonzero(uncertain)),
        "parsing_warnings": warnings, "layers": layer_metadata,
        "component_overlap_conflicts": len(conflict_records),
    }
    _json(output_dir / "metadata.json", metadata)
    return metadata


def reorient_converted_sample(
    output_dir: str | Path,
    external_image: np.ndarray,
    original_image_path: str | Path,
    transform: str,
    evidence: dict[str, Any],
) -> dict:
    """Archive native PSD outputs, then rotate every target mask into external-image coordinates."""
    output_dir = Path(output_dir)
    native_dir = output_dir / "psd_coordinates"
    native_dir.mkdir(exist_ok=False)
    mask_names = ("semantic_mask.tif", "instance_mask.tif", "uncertain_mask.tif", "background_mask.tif", "conflict_mask.tif")
    for filename in (*mask_names, "qc_overlay.png", "metadata.json", "conflicts.json"):
        shutil.copy2(output_dir / filename, native_dir / filename)
    shutil.copytree(output_dir / "components", native_dir / "components")

    native = {name: tifffile.imread(native_dir / name) for name in mask_names}
    target = {name: apply_orientation_transform(array, transform) for name, array in native.items()}
    for name in mask_names:
        if not np.array_equal(np.unique(target[name]), np.unique(native[name])):
            raise AssertionError(f"Orientation changed label values in {name}")
        if np.count_nonzero(target[name]) != np.count_nonzero(native[name]):
            raise AssertionError(f"Orientation changed pixel support in {name}")

    semantic = target["semantic_mask.tif"]
    instance = target["instance_mask.tif"]
    uncertain = target["uncertain_mask.tif"]
    background = target["background_mask.tif"]
    conflicts = target["conflict_mask.tif"] > 0
    _write_mask(output_dir / "semantic_mask.tif", semantic, {"axes": "YX", "background": 0, "soma": 1, "process": 2, "ignore_unknown": 255})
    _write_mask(output_dir / "instance_mask.tif", instance, {"axes": "YX", "confirmed_background": 0, "unknown_uncertain_conflict": int(UNKNOWN_INSTANCE)})
    _write_mask(output_dir / "uncertain_mask.tif", uncertain, {"axes": "YX", "uncertain": 255})
    _write_mask(output_dir / "background_mask.tif", background, {"axes": "YX", "explicit_confirmed_background": 255})
    _write_mask(output_dir / "conflict_mask.tif", target["conflict_mask.tif"], {"axes": "YX", "conflict": 255})

    for native_path in sorted((native_dir / "components").glob("*.tif")):
        match = re.fullmatch(r"cell(\d+)_(soma|process)\.tif", native_path.name)
        if match is None:
            raise ValueError(f"Unexpected component filename: {native_path.name}")
        raw = tifffile.imread(native_path)
        oriented = apply_orientation_transform(raw, transform)
        if np.count_nonzero(oriented) != np.count_nonzero(raw):
            raise AssertionError(f"Orientation changed component support: {native_path.name}")
        _write_mask(output_dir / "components" / native_path.name, oriented, {
            "axes": "YX", "cell_id": int(match.group(1)), "component": match.group(2),
            "coordinate_system": "external_image", "orientation_transform": transform,
            "review_status": "unverified",
        })

    external_image = np.asarray(external_image)
    if external_image.shape[:2] != semantic.shape:
        raise ValueError(f"External image shape {external_image.shape[:2]} does not match oriented masks {semantic.shape}")
    (output_dir / "qc_overlay.png").unlink()  # Native QC is archived; replace only this new run's target QC.
    _qc_visualization(output_dir / "qc_overlay.png", external_image, semantic, instance, uncertain, conflicts)

    metadata = json.loads((native_dir / "metadata.json").read_text(encoding="utf-8"))
    rotated_counts = {str(value): int(np.count_nonzero(semantic == value)) for value in (0, 1, 2, 255)}
    if rotated_counts != metadata["semantic_pixel_counts"]:
        raise AssertionError("Orientation changed semantic class pixel counts")
    metadata["psd_canvas_width"] = metadata["original_width"]
    metadata["psd_canvas_height"] = metadata["original_height"]
    metadata["original_width"] = int(semantic.shape[1])
    metadata["original_height"] = int(semantic.shape[0])
    metadata["original_image_filename"] = Path(original_image_path).name
    metadata["coordinate_system"] = "External image pixels; annotation arrays transformed losslessly from archived PSD coordinates"
    metadata["orientation_transform"] = transform
    metadata["orientation_evidence"] = evidence
    metadata["orientation_operation"] = "lossless_array_dihedral_transform_no_interpolation"
    metadata["psd_coordinate_archive"] = "psd_coordinates"
    metadata["transformed_outputs"] = ["Soma components", "Process components", "Uncertain", "semantic", "instance", "Background", "conflicts"]
    for cell in metadata["cells"]:
        cell["raw_component_mask_paths"] = {
            role: None if relative is None else (Path("psd_coordinates") / relative).as_posix()
            for role, relative in cell["component_mask_paths"].items()
        }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return metadata


def convert_psd_file(psd_path: str | Path, output_dir: str | Path, original_image_path: str | Path | None = None) -> dict:
    from psd_tools import PSDImage
    from src.preprocessing import load_image

    psd_path = Path(psd_path)
    psd = PSDImage.open(psd_path)
    originals = [layer for layer in psd if parse_layer_name(layer.name)["role"] == "original" and layer.visible and layer.kind == "pixel"]
    if len(originals) != 1:
        raise AlignmentError("Reliable conversion requires exactly one visible supported Original layer")
    embedded = np.asarray(originals[0].topil().convert("RGB"))
    if original_image_path is not None:
        external = load_image(original_image_path)
        transform, evidence = detect_orientation_transform(embedded, external)
    metadata = convert_psd(psd, psd_path, output_dir, embedded)
    if original_image_path is None:
        return metadata
    return reorient_converted_sample(output_dir, external, original_image_path, transform, evidence)

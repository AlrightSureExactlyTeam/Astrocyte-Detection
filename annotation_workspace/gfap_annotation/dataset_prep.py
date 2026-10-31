"""Validate manual GFAP morphology and ImageJ pseudo-label datasets.

Manual dense labels use 0=confirmed background, 1=soma, 2=dense process and
255=ignore. This project has no explicit verified background layer, so empty
pixels remain 255. ImageJ masks stay separate, unreviewed foreground candidates.
"""
from __future__ import annotations

import csv
import json
import re
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

from .workflow import PROJECT_ROOT, relative_path, resolve_path, sha256, utc_now, write_json_new

MANIFEST_COLUMNS = [
    "sample_id", "image_filename", "image_path", "annotation_filename", "annotation_path",
    "annotation_source", "annotation_type", "width", "height", "available_label_classes",
    "validation_status", "warnings", "slide_sample_id", "tile_x", "tile_y",
    "training_label_path", "training_suitable", "review_required", "pixel_count",
    "annotated_pixels", "soma_pixels", "process_pixels", "uncertain_pixels", "ignore_pixels",
    "unexpected_color_pixels", "pseudo_foreground_pixels", "source_sha256", "psd_path", "psd_layers",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _empty_row(sample_id: str, source: str) -> dict:
    row = {name: "" for name in MANIFEST_COLUMNS}
    row.update({
        "sample_id": sample_id,
        "annotation_source": source,
        "validation_status": "invalid",
        "training_suitable": "false",
        "review_required": "true",
    })
    for name in (
        "pixel_count", "annotated_pixels", "soma_pixels", "process_pixels", "uncertain_pixels",
        "ignore_pixels", "unexpected_color_pixels", "pseudo_foreground_pixels",
    ):
        row[name] = "0"
    return row


def _slide_and_tile(filename: str) -> tuple[str, str, str]:
    match = re.match(r"(.+_Scan\d+)_x(\d+)_y(\d+)", Path(filename).stem)
    if not match:
        return Path(filename).stem, "", ""
    return match.group(1), str(int(match.group(2))), str(int(match.group(3)))


def read_psd_metadata(path: str | Path) -> dict:
    """Read PSD dimensions and layer names without changing the source file."""
    path = resolve_path(path)
    data = path.read_bytes()
    if len(data) < 30 or data[:4] != b"8BPS" or struct.unpack_from(">H", data, 4)[0] != 1:
        raise ValueError(f"Only valid PSD version 1 files are supported: {path}")
    channels, height, width = struct.unpack_from(">HII", data, 12)
    offset = 26

    def take(length: int) -> bytes:
        nonlocal offset
        if length < 0 or offset + length > len(data):
            raise ValueError(f"Truncated PSD structure: {path}")
        value = data[offset:offset + length]
        offset += length
        return value

    def uint32() -> int:
        return struct.unpack(">I", take(4))[0]

    for _ in range(2):
        take(uint32())
    take(uint32())  # Layer/mask section; layer records are parsed below.
    # The previous take skips the section, so reopen at its recorded offset.
    # Reparse from the start of the layer/mask section to avoid interpreting image data.
    offset = 26
    for _ in range(2):
        take(uint32())
    section_length = uint32()
    section_end = offset + section_length
    layer_info_length = uint32()
    layer_info_end = offset + layer_info_length
    if layer_info_end > section_end or layer_info_end > len(data):
        raise ValueError(f"Invalid PSD layer information length: {path}")
    layer_count = abs(struct.unpack(">h", take(2))[0])
    layers = []
    for _ in range(layer_count):
        take(16)
        channel_count = struct.unpack(">H", take(2))[0]
        take(channel_count * 6 + 12)
        extra_length = uint32()
        extra_end = offset + extra_length
        take(uint32())
        take(uint32())
        name_length = take(1)[0]
        pascal_name = take(name_length).decode("latin1")
        take(4 * ((name_length + 4) // 4) - name_length - 1)
        unicode_name = None
        while offset + 12 <= extra_end:
            signature, key, length = struct.unpack(">4s4sI", take(12))
            if signature not in {b"8BIM", b"8B64"}:
                raise ValueError(f"Unexpected PSD additional-layer signature: {signature!r}")
            payload = take(length)
            if length % 2:
                take(1)
            if key == b"luni" and len(payload) >= 4:
                count = struct.unpack_from(">I", payload, 0)[0]
                unicode_name = payload[4:4 + count * 2].decode("utf-16-be")
        if offset != extra_end:
            raise ValueError(f"Inconsistent PSD layer record: {path}")
        layers.append((unicode_name or pascal_name).strip())
    return {"width": int(width), "height": int(height), "channels": int(channels), "layer_names": layers}


def classify_manual_rgba(rgba: np.ndarray) -> tuple[np.ndarray, dict]:
    """Classify core and antialiased red/green/yellow pixels using RGB and alpha."""
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError("Manual annotation must be a uint8 RGBA image")
    alpha = rgba[..., 3]
    rgb = rgba[..., :3].astype(np.float32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    annotated = alpha > 0
    classes = np.zeros(alpha.shape, dtype=np.uint8)  # 0 transparent, 1 soma, 2 process, 3 uncertain, 4 unexpected
    soma = annotated & (green >= 100) & (green > 1.25 * red) & (green > 1.25 * blue)
    process = annotated & (red >= 100) & (red > 1.25 * green) & (red > 1.25 * blue)
    uncertain = (
        annotated & (red >= 100) & (green >= 100)
        & (blue < 0.8 * np.minimum(red, green))
        & (np.abs(red - green) <= 0.45 * np.maximum(red, green))
    )
    # Yellow takes precedence: its antialiased fringe can otherwise look red.
    classes[uncertain] = 3
    classes[soma & (classes == 0)] = 1
    classes[process & (classes == 0)] = 2
    classes[annotated & (classes == 0)] = 4
    unexpected = annotated & (classes == 4)
    colors = Counter(map(tuple, rgba[unexpected, :3].tolist()))
    stats = {
        "annotated_pixels": int(annotated.sum()),
        "soma_pixels": int((classes == 1).sum()),
        "process_pixels": int((classes == 2).sum()),
        "uncertain_pixels": int((classes == 3).sum()),
        "unexpected_color_pixels": int(unexpected.sum()),
        "semi_transparent_pixels": int(((alpha > 0) & (alpha < 255)).sum()),
        "unexpected_colors": [{"rgb": list(color), "count": count} for color, count in colors.most_common(10)],
    }
    return classes, stats


def _psd_roles(layer_names: list[str]) -> tuple[dict, list[str]]:
    roles = {"soma": False, "process_dense": False, "process_point": False, "uncertain": False}
    warnings = []
    for name in layer_names:
        normalized = name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
        if not normalized.startswith("cell"):
            continue
        if "process" in normalized:
            if "point" in normalized or "点" in normalized:
                roles["process_point"] = True
            else:
                roles["process_dense"] = True
        elif "soma" in normalized:
            roles["soma"] = True
        elif "sima" in normalized:
            roles["soma"] = True
            warnings.append(f"possible_psd_layer_typo:{name}")
        elif "uncertain" in normalized:
            roles["uncertain"] = True
        else:
            warnings.append(f"unrecognized_psd_cell_layer:{name}")
    return roles, warnings


def _manual_annotation_type(roles: dict) -> str:
    dense = roles["soma"] or roles["process_dense"]
    point = roles["process_point"]
    if dense and point:
        return "mixed"
    if point:
        return "point"
    return "dense"


def _label_path(output_root: Path, sample_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    return output_root / "labels" / "manual" / f"{safe}.tif"


def _save_manual_label(path: Path, classes: np.ndarray, roles: dict, sample_id: str) -> dict:
    label = np.full(classes.shape, 255, dtype=np.uint8)
    label[classes == 1] = 1
    if roles["process_dense"]:
        label[classes == 2] = 2
    # Process points, uncertain, unexpected and unannotated pixels intentionally remain 255.
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path, label, photometric="minisblack",
        metadata={"axes": "YX", "sample_id": sample_id, "ignore_value": 255, "annotation_source": "manual"},
    )
    counts = {str(value): int((label == value).sum()) for value in (0, 1, 2, 255)}
    return {"label": label, "class_pixel_counts": counts}


def _save_visualization(
    path: Path,
    image: np.ndarray,
    annotation: np.ndarray,
    title: str,
    pseudo: bool = False,
) -> None:
    rgb = image[..., :3].astype(np.float32) if image.ndim == 3 else np.repeat(image[..., None], 3, axis=2).astype(np.float32)
    if rgb.max(initial=0) > 1:
        rgb /= 255.0
    color = np.zeros((*annotation.shape, 3), dtype=np.float32)
    if pseudo:
        foreground = annotation > 0
        color[foreground] = (0, 1, 0)
        middle = np.repeat(annotation[..., None], 3, axis=2).astype(np.float32) / 255.0
    else:
        color[annotation == 1] = (0, 1, 0)
        color[annotation == 2] = (1, 0, 0)
        color[annotation == 3] = (1, 1, 0)
        color[annotation == 4] = (1, 0, 1)
        foreground = annotation > 0
        middle = color
    overlay = rgb.copy()
    overlay[foreground] = 0.55 * rgb[foreground] + 0.45 * color[foreground]
    panels = []
    for content in (rgb, middle, overlay):
        panel = np.clip(content * 255, 0, 255).astype(np.uint8)
        panels.append(cv2.resize(panel, (512, 512), interpolation=cv2.INTER_NEAREST))
    canvas = np.zeros((570, 1536, 3), dtype=np.uint8)
    canvas[58:, :] = np.concatenate(panels, axis=1)
    cv2.putText(canvas, title[:105], (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    for index, heading in enumerate(("Original image", "Annotation / pseudo mask", "Overlay")):
        cv2.putText(canvas, heading, (12 + 512 * index, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not success:
        raise OSError("Could not encode validation visualization")
    with path.open("xb") as handle:
        handle.write(encoded.tobytes())


def _manual_records(manual_root: Path, output_root: Path, pilot: bool) -> tuple[list[dict], list[dict]]:
    images = {path.name: path for path in (manual_root / "images").glob("*.tif")}
    masks = sorted((manual_root / "masks").glob("*.png"))
    grouped: dict[str, list[Path]] = {}
    for mask in masks:
        image_name = mask.name[:-4] + ".tif"
        if image_name not in images and re.search(r" \d+$", mask.stem):
            # Procreate copy suffixes identify annotation revisions, not new source images.
            base_name = re.sub(r" \d+$", "", mask.stem) + ".tif"
            if base_name in images:
                image_name = base_name
        grouped.setdefault(image_name, []).append(mask)
    selected_image = "GBM664_S5_Scan1_x043008_y109568_RGB.tif"
    records = []
    visualizations = []
    for image_name in sorted(set(images) | set(grouped)):
        if pilot and image_name != selected_image:
            continue
        image_path = images.get(image_name)
        annotation_paths = grouped.get(image_name, [])
        if not annotation_paths:
            row = _empty_row(Path(image_name).stem, "manual")
            row.update({
                "image_filename": image_name,
                "image_path": relative_path(image_path) if image_path else "",
                "warnings": "missing_annotation_png",
                "slide_sample_id": _slide_and_tile(image_name)[0],
            })
            records.append(row)
            continue
        for annotation_path in annotation_paths:
            sample_id = annotation_path.stem
            row = _empty_row(sample_id, "manual")
            row.update({
                "image_filename": image_name,
                "annotation_filename": annotation_path.name,
                "annotation_path": relative_path(annotation_path),
                "annotation_type": "dense",
                "slide_sample_id": _slide_and_tile(image_name)[0],
                "tile_x": _slide_and_tile(image_name)[1],
                "tile_y": _slide_and_tile(image_name)[2],
                "validation_status": "valid",
                "review_required": "false",
            })
            warnings = []
            errors = []
            if image_path is None:
                errors.append("missing_original_tiff")
            if len(annotation_paths) > 1:
                errors.append("duplicate_annotation_for_same_image")
            psd_path = manual_root / "psd" / f"{annotation_path.stem}.psd"
            row["psd_path"] = relative_path(psd_path) if psd_path.is_file() else ""
            try:
                if image_path is not None:
                    row["image_path"] = relative_path(image_path)
                    row["source_sha256"] = sha256(image_path)
                    image = tifffile.imread(image_path)
                    image_hw = image.shape[:2]
                    row["height"], row["width"] = str(image_hw[0]), str(image_hw[1])
                with Image.open(annotation_path) as handle:
                    if "A" not in handle.getbands():
                        raise ValueError("manual_annotation_missing_alpha_channel")
                    rgba = np.asarray(handle.convert("RGBA"), dtype=np.uint8)
                if image_path is not None and rgba.shape[:2] != image_hw:
                    errors.append(f"image_mask_dimension_mismatch:{image_hw}!={rgba.shape[:2]}")
                classes, color_stats = classify_manual_rgba(rgba)
                row.update({name: str(value) for name, value in color_stats.items() if isinstance(value, int)})
                row["pixel_count"] = str(int(classes.size))
                if color_stats["unexpected_color_pixels"]:
                    warnings.append("unexpected_annotation_colors")
                if color_stats["annotated_pixels"] == 0:
                    errors.append("empty_manual_annotation")
                roles = {"soma": False, "process_dense": False, "process_point": False, "uncertain": False}
                if psd_path.is_file():
                    psd = read_psd_metadata(psd_path)
                    row["psd_layers"] = json.dumps(psd["layer_names"], ensure_ascii=False)
                    if image_path is not None and (psd["height"], psd["width"]) != image_hw:
                        errors.append("psd_image_dimension_mismatch")
                    roles, role_warnings = _psd_roles(psd["layer_names"])
                    warnings.extend(role_warnings)
                else:
                    warnings.append("missing_psd_layer_metadata")
                row["annotation_type"] = _manual_annotation_type(roles)
                if color_stats["process_pixels"] and not (roles["process_dense"] or roles["process_point"]):
                    if roles["uncertain"] and not roles["soma"]:
                        # Isolated red fringe pixels can remain where yellow uncertain regions
                        # were antialiased. They stay ignore=255, not inferred process truth.
                        warnings.append("red_pixels_on_uncertain_only_layers_set_to_ignore")
                    else:
                        errors.append("process_dense_or_point_type_undetermined")
                if color_stats["uncertain_pixels"]:
                    warnings.append("uncertain_regions_set_to_ignore")
                if not any((roles["soma"], roles["process_dense"], roles["process_point"], roles["uncertain"])):
                    warnings.append("no_training_roles_found_in_psd")
                row["ignore_pixels"] = str(int(classes.size - color_stats["soma_pixels"] - (
                    color_stats["process_pixels"] if roles["process_dense"] else 0
                )))
                classes_present = ["ignore"]
                if color_stats["soma_pixels"]:
                    classes_present.append("soma")
                if roles["process_dense"] and color_stats["process_pixels"]:
                    classes_present.append("process")
                row["available_label_classes"] = ";".join(classes_present)
                has_dense_target = bool(color_stats["soma_pixels"] or (
                    roles["process_dense"] and color_stats["process_pixels"]
                ))
                if not has_dense_target:
                    warnings.append("no_dense_soma_or_process_training_target")
                if errors:
                    row["validation_status"] = "requires_review" if "duplicate_annotation_for_same_image" in errors else "invalid"
                elif warnings:
                    row["validation_status"] = "valid_with_warnings"
                suitable = not errors and has_dense_target
                row["training_suitable"] = str(suitable).lower()
                row["review_required"] = str(bool(errors or color_stats["uncertain_pixels"] or not has_dense_target)).lower()
                if suitable:
                    label_path = _label_path(output_root, sample_id)
                    _save_manual_label(label_path, classes, roles, sample_id)
                    row["training_label_path"] = relative_path(label_path)
                if not records or all(item["annotation_source"] != "manual" for item in records):
                    visualization_path = output_root / "visualizations" / f"manual_{sample_id}.png"
                    _save_visualization(visualization_path, image, classes, f"{sample_id} (unverified background=ignore)")
                    visualizations.append(relative_path(visualization_path))
            except Exception as exc:
                errors.append(f"read_or_validation_error:{exc}")
                row["validation_status"] = "invalid"
                row["training_suitable"] = "false"
                row["review_required"] = "true"
            row["warnings"] = ";".join(errors + warnings)
            records.append(row)
    return records, visualizations


def _imagej_records(imagej_root: Path, output_root: Path, pilot: bool) -> tuple[list[dict], list[dict]]:
    mapping_rows = _read_csv(imagej_root / "source_mapping.csv")
    manifest_rows = {Path(row["image_name"]).stem: row for row in _read_csv(imagej_root / "manifest.csv")}
    images = {path.name: path for path in (imagej_root / "images").glob("*.tif")}
    hash_index: dict[str, list[Path]] = {}
    for path in images.values():
        hash_index.setdefault(sha256(path), []).append(path)
    records = []
    visualizations = []
    mapped_images = set()
    selected_id = "slide_02"
    for mapping in mapping_rows:
        image_id = mapping["image_id"]
        if pilot and image_id != selected_id:
            continue
        mask_path = imagej_root / "Mask" / f"{image_id}_mask.tif"
        positions_path = imagej_root / "CSV" / f"{image_id}_positions.csv"
        original_name = mapping.get("original_name", "")
        candidates = [images[original_name]] if original_name in images else hash_index.get(mapping["sha256"], [])
        row = _empty_row(image_id, "imagej_pseudo")
        row.update({
            "annotation_source": "imagej_pseudo",
            "annotation_type": "pseudo",
            "annotation_filename": mask_path.name,
            "annotation_path": relative_path(mask_path) if mask_path.is_file() else "",
            "available_label_classes": "GFAP_foreground_candidate;ignore_not_confirmed_background",
            "validation_status": "valid",
            "training_suitable": "false",
            "review_required": "true",
            "source_sha256": mapping["sha256"],
        })
        warnings = ["unreviewed_imagej_pseudo_label", "outside_foreground_is_not_confirmed_background"]
        errors = []
        try:
            if len(candidates) != 1:
                errors.append(f"source_image_match_count:{len(candidates)}")
                image_path = None
            else:
                image_path = candidates[0]
                mapped_images.add(image_path.name)
                row["image_filename"] = image_path.name
                row["image_path"] = relative_path(image_path)
                slide, tile_x, tile_y = _slide_and_tile(image_path.name)
                row["slide_sample_id"], row["tile_x"], row["tile_y"] = slide, tile_x, tile_y
                if sha256(image_path) != mapping["sha256"]:
                    errors.append("source_checksum_mismatch")
                image = tifffile.imread(image_path)
                image_hw = image.shape[:2]
                row["height"], row["width"] = str(image_hw[0]), str(image_hw[1])
                if (str(image_hw[1]), str(image_hw[0])) != (mapping["width_px"], mapping["height_px"]):
                    errors.append("source_mapping_dimension_mismatch")
            if not mask_path.is_file():
                errors.append("missing_imagej_mask")
                mask = None
            else:
                mask = tifffile.imread(mask_path)
                values = set(int(value) for value in np.unique(mask))
                if mask.ndim != 2:
                    errors.append("imagej_mask_not_2d")
                elif image_path is not None and mask.shape != image_hw:
                    errors.append(f"image_mask_dimension_mismatch:{image_hw}!={mask.shape}")
                if not values.issubset({0, 255}):
                    errors.append(f"unexpected_imagej_mask_values:{sorted(values)}")
                foreground = int((mask == 255).sum())
                row["pseudo_foreground_pixels"] = str(foreground)
                if foreground == 0:
                    errors.append("empty_imagej_mask")
            positions = _read_csv(positions_path) if positions_path.is_file() else []
            if not positions_path.is_file():
                errors.append("missing_imagej_positions_csv")
            expected = manifest_rows.get(image_id)
            if expected is None:
                errors.append("missing_imagej_manifest_row")
            elif int(expected["n_regions"]) != len(positions):
                errors.append("region_count_manifest_csv_mismatch")
            if any(item.get("status") != "unreviewed" for item in positions):
                warnings.append("position_review_status_differs_from_unreviewed")
            if errors:
                row["validation_status"] = "invalid"
            if image_path is not None and mask is not None and mask.ndim == 2:
                visualization_path = output_root / "visualizations" / f"imagej_{image_id}.png"
                _save_visualization(visualization_path, image, mask, f"{image_id}: GFAP foreground candidates (unreviewed)", pseudo=True)
                visualizations.append(relative_path(visualization_path))
        except Exception as exc:
            errors.append(f"read_or_validation_error:{exc}")
            row["validation_status"] = "invalid"
        row["warnings"] = ";".join(errors + warnings)
        records.append(row)
    if not pilot:
        for name, path in sorted(images.items()):
            if name not in mapped_images:
                row = _empty_row(Path(name).stem, "imagej_pseudo")
                row.update({
                    "image_filename": name, "image_path": relative_path(path),
                    "annotation_type": "pseudo", "warnings": "image_not_in_source_mapping",
                    "slide_sample_id": _slide_and_tile(name)[0],
                })
                records.append(row)
    return records, visualizations


def prepare_dataset_validation(manual_root: str | Path, imagej_root: str | Path, output_root: str | Path, pilot: bool = False) -> dict:
    """Validate selected samples or all available samples without modifying inputs."""
    manual_root = resolve_path(manual_root)
    imagej_root = resolve_path(imagej_root)
    output_root = resolve_path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    manual, manual_views = _manual_records(manual_root, output_root, pilot)
    imagej, imagej_views = _imagej_records(imagej_root, output_root, pilot)
    rows = manual + imagej
    _write_csv(output_root / "dataset_manifest.csv", rows, MANIFEST_COLUMNS)
    stats_columns = [
        "sample_id", "annotation_source", "annotation_type", "validation_status", "training_suitable",
        "pixel_count", "annotated_pixels", "soma_pixels", "process_pixels", "uncertain_pixels",
        "ignore_pixels", "unexpected_color_pixels", "pseudo_foreground_pixels",
    ]
    _write_csv(output_root / "annotation_statistics.csv", rows, stats_columns)
    problems = [row for row in rows if row["validation_status"] in {"invalid", "requires_review"}]
    review = [row for row in rows if row["review_required"] == "true"]
    _write_csv(output_root / "missing_mismatched.csv", problems, MANIFEST_COLUMNS)
    _write_csv(output_root / "review_required.csv", review, MANIFEST_COLUMNS)
    valid_status = {"valid", "valid_with_warnings"}
    valid_manual = [row for row in manual if row["validation_status"] in valid_status]
    valid_imagej = [row for row in imagej if row["validation_status"] in valid_status]
    suitable = [row for row in manual if row["training_suitable"] == "true"]
    slide_groups = sorted({row["slide_sample_id"] for row in suitable if row["slide_sample_id"]})
    summary = {
        "created_utc": utc_now(),
        "scope": "one_manual_and_one_imagej_pilot" if pilot else "all_available_samples",
        "manual_record_count": len(manual),
        "valid_manual_sample_count": len(valid_manual),
        "valid_imagej_sample_count": len(valid_imagej),
        "problematic_sample_count": len(problems),
        "manual_training_suitable_count": len(suitable),
        "imagej_unreviewed_pseudo_candidate_count": len(valid_imagej),
        "review_required_count": len(review),
        "manual_training_slide_groups": slide_groups,
        "visualizations": manual_views + imagej_views,
        "label_semantics": {
            "0": "confirmed_background_only",
            "1": "manual_soma",
            "2": "manual_dense_process_only",
            "255": "ignore_unlabeled_uncertain_point_or_unverified_background",
        },
        "training_started": False,
        "train_test_split_created": False,
        "split_policy": "No split was created; future splits must keep source slide/sample groups together.",
        "imagej_usage": "Unreviewed GFAP foreground pseudo-labels only; not soma/process truth or final evaluation labels.",
        "direct_current_unet_loader_compatible": False,
        "compatibility_note": (
            "Generated manual labels contain ignore=255 and no confirmed background. An ignore-aware dataset adapter "
            "and explicit semantic_valid_mask are needed before training; existing loss and loader were not changed."
        ),
        "readiness_note": (
            "File-level validation does not establish biological correctness. A slide-independent validation set is "
            "required before interpreting first-training performance."
        ),
    }
    write_json_new(output_root / "validation_report.json", summary)
    write_json_new(output_root / "training_readiness.json", summary)
    lines = [
        "# GFAP dataset validation summary", "",
        f"- Scope: {summary['scope']}",
        f"- Valid manual records: {len(valid_manual)} / {len(manual)}",
        f"- Valid ImageJ pseudo-label records: {len(valid_imagej)} / {len(imagej)}",
        f"- Manual samples with dense training targets: {len(suitable)}",
        f"- Problematic records: {len(problems)}",
        f"- Records requiring review: {len(review)}", "",
        "No training was started and no train/test split was created.",
        "ImageJ outputs remain unreviewed GFAP foreground pseudo-labels, not final evaluation truth.",
        "Manual unlabeled and uncertain regions are ignore=255; no confirmed background was inferred.",
        summary["compatibility_note"], summary["readiness_note"], "",
    ]
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary

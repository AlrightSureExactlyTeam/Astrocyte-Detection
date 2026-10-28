"""Reusable real-ROI validation, prediction, evaluation and reporting utilities.

Coordinates are always original-ROI ``(x, y)`` pixels. Accuracy is based only on
confirmed human centers; uncertain annotations can exclude otherwise-unmatched
predictions but can never create or suppress a confirmed match.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from baseline_cellpose import mask_centers, run_cellpose
from evaluate import detection_metrics, match_points
from postprocessing import extract_centers
from predict import load_trained_model, predict_heatmap
from preprocessing import load_image, prepare_image

IMAGE_SUFFIXES = {".png", ".tif", ".tiff"}
PREDICTION_FIELDS = ["image_id", "prediction_id", "x", "y", "model", "confidence"]
METRIC_FIELDS = [
    "image_id", "model", "tp", "fp", "fn", "precision", "recall", "f1",
    "predicted_count", "confirmed_count", "uncertain_count", "ignored_count",
    "count_error", "match_distance_pixels",
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_unique_dir(root, prefix="experiment"):
    """Atomically create a new timestamped directory without replacing old runs."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(10000):
        name = f"{prefix}_{stamp}" if suffix == 0 else f"{prefix}_{stamp}_{suffix:04d}"
        path = root / name
        try:
            path.mkdir()
            return path
        except FileExistsError:
            pass
    raise RuntimeError(f"Cannot allocate a unique experiment directory under {root}")


def _image_index(directory):
    if not directory.is_dir():
        return {}, [f"Missing image directory: {directory}"]
    index, errors = {}, []
    for path in sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        if path.stem in index:
            errors.append(f"Duplicate image stem {path.stem}: {index[path.stem]} and {path}")
        else:
            index[path.stem] = path
    return index, errors


def _annotation_index(directory):
    if not directory.is_dir():
        return {}, [f"Missing annotation directory: {directory}"]
    index, errors = {}, []
    for path in sorted(directory.rglob("*.csv")):
        if path.stem in index:
            errors.append(f"Duplicate annotation stem {path.stem}: {index[path.stem]} and {path}")
        else:
            index[path.stem] = path
    return index, errors


def read_annotations(path):
    """Read strict annotation CSV and return confirmed/uncertain point arrays."""
    confirmed, uncertain = [], []
    seen_ids, seen_points = set(), set()
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"cell_id", "x", "y", "status"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: required columns are cell_id,x,y,status")
        for line, row in enumerate(reader, start=2):
            try:
                cell_id = int(row["cell_id"])
                x, y = float(row["x"]), float(row["y"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: invalid numeric annotation") from exc
            status = (row["status"] or "").strip().lower()
            if cell_id <= 0 or not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"{path}:{line}: IDs must be positive and coordinates finite")
            if status not in {"confirmed", "uncertain"}:
                raise ValueError(f"{path}:{line}: unsupported status {status!r}")
            point = (x, y)
            if cell_id in seen_ids or point in seen_points:
                raise ValueError(f"{path}:{line}: duplicate cell_id or coordinate")
            seen_ids.add(cell_id)
            seen_points.add(point)
            (confirmed if status == "confirmed" else uncertain).append(point)
    return {
        "confirmed": np.asarray(confirmed, dtype=np.float64).reshape(-1, 2),
        "uncertain": np.asarray(uncertain, dtype=np.float64).reshape(-1, 2),
    }


def _check_bounds(points, width, height, label):
    bad = (points[:, 0] < 0) | (points[:, 0] >= width) | (points[:, 1] < 0) | (points[:, 1] >= height)
    if np.any(bad):
        raise ValueError(f"{label} outside {width}x{height} image: {points[bad][:3].tolist()}")


def validate_real_data(data_root, output_path=None):
    """Return a report even when directories/images are absent; never invent GT."""
    root = Path(data_root)
    images, errors = _image_index(root / "images")
    annotations, annotation_errors = _annotation_index(root / "annotations")
    errors.extend(annotation_errors)
    missing_annotations = sorted(set(images) - set(annotations))
    missing_images = sorted(set(annotations) - set(images))
    records = []
    for image_id in sorted(set(images) & set(annotations)):
        record = {
            "image_id": image_id, "image": str(images[image_id]),
            "annotation": str(annotations[image_id]), "status": "valid",
            "width": None, "height": None, "channels": None,
            "confirmed_count": 0, "uncertain_count": 0,
        }
        try:
            image = load_image(images[image_id])
            if image.size == 0 or image.ndim not in (2, 3):
                raise ValueError(f"Empty or unsupported image shape: {image.shape}")
            record["height"], record["width"] = map(int, image.shape[:2])
            record["channels"] = 1 if image.ndim == 2 else int(image.shape[2])
            labels = read_annotations(annotations[image_id])
            _check_bounds(labels["confirmed"], record["width"], record["height"], "Confirmed point")
            _check_bounds(labels["uncertain"], record["width"], record["height"], "Uncertain point")
            record["confirmed_count"] = int(len(labels["confirmed"]))
            record["uncertain_count"] = int(len(labels["uncertain"]))
        except (OSError, ValueError) as exc:
            record["status"] = "invalid"
            record["error"] = str(exc)
            errors.append(f"{image_id}: {exc}")
        records.append(record)
    for image_id in missing_annotations:
        errors.append(f"{image_id}: missing annotation CSV")
    for image_id in missing_images:
        errors.append(f"{image_id}: missing original image")
    status = "valid" if records and not errors else "missing_input" if not records and not errors else "invalid"
    report = {
        "status": status, "data_root": str(root), "image_count": len(images),
        "annotation_count": len(annotations), "paired_count": len(records),
        "confirmed_count": sum(r["confirmed_count"] for r in records),
        "uncertain_count": sum(r["uncertain_count"] for r in records),
        "missing_images": missing_images, "missing_annotations": missing_annotations,
        "records": records, "errors": errors,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


def read_predictions(path, expected_model=None):
    points, confidences = [], []
    seen_ids = set()
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(PREDICTION_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"{path}: incompatible prediction CSV schema")
        image_ids = set()
        for line, row in enumerate(reader, start=2):
            try:
                prediction_id = int(row["prediction_id"])
                x, y = float(row["x"]), float(row["y"])
                confidence = float(row["confidence"]) if row["confidence"].strip() else None
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"{path}:{line}: invalid prediction values") from exc
            image_id, model = row["image_id"], row["model"]
            if not image_id or prediction_id <= 0 or prediction_id in seen_ids:
                raise ValueError(f"{path}:{line}: empty image_id or duplicate/invalid prediction_id")
            if expected_model is not None and model != expected_model:
                raise ValueError(f"{path}:{line}: expected model {expected_model}, got {model}")
            if not all(math.isfinite(v) for v in (x, y)) or confidence is not None and not math.isfinite(confidence):
                raise ValueError(f"{path}:{line}: non-finite prediction")
            image_ids.add(image_id)
            seen_ids.add(prediction_id)
            points.append((x, y))
            confidences.append(confidence)
    if len(image_ids) > 1:
        raise ValueError(f"{path}: each prediction file must contain exactly one image_id")
    return np.asarray(points, dtype=np.float64).reshape(-1, 2), confidences


def write_predictions(path, image_id, points, model, confidences=None, shape=None):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError("Prediction coordinates must be finite")
    if shape is not None:
        _check_bounds(points, shape[1], shape[0], "Prediction")
    if confidences is None:
        confidences = [None] * len(points)
    if len(confidences) != len(points):
        raise ValueError("Confidence count must match predictions")
    rows = [{
        "image_id": image_id, "prediction_id": index, "x": float(x), "y": float(y),
        "model": model, "confidence": "" if confidence is None else float(confidence),
    } for index, ((x, y), confidence) in enumerate(zip(points, confidences), start=1)]
    write_csv(path, rows, PREDICTION_FIELDS)


def save_overlay(image, points, path, color=(0, 0, 255), extra_points=None, channel=None):
    gray = prepare_image(image, channel=channel).squeeze(0).numpy()
    canvas = cv2.cvtColor(np.rint(gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for x, y in np.asarray(points).reshape(-1, 2):
        cv2.circle(canvas, (int(round(x)), int(round(y))), 4, color, 1, cv2.LINE_AA)
    if extra_points:
        for label_points, label_color in extra_points:
            for x, y in np.asarray(label_points).reshape(-1, 2):
                cv2.circle(canvas, (int(round(x)), int(round(y))), 4, label_color, 1, cv2.LINE_AA)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write overlay: {path}")


def predict_unet_image(image_path, checkpoint_path, config, output_dir, device="cpu"):
    """Run center-only U-Net inference, with explicit resize and inverse mapping."""
    started = time.perf_counter()
    image = load_image(image_path)
    channel = config.get("inference", {}).get("channel", config.get("data", {}).get("channel"))
    tensor = prepare_image(image, channel=channel)
    original_height, original_width = image.shape[:2]
    requested_size = config.get("data", {}).get("image_size")
    resized = False
    if requested_size:
        target = tuple(map(int, requested_size))
        if tensor.shape[-2:] != target:
            tensor = F.interpolate(tensor.unsqueeze(0), size=target, mode="bilinear", align_corners=False)[0]
            resized = True
    model = load_trained_model(str(checkpoint_path), config, device)
    probabilities = predict_heatmap(model, tensor, device)
    heatmap = probabilities["center_prob"]
    inference = config.get("inference", {})
    points = extract_centers(
        heatmap, threshold=float(inference.get("threshold", 0.5)),
        min_distance=int(inference.get("min_distance_pixels", 5)),
    )
    map_height, map_width = heatmap.shape
    scale_x, scale_y = original_width / map_width, original_height / map_height
    if len(points):
        points = points * np.asarray([scale_x, scale_y])
        points = np.minimum(points, [original_width - 1e-6, original_height - 1e-6])
    confidences = [float(heatmap[int(round(y / scale_y)), int(round(x / scale_x))]) for x, y in points]
    image_id = Path(image_path).stem
    output_dir = Path(output_dir)
    write_predictions(output_dir / "predictions" / f"{image_id}.csv", image_id, points, "unet", confidences, image.shape)
    save_overlay(image, points, output_dir / "overlays" / f"{image_id}.png", channel=channel)
    metadata = {
        "image_id": image_id, "model": "unet", "checkpoint": str(checkpoint_path),
        "device": str(device), "original_shape": [original_height, original_width],
        "inference_shape": list(heatmap.shape), "resized": resized,
        "coordinate_scale_xy": [scale_x, scale_y], "channel": channel,
        "threshold": float(inference.get("threshold", 0.5)),
        "min_distance_pixels": int(inference.get("min_distance_pixels", 5)),
        "semantic_head_used_for_filtering": False, "prediction_count": len(points),
        "seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "metadata" / f"{image_id}.json", metadata)
    return points, metadata


def predict_cellpose_image(image_path, config, output_dir, device="cpu"):
    if str(device) != "cpu":
        raise ValueError("GPU Cellpose inference requires explicit code review and request")
    started = time.perf_counter()
    image = load_image(image_path)
    baseline = config.get("baseline", {})
    channel = baseline.get("channel", config.get("data", {}).get("channel"))
    if image.ndim == 3 and image.shape[-1] > 1 and channel is None:
        raise ValueError("Multichannel fluorescence input requires baseline.channel")
    masks = run_cellpose(
        image, use_gpu=False, channel=channel, diameter=baseline.get("diameter"),
        min_size=int(baseline.get("min_size", 15)),
        max_size_fraction=float(baseline.get("max_size_fraction", 0.4)),
    )
    centers = mask_centers(masks)
    points = np.asarray([(x, y) for _, x, y in centers], dtype=np.float64).reshape(-1, 2)
    image_id = Path(image_path).stem
    output_dir = Path(output_dir)
    write_predictions(output_dir / "predictions" / f"{image_id}.csv", image_id, points, "cellpose", shape=image.shape)
    save_overlay(image, points, output_dir / "overlays" / f"{image_id}.png", channel=channel)
    metadata = {
        "image_id": image_id, "model": "cellpose", "device": "cpu",
        "original_shape": list(image.shape[:2]), "coordinate_scale_xy": [1.0, 1.0],
        "channel": channel, "diameter": baseline.get("diameter"),
        "min_size": int(baseline.get("min_size", 15)),
        "max_size_fraction": float(baseline.get("max_size_fraction", 0.4)),
        "prediction_count": len(points), "seconds": time.perf_counter() - started,
        "limitation": "General cell baseline; detections are not necessarily GFAP-positive astrocytes.",
    }
    write_json(output_dir / "metadata" / f"{image_id}.json", metadata)
    return points, metadata


def evaluate_points(predictions, labels, max_distance, image_id="", model=""):
    """Confirmed-first one-to-one matching, then uncertain exclusion for leftovers."""
    confirmed_result = match_points(predictions, labels["confirmed"], max_distance)
    matched_predictions = {match[0] for match in confirmed_result["matches"]}
    remaining_indices = [i for i in range(len(predictions)) if i not in matched_predictions]
    remaining = predictions[remaining_indices]
    uncertain_result = match_points(remaining, labels["uncertain"], max_distance)
    ignored_local = {match[0] for match in uncertain_result["matches"]}
    ignored_indices = [remaining_indices[i] for i in sorted(ignored_local)]
    fp_indices = [i for i in remaining_indices if i not in set(ignored_indices)]
    matched_gt = {match[1] for match in confirmed_result["matches"]}
    fn_indices = [i for i in range(len(labels["confirmed"])) if i not in matched_gt]
    tp, fp, fn = len(matched_predictions), len(fp_indices), len(fn_indices)
    row = {
        "image_id": image_id, "model": model, "tp": tp, "fp": fp, "fn": fn,
        **detection_metrics(tp, fp, fn), "predicted_count": len(predictions),
        "confirmed_count": len(labels["confirmed"]), "uncertain_count": len(labels["uncertain"]),
        "ignored_count": len(ignored_indices), "count_error": len(predictions) - len(ignored_indices) - len(labels["confirmed"]),
        "match_distance_pixels": float(max_distance),
        "_tp_indices": sorted(matched_predictions), "_fp_indices": fp_indices,
        "_fn_indices": fn_indices, "_ignored_indices": ignored_indices,
        "_matches": confirmed_result["matches"],
    }
    return row


def evaluate_experiment(data_root, experiment_dir, models=("unet", "cellpose"), max_distance=8.0):
    if not np.isfinite(max_distance) or max_distance < 0:
        raise ValueError("max_distance must be finite and nonnegative")
    root, experiment = Path(data_root), Path(experiment_dir)
    images, image_errors = _image_index(root / "images")
    annotations, annotation_errors = _annotation_index(root / "annotations")
    if image_errors or annotation_errors:
        raise ValueError("; ".join(image_errors + annotation_errors))
    all_rows = []
    for model in models:
        prediction_dir = experiment / model / "predictions"
        if not prediction_dir.is_dir():
            continue
        rows = []
        for path in sorted(prediction_dir.glob("*.csv")):
            image_id = path.stem
            if image_id not in images or image_id not in annotations:
                raise ValueError(f"Prediction {image_id} lacks a paired original image/annotation")
            predictions, _ = read_predictions(path, expected_model=model)
            image = load_image(images[image_id])
            _check_bounds(predictions, image.shape[1], image.shape[0], "Prediction")
            row = evaluate_points(predictions, read_annotations(annotations[image_id]), max_distance, image_id, model)
            rows.append(row)
        write_csv(experiment / "metrics" / f"{model}_metrics.csv", rows, METRIC_FIELDS)
        all_rows.extend(rows)
    return all_rows


def micro_metrics(rows):
    totals = {key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn")}
    return {**totals, **detection_metrics(**totals)}


def compare_metrics(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["image_id"], {})[row["model"]] = row
    comparisons = []
    for image_id, models in sorted(grouped.items()):
        reference = models.get("unet") or next(iter(models.values()))
        result = {"image_id": image_id, "ground_truth_count": reference["confirmed_count"]}
        for model in ("cellpose", "unet"):
            row = models.get(model)
            for field in ("predicted_count", "tp", "fp", "fn", "precision", "recall", "f1", "count_error"):
                result[f"{model}_{field}"] = "" if row is None else row[field]
        result["f1_difference_unet_minus_cellpose"] = (
            models["unet"]["f1"] - models["cellpose"]["f1"] if "unet" in models and "cellpose" in models else ""
        )
        comparisons.append(result)
    return comparisons


def _public_row(row):
    return {key: value for key, value in row.items() if not key.startswith("_")}


def generate_summary(experiment_dir, rows, comparisons, validation=None):
    experiment = Path(experiment_dir)
    models = {}
    for model in ("unet", "cellpose"):
        selected = [row for row in rows if row["model"] == model]
        if selected:
            models[model] = {"roi_count": len(selected), **micro_metrics(selected)}
    summary = {
        "experiment": experiment.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if len(models) == 2 else "incomplete_comparison" if models else "predictions_only",
        "roi_count": len({row["image_id"] for row in rows}),
        "ground_truth": {
            "confirmed": sum(row["confirmed_count"] for row in rows if row["model"] == "unet") or sum(row["confirmed_count"] for row in rows if row["model"] == "cellpose"),
            "uncertain": sum(row["uncertain_count"] for row in rows if row["model"] == "unet") or sum(row["uncertain_count"] for row in rows if row["model"] == "cellpose"),
        },
        "models": models, "comparison": comparisons,
        "match_distance_pixels": rows[0]["match_distance_pixels"] if rows else None,
        "limitations": [
            "Pilot ROIs from one tissue section are not independent biological samples.",
            "The U-Net semantic head is not evaluated or used to filter centers.",
            "Cellpose is a general cell baseline; its objects are not necessarily GFAP-positive astrocytes.",
            "No generalization claim or inferential p-value is supported by this pipeline output.",
        ],
    }
    if validation is not None:
        summary["validation_status"] = validation.get("status")
    write_json(experiment / "metrics" / "summary.json", summary)
    return summary


def generate_visualizations(data_root, experiment_dir, rows):
    """Create ROI panels and metric bars with OpenCV (no plotting dependency)."""
    root, experiment = Path(data_root), Path(experiment_dir)
    figures = experiment / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    images, _ = _image_index(root / "images")
    annotations, _ = _annotation_index(root / "annotations")
    channel = None
    config_path = experiment / "experiment_config.yaml"
    if config_path.is_file():
        import yaml
        with config_path.open("r", encoding="utf-8") as handle:
            saved_config = yaml.safe_load(handle) or {}
        channel = saved_config.get("inference", {}).get("channel", saved_config.get("data", {}).get("channel"))
    by_image = {}
    for row in rows:
        by_image.setdefault(row["image_id"], {})[row["model"]] = row

    def base_panel(image, title, size=(400, 300)):
        gray = prepare_image(image, channel=channel).squeeze(0).numpy()
        panel = cv2.cvtColor(np.rint(gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(panel, size, interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(panel, (0, 0), (size[0] - 1, 28), (30, 30, 30), -1)
        cv2.putText(panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return panel

    def mark(panel, points, color, image_shape, cross=False):
        height, width = image_shape[:2]
        for x, y in np.asarray(points).reshape(-1, 2):
            px, py = int(round(x * 399 / max(width - 1, 1))), int(round(y * 299 / max(height - 1, 1)))
            if cross:
                cv2.line(panel, (px - 4, py - 4), (px + 4, py + 4), color, 1)
                cv2.line(panel, (px - 4, py + 4), (px + 4, py - 4), color, 1)
            else:
                cv2.circle(panel, (px, py), 5, color, 1, cv2.LINE_AA)

    for image_id, model_rows in by_image.items():
        image = load_image(images[image_id])
        labels = read_annotations(annotations[image_id])
        panels = [base_panel(image, title) for title in ("Original", "Human: green confirmed / yellow uncertain", "U-Net", "Cellpose", "U-Net: TP green FP red FN magenta ignored yellow", "Cellpose: TP green FP red FN magenta ignored yellow")]
        mark(panels[1], labels["confirmed"], (0, 255, 0), image.shape)
        mark(panels[1], labels["uncertain"], (0, 200, 255), image.shape)
        for panel_index, model in ((2, "unet"), (3, "cellpose")):
            if model in model_rows:
                points, _ = read_predictions(experiment / model / "predictions" / f"{image_id}.csv")
                mark(panels[panel_index], points, (255, 255, 0), image.shape)
        for panel_index, model in ((4, "unet"), (5, "cellpose")):
            row = model_rows.get(model)
            if row is None:
                continue
            points, _ = read_predictions(experiment / model / "predictions" / f"{image_id}.csv")
            for key, color in (("_tp_indices", (0, 255, 0)), ("_fp_indices", (0, 0, 255)), ("_ignored_indices", (0, 200, 255))):
                mark(panels[panel_index], points[row[key]] if row[key] else np.empty((0, 2)), color, image.shape)
            mark(panels[panel_index], labels["confirmed"][row["_fn_indices"]] if row["_fn_indices"] else np.empty((0, 2)), (255, 0, 255), image.shape, cross=True)
        comparison = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
        if not cv2.imwrite(str(figures / f"{image_id}_comparison.png"), comparison):
            raise OSError(f"Could not save comparison figure for {image_id}")

    models = sorted({row["model"] for row in rows})
    image_ids = sorted(by_image)
    colors = {"unet": (255, 180, 0), "cellpose": (180, 100, 0)}
    for metric in ("f1", "precision", "recall"):
        width = max(600, 100 * len(image_ids) + 160)
        chart = np.full((420, width, 3), 255, dtype=np.uint8)
        cv2.putText(chart, f"Per-ROI {metric}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.line(chart, (70, 350), (width - 30, 350), (0, 0, 0), 1)
        for index, image_id in enumerate(image_ids):
            for model_index, model in enumerate(models):
                row = by_image[image_id].get(model)
                if row is None:
                    continue
                x = 90 + index * 100 + model_index * 32
                bar_height = int(round(280 * row[metric]))
                cv2.rectangle(chart, (x, 350 - bar_height), (x + 28, 350), colors.get(model, (100, 100, 100)), -1)
                cv2.putText(chart, f"{row[metric]:.2f}", (x - 4, 342 - bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            cv2.putText(chart, image_id[:14], (75 + index * 100, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        for index, model in enumerate(models):
            cv2.rectangle(chart, (width - 150, 50 + 25 * index), (width - 130, 65 + 25 * index), colors.get(model, (100, 100, 100)), -1)
            cv2.putText(chart, model, (width - 120, 64 + 25 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        if not cv2.imwrite(str(figures / f"{metric}_comparison.png"), chart):
            raise OSError(f"Could not save {metric} chart")


def generate_report(experiment_dir, rows, comparisons, summary, validation=None):
    experiment = Path(experiment_dir)
    lines = [
        f"# Experiment {experiment.name}", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        "## Scope", "",
        "Astrocyte center-detection pilot evaluation. This report contains observed pipeline outputs only.", "",
        "## Data and validation", "",
        f"- ROI images: {summary['roi_count']}",
        f"- Confirmed annotations: {summary['ground_truth']['confirmed']}",
        f"- Uncertain annotations: {summary['ground_truth']['uncertain']}",
        f"- Validation status: {validation['status'] if validation else 'not recorded'}",
        f"- Matching threshold: {summary['match_distance_pixels']} pixels", "",
        "## Overall results", "",
        "| Model | TP | FP | FN | Micro-Precision | Micro-Recall | Micro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, metrics in summary["models"].items():
        lines.append(f"| {model} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |")
    if not rows:
        lines.extend(["", "No evaluated predictions were available; accuracy was not calculated."])
    elif len(summary["models"]) < 2:
        lines.extend(["", "Only one model was available; model comparison is incomplete."])
    lines.extend(["", "## Per-ROI results", "", "| Image | Model | TP | FP | FN | Precision | Recall | F1 | Ignored |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(f"| {row['image_id']} | {row['model']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['ignored_count']} |")
    lines.extend(["", "## Limitations", ""] + [f"- {item}" for item in summary["limitations"]])
    lines.extend(["", "## Conclusion", "", "Results apply only to the supplied pilot ROIs and selected checkpoint/configuration. No semantic-segmentation or cross-slide generalization claim is made.", ""])
    (experiment / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_csv(experiment / "metrics" / "metrics.csv", rows, METRIC_FIELDS)
    fields = list(comparisons[0]) if comparisons else ["image_id"]
    write_csv(experiment / "metrics" / "comparison.csv", comparisons, fields)

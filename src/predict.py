"""Run trained AstrocyteUNet inference on images and tiled TIFF/QPTIFF slides."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import yaml

if __package__:
    from .model import create_model
    from .postprocessing import (
        centers_to_global_coordinates,
        extract_centers,
        separate_instances,
    )
    from .preprocessing import iter_qptiff_tiles, load_image, prepare_image
else:
    from model import create_model
    from postprocessing import centers_to_global_coordinates, extract_centers, separate_instances
    from preprocessing import iter_qptiff_tiles, load_image, prepare_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(
        "cuda" if str(device) == "auto" and torch.cuda.is_available()
        else "cpu" if str(device) == "auto" else device
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    return resolved


def load_trained_model(checkpoint_path: str, config: dict, device: str):
    """Load a compatible two-head checkpoint and return an evaluation model."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    resolved_device = _resolve_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict")
    expected_heads = {"semantic_classes": 3, "center_channels": 1}
    if checkpoint.get("output_heads") != expected_heads:
        raise ValueError(
            f"Checkpoint output_heads must be {expected_heads}; "
            "legacy single-heatmap weights are not compatible"
        )

    model_cfg = config.get("model", {})
    if model_cfg.get("architecture", "unet") != "unet":
        raise ValueError("Only the unet architecture is supported")
    if int(model_cfg.get("output_channels", 4)) != 4:
        raise ValueError("model.output_channels must be 4 (3 semantic + 1 center)")
    model = create_model(
        in_channels=int(config.get("data", {}).get("input_channels", 1)),
        encoder_name=model_cfg.get("encoder", "resnet18"),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(resolved_device)
    model.eval()
    return model


@torch.no_grad()
def predict_heatmap(model, image_tensor: torch.Tensor, device: str) -> dict:
    """Return semantic probabilities ``[3,H,W]`` and center probability ``[H,W]``."""
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor must be a torch.Tensor")
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError("image_tensor must have shape [C,H,W] or [1,C,H,W]")
    if not torch.is_floating_point(image_tensor) or not torch.isfinite(image_tensor).all():
        raise ValueError("image_tensor must contain finite floating-point values")

    outputs = model(image_tensor.to(_resolve_device(device)))
    if not isinstance(outputs, dict) or not {"semantic_logits", "center_logits"} <= outputs.keys():
        raise ValueError("Model must return semantic_logits and center_logits")
    semantic_logits = outputs["semantic_logits"]
    center_logits = outputs["center_logits"]
    if semantic_logits.ndim != 4 or semantic_logits.shape[:2] != (1, 3):
        raise ValueError("semantic_logits must have shape [1,3,H,W]")
    if center_logits.ndim != 4 or center_logits.shape[:2] != (1, 1):
        raise ValueError("center_logits must have shape [1,1,H,W]")
    if semantic_logits.shape[2:] != center_logits.shape[2:]:
        raise ValueError("Semantic and center output sizes must match")

    # IMPLEMENTED [Liu]: softmax only the semantic head and sigmoid only centers.
    semantic_probability = torch.softmax(semantic_logits, dim=1)[0].cpu().numpy()
    center_probability = torch.sigmoid(center_logits)[0, 0].cpu().numpy()
    return {
        "semantic_prob": semantic_probability.astype(np.float32),
        "center_prob": center_probability.astype(np.float32),
    }


def heatmap_to_points(heatmap: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    """Decode center probabilities using the shared postprocessing implementation."""
    return extract_centers(heatmap, threshold=threshold, min_distance=min_distance)


def _semantic_from_probability(probability: np.ndarray, threshold: float | None) -> np.ndarray:
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("semantic_confidence_threshold must be in [0,1]")
    mask = probability.argmax(axis=0).astype(np.uint8)
    if threshold is not None:
        mask[probability.max(axis=0) < threshold] = 0
    return mask


def _predict_array(model, image: np.ndarray, config: dict, device: str) -> dict:
    inference_cfg = config.get("inference", {})
    channel = inference_cfg.get("channel", config.get("data", {}).get("channel"))
    tensor = prepare_image(image, channel=channel)
    probabilities = predict_heatmap(model, tensor, device)
    raw_points = heatmap_to_points(
        probabilities["center_prob"],
        threshold=float(inference_cfg.get("threshold", 0.5)),
        min_distance=int(inference_cfg.get("min_distance_pixels", 5)),
    )
    semantic_mask = _semantic_from_probability(
        probabilities["semantic_prob"], inference_cfg.get("semantic_confidence_threshold")
    )
    # FIX [Liu]: center detections and counts must not depend on semantic assignment.
    instance_mask, assigned_points = separate_instances(semantic_mask, raw_points)
    instance_count = int(np.count_nonzero(np.unique(instance_mask)))
    return {
        "points": raw_points,
        "assigned_points": assigned_points,
        "count": int(len(raw_points)),
        "instance_count": instance_count,
        "center_prob": probabilities["center_prob"],
        "semantic_prob": probabilities["semantic_prob"],
        "semantic_mask": semantic_mask,
        "instance_mask": instance_mask,
        "raw_peak_count": int(len(raw_points)),
    }


def predict_image(
    image_path: str,
    checkpoint_path: str,
    config: dict,
    device: str = "auto",
    model=None,
):
    """Run the complete single-image pipeline and return predictions."""
    image = load_image(image_path)
    if model is None:
        model = load_trained_model(checkpoint_path, config, device)
    result = _predict_array(model, image, config, device)
    result["metadata"] = {
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "image_size": tuple(image.shape[:2]),
        "scale_xy": (1.0, 1.0),
        "tile_offset": (0, 0),
        "instance_assignment": "center-marker watershed; crossing processes require review",
    }
    return result


def _deduplicate_global_detections(detections, tile_bounds, max_distance):
    """Merge cross-tile duplicates with spatial hashing and overlap checks."""
    if not detections:
        return np.empty((0, 2), dtype=np.float64), [], {}
    parent = list(range(len(detections)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    def relevant_pair(first, second):
        a, b = detections[first], detections[second]
        if a["tile_id"] == b["tile_id"]:
            return False
        ax0, ay0, ax1, ay1 = tile_bounds[a["tile_id"]]
        bx0, by0, bx1, by1 = tile_bounds[b["tile_id"]]
        x0, y0 = max(ax0, bx0), max(ay0, by0)
        x1, y1 = min(ax1, bx1), min(ay1, by1)
        if x1 <= x0 or y1 <= y0:
            return False
        # Both detections must lie in the shared tile region (within tolerance).
        return all(
            x0 - max_distance <= point["x"] <= x1 + max_distance
            and y0 - max_distance <= point["y"] <= y1 + max_distance
            for point in (a, b)
        )

    grid = {}
    exact_locations = {}
    cell_size = max(float(max_distance), 1.0)
    for index, point in enumerate(detections):
        candidates = set()
        if max_distance == 0:
            candidates.update(exact_locations.get((point["x"], point["y"]), []))
        else:
            grid_x, grid_y = int(np.floor(point["x"] / cell_size)), int(np.floor(point["y"] / cell_size))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    candidates.update(grid.get((grid_x + dx, grid_y + dy), []))
        for other in candidates:
            if not relevant_pair(index, other):
                continue
            distance = np.hypot(point["x"] - detections[other]["x"], point["y"] - detections[other]["y"])
            if distance <= max_distance:
                union(index, other)
        if max_distance == 0:
            exact_locations.setdefault((point["x"], point["y"]), []).append(index)
        else:
            grid.setdefault((grid_x, grid_y), []).append(index)

    components = {}
    for index in range(len(detections)):
        components.setdefault(find(index), []).append(index)
    ordered = []
    for members in components.values():
        representative = min(
            members,
            key=lambda i: (
                detections[i]["y"], detections[i]["x"], detections[i]["tile_id"], i
            ),
        )
        ordered.append((representative, members))
    ordered.sort(key=lambda item: (
        detections[item[0]]["y"], detections[item[0]]["x"], detections[item[0]]["tile_id"]
    ))
    points, sources, accepted_by_tile = [], [], {tile_id: 0 for tile_id in tile_bounds}
    for representative, members in ordered:
        point = detections[representative]
        points.append((point["x"], point["y"]))
        sources.append(sorted({detections[index]["tile_id"] for index in members}))
        accepted_by_tile[point["tile_id"]] += 1
    return np.asarray(points, dtype=np.float64).reshape(-1, 2), sources, accepted_by_tile


def predict_whole_slide(
    slide_path: str,
    checkpoint_path: str,
    config: dict,
    device: str = "auto",
    model=None,
    output_dir: str | None = None,
):
    """Predict tiles and deduplicate raw centers only in geometric overlaps.

    Coordinates use the selected level's grid. Extremely close real cells in an
    overlap remain ambiguous; distance alone cannot prove biological identity.
    Full-slide masks are not materialized and cross-tile instances are not stitched.
    """
    if model is None:
        model = load_trained_model(checkpoint_path, config, device)
    inference_cfg = config.get("inference", {})
    tile_size = int(inference_cfg.get("tile_size", 512))
    overlap = int(inference_cfg.get("overlap", 32))
    level = int(inference_cfg.get("level", 0))
    channel = inference_cfg.get("channel", config.get("data", {}).get("channel"))
    series_index = int(inference_cfg.get("series_index", 0))
    dedup_distance = float(inference_cfg.get("dedup_distance_pixels", inference_cfg.get("min_distance_pixels", 5)))
    if not np.isfinite(dedup_distance) or dedup_distance < 0:
        raise ValueError("dedup_distance_pixels must be finite and nonnegative")

    tile_output = Path(output_dir) / "tiles" if output_dir is not None else None
    if tile_output is not None:
        tile_output.mkdir(parents=True, exist_ok=True)
    detections = []
    tile_bounds = {}
    tile_records = []
    next_instance_id = 1
    for item in iter_qptiff_tiles(
        slide_path,
        tile_size=tile_size,
        level=level,
        overlap=overlap,
        channel=channel,
        series_index=series_index,
        with_meta=True,
    ):
        tile_id = str(item["tile_id"])
        if tile_id in tile_bounds:
            raise ValueError(f"Duplicate tile ID: {tile_id}")
        prediction = _predict_array(model, item["tile"], config, device)
        global_points = centers_to_global_coordinates(
            prediction["points"], tile_offset=(item["x"], item["y"])
        )
        tile_height, tile_width = item["tile"].shape[:2]
        tile_bounds[tile_id] = (
            float(item["x"]), float(item["y"]),
            float(item["x"] + tile_width), float(item["y"] + tile_height),
        )
        seen_within_tile = set()
        for x, y in global_points:
            location = (float(x), float(y))
            if location in seen_within_tile:
                continue
            seen_within_tile.add(location)
            detections.append({"x": location[0], "y": location[1], "tile_id": tile_id})

        instance_mask = prediction["instance_mask"]
        remapped = np.zeros_like(instance_mask, dtype=np.uint32)
        instance_id_map = {}
        for local_id in np.unique(instance_mask):
            if local_id == 0:
                continue
            if next_instance_id > np.iinfo(np.uint32).max:
                raise ValueError("Global instance ID exceeds uint32 range")
            instance_id_map[str(int(local_id))] = int(next_instance_id)
            remapped[instance_mask == local_id] = next_instance_id
            next_instance_id += 1
        if tile_output is not None:
            stem = f"{tile_id}_level{level}"
            tifffile.imwrite(tile_output / f"{stem}_semantic.tif", prediction["semantic_mask"])
            tifffile.imwrite(tile_output / f"{stem}_instance.tif", remapped)
        tile_records.append({
            "tile_id": tile_id,
            "x": item["x"],
            "y": item["y"],
            "width": tile_width,
            "height": tile_height,
            "raw_peak_count": prediction["raw_peak_count"],
            "assigned_count": int(len(prediction["assigned_points"])),
            "instance_count": prediction["instance_count"],
            "instance_id_map": instance_id_map,
            "accepted_after_dedup": 0,
            "level": level,
            "series_index": series_index,
            "channel": channel,
            "scale_xy": item["scale_xy"],
        })

    points, point_sources, accepted_by_tile = _deduplicate_global_detections(
        detections, tile_bounds, dedup_distance
    )
    for record in tile_records:
        record["accepted_after_dedup"] = accepted_by_tile.get(record["tile_id"], 0)
    return {
        "points": points,
        "point_sources": point_sources,
        "count": int(len(points)),
        "tiles": tile_records,
        "metadata": {
            "slide_path": str(slide_path),
            "checkpoint_path": str(checkpoint_path),
            "level": level,
            "series_index": series_index,
            "channel": channel,
            "coordinate_grid": "selected level",
            "tile_size": tile_size,
            "overlap": overlap,
            "dedup_distance_pixels": dedup_distance,
            "dedup_policy": "exact within-tile duplicates; distance only between geometrically overlapping tiles",
            "instance_id_policy": "globally unique per tile-local instance; cross-tile instances are not stitched",
        },
    }


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_predictions(
    points: np.ndarray,
    image,
    output_dir: str,
    semantic_mask: np.ndarray | None = None,
    instance_mask: np.ndarray | None = None,
    metadata: dict | None = None,
):
    """Validate all outputs, then atomically publish CSV, masks, and overlay."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be a finite (N,2) array")
    if output_dir is None or not str(output_dir).strip():
        raise ValueError("output_dir must not be empty")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")

    gray = None
    if image is not None:
        gray = prepare_image(image).squeeze(0).numpy()
        out_of_bounds = (
            (points[:, 0] < 0) | (points[:, 0] >= gray.shape[1])
            | (points[:, 1] < 0) | (points[:, 1] >= gray.shape[0])
        )
        if np.any(out_of_bounds):
            raise ValueError("points must lie inside the supplied image")
    if semantic_mask is not None:
        semantic_mask = np.asarray(semantic_mask)
        if (
            semantic_mask.ndim != 2 or semantic_mask.size == 0
            or not np.issubdtype(semantic_mask.dtype, np.integer)
            or np.any(semantic_mask < 0) or np.any(semantic_mask > 2)
        ):
            raise ValueError("semantic_mask must be a non-empty 2D integer label map with values 0, 1, 2")
        if gray is not None and semantic_mask.shape != gray.shape:
            raise ValueError("semantic_mask shape must match the image")
    if instance_mask is not None:
        instance_mask = np.asarray(instance_mask)
        if (
            instance_mask.ndim != 2 or instance_mask.size == 0
            or not np.issubdtype(instance_mask.dtype, np.integer)
            or np.any(instance_mask < 0)
        ):
            raise ValueError("instance_mask must be a non-empty non-negative 2D integer label map")
        if instance_mask.size and instance_mask.max() > np.iinfo(np.uint32).max:
            raise ValueError("Instance ID exceeds uint32 range")
        if gray is not None and instance_mask.shape != gray.shape:
            raise ValueError("instance_mask shape must match the image")
    if semantic_mask is not None and instance_mask is not None and semantic_mask.shape != instance_mask.shape:
        raise ValueError("semantic_mask and instance_mask shapes must match")
    summary_json = json.dumps(
        {"count": int(len(points)), "metadata": _json_value(metadata or {})},
        indent=2,
        ensure_ascii=False,
    )

    # FIX [Liu]: invalid inputs and rendering failures cannot leave partial finals.
    import tempfile
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".predictions_", dir=out_dir) as temp_dir:
        staging = Path(temp_dir)
        with (staging / "centers.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cell_id", "x", "y", "soma_area", "process_area", "status"])
            for index, (x, y) in enumerate(points, start=1):
                soma_area = process_area = ""
                status = "center_only"
                if instance_mask is not None and semantic_mask is not None:
                    col, row = int(round(x)), int(round(y))
                    if 0 <= row < instance_mask.shape[0] and 0 <= col < instance_mask.shape[1]:
                        instance_id = int(instance_mask[row, col])
                        if instance_id > 0:
                            selected = instance_mask == instance_id
                            soma_area = int(np.count_nonzero(selected & (semantic_mask == 1)))
                            process_area = int(np.count_nonzero(selected & (semantic_mask == 2)))
                            status = "assigned"
                        else:
                            status = "unassigned"
                writer.writerow([index, float(x), float(y), soma_area, process_area, status])
        (staging / "summary.json").write_text(summary_json, encoding="utf-8")
        if semantic_mask is not None:
            tifffile.imwrite(staging / "semantic_mask.tif", semantic_mask.astype(np.uint8))
        if instance_mask is not None:
            tifffile.imwrite(staging / "instance_mask.tif", instance_mask.astype(np.uint32))
        if gray is not None:
            overlay = cv2.cvtColor((gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            if instance_mask is not None:
                boundaries = np.zeros(gray.shape, dtype=bool)
                boundaries[:-1, :] |= instance_mask[:-1, :] != instance_mask[1:, :]
                boundaries[:, :-1] |= instance_mask[:, :-1] != instance_mask[:, 1:]
                overlay[boundaries] = (0, 255, 255)
            for index, (x, y) in enumerate(points, start=1):
                col, row = int(round(x)), int(round(y))
                if 0 <= col < overlay.shape[1] and 0 <= row < overlay.shape[0]:
                    cv2.circle(overlay, (col, row), 3, (0, 0, 255), -1)
                    cv2.putText(
                        overlay, str(index), (col + 4, max(10, row - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA
                    )
            if not cv2.imwrite(str(staging / "overlay.png"), overlay):
                raise OSError(f"Could not save overlay to {out_dir}")
        for temporary in staging.iterdir():
            temporary.replace(out_dir / temporary.name)
    return str(out_dir)


def main():
    parser = argparse.ArgumentParser(description="Predict astrocyte centers and masks")
    parser.add_argument("--image", required=True, help="PNG/TIF/TIFF image or QPTIFF slide")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt/last.pt")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out", default="results/predictions")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--channel", type=int, default=None)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.channel is not None:
        config.setdefault("inference", {})["channel"] = args.channel

    # IMPLEMENTED [Liu]: one CLI handles small images and lazy whole-slide tiles.
    if Path(args.image).suffix.lower() == ".qptiff":
        result = predict_whole_slide(
            args.image, args.checkpoint, config, device=args.device, output_dir=args.out
        )
        save_predictions(result["points"], None, args.out, metadata=result["metadata"])
        (Path(args.out) / "tiles.json").write_text(
            json.dumps(_json_value(result["tiles"]), indent=2), encoding="utf-8"
        )
    else:
        result = predict_image(args.image, args.checkpoint, config, device=args.device)
        save_predictions(
            result["points"],
            load_image(args.image),
            args.out,
            semantic_mask=result["semantic_mask"],
            instance_mask=result["instance_mask"],
            metadata=result["metadata"],
        )
    print(f"Detected {result['count']} cells. Results saved to: {args.out}")


if __name__ == "__main__":
    main()

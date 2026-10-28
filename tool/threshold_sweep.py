"""Threshold sweep for the fine-tuned U-Net center head.

Runs the model forward once per ROI, then extracts centers from the SAME heatmap
at multiple thresholds and reports micro-averaged detection metrics. This does
not retrain anything and does not touch Cellpose.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from postprocessing import extract_centers
from predict import load_trained_model, predict_heatmap
from preprocessing import load_image, prepare_image
from real_pipeline import (
    _annotation_index,
    _image_index,
    evaluate_points,
    micro_metrics,
    read_annotations,
)
from train import get_device, load_config


def frange(lo, hi, step):
    values = []
    n = int(round((hi - lo) / step))
    for i in range(n + 1):
        values.append(round(lo + i * step, 4))
    return values


def main():
    parser = argparse.ArgumentParser(description="Sweep U-Net center threshold")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/run_20261029_033330/best.pt")
    parser.add_argument("--lo", type=float, default=0.10)
    parser.add_argument("--hi", type=float, default=0.20)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--match-distance", type=float, default=8.0)
    parser.add_argument("--image-size", type=int, nargs=2, default=None,
                        help="override data.image_size, e.g. 512 512")
    parser.add_argument("--min-distance", type=int, default=None,
                        help="override inference.min_distance_pixels")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    if args.image_size is not None:
        config["data"]["image_size"] = list(args.image_size)
    if args.min_distance is not None:
        config.setdefault("inference", {})["min_distance_pixels"] = args.min_distance
    device = get_device("auto")
    print(f"device: {device}")

    data_root = Path(config["data"]["root"])
    images, _ = _image_index(data_root / "images")
    annotations, _ = _annotation_index(data_root / "annotations")
    ids = sorted(set(images) & set(annotations))

    model = load_trained_model(args.checkpoint, config, device)
    inference = config.get("inference", {})
    channel = inference.get("channel", config.get("data", {}).get("channel"))
    min_distance = int(inference.get("min_distance_pixels", 5))
    requested_size = config.get("data", {}).get("image_size")

    # Build heatmaps once.
    heatmaps = []
    for image_id in ids:
        image = load_image(images[image_id])
        tensor = prepare_image(image, channel=channel)
        oh, ow = image.shape[:2]
        if requested_size:
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=tuple(map(int, requested_size)),
                mode="bilinear", align_corners=False,
            )[0]
        probs = predict_heatmap(model, tensor, device)
        hm = probs["center_prob"]
        labels = read_annotations(annotations[image_id])
        scale_x, scale_y = ow / hm.shape[1], oh / hm.shape[0]
        heatmaps.append((image_id, hm, labels, scale_x, scale_y, oh, ow))

    print(f"\n{'thr':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    best = None
    for thr in frange(args.lo, args.hi, args.step):
        rows = []
        for image_id, hm, labels, sx, sy, oh, ow in heatmaps:
            pts = extract_centers(hm, threshold=thr, min_distance=min_distance)
            if len(pts):
                pts = pts * np.asarray([sx, sy])
                pts = np.minimum(pts, [ow - 1e-6, oh - 1e-6])
            rows.append(evaluate_points(pts, labels, args.match_distance, image_id, "unet"))
        m = micro_metrics(rows)
        print(f"{thr:>6.2f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}")
        if best is None or m["f1"] > best[1]["f1"]:
            best = (thr, m)

    print(f"\nBest F1 = {best[1]['f1']:.4f} at threshold {best[0]:.2f}")
    print("(Cellpose reference F1 = 0.1174)")


if __name__ == "__main__":
    main()

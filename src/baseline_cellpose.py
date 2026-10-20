"""Cellpose baseline: segmentation -> centroid coordinates -> CSV/overlay."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import regionprops

try:
    from .preprocessing import load_image, prepare_image
except ImportError:
    from preprocessing import load_image, prepare_image


def run_cellpose(image: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """Run Cellpose-SAM (`cpsam`) on a single 2D grayscale image."""
    from cellpose import models

    model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
    gray = prepare_image(image).squeeze(0).numpy()
    masks, flows, styles = model.eval(gray)
    return np.asarray(masks)


def mask_centers(masks: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (instance_id, x, y) for each Cellpose mask."""
    centers = []
    for region in regionprops(masks.astype(np.int32)):
        cy, cx = region.centroid
        centers.append((int(region.label), int(round(cx)), int(round(cy))))
    return centers


def save_results(image: np.ndarray, centers: list[tuple[int, int, int]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "cellpose_centers.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "x", "y"])
        writer.writerows(centers)

    gray = prepare_image(image).squeeze(0).numpy()
    canvas = cv2.cvtColor((gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for _, x, y in centers:
        cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)
    cv2.imwrite(str(out_dir / "cellpose_overlay.png"), canvas)
    print(f"Detected {len(centers)} instances. Saved results in: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Cellpose baseline on a small image")
    parser.add_argument("--image", required=True, help="Path to PNG/TIF/TIFF image")
    parser.add_argument("--out", default="results/cellpose", help="Output directory")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA if Cellpose supports it")
    args = parser.parse_args()

    image = load_image(args.image)
    masks = run_cellpose(image, use_gpu=args.gpu)
    centers = mask_centers(masks)
    save_results(image, centers, Path(args.out))


if __name__ == "__main__":
    main()

"""Cellpose baseline: instance masks, centroid coordinates and CSV/overlay.
REDESIGN: compare instance masks and visible processes with the U-Net on the
same expert-labeled test slides; Cellpose IDs do not imply astrocyte identity.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import regionprops
import tifffile

# FIX: Use an explicit package/script import branch; avoid swallowing errors inside preprocessing.
if __package__:
    from .preprocessing import load_image, prepare_image
else:
    from preprocessing import load_image, prepare_image


def run_cellpose(image: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """Run Cellpose-SAM (`cpsam`) on a single 2D grayscale image."""
    from cellpose import models

    model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
    gray = prepare_image(image).squeeze(0).numpy()
    # FIX: API outputs vary across Cellpose versions; only the first result is needed.
    result = model.eval(gray)
    masks = result[0] if isinstance(result, (tuple, list)) else result
    masks = np.asarray(masks)
    if masks.ndim != 2 or masks.shape != gray.shape:
        raise ValueError(f"Unexpected Cellpose mask shape {masks.shape}; expected {gray.shape}")
    # REDESIGN: Retain the original instance labels for mask-level evaluation.
    return masks


def mask_centers(masks: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (instance_id, x, y) for each Cellpose mask."""
    # FIX: Empty label maps with valid H,W are allowed; malformed inputs are not.
    if masks is None:
        raise ValueError("masks must not be None")
    masks = np.asarray(masks)
    if masks.ndim != 2 or masks.size == 0:
        raise ValueError(f"Expected non-empty 2D label map, got {masks.shape}")
    if not np.issubdtype(masks.dtype, np.number) or not np.all(np.isfinite(masks)):
        raise ValueError("masks must contain finite numeric labels")
    if np.any(masks < 0) or np.any(masks != np.floor(masks)):
        raise ValueError("masks must contain non-negative integer labels")
    centers = []
    for region in regionprops(masks.astype(np.int32)):
        cy, cx = region.centroid
        centers.append((int(region.label), int(round(cx)), int(round(cy))))
    return centers


def save_results(image: np.ndarray, centers: list[tuple[int, int, int]], out_dir: Path) -> None:
    # FIX: Validate image/coordinates before writing any partial output files.
    gray = prepare_image(image).squeeze(0).numpy()
    if centers is None:
        raise ValueError("centers must not be None")
    for row in centers:
        if len(row) != 3 or not all(isinstance(v, (int, np.integer)) for v in row):
            raise ValueError("Each center must be (integer id, integer x, integer y)")
        _, x, y = row
        if not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]):
            raise ValueError(f"Center outside image bounds: ({x}, {y})")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "cellpose_centers.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "x", "y"])
        writer.writerows(centers)

    canvas = cv2.cvtColor((gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for _, x, y in centers:
        cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)
    # FIX: OpenCV returns False on some write failures without raising.
    if not cv2.imwrite(str(out_dir / "cellpose_overlay.png"), canvas):
        raise OSError(f"Could not save overlay to {out_dir}")
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
    # IMPLEMENTED [Liu]: save full-resolution instance label mask losslessly;
    # compare Cellpose morphology/instance metrics to U-Net on identical images.
    if masks.max() > np.iinfo(np.uint32).max:
        raise ValueError("Instance ID exceeds uint32 range.")

    mask_path = Path(args.out) / "cellpose_instance_mask.tif"
    tifffile.imwrite(str(mask_path), masks.astype(np.uint32))
    print(f"Instance mask saved: {mask_path}")

if __name__ == "__main__":
    main()

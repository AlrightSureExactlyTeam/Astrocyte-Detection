"""Check BBBC042 image/annotation inventory and optional coordinate validity."""

import argparse
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset import read_point_annotations

DATA_ROOT = ROOT / "datasets" / "BBBC042"
IMAGE_DIR = DATA_ROOT / "images"
POSITION_DIR = DATA_ROOT / "positions"
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png"}


def collect_files(folder: Path, extensions: set[str]) -> list[Path]:
    """Collect matching files recursively, sorted for reproducibility."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def summarize_dataset() -> dict:
    """Report file counts and filename mismatches without parsing all images."""
    images = collect_files(IMAGE_DIR, IMAGE_EXTENSIONS)
    positions = collect_files(POSITION_DIR, {".txt"})
    ignored = sorted(
        str(p.relative_to(IMAGE_DIR))
        for p in IMAGE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() not in IMAGE_EXTENSIONS
    )
    image_ids = Counter(p.stem for p in images)
    position_ids = Counter(p.stem for p in positions)
    return {
        "image_count": len(images),
        "position_count": len(positions),
        "duplicate_image_stems": sorted(k for k, v in image_ids.items() if v > 1),
        "duplicate_position_stems": sorted(k for k, v in position_ids.items() if v > 1),
        "images_without_matching_stem": sorted(image_ids.keys() - position_ids.keys()),
        "positions_without_matching_stem": sorted(position_ids.keys() - image_ids.keys()),
        "unsupported_files_ignored": ignored,
    }


def _image_size(path: Path) -> tuple[int, int]:
    """Return ``(width,height)`` from image metadata when available."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(str(path)) as tif:
            if not tif.series:
                raise ValueError(f"No image series found: {path}")
            series = tif.series[0]
            axes = series.axes
            if "Y" not in axes or "X" not in axes:
                raise ValueError(f"Cannot determine YX dimensions for {path}: {axes}")
            return int(series.shape[axes.index("X")]), int(series.shape[axes.index("Y")])
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise ValueError(f"Cannot read image: {path}")
    return int(image.shape[1]), int(image.shape[0])


def validate_coordinates(image_path: Path, annotation_path: Path) -> None:
    """Require finite centers inside ``0 <= x < width`` and ``0 <= y < height``."""
    # IMPLEMENTED [Liu]: parse confirmed BBBC042 boxes and check their centers.
    width, height = _image_size(Path(image_path))
    points = read_point_annotations(annotation_path)
    invalid = (
        (points[:, 0] < 0) | (points[:, 0] >= width)
        | (points[:, 1] < 0) | (points[:, 1] >= height)
    )
    if np.any(invalid):
        examples = points[invalid][:10].tolist()
        raise ValueError(
            f"Out-of-bounds annotations in {annotation_path} for image "
            f"{image_path} ({width}x{height}): {examples}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-coordinates", action="store_true",
        help="Also parse every matched annotation and check image bounds",
    )
    args = parser.parse_args()
    summary = summarize_dataset()
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} (first 10: {value[:10]})")
        else:
            print(f"{key}: {value}")

    if args.validate_coordinates:
        if any(summary[key] for key in (
            "duplicate_image_stems", "duplicate_position_stems",
            "images_without_matching_stem", "positions_without_matching_stem",
        )):
            raise SystemExit("Resolve filename mismatches before coordinate validation")
        errors = []
        for annotation_path in collect_files(POSITION_DIR, {".txt"}):
            image_path = IMAGE_DIR / f"{annotation_path.stem}.tif"
            if not image_path.is_file():
                matches = [p for p in collect_files(IMAGE_DIR, IMAGE_EXTENSIONS) if p.stem == annotation_path.stem]
                image_path = matches[0]
            try:
                validate_coordinates(image_path, annotation_path)
            except (ValueError, OSError) as exc:
                errors.append(str(exc))
        if errors:
            for error in errors[:20]:
                print(f"ERROR: {error}")
            raise SystemExit(f"Coordinate validation failed for {len(errors)} files")
        print(f"Coordinate validation passed for {summary['position_count']} files")


if __name__ == "__main__":
    main()

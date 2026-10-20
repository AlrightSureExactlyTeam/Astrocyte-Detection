"""Check BBBC042 image/annotation inventory before training.

Run from project root: python tools/check_dataset.py

Owner: Zhang (data), with Zhu (annotation/image QC).
This script deliberately does not guess the BBBC042 position-file format.
"""
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "BBBC042"
IMAGE_DIR = DATA_ROOT / "images"
POSITION_DIR = DATA_ROOT / "positions"
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def collect_files(folder: Path, extensions: set[str]) -> list[Path]:
    """Collect matching files recursively, sorted for reproducibility."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def summarize_dataset() -> dict:
    """Report file counts and possible filename mismatches (not coordinate validity)."""
    images = collect_files(IMAGE_DIR, IMAGE_EXTENSIONS)
    positions = collect_files(POSITION_DIR, {".txt"})
    image_ids = Counter(p.stem for p in images)
    position_ids = Counter(p.stem for p in positions)
    return {
        "image_count": len(images),
        "position_count": len(positions),
        "duplicate_image_stems": sorted(k for k, v in image_ids.items() if v > 1),
        "duplicate_position_stems": sorted(k for k, v in position_ids.items() if v > 1),
        "images_without_matching_stem": sorted(image_ids.keys() - position_ids.keys()),
        "positions_without_matching_stem": sorted(position_ids.keys() - image_ids.keys()),
    }


def validate_coordinates(image_path: Path, annotation_path: Path) -> None:
    """Validate point coordinates against image width and height."""
    # TODO [Zhang]: inspect real positions/*.txt format before implementing parsing.
    # TODO [Zhang]: read image dimensions without loading unnecessarily large images.
    # TODO [Zhang]: verify finite (x, y) values and 0 <= x < W, 0 <= y < H.
    # TODO [Zhu]: review out-of-bounds, empty or ambiguous annotations manually.
    raise NotImplementedError("Coordinate validation requires confirmed annotation format")


def main() -> None:
    summary = summarize_dataset()
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} (first 10: {value[:10]})")
        else:
            print(f"{key}: {value}")
    print("NOTE: matching filename stems does not prove matching image IDs.")
    print("TODO: confirm BBBC042 image/annotation naming and coordinate convention.")


if __name__ == "__main__":
    main()

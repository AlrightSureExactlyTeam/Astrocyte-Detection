
import sys
import csv
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

# Allow importing Python files from src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import baseline_cellpose as baseline


class TestBaselineCellpose(unittest.TestCase):

    def setUp(self):
        # Artificial grayscale image
        self.image = np.zeros(
            (64, 64),
            dtype=np.uint8
        )

        # Artificial instance mask
        # 0 = background
        # 1 = first cell
        # 2 = second cell
        self.masks = np.zeros(
            (64, 64),
            dtype=np.int32
        )

        self.masks[10:20, 10:20] = 1
        self.masks[35:45, 35:45] = 2

    def test_mask_centers(self):
        centers = baseline.mask_centers(self.masks)

        self.assertEqual(len(centers), 2)

        ids = [item[0] for item in centers]

        self.assertIn(1, ids)
        self.assertIn(2, ids)

    def test_empty_mask(self):
        empty_mask = np.zeros(
            (64, 64),
            dtype=np.int32
        )

        centers = baseline.mask_centers(empty_mask)

        self.assertEqual(len(centers), 0)

    def test_invalid_mask(self):
        invalid_mask = np.array([
            [-1, 0],
            [0, 1]
        ])

        with self.assertRaises(ValueError):
            baseline.mask_centers(invalid_mask)

    def test_save_results(self):
        centers = baseline.mask_centers(self.masks)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            baseline.save_results(
                self.image,
                centers,
                output_dir
            )

            csv_path = output_dir / "cellpose_centers.csv"
            png_path = output_dir / "cellpose_overlay.png"

            self.assertTrue(csv_path.exists())
            self.assertTrue(png_path.exists())

            with csv_path.open(
                "r",
                newline="",
                encoding="utf-8"
            ) as f:
                rows = list(csv.reader(f))

            # Header + 2 detected cells
            self.assertEqual(len(rows), 3)

    def test_instance_mask_tiff(self):
        # Simulate Cellpose prediction.
        # The real model is not loaded.

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            with patch.object(
                sys,
                "argv",
                [
                    "baseline_cellpose.py",
                    "--image",
                    "dummy.png",
                    "--out",
                    str(output_dir)
                ]
            ):
                with patch.object(
                    baseline,
                    "load_image",
                    return_value=self.image
                ):
                    with patch.object(
                        baseline,
                        "run_cellpose",
                        return_value=self.masks
                    ):
                        baseline.main()

            mask_path = (
                output_dir / "cellpose_instance_mask.tif"
            )

            self.assertTrue(mask_path.exists())

            saved_mask = tifffile.imread(
                str(mask_path)
            )

            np.testing.assert_array_equal(
                saved_mask,
                self.masks
            )

            self.assertEqual(
                saved_mask.dtype,
                np.uint32
            )


if __name__ == "__main__":
    unittest.main()

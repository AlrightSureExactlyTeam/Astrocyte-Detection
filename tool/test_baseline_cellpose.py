import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import tifffile

# Allow importing Python files from src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import baseline_cellpose as baseline


class TestBaselineCellpose(unittest.TestCase):
    """Unit tests for the current baseline; no model weights are downloaded."""

    def setUp(self):
        self.image = np.zeros((64, 64), dtype=np.uint8)
        self.masks = np.zeros((64, 64), dtype=np.int32)
        self.masks[10:20, 10:20] = 1
        self.masks[35:45, 35:45] = 2

    def test_run_cellpose_passes_current_parameters(self):
        with patch("cellpose.models.CellposeModel") as model_class:
            model = model_class.return_value
            model.eval.return_value = (self.masks, None, None)

            result = baseline.run_cellpose(
                self.image,
                use_gpu=True,
                channel=None,
                diameter=32.0,
                min_size=8,
                max_size_fraction=0.5,
            )

        model_class.assert_called_once_with(gpu=True, pretrained_model="cpsam")
        model.eval.assert_called_once()
        args, kwargs = model.eval.call_args
        self.assertEqual(args[0].shape, self.image.shape)
        self.assertEqual(args[0].dtype, np.float32)
        self.assertEqual(
            kwargs,
            {
                "normalize": False,
                "diameter": 32.0,
                "min_size": 8,
                "max_size_fraction": 0.5,
            },
        )
        np.testing.assert_array_equal(result, self.masks)

    def test_run_cellpose_selects_requested_channel(self):
        rgb = np.zeros((16, 20, 3), dtype=np.uint8)
        rgb[..., 1] = 100

        with patch("cellpose.models.CellposeModel") as model_class:
            model = model_class.return_value
            model.eval.return_value = (
                np.zeros((16, 20), dtype=np.int32),
                None,
                None,
            )
            baseline.run_cellpose(rgb, channel=1)

        gray = model.eval.call_args.args[0]
        self.assertEqual(gray.shape, (16, 20))
        self.assertTrue(np.all(gray == 0))  # Constant selected channel normalizes to zero.

    def test_run_cellpose_rejects_invalid_model_output(self):
        cases = [
            (np.zeros((8, 8), dtype=np.int32), ValueError),
            (np.zeros((64, 64), dtype=np.float32), TypeError),
            (np.full((64, 64), -1, dtype=np.int32), ValueError),
        ]
        for bad_masks, exception in cases:
            with self.subTest(exception=exception, shape=bad_masks.shape):
                with patch("cellpose.models.CellposeModel") as model_class:
                    model_class.return_value.eval.return_value = (bad_masks, None, None)
                    with self.assertRaises(exception):
                        baseline.run_cellpose(self.image)

    def test_mask_centers_returns_exact_id_xy_rows(self):
        self.assertEqual(
            baseline.mask_centers(self.masks),
            [(1, 14, 14), (2, 40, 40)],
        )

    def test_mask_centers_accepts_empty_label_map(self):
        empty_mask = np.zeros((64, 64), dtype=np.int32)
        self.assertEqual(baseline.mask_centers(empty_mask), [])

    def test_mask_centers_rejects_malformed_labels(self):
        cases = [
            None,
            np.array([], dtype=np.int32),
            np.zeros(4, dtype=np.int32),
            np.array([[-1, 0], [0, 1]], dtype=np.int32),
            np.array([[np.nan, 0], [0, 1]], dtype=float),
            np.array([[0.5, 0], [0, 1]], dtype=float),
        ]
        for masks in cases:
            with self.subTest(masks=masks):
                with self.assertRaises(ValueError):
                    baseline.mask_centers(masks)

    def test_save_results_writes_exact_csv_and_overlay(self):
        centers = baseline.mask_centers(self.masks)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nested"
            baseline.save_results(self.image, centers, output_dir)

            with (output_dir / "cellpose_centers.csv").open(
                "r", newline="", encoding="utf-8"
            ) as f:
                self.assertEqual(
                    list(csv.reader(f)),
                    [
                        ["cell_id", "x", "y"],
                        ["1", "14", "14"],
                        ["2", "40", "40"],
                    ],
                )

            overlay = cv2.imread(
                str(output_dir / "cellpose_overlay.png"), cv2.IMREAD_COLOR
            )
            self.assertIsNotNone(overlay)
            self.assertEqual(overlay.shape, (64, 64, 3))
            for _, x, y in centers:
                np.testing.assert_array_equal(overlay[y, x], [0, 0, 255])

    def test_save_results_rejects_invalid_centers_before_creating_output(self):
        cases = [None, [(1, 64, 0)], [(1, 0, -1)], [(1, 2.5, 3)], [(1, 2)]]
        for centers in cases:
            with self.subTest(centers=centers):
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_dir = Path(temp_dir) / "results"
                    with self.assertRaises(ValueError):
                        baseline.save_results(self.image, centers, output_dir)
                    self.assertFalse(output_dir.exists())

    def test_main_forwards_cli_parameters_and_saves_uint32_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            argv = [
                "baseline_cellpose.py",
                "--image", "dummy.png",
                "--out", str(output_dir),
                "--gpu",
                "--channel", "2",
                "--diameter", "40",
                "--min-size", "5",
                "--max-size-fraction", "0.6",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(baseline, "load_image", return_value=self.image),
                patch.object(baseline, "run_cellpose", return_value=self.masks) as run_mock,
            ):
                baseline.main()

            run_mock.assert_called_once_with(
                self.image,
                use_gpu=True,
                channel=2,
                diameter=40.0,
                min_size=5,
                max_size_fraction=0.6,
            )
            saved_mask = tifffile.imread(output_dir / "cellpose_instance_mask.tif")
            self.assertEqual(saved_mask.dtype, np.uint32)
            np.testing.assert_array_equal(saved_mask, self.masks)


if __name__ == "__main__":
    unittest.main()

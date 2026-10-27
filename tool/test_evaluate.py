
import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np

# Allow importing Python files from src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from evaluate import (
    match_points,
    detection_metrics,
    counting_metrics,
    export_report,
)


class TestEvaluate(unittest.TestCase):

    def test_match_points(self):
        # Two correct predictions and one false positive
        pred = np.array([
            [10, 10],
            [30, 30],
            [90, 90]
        ], dtype=float)

        gt = np.array([
            [11, 10],
            [31, 29],
            [50, 50]
        ], dtype=float)

        result = match_points(
            pred,
            gt,
            max_distance=3
        )

        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(len(result["matches"]), 2)

    def test_empty_points(self):
        # Both prediction and GT are empty
        pred = np.empty((0, 2))
        gt = np.empty((0, 2))

        result = match_points(pred, gt, max_distance=5)

        self.assertEqual(result["tp"], 0)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["fn"], 0)

    def test_one_to_one_matching(self):
        # Two predictions near one GT:
        # only one prediction can match
        pred = np.array([
            [10, 10],
            [11, 10]
        ], dtype=float)

        gt = np.array([[10, 10]], dtype=float)

        result = match_points(pred, gt, max_distance=3)

        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 0)

    def test_detection_metrics(self):
        result = detection_metrics(
            tp=8,
            fp=2,
            fn=2
        )

        self.assertAlmostEqual(
            result["precision"], 0.8
        )
        self.assertAlmostEqual(
            result["recall"], 0.8
        )
        self.assertAlmostEqual(
            result["f1"], 0.8
        )

    def test_counting_metrics(self):
        pred_counts = [10, 20, 30]
        gt_counts = [12, 18, 30]

        result = counting_metrics(
            pred_counts,
            gt_counts
        )

        self.assertAlmostEqual(
            result["mae"], 4 / 3
        )
        self.assertAlmostEqual(
            result["rmse"],
            np.sqrt(8 / 3)
        )

    def test_export_report(self):
        metrics = {
            "precision": 0.8,
            "recall": 0.9,
            "f1": 0.847
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.json"

            export_report(metrics, str(output_path))

            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()

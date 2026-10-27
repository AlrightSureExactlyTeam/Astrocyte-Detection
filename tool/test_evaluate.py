import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Allow importing Python files from src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from evaluate import counting_metrics, detection_metrics, export_report, match_points


class TestMatchPoints(unittest.TestCase):
    def test_matches_indices_distances_and_counts(self):
        pred = np.array([[10, 10], [30, 30], [90, 90]], dtype=float)
        gt = np.array([[11, 10], [31, 29], [50, 50]], dtype=float)

        result = match_points(pred, gt, max_distance=3)

        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 1, 1))
        self.assertEqual([match[:2] for match in result["matches"]], [(0, 0), (1, 1)])
        self.assertAlmostEqual(result["matches"][0][2], 1.0)
        self.assertAlmostEqual(result["matches"][1][2], np.sqrt(2))

    def test_empty_point_sets(self):
        empty = np.empty((0, 2))
        one = np.array([[1, 2]], dtype=float)
        cases = [
            (empty, empty, (0, 0, 0)),
            (one, empty, (0, 1, 0)),
            (empty, one, (0, 0, 1)),
        ]
        for pred, gt, expected in cases:
            with self.subTest(expected=expected):
                result = match_points(pred, gt, max_distance=5)
                self.assertEqual((result["tp"], result["fp"], result["fn"]), expected)
                self.assertEqual(result["matches"], [])

    def test_one_to_one_matching_selects_nearest_prediction(self):
        pred = np.array([[10, 10], [11, 10]], dtype=float)
        gt = np.array([[10, 10]], dtype=float)

        result = match_points(pred, gt, max_distance=3)

        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertEqual(result["matches"], [(0, 0, 0.0)])

    def test_maximizes_match_count_before_minimizing_distance(self):
        # Greedy nearest-neighbor matching would find only one pair here.
        pred = np.array([[0, 0], [1.9, 0]])
        gt = np.array([[1, 0], [3, 0]])

        result = match_points(pred, gt, max_distance=1.1)

        self.assertEqual(result["tp"], 2)
        self.assertEqual([match[:2] for match in result["matches"]], [(0, 0), (1, 1)])

    def test_minimizes_distance_among_maximum_matchings(self):
        pred = np.array([[0, 0], [10, 0]], dtype=float)
        gt = np.array([[1, 0], [9, 0]], dtype=float)

        result = match_points(pred, gt, max_distance=20)

        self.assertEqual([match[:2] for match in result["matches"]], [(0, 0), (1, 1)])
        self.assertAlmostEqual(sum(match[2] for match in result["matches"]), 2.0)

    def test_distance_threshold_is_inclusive(self):
        pred = np.array([[0, 0]], dtype=float)
        gt = np.array([[3, 4]], dtype=float)

        matched = match_points(pred, gt, max_distance=5)
        unmatched = match_points(pred, gt, max_distance=4.999)

        self.assertEqual(matched["tp"], 1)
        self.assertAlmostEqual(matched["matches"][0][2], 5.0)
        self.assertEqual((unmatched["tp"], unmatched["fp"], unmatched["fn"]), (0, 1, 1))

    def test_pixel_size_converts_threshold_but_reports_pixel_distance(self):
        pred = np.array([[0, 0]], dtype=float)
        gt = np.array([[3, 4]], dtype=float)

        result = match_points(pred, gt, max_distance=10, pixel_size_um=2)

        self.assertEqual(result["tp"], 1)  # 10 um / 2 um per pixel = 5 pixels.
        self.assertAlmostEqual(result["matches"][0][2], 5.0)

    def test_rejects_invalid_points_thresholds_and_pixel_size(self):
        valid = np.array([[0, 0]], dtype=float)
        invalid_point_cases = [
            (None, valid),
            (valid, None),
            (np.array([1, 2]), valid),
            (np.array([[np.nan, 0]]), valid),
            (np.array([[np.inf, 0]]), valid),
        ]
        for pred, gt in invalid_point_cases:
            with self.subTest(pred=pred, gt=gt):
                with self.assertRaises(ValueError):
                    match_points(pred, gt, max_distance=1)

        for threshold in (-1, np.nan, np.inf):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    match_points(valid, valid, max_distance=threshold)

        for pixel_size in (0, -1, np.nan, np.inf):
            with self.subTest(pixel_size=pixel_size):
                with self.assertRaises(ValueError):
                    match_points(valid, valid, max_distance=1, pixel_size_um=pixel_size)


class TestMetrics(unittest.TestCase):
    def test_detection_metrics(self):
        result = detection_metrics(tp=8, fp=2, fn=2)
        self.assertAlmostEqual(result["precision"], 0.8)
        self.assertAlmostEqual(result["recall"], 0.8)
        self.assertAlmostEqual(result["f1"], 0.8)

    def test_detection_metrics_empty_conventions(self):
        self.assertEqual(
            detection_metrics(tp=0, fp=0, fn=0),
            {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        )
        for counts in ((0, 1, 0), (0, 0, 1)):
            with self.subTest(counts=counts):
                self.assertEqual(
                    detection_metrics(*counts),
                    {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                )

    def test_detection_metrics_rejects_invalid_counts(self):
        for counts in ((-1, 0, 0), (1.5, 0, 0), (True, 0, 0)):
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    detection_metrics(*counts)

    def test_counting_metrics(self):
        result = counting_metrics([10, 20, 30], [12, 18, 30])
        self.assertAlmostEqual(result["mae"], 4 / 3)
        self.assertAlmostEqual(result["rmse"], np.sqrt(8 / 3))
        self.assertEqual(counting_metrics([0, 0], [0, 0]), {"mae": 0.0, "rmse": 0.0})

    def test_counting_metrics_rejects_invalid_inputs(self):
        cases = [
            ([], []),
            ([1], [1, 2]),
            ([[1]], [[1]]),
            ([-1], [0]),
            ([1.5], [1]),
            ([np.nan], [1]),
            ([np.inf], [1]),
            (["bad"], [1]),
        ]
        for pred, gt in cases:
            with self.subTest(pred=pred, gt=gt):
                with self.assertRaises(ValueError):
                    counting_metrics(pred, gt)


class TestExportReport(unittest.TestCase):
    def setUp(self):
        self.metrics = {"precision": 0.8, "recall": 0.9, "f1": 0.847}

    def test_exports_json_content_and_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "report.json"
            returned = export_report(self.metrics, str(output_path))

            self.assertEqual(returned, str(output_path))
            with output_path.open("r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), self.metrics)

    def test_exports_csv_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.csv"
            export_report(self.metrics, str(output_path))

            with output_path.open("r", newline="", encoding="utf-8") as f:
                self.assertEqual(
                    list(csv.reader(f)),
                    [
                        ["metric", "value"],
                        ["precision", "0.8"],
                        ["recall", "0.9"],
                        ["f1", "0.847"],
                    ],
                )

    def test_rejects_invalid_report_arguments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                export_report(self.metrics, str(Path(temp_dir) / "report.txt"))
            with self.assertRaises(TypeError):
                export_report([], str(Path(temp_dir) / "report.json"))


if __name__ == "__main__":
    unittest.main()

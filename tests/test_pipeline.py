import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_pipeline import (
    compare_metrics, create_unique_dir, evaluate_points, evaluate_experiment,
    generate_report, generate_summary, generate_visualizations, micro_metrics,
    predict_cellpose_image, predict_unet_image, read_annotations,
    read_predictions, validate_real_data, write_predictions,
)


class PipelineCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        (self.data / "images").mkdir(parents=True)
        (self.data / "annotations").mkdir()
        self.image = np.zeros((40, 60), dtype=np.uint8)
        self.image[10, 20] = 255
        tifffile.imwrite(self.data / "images" / "roi_01.tif", self.image)
        self.write_annotation("roi_01", [(1, 20, 10, "confirmed"), (2, 40, 20, "uncertain")])

    def tearDown(self):
        self.temp.cleanup()

    def write_annotation(self, image_id, rows):
        with (self.data / "annotations" / f"{image_id}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cell_id", "x", "y", "status"])
            writer.writerows(rows)

    def write_prediction(self, experiment, model, image_id, points):
        write_predictions(experiment / model / "predictions" / f"{image_id}.csv", image_id, np.asarray(points).reshape(-1, 2), model)

    def test_missing_directories_and_images_are_reported(self):
        report = validate_real_data(self.root / "absent")
        self.assertEqual(report["status"], "invalid")
        self.assertTrue(any("Missing image directory" in error for error in report["errors"]))
        (self.data / "annotations" / "roi_01.csv").unlink()
        report = validate_real_data(self.data)
        self.assertEqual(report["missing_annotations"], ["roi_01"])

    def test_missing_annotation_directory_is_reported(self):
        for path in (self.data / "annotations").iterdir():
            path.unlink()
        (self.data / "annotations").rmdir()
        report = validate_real_data(self.data)
        self.assertTrue(any("Missing annotation directory" in error for error in report["errors"]))

    def test_empty_image_is_invalid(self):
        with patch("real_pipeline.load_image", return_value=np.empty((0, 0))):
            report = validate_real_data(self.data)
        self.assertEqual(report["records"][0]["status"], "invalid")

    def test_empty_annotations_and_predictions_are_valid(self):
        self.write_annotation("roi_01", [])
        labels = read_annotations(self.data / "annotations" / "roi_01.csv")
        self.assertEqual(labels["confirmed"].shape, (0, 2))
        path = self.root / "empty.csv"
        write_predictions(path, "roi_01", np.empty((0, 2)), "unet")
        points, _ = read_predictions(path)
        self.assertEqual(points.shape, (0, 2))

    def test_duplicate_annotations_are_rejected(self):
        self.write_annotation("roi_01", [(1, 20, 10, "confirmed"), (1, 22, 10, "confirmed")])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_annotations(self.data / "annotations" / "roi_01.csv")

    def test_invalid_numeric_and_out_of_bounds_annotations_are_rejected(self):
        self.write_annotation("roi_01", [(1, "bad", 10, "confirmed")])
        with self.assertRaisesRegex(ValueError, "numeric"):
            read_annotations(self.data / "annotations" / "roi_01.csv")
        self.write_annotation("roi_01", [(1, 60, 10, "confirmed")])
        self.assertEqual(validate_real_data(self.data)["status"], "invalid")

    def test_perfect_predictions(self):
        result = evaluate_points(np.array([[20, 10]]), read_annotations(self.data / "annotations" / "roi_01.csv"), 2)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["f1"]), (1, 0, 0, 1.0))

    def test_empty_predictions_false_positives_and_false_negatives(self):
        labels = read_annotations(self.data / "annotations" / "roi_01.csv")
        empty = evaluate_points(np.empty((0, 2)), labels, 2)
        self.assertEqual((empty["tp"], empty["fp"], empty["fn"]), (0, 0, 1))
        false_positive = evaluate_points(np.array([[5, 5], [20, 10]]), labels, 2)
        self.assertEqual((false_positive["tp"], false_positive["fp"], false_positive["fn"]), (1, 1, 0))

    def test_uncertain_excludes_only_unmatched_prediction(self):
        labels = read_annotations(self.data / "annotations" / "roi_01.csv")
        result = evaluate_points(np.array([[20, 10], [41, 20], [50, 30]]), labels, 2)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["ignored_count"]), (1, 1, 0, 1))

    def test_uncertain_cannot_suppress_confirmed_match(self):
        self.write_annotation("roi_01", [(1, 20, 10, "confirmed"), (2, 21, 10, "uncertain")])
        labels = read_annotations(self.data / "annotations" / "roi_01.csv")
        result = evaluate_points(np.array([[20, 10]]), labels, 2)
        self.assertEqual((result["tp"], result["ignored_count"]), (1, 0))

    def test_one_to_one_matching(self):
        labels = {"confirmed": np.array([[10.0, 10.0]]), "uncertain": np.empty((0, 2))}
        result = evaluate_points(np.array([[10.0, 10.0], [11.0, 10.0]]), labels, 3)
        self.assertEqual((result["tp"], result["fp"]), (1, 1))

    def test_prediction_schema_and_duplicate_ids(self):
        path = self.root / "predictions.csv"
        write_predictions(path, "roi_01", [[20, 10], [30, 10]], "unet", [0.9, None])
        points, confidence = read_predictions(path, "unet")
        self.assertEqual(points.shape, (2, 2))
        self.assertIsNone(confidence[1])
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[1]["prediction_id"] = rows[0]["prediction_id"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_predictions(path)

    def test_unet_coordinate_conversion_and_center_only_inference(self):
        heatmap = np.zeros((20, 30), dtype=np.float32)
        heatmap[5, 10] = 0.9
        output = self.root / "prediction"
        config = {"data": {"input_channels": 1, "image_size": [20, 30]}, "model": {"output_channels": 4}, "inference": {"threshold": 0.5, "min_distance_pixels": 2}}
        with patch("real_pipeline.load_trained_model", return_value=object()), patch("real_pipeline.predict_heatmap", return_value={"center_prob": heatmap}):
            points, metadata = predict_unet_image(self.data / "images" / "roi_01.tif", "unused.pt", config, output)
        np.testing.assert_allclose(points, [[20, 10]])
        self.assertEqual(metadata["coordinate_scale_xy"], [2.0, 2.0])
        self.assertFalse(metadata["semantic_head_used_for_filtering"])

    def test_cellpose_output_compatibility(self):
        masks = np.zeros((40, 60), dtype=np.uint16)
        masks[8:13, 18:23] = 1
        config = {"baseline": {"channel": None}}
        with patch("real_pipeline.run_cellpose", return_value=masks):
            points, metadata = predict_cellpose_image(self.data / "images" / "roi_01.tif", config, self.root / "cellpose")
        self.assertEqual(points.shape, (1, 2))
        self.assertIn("General cell baseline", metadata["limitation"])
        read_predictions(self.root / "cellpose" / "predictions" / "roi_01.csv", "cellpose")

    def test_evaluation_and_micro_f1(self):
        experiment = self.root / "experiment"
        self.write_prediction(experiment, "unet", "roi_01", [[20, 10], [50, 30]])
        self.write_prediction(experiment, "cellpose", "roi_01", [[20, 10]])
        rows = evaluate_experiment(self.data, experiment, max_distance=2)
        micro = micro_metrics([row for row in rows if row["model"] == "unet"])
        self.assertAlmostEqual(micro["f1"], 2 / 3)
        comparisons = compare_metrics(rows)
        self.assertEqual(comparisons[0]["unet_tp"], 1)
        self.assertEqual(comparisons[0]["cellpose_f1"], 1.0)

    def test_single_model_comparison_is_incomplete(self):
        rows = [evaluate_points(np.array([[20, 10]]), read_annotations(self.data / "annotations" / "roi_01.csv"), 2, "roi_01", "unet")]
        comparisons = compare_metrics(rows)
        summary = generate_summary(self.root / "single", rows, comparisons)
        self.assertEqual(summary["status"], "incomplete_comparison")
        self.assertEqual(comparisons[0]["cellpose_f1"], "")

    def test_report_figures_and_metrics(self):
        rows = [evaluate_points(np.array([[20, 10]]), read_annotations(self.data / "annotations" / "roi_01.csv"), 2, "roi_01", model) for model in ("unet", "cellpose")]
        experiment = self.root / "reporting"
        for model in ("unet", "cellpose"):
            self.write_prediction(experiment, model, "roi_01", [[20, 10]])
        comparisons = compare_metrics(rows)
        summary = generate_summary(experiment, rows, comparisons)
        generate_visualizations(self.data, experiment, rows)
        generate_report(experiment, rows, comparisons, summary, {"status": "valid"})
        self.assertIn("Micro-F1", (experiment / "report.md").read_text(encoding="utf-8"))
        self.assertTrue((experiment / "figures" / "f1_comparison.png").is_file())
        self.assertTrue((experiment / "metrics" / "comparison.csv").is_file())

    def test_report_without_ground_truth_does_not_fabricate_metrics(self):
        experiment = self.root / "predictions_only"
        summary = generate_summary(experiment, [], [])
        generate_report(experiment, [], [], summary)
        self.assertEqual(summary["status"], "predictions_only")
        self.assertIn("accuracy was not calculated", (experiment / "report.md").read_text(encoding="utf-8"))

    def test_unique_directories_do_not_overwrite(self):
        first = create_unique_dir(self.root / "results", "pilot")
        second = create_unique_dir(self.root / "results", "pilot")
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_dir() and second.is_dir())

    def test_validation_cli(self):
        output = self.root / "cli"
        command = [sys.executable, str(ROOT / "tool" / "run_experiment.py"), "--stage", "validate", "--data-root", str(self.data), "--experiment-dir", str(output), "--config", str(ROOT / "configs" / "config.yaml")]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads((output / "validation" / "validation_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "valid")


if __name__ == "__main__":
    unittest.main()

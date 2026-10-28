import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import tifffile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Prediction tests use a dummy model and do not require the optional U-Net SDK.
try:
    import segmentation_models_pytorch  # noqa: F401
except ImportError:
    sys.modules["segmentation_models_pytorch"] = types.SimpleNamespace(Unet=None)

import predict


class DummyModel(torch.nn.Module):
    def forward(self, images):
        batch, _, height, width = images.shape
        semantic = torch.zeros(
            (batch, 3, height, width), dtype=images.dtype, device=images.device
        )
        semantic[:, 1] = 2.0
        center = torch.full(
            (batch, 1, height, width), -10.0, dtype=images.dtype, device=images.device
        )
        center[:, :, height // 2, width // 2] = 10.0
        return {"semantic_logits": semantic, "center_logits": center}


def test_config():
    return {
        "data": {"input_channels": 1},
        "model": {"architecture": "unet", "encoder": "resnet18", "output_channels": 4},
        "inference": {"threshold": 0.5, "min_distance_pixels": 2, "dedup_distance_pixels": 5},
    }


class TestPredict(unittest.TestCase):
    def test_predict_heatmap_uses_correct_activations(self):
        tensor = torch.zeros((1, 16, 20), dtype=torch.float32)
        result = predict.predict_heatmap(DummyModel(), tensor, "cpu")
        self.assertEqual(result["semantic_prob"].shape, (3, 16, 20))
        self.assertEqual(result["center_prob"].shape, (16, 20))
        np.testing.assert_allclose(result["semantic_prob"].sum(axis=0), 1.0, atol=1e-6)
        self.assertGreater(result["center_prob"][8, 10], 0.99)

    def test_background_semantics_do_not_remove_center(self):
        image = np.zeros((32, 40), dtype=np.uint8)
        center_prob = np.zeros((32, 40), dtype=np.float32)
        center_prob[16, 20] = 0.99
        semantic_prob = np.zeros((3, 32, 40), dtype=np.float32)
        semantic_prob[0] = 1.0
        probabilities = {"semantic_prob": semantic_prob, "center_prob": center_prob}
        with patch.object(predict, "predict_heatmap", return_value=probabilities):
            result = predict._predict_array(DummyModel(), image, test_config(), "cpu")

        np.testing.assert_array_equal(result["points"], [[20, 16]])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["raw_peak_count"], 1)
        self.assertEqual(result["assigned_points"].shape, (0, 2))
        self.assertEqual(result["instance_count"], 0)
        self.assertTrue(np.all(result["instance_mask"] == 0))

    def test_checkpoint_schema_and_strict_loading(self):
        config = test_config()
        model = Mock()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "best.pt"
            torch.save(
                {
                    "model_state_dict": {"weight": torch.tensor([1.0])},
                    "output_heads": {"semantic_classes": 3, "center_channels": 1},
                },
                path,
            )
            with patch.object(predict, "create_model", return_value=model) as create:
                loaded = predict.load_trained_model(path, config, "cpu")
            self.assertIs(loaded, model)
            create.assert_called_once_with(in_channels=1, encoder_name="resnet18")
            model.load_state_dict.assert_called_once()
            model.eval.assert_called_once()

            torch.save({"model_state_dict": {}}, path)
            with self.assertRaises(ValueError):
                predict.load_trained_model(path, config, "cpu")

    def test_single_image_prediction_and_saved_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((32, 40), dtype=np.uint8)
            image[16, 20] = 255
            image_path = root / "image.tif"
            tifffile.imwrite(image_path, image)
            result = predict.predict_image(
                image_path, "unused.pt", test_config(), device="cpu", model=DummyModel()
            )
            self.assertEqual(result["count"], 1)
            np.testing.assert_array_equal(result["points"], [[20, 16]])
            np.testing.assert_array_equal(result["assigned_points"], [[20, 16]])
            self.assertEqual(result["instance_count"], 1)

            predict.save_predictions(
                result["points"], image, root / "results",
                semantic_mask=result["semantic_mask"],
                instance_mask=result["instance_mask"], metadata=result["metadata"],
            )
            with (root / "results" / "centers.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["cell_id", "x", "y", "soma_area", "process_area", "status"])
            self.assertEqual(len(rows), 2)
            summary = json.loads((root / "results" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["count"], 1)
            self.assertTrue((root / "results" / "overlay.png").is_file())
            self.assertEqual(tifffile.imread(root / "results" / "instance_mask.tif").dtype, np.uint32)

    def test_whole_slide_deduplicates_overlap_and_saves_tiles(self):
        tile = np.zeros((16, 16), dtype=np.uint8)
        items = [
            {"tile": tile, "x": 0, "y": 0, "tile_id": "r0_c0", "level": 0,
             "series_index": 0, "channel": None, "scale_xy": (1.0, 1.0)},
            {"tile": tile, "x": 2, "y": 0, "tile_id": "r0_c1", "level": 0,
             "series_index": 0, "channel": None, "scale_xy": (1.0, 1.0)},
        ]
        prediction = {
            "points": np.array([[5.0, 5.0]]),
            "assigned_points": np.array([[5.0, 5.0]]),
            "count": 1, "raw_peak_count": 1, "instance_count": 1,
            "semantic_mask": np.ones((16, 16), dtype=np.uint8),
            "instance_mask": np.ones((16, 16), dtype=np.uint32),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(predict, "iter_qptiff_tiles", return_value=iter(items)),
                patch.object(predict, "_predict_array", return_value=prediction),
            ):
                result = predict.predict_whole_slide(
                    "slide.qptiff", "unused.pt", test_config(),
                    device="cpu", model=DummyModel(), output_dir=temp_dir,
                )
            self.assertEqual(result["count"], 1)
            np.testing.assert_array_equal(result["points"], [[5.0, 5.0]])
            self.assertEqual([r["accepted_after_dedup"] for r in result["tiles"]], [1, 0])
            self.assertEqual([r["assigned_count"] for r in result["tiles"]], [1, 1])
            self.assertEqual(len(list((Path(temp_dir) / "tiles").glob("*.tif"))), 4)

    def test_whole_slide_keeps_raw_centers_when_assignment_fails(self):
        tile = np.zeros((16, 16), dtype=np.uint8)
        items = [
            {"tile": tile, "x": 0, "y": 0, "tile_id": "r0_c0", "level": 0,
             "series_index": 0, "channel": None, "scale_xy": (1.0, 1.0)},
        ]
        prediction = {
            "points": np.array([[5.0, 5.0]]),
            "assigned_points": np.empty((0, 2)),
            "count": 1, "raw_peak_count": 1, "instance_count": 0,
            "semantic_mask": np.zeros((16, 16), dtype=np.uint8),
            "instance_mask": np.zeros((16, 16), dtype=np.uint32),
        }
        with (
            patch.object(predict, "iter_qptiff_tiles", return_value=iter(items)),
            patch.object(predict, "_predict_array", return_value=prediction),
        ):
            result = predict.predict_whole_slide(
                "slide.qptiff", "unused.pt", test_config(), device="cpu", model=DummyModel()
            )

        self.assertEqual(result["count"], 1)
        np.testing.assert_array_equal(result["points"], [[5.0, 5.0]])
        self.assertEqual(result["tiles"][0]["assigned_count"], 0)
        self.assertEqual(result["tiles"][0]["instance_count"], 0)
        self.assertEqual(result["tiles"][0]["accepted_after_dedup"], 1)

    def test_global_dedup_uses_overlap_provenance_and_deterministic_representatives(self):
        bounds = {"a": (0, 0, 20, 20), "b": (10, 0, 30, 20), "c": (15, 0, 35, 20)}
        detections = [
            {"x": 16.0, "y": 5.0, "tile_id": "c"},
            {"x": 15.0, "y": 5.0, "tile_id": "a"},
            {"x": 15.5, "y": 5.0, "tile_id": "b"},
        ]
        points, sources, accepted = predict._deduplicate_global_detections(detections, bounds, 2.0)
        np.testing.assert_array_equal(points, [[15.0, 5.0]])
        self.assertEqual(sources, [["a", "b", "c"]])
        self.assertEqual(accepted, {"a": 1, "b": 0, "c": 0})
        reversed_points, reversed_sources, _ = predict._deduplicate_global_detections(
            list(reversed(detections)), bounds, 2.0
        )
        np.testing.assert_array_equal(reversed_points, points)
        self.assertEqual(reversed_sources, sources)

        separate_bounds = {"a": (0, 0, 10, 10), "b": (12, 0, 22, 10)}
        nearby = [
            {"x": 9.9, "y": 5.0, "tile_id": "a"},
            {"x": 12.1, "y": 5.0, "tile_id": "b"},
        ]
        distinct, _, _ = predict._deduplicate_global_detections(nearby, separate_bounds, 5.0)
        self.assertEqual(len(distinct), 2)

    def test_whole_slide_deduplicates_only_exact_within_tile_points(self):
        tile = np.zeros((16, 16), dtype=np.uint8)
        items = [{"tile": tile, "x": 0, "y": 0, "tile_id": "tile", "scale_xy": (1, 1)}]
        prediction = {
            "points": np.array([[5.0, 5.0], [5.0, 5.0], [5.5, 5.0]]),
            "assigned_points": np.empty((0, 2)), "count": 3, "raw_peak_count": 3,
            "instance_count": 0, "semantic_mask": np.zeros((16, 16), dtype=np.uint8),
            "instance_mask": np.zeros((16, 16), dtype=np.uint32),
        }
        with (
            patch.object(predict, "iter_qptiff_tiles", return_value=iter(items)),
            patch.object(predict, "_predict_array", return_value=prediction),
        ):
            result = predict.predict_whole_slide(
                "slide.qptiff", "unused.pt", test_config(), device="cpu", model=DummyModel()
            )
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["point_sources"], [["tile"], ["tile"]])

    def test_invalid_prediction_outputs_do_not_create_final_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results"
            with self.assertRaises(ValueError):
                predict.save_predictions(
                    np.array([[2.0, 2.0]]), np.zeros((8, 8), dtype=np.uint8), output,
                    semantic_mask=np.zeros((4, 4), dtype=np.uint8),
                )
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                predict.save_predictions(
                    np.array([[8.0, 2.0]]), np.zeros((8, 8), dtype=np.uint8), output,
                )
            self.assertFalse(output.exists())

    def test_save_predictions_accepts_verified_empty_points(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            predict.save_predictions(np.empty((0, 2)), None, temp_dir)
            summary = json.loads((Path(temp_dir) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["count"], 0)
            with (Path(temp_dir) / "centers.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(
                rows, [["cell_id", "x", "y", "soma_area", "process_area", "status"]]
            )


if __name__ == "__main__":
    unittest.main()

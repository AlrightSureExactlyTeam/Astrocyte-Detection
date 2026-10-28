import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import AstrocyteDataset, build_dataloaders, points_to_heatmap, read_point_annotations


class TestAnnotationsAndHeatmaps(unittest.TestCase):
    def test_reads_bbb042_boxes_as_centers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "1.txt"
            path.write_text(
                "Cell 0 0 0 4 2 8 6 0 0 0 0 0 0 0\n"
                "Cell 1 0 0 10 20 14 24 0 0 0 0 0 0 0\n",
                encoding="utf-8",
            )
            np.testing.assert_array_equal(
                read_point_annotations(path), [[6.0, 4.0], [12.0, 22.0]]
            )

    def test_reads_xy_csv_and_explicit_empty_annotation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "points.csv"
            csv_path.write_text("x,y\n1.5,2.5\n", encoding="utf-8")
            empty_path = Path(temp_dir) / "empty.txt"
            empty_path.write_text("", encoding="utf-8")
            np.testing.assert_array_equal(read_point_annotations(csv_path), [[1.5, 2.5]])
            self.assertEqual(read_point_annotations(empty_path).shape, (0, 2))

    def test_rejects_malformed_annotation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.txt"
            path.write_text("Cell 0 0 0 8 2 4 6\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_point_annotations(path)

    def test_heatmap_has_gaussian_peak_and_handles_empty_points(self):
        heatmap = points_to_heatmap([[4, 3]], 8, 9, sigma=1.5)
        self.assertEqual(heatmap.shape, (8, 9))
        self.assertEqual(np.unravel_index(heatmap.argmax(), heatmap.shape), (3, 4))
        self.assertAlmostEqual(float(heatmap.max()), 1.0)
        self.assertTrue(np.array_equal(points_to_heatmap([], 8, 9), np.zeros((8, 9))))
        self.assertTrue(np.array_equal(points_to_heatmap([[-1, 100]], 8, 9), np.zeros((8, 9))))


class TestAstrocyteDataset(unittest.TestCase):
    def test_resize_transforms_image_points_and_masks_consistently(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            annotation_dir = root / "positions"
            semantic_dir = root / "semantic"
            instance_dir = root / "instances"
            for folder in (image_dir, annotation_dir, semantic_dir, instance_dir):
                folder.mkdir()
            image = np.arange(20 * 40, dtype=np.uint16).reshape(20, 40)
            tifffile.imwrite(image_dir / "sample.tif", image)
            (annotation_dir / "sample.txt").write_text(
                "Cell 0 0 0 4 2 8 6 0 0 0 0 0 0 0\n", encoding="utf-8"
            )
            semantic = np.ones((20, 40), dtype=np.uint8)
            semantic[:, 20:] = 2
            instance = np.ones((20, 40), dtype=np.uint32)
            tifffile.imwrite(semantic_dir / "sample.tif", semantic)
            tifffile.imwrite(instance_dir / "sample.tif", instance)

            dataset = AstrocyteDataset(
                image_dir, annotation_dir, image_size=(10, 20), sigma=1.0,
                semantic_mask_dir=semantic_dir, instance_mask_dir=instance_dir,
            )
            image_tensor, targets, metadata = dataset[0]

            self.assertEqual(image_tensor.shape, (1, 10, 20))
            self.assertEqual(targets["center_heatmap"].shape, (1, 10, 20))
            self.assertTrue(targets["valid_mask"].all())
            self.assertTrue(targets["semantic_valid_mask"].all())
            self.assertEqual(set(np.unique(targets["semantic_mask"].numpy())), {1, 2})
            self.assertEqual(metadata["scale_xy"], (0.5, 0.5))
            peak = np.unravel_index(
                targets["center_heatmap"].numpy().argmax(), (10, 20)
            )
            self.assertEqual(peak, (2, 3))  # Original center (6,4) -> resized (3,2).

    def test_point_only_sample_disables_semantic_supervision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images").mkdir()
            (root / "positions").mkdir()
            tifffile.imwrite(root / "images" / "1.tif", np.zeros((16, 16), dtype=np.uint8))
            (root / "positions" / "1.txt").write_text("", encoding="utf-8")
            dataset = AstrocyteDataset(root / "images", root / "positions", image_size=(16, 16))
            _, targets, metadata = dataset[0]
            self.assertFalse(targets["semantic_valid_mask"].any())
            self.assertTrue(targets["valid_mask"].all())
            self.assertTrue(np.all(targets["center_heatmap"].numpy() == 0))
            self.assertFalse(metadata["semantic_available"])
            self.assertTrue(metadata["annotation_available"])
            self.assertTrue(metadata["annotation_empty"])

    def test_rejects_out_of_bounds_source_annotations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images").mkdir()
            (root / "positions").mkdir()
            tifffile.imwrite(
                root / "images" / "sample_oob.tif", np.zeros((10, 20), dtype=np.uint8)
            )
            (root / "positions" / "sample_oob.txt").write_text("20,5\n", encoding="utf-8")
            dataset = AstrocyteDataset(root / "images", root / "positions", image_size=(5, 10))

            with self.assertRaises(ValueError) as context:
                dataset[0]

            message = str(context.exception)
            self.assertIn("sample_oob", message)
            self.assertIn("width=20", message)
            self.assertIn("height=10", message)
            self.assertIn("[20.0, 5.0]", message)

    def test_rejects_unpaired_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images").mkdir()
            (root / "positions").mkdir()
            tifffile.imwrite(root / "images" / "1.tif", np.zeros((8, 8), dtype=np.uint8))
            with self.assertRaises(ValueError):
                AstrocyteDataset(root / "images", root / "positions")

    def test_build_dataloaders_splits_by_source_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images").mkdir()
            (root / "positions").mkdir()
            for index in range(8):
                tifffile.imwrite(
                    root / "images" / f"{index}.tif", np.full((16, 16), index, dtype=np.uint8)
                )
                (root / "positions" / f"{index}.txt").write_text("", encoding="utf-8")
            config = {
                "project": {"seed": 42},
                "data": {
                    "images_dir": str(root / "images"),
                    "annotations_dir": str(root / "positions"),
                    "input_channels": 1,
                    "image_size": [16, 16],
                    "train_fraction": 0.5,
                    "val_fraction": 0.25,
                    "test_fraction": 0.25,
                },
                "heatmap": {"sigma_pixels": 2.0},
                "training": {"batch_size": 2, "num_workers": 0},
            }
            train_loader, val_loader = build_dataloaders(config)
            train_ids = set(train_loader.dataset.indices)
            val_ids = set(val_loader.dataset.indices)
            self.assertEqual((len(train_ids), len(val_ids)), (4, 2))
            self.assertTrue(train_ids.isdisjoint(val_ids))
            images, targets, _ = next(iter(train_loader))
            self.assertEqual(images.shape, (2, 1, 16, 16))
            self.assertEqual(targets["center_heatmap"].shape, (2, 1, 16, 16))


if __name__ == "__main__":
    unittest.main()

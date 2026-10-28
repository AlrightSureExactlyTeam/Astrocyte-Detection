import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import AstrocyteDataset, build_dataloaders, points_to_heatmap, read_point_annotations


class MaskGeometryTransform:
    def __init__(self, operation):
        self.operation = operation

    def transform_with_masks(self, image, points, semantic, instance):
        height, width = image.shape[-2:]
        points = points.copy()
        if self.operation == "horizontal":
            image = torch.flip(image, dims=(2,))
            semantic, instance = np.fliplr(semantic), np.fliplr(instance)
            points[:, 0] = width - 1 - points[:, 0]
        elif self.operation == "vertical":
            image = torch.flip(image, dims=(1,))
            semantic, instance = np.flipud(semantic), np.flipud(instance)
            points[:, 1] = height - 1 - points[:, 1]
        elif self.operation == "rotation":
            image = torch.rot90(image, k=1, dims=(1, 2))
            semantic, instance = np.rot90(semantic), np.rot90(instance)
            points[:] = points[:, [1, 0]]
            points[:, 1] = width - 1 - points[:, 1]
        elif self.operation == "crop":
            image = image[:, 1:3, 1:5]
            semantic, instance = semantic[1:3, 1:5], instance[1:3, 1:5]
            points -= np.array([1, 1])
        return image, points, semantic.copy(), instance.copy()


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

    def test_mask_aware_geometric_transforms_keep_all_targets_aligned(self):
        expected = {
            "horizontal": ((4, 6), (2, 4)),
            "vertical": ((4, 6), (1, 1)),
            "rotation": ((6, 4), (4, 2)),
            "crop": ((2, 4), (1, 0)),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("images", "positions", "semantic", "instances"):
                (root / name).mkdir()
            image = np.arange(24, dtype=np.uint8).reshape(4, 6)
            semantic = np.zeros((4, 6), dtype=np.uint8)
            instance = np.zeros((4, 6), dtype=np.uint32)
            semantic[2, 1], instance[2, 1] = 1, 7
            tifffile.imwrite(root / "images" / "sample.tif", image)
            tifffile.imwrite(root / "semantic" / "sample.tif", semantic)
            tifffile.imwrite(root / "instances" / "sample.tif", instance)
            (root / "positions" / "sample.txt").write_text("1,2\n", encoding="utf-8")
            for operation, (shape, expected_peak) in expected.items():
                with self.subTest(operation=operation):
                    dataset = AstrocyteDataset(
                        root / "images", root / "positions", image_size=(4, 6), sigma=1.0,
                        transform=MaskGeometryTransform(operation),
                        semantic_mask_dir=root / "semantic", instance_mask_dir=root / "instances",
                    )
                    _, targets, metadata = dataset[0]
                    heatmap = targets["center_heatmap"].numpy()[0]
                    peak = np.unravel_index(heatmap.argmax(), heatmap.shape)
                    self.assertEqual(metadata["image_size"], shape)
                    self.assertEqual(peak, expected_peak)
                    self.assertEqual(targets["semantic_mask"].numpy()[peak], 1)
                    self.assertEqual(targets["instance_mask"].numpy()[peak], 7)

    def test_legacy_transform_is_rejected_when_masks_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("images", "positions", "semantic"):
                (root / name).mkdir()
            tifffile.imwrite(root / "images" / "1.tif", np.zeros((4, 6), dtype=np.uint8))
            tifffile.imwrite(root / "semantic" / "1.tif", np.zeros((4, 6), dtype=np.uint8))
            (root / "positions" / "1.txt").write_text("1,2\n", encoding="utf-8")
            dataset = AstrocyteDataset(
                root / "images", root / "positions", image_size=(4, 6),
                transform=lambda image, points: (image, points), semantic_mask_dir=root / "semantic",
            )
            with self.assertRaisesRegex(ValueError, "transform_with_masks"):
                dataset[0]

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

    def test_explicit_group_split_is_reproducible_and_leakage_free(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images").mkdir()
            (root / "positions").mkdir()
            groups = {}
            for group in range(3):
                for suffix in ("a", "b"):
                    sample_id = f"slide{group}_{suffix}"
                    groups[sample_id] = f"slide{group}"
                    tifffile.imwrite(
                        root / "images" / f"{sample_id}.tif", np.zeros((8, 8), dtype=np.uint8)
                    )
                    (root / "positions" / f"{sample_id}.txt").write_text("", encoding="utf-8")
            config = {
                "project": {"seed": 42},
                "data": {
                    "images_dir": str(root / "images"), "annotations_dir": str(root / "positions"),
                    "input_channels": 1, "image_size": [8, 8], "groups": groups,
                    "train_fraction": 0.34, "val_fraction": 0.33, "test_fraction": 0.33,
                },
                "training": {"batch_size": 1, "num_workers": 0},
            }
            with self.assertWarns(UserWarning):
                train_loader, val_loader = build_dataloaders(config)
            train_indices = list(train_loader.dataset.indices)
            val_indices = list(val_loader.dataset.indices)
            train_groups = {groups[train_loader.dataset.dataset.samples[i]["sample_id"]] for i in train_indices}
            val_groups = {groups[val_loader.dataset.dataset.samples[i]["sample_id"]] for i in val_indices}
            self.assertTrue(train_groups.isdisjoint(val_groups))
            self.assertEqual(train_loader.split_summary["source_group_count"], 3)
            with self.assertWarns(UserWarning):
                repeated_train, _ = build_dataloaders(config)
            self.assertEqual(list(repeated_train.dataset.indices), train_indices)
            config["data"]["groups"] = {sample_id: "one_slide" for sample_id in groups}
            with self.assertRaises(ValueError):
                build_dataloaders(config)


if __name__ == "__main__":
    unittest.main()

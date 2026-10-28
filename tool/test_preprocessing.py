import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import _select_plane_index, iter_qptiff_tiles, load_image, prepare_image


class TestImageLoading(unittest.TestCase):
    def test_grayscale_and_rgb_png(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cv2.imwrite(str(root / "gray.png"), np.full((4, 5), 127, dtype=np.uint8))
            rgb = np.zeros((2, 3, 3), dtype=np.uint8)
            rgb[0, 0] = (30, 20, 10)
            bgr = rgb[..., ::-1]
            cv2.imwrite(str(root / "rgb.png"), bgr)
            self.assertEqual(load_image(root / "gray.png").shape, (4, 5))
            np.testing.assert_array_equal(load_image(root / "rgb.png"), rgb)

    def test_explicit_planar_channels_and_ambiguous_tiff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            planar = np.arange(2 * 4 * 5, dtype=np.uint8).reshape(2, 4, 5)
            tifffile.imwrite(root / "channels.tif", planar, metadata={"axes": "CYX"})
            loaded = load_image(root / "channels.tif")
            self.assertEqual(loaded.shape, (4, 5, 2))
            np.testing.assert_array_equal(loaded, np.moveaxis(planar, 0, -1))

            tifffile.imwrite(root / "ambiguous.tif", np.zeros((3, 4, 5), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "Ambiguous TIFF axes"):
                load_image(root / "ambiguous.tif")
            tifffile.imwrite(
                root / "stack.tif", np.zeros((3, 4, 5), dtype=np.uint8),
                metadata={"axes": "ZYX"},
            )
            with self.assertRaisesRegex(ValueError, "stack axis"):
                load_image(root / "stack.tif")

    def test_corrupted_tiff_is_not_an_empty_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.tif"
            path.write_text("not a tiff", encoding="utf-8")
            with self.assertRaises((ValueError, OSError, tifffile.TiffFileError)):
                load_image(path)


class TestPreparationAndTiling(unittest.TestCase):
    def test_empty_constant_and_single_pixel_images(self):
        for image in (None, np.empty((0, 2)), np.empty((2, 0))):
            with self.subTest(image=image):
                with self.assertRaises(ValueError):
                    prepare_image(image)
        constant = prepare_image(np.full((3, 4), 7, dtype=np.uint8))
        self.assertTrue(torch_is_finite(constant))
        self.assertTrue(np.all(constant.numpy() == 0))
        single = prepare_image(np.array([[255]], dtype=np.uint8))
        self.assertEqual(tuple(single.shape), (1, 1, 1))

    def test_stack_axes_are_not_channels(self):
        with self.assertRaises(ValueError):
            _select_plane_index("ZYX", (3, 10, 10), channel=0)
        with self.assertRaises(ValueError):
            _select_plane_index("TYX", (3, 10, 10), channel=0)

    def test_small_tiled_image_preserves_coordinates_and_edge_shapes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "slide.qptiff"
            tifffile.imwrite(
                path, np.zeros((40, 48), dtype=np.uint8), tile=(16, 16),
                photometric="minisblack",
            )
            items = list(iter_qptiff_tiles(path, tile_size=16, overlap=0, with_meta=True))
            self.assertEqual(len(items), 9)
            self.assertEqual(items[-1]["tile"].shape, (8, 16))
            self.assertEqual((items[-1]["x"], items[-1]["y"]), (32, 32))
            self.assertTrue(all(item["tile"].ndim == 2 for item in items))


def torch_is_finite(tensor):
    import torch
    return bool(torch.isfinite(tensor).all())


if __name__ == "__main__":
    unittest.main()

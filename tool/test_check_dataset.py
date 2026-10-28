import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from tool.check_dataset import validate_coordinates


class TestCoordinateValidation(unittest.TestCase):
    def test_accepts_in_bounds_centers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "1.tif"
            annotation = root / "1.txt"
            tifffile.imwrite(image, np.zeros((20, 30), dtype=np.uint8))
            annotation.write_text("Cell 0 0 0 4 2 8 6 0 0 0 0 0 0 0\n", encoding="utf-8")
            validate_coordinates(image, annotation)

    def test_rejects_out_of_bounds_centers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "1.tif"
            annotation = root / "1.txt"
            tifffile.imwrite(image, np.zeros((20, 30), dtype=np.uint8))
            annotation.write_text("Cell 0 0 0 50 2 60 6 0 0 0 0 0 0 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_coordinates(image, annotation)


if __name__ == "__main__":
    unittest.main()

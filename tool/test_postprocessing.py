import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from postprocessing import (
    centers_to_global_coordinates,
    decode_semantic_mask,
    extract_centers,
    separate_instances,
    to_probability,
)


class TestCenterPostprocessing(unittest.TestCase):
    def test_probability_conversion(self):
        probability = to_probability(np.zeros((1, 4, 5), dtype=np.float32))
        self.assertEqual(probability.shape, (4, 5))
        self.assertTrue(np.all(probability == 0.5))

    def test_extracts_xy_centers_and_empty_result(self):
        probability = np.zeros((20, 30), dtype=np.float32)
        probability[5, 10] = 0.9
        probability[15, 25] = 0.8
        centers = extract_centers(probability, threshold=0.5, min_distance=3)
        np.testing.assert_array_equal(centers, [[10, 5], [25, 15]])
        self.assertEqual(extract_centers(probability, threshold=0.95).shape, (0, 2))

    def test_extract_centers_rejects_invalid_input(self):
        for probability in (
            None, np.zeros(3), np.empty((0, 3)), np.empty((3, 0)),
            np.full((3, 3), np.nan),
        ):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    extract_centers(probability)
        with self.assertRaises(ValueError):
            extract_centers(np.zeros((3, 3)), min_distance=0)

    def test_global_coordinate_conversion(self):
        points = centers_to_global_coordinates([[1, 2]], (10, 20), (2, 3))
        np.testing.assert_array_equal(points, [[22, 66]])


class TestSemanticAndInstances(unittest.TestCase):
    def test_decodes_semantic_labels_and_uncertainty(self):
        logits = np.zeros((3, 2, 2), dtype=np.float32)
        logits[1, 0, 0] = 5
        logits[2, 0, 1] = 5
        logits[0, 1, :] = 5
        mask = decode_semantic_mask(logits)
        np.testing.assert_array_equal(mask, [[1, 2], [0, 0]])
        uncertain = decode_semantic_mask(np.zeros((3, 2, 2)), confidence_threshold=0.5)
        self.assertTrue(np.all(uncertain == 0))

    def test_separates_two_foreground_instances(self):
        semantic = np.zeros((20, 30), dtype=np.uint8)
        yy, xx = np.mgrid[:20, :30]
        semantic[(yy - 10) ** 2 + (xx - 7) ** 2 <= 9] = 1
        semantic[(yy - 10) ** 2 + (xx - 22) ** 2 <= 9] = 2
        instances, assigned = separate_instances(semantic, [[7, 10], [22, 10]])
        self.assertEqual(len(np.unique(instances)), 3)
        self.assertEqual(assigned.shape, (2, 2))
        self.assertNotEqual(instances[10, 7], instances[10, 22])

    def test_unassigned_centers_do_not_create_instances(self):
        semantic = np.zeros((10, 10), dtype=np.uint8)
        instances, assigned = separate_instances(semantic, [[2, 2], [100, 100]])
        self.assertEqual(assigned.shape, (0, 2))
        self.assertTrue(np.all(instances == 0))
        empty_instances, empty_assigned = separate_instances(semantic, np.empty((0, 2)))
        self.assertEqual(empty_assigned.shape, (0, 2))
        self.assertTrue(np.all(empty_instances == 0))


if __name__ == "__main__":
    unittest.main()

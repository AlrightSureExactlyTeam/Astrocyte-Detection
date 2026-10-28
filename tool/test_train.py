import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The loss test does not instantiate a U-Net; avoid requiring its optional SDK.
try:
    import segmentation_models_pytorch  # noqa: F401
except ImportError:
    sys.modules["segmentation_models_pytorch"] = types.SimpleNamespace(Unet=None)

from train import CompositeLoss


class TestCompositeLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.outputs = {
            "semantic_logits": torch.randn(1, 3, 4, 5, requires_grad=True),
            "center_logits": torch.randn(1, 1, 4, 5, requires_grad=True),
        }
        self.targets = {
            "semantic_mask": torch.zeros(1, 4, 5, dtype=torch.long),
            "center_heatmap": torch.zeros(1, 1, 4, 5),
            "valid_mask": torch.ones(1, 4, 5, dtype=torch.bool),
        }

    def test_point_only_targets_exclude_semantic_loss(self):
        self.targets["semantic_valid_mask"] = torch.zeros(1, 4, 5, dtype=torch.bool)
        losses = CompositeLoss()(self.outputs, self.targets)
        self.assertEqual(float(losses["semantic_loss"].detach()), 0.0)
        self.assertGreater(float(losses["center_loss"].detach()), 0.0)
        self.assertAlmostEqual(
            float(losses["loss"].detach()), float(losses["center_loss"].detach())
        )
        losses["loss"].backward()
        semantic_grad = self.outputs["semantic_logits"].grad
        self.assertTrue(torch.all(semantic_grad == 0))
        self.assertTrue(torch.isfinite(self.outputs["center_logits"].grad).all())

    def test_fully_ignored_batch_has_finite_zero_loss(self):
        self.targets["valid_mask"][:] = False
        self.targets["semantic_valid_mask"] = torch.zeros(1, 4, 5, dtype=torch.bool)
        losses = CompositeLoss()(self.outputs, self.targets)
        for value in losses.values():
            self.assertTrue(np.isfinite(float(value.detach())))
            self.assertEqual(float(value.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()

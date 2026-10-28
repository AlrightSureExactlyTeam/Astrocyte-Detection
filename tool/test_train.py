import sys
import types
import unittest
from unittest.mock import patch
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

from train import CompositeLoss, build_criterion, get_device, train, validate_config


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

    def test_shared_valid_mask_excludes_semantic_pixels(self):
        self.targets["valid_mask"][:] = False
        self.targets["semantic_valid_mask"] = torch.ones(1, 4, 5, dtype=torch.bool)
        losses = CompositeLoss()(self.outputs, self.targets)
        self.assertEqual(float(losses["semantic_loss"].detach()), 0.0)
        self.assertEqual(float(losses["loss"].detach()), 0.0)

    def test_invalid_supervision_is_rejected(self):
        self.targets["semantic_mask"][0, 0, 0] = 3
        with self.assertRaises(ValueError):
            CompositeLoss()(self.outputs, self.targets)
        self.targets["semantic_mask"].zero_()
        self.targets["center_heatmap"][0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            CompositeLoss()(self.outputs, self.targets)

    def test_criterion_uses_configured_loss_weights(self):
        criterion = build_criterion({
            "training": {"loss": "bce_with_logits", "semantic_weight": 0.25, "center_weight": 2.0}
        })
        self.assertEqual(criterion.semantic_weight, 0.25)
        self.assertEqual(criterion.center_weight, 2.0)


class TestTrainingConfig(unittest.TestCase):
    def valid_config(self):
        return {
            "project": {"seed": 42},
            "data": {"input_channels": 1, "image_size": [32, 32]},
            "model": {"output_channels": 4},
            "training": {
                "epochs": 2, "batch_size": 2, "num_workers": 0,
                "early_stopping_patience": 1, "learning_rate": 1e-4,
            },
        }

    def test_rejects_fractional_and_nonfinite_configuration(self):
        cases = [
            (("training", "epochs"), 1.5), (("training", "batch_size"), 0),
            (("training", "learning_rate"), np.nan), (("data", "input_channels"), 1.5),
            (("project", "seed"), -1), (("data", "image_size"), [32, 0]),
        ]
        for (section, key), value in cases:
            with self.subTest(section=section, key=key, value=value):
                config = self.valid_config()
                config[section][key] = value
                with self.assertRaises(ValueError):
                    validate_config(config)


class TestCudaTrainingDevice(unittest.TestCase):
    def test_default_auto_and_indexed_device_resolve_to_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(get_device().type, "cuda")
            self.assertEqual(get_device("auto").type, "cuda")
            self.assertEqual(str(get_device("cuda:1")), "cuda:1")

    def test_missing_cuda_fails_instead_of_falling_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA training requested"):
                get_device()

    def test_cpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "CPU training is disabled"):
            get_device("cpu")

    def test_real_training_api_also_rejects_cpu(self):
        with self.assertRaisesRegex(ValueError, "CPU training is disabled"):
            train({}, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()

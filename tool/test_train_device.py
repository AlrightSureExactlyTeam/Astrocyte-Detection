import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import segmentation_models_pytorch  # noqa: F401
except ImportError:
    sys.modules["segmentation_models_pytorch"] = types.SimpleNamespace(Unet=None)

from train import get_device, train


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

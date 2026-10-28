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

import model


class FakeUnet(torch.nn.Module):
    def forward(self, images):
        return torch.cat((images, images, images, images), dim=1)


class TestModelInterface(unittest.TestCase):
    def test_two_head_forward_contract_without_pretrained_weights(self):
        with patch.object(model.smp, "Unet", return_value=FakeUnet()):
            network = model.create_model(in_channels=1)
        outputs = network(torch.randn(2, 1, 16, 20))
        self.assertEqual(outputs["semantic_logits"].shape, (2, 3, 16, 20))
        self.assertEqual(outputs["center_logits"].shape, (2, 1, 16, 20))

    def test_invalid_image_tensor_is_rejected(self):
        with patch.object(model.smp, "Unet", return_value=FakeUnet()):
            network = model.create_model(in_channels=1)
        for images in (None, torch.empty(0, 1, 4, 4), torch.randn(1, 2, 4, 4)):
            with self.subTest(images=images):
                with self.assertRaises((TypeError, ValueError)):
                    network(images)


if __name__ == "__main__":
    unittest.main()

"""U-Net for astrocyte visible morphology segmentation and center detection.

REDESIGN: semantic classes = background(0), soma(1), visible processes(2).
A separate center-logit channel supports later instance separation; it is NOT
itself an instance segmentation prediction.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class AstrocyteUNet(nn.Module):
    """Predict semantic logits and auxiliary center logits for each image."""

    def __init__(self, in_channels: int = 1, encoder_name: str = "resnet18"):
        super().__init__()
        # FIX: Keep public input-channel configuration for robust forward validation.
        self.in_channels = in_channels
        # REDESIGN: One shared U-Net decoder predicts 3 semantic logits + 1 center logit.
        # REDESIGN: Do not apply softmax/sigmoid here; training/inference handle each head.
        self.network = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=4,  # REDESIGN: 3 mutually exclusive semantic classes + 1 center channel.
            activation=None,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        # FIX: Validate input before the encoder fails with opaque shape/channel errors.
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor [B,C,H,W]")
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[2] == 0 or images.shape[3] == 0:
            raise ValueError(f"Expected non-empty [B,C,H,W] tensor, got {tuple(images.shape)}")
        if images.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {images.shape[1]}")
        if not torch.is_floating_point(images):
            raise TypeError("images must be a floating-point tensor")
        if not torch.isfinite(images).all().item():
            raise ValueError("images contain NaN or infinity")
        # REDESIGN: Explicit output contract consumed by train.py and predict.py.
        logits = self.network(images)
        return {
            "semantic_logits": logits[:, :3, :, :],  # [B,3,H,W]; apply softmax for inference.
            "center_logits": logits[:, 3:4, :, :],  # [B,1,H,W]; apply sigmoid for inference.
        }


def create_model(in_channels: int = 1, encoder_name: str = "resnet18") -> AstrocyteUNet:
    return AstrocyteUNet(in_channels=in_channels, encoder_name=encoder_name)


if __name__ == "__main__":
    model = create_model(in_channels=1)
    # FIX: The standalone smoke test must not update BatchNorm running statistics.
    model.eval()
    example = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        outputs = model(example)
    print("Input:", tuple(example.shape))
    # REDESIGN: Smoke-test both heads, rather than assuming a single heatmap tensor.
    print("Semantic logits:", tuple(outputs["semantic_logits"].shape))
    print("Center logits:", tuple(outputs["center_logits"].shape))
    center_probability = torch.sigmoid(outputs["center_logits"])
    print("Center probability range:", float(center_probability.min()), float(center_probability.max()))

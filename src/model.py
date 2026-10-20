"""U-Net model for astrocyte center heatmap regression."""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class AstrocyteUNet(nn.Module):
    """Predict one unnormalized heatmap (logits) per input image."""

    def __init__(self, in_channels: int = 1, encoder_name: str = "resnet18"):
        super().__init__()
        self.network = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def create_model(in_channels: int = 1, encoder_name: str = "resnet18") -> AstrocyteUNet:
    return AstrocyteUNet(in_channels=in_channels, encoder_name=encoder_name)


if __name__ == "__main__":
    model = create_model(in_channels=1)
    example = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        logits = model(example)
    print("Input:", tuple(example.shape))
    print("Output:", tuple(logits.shape))
    print("Heatmap range after sigmoid:",
          float(torch.sigmoid(logits).min()),
          float(torch.sigmoid(logits).max()))

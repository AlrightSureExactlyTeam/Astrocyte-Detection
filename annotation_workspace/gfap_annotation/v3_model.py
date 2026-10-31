"""Small shared-encoder U-Net with morphology and instance-embedding heads.

V3 is intentionally a first-stage annotated-cell model. It has no Background
logit and therefore is not a free-standing full-image instance detector.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((self.reduce(x), skip), dim=1))


class AstrocyteInstanceV3(nn.Module):
    """Two-class morphology logits plus dense D-dimensional cell embeddings."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, embedding_dim: int = 8) -> None:
        super().__init__()
        if in_channels <= 0 or base_channels <= 0 or embedding_dim < 2:
            raise ValueError("Invalid V3 channel or embedding configuration")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.embedding_dim = int(embedding_dim)
        b = base_channels
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.bottleneck = ConvBlock(b * 4, b * 8)
        self.dec3 = UpBlock(b * 8, b * 4, b * 4)
        self.dec2 = UpBlock(b * 4, b * 2, b * 2)
        self.dec1 = UpBlock(b * 2, b, b)
        self.morphology_head = nn.Conv2d(b, 2, 1)  # 0=Soma, 1=Process; no Background
        self.embedding_head = nn.Conv2d(b, embedding_dim, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor [B,C,H,W]")
        if images.ndim != 4 or min(images.shape) <= 0:
            raise ValueError(f"Expected non-empty [B,C,H,W], got {tuple(images.shape)}")
        if images.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {images.shape[1]}")
        if not torch.is_floating_point(images) or not torch.isfinite(images).all():
            raise ValueError("images must be finite floating-point values")
        e1 = self.enc1(images)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
        decoded = self.dec1(self.dec2(self.dec3(bottleneck, e3), e2), e1)
        return {
            "morphology_logits": self.morphology_head(decoded),
            "instance_embedding": self.embedding_head(decoded),
        }


def create_v3_model(in_channels: int = 1, base_channels: int = 16, embedding_dim: int = 8) -> AstrocyteInstanceV3:
    return AstrocyteInstanceV3(in_channels, base_channels, embedding_dim)

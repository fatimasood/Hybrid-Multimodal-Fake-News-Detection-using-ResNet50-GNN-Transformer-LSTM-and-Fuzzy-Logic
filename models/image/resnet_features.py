from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50AvgPoolExtractor(nn.Module):
    """Extracts ResNet-50 average pooling features: [batch, 2048]."""

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [batch, 3, 224, 224]; output: [batch, 2048]
        pooled = self.features(images)
        return torch.flatten(pooled, 1)


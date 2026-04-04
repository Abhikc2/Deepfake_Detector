"""
CNN Feature Extractor.

Wraps a pretrained ResNet18 (or other backbone) to produce
a fixed-size feature vector per input image frame.
"""

import torch
import torch.nn as nn
from torchvision import models


class CNNFeatureExtractor(nn.Module):
    """
    Extracts spatial features from a single image frame.

    Uses a pretrained ResNet18 backbone with the final classification
    layer removed, producing a 512-dimensional feature vector.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 512,
    ):
        """
        Args:
            backbone:        Model name — 'resnet18' or 'efficientnet_b0'.
            pretrained:      Use ImageNet pretrained weights.
            freeze_backbone: If True, freeze all backbone parameters.
            output_dim:      Dimensionality of the output feature vector.
        """
        super().__init__()
        self.output_dim = output_dim

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            in_features = base.fc.in_features
            # Remove the final FC layer
            self.features = nn.Sequential(*list(base.children())[:-1])

        elif backbone == "efficientnet_b0":
            weights = (
                models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            self.features = nn.Sequential(base.features, base.avgpool)

        elif backbone == "efficientnet_b2":
            weights = (
                models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            )
            base = models.efficientnet_b2(weights=weights)
            in_features = base.classifier[1].in_features
            self.features = nn.Sequential(base.features, base.avgpool)

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Projection layer if backbone dim ≠ desired output_dim
        if in_features != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(in_features, output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.projection = nn.Identity()

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, C, H, W).

        Returns:
            Feature tensor of shape (batch, output_dim).
        """
        features = self.features(x)          # (batch, in_features, 1, 1)
        features = features.flatten(1)       # (batch, in_features)
        features = self.projection(features) # (batch, output_dim)
        return features

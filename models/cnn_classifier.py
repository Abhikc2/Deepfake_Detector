"""
CNN-only Deepfake Image Classifier.

Uses the shared CNNFeatureExtractor backbone with a classifier head
to classify single images as Real or Fake — no temporal (LSTM) layer.

Default backbone: EfficientNet-B2 (ImageNet pretrained).
"""

import torch
import torch.nn as nn
from .cnn import CNNFeatureExtractor


class DeepfakeImageClassifier(nn.Module):
    """
    Image-level deepfake classifier.

    Pipeline:  (batch, C, H, W)
                       ↓
               CNN backbone  →  (batch, feature_dim)
                       ↓
               Classifier    →  (batch, num_classes)
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b2",
        pretrained: bool = True,
        freeze_cnn: bool = False,
        feature_dim: int = 1408,
        num_classes: int = 2,
        classifier_dropout: float = 0.4,
    ):
        """
        Args:
            backbone:           CNN backbone name ('efficientnet_b2', 'efficientnet_b0', 'resnet18').
            pretrained:         Use ImageNet pretrained weights.
            freeze_cnn:         Freeze CNN backbone parameters.
            feature_dim:        CNN output feature dimension (1408 for B2, 1280 for B0, 512 for ResNet18).
            num_classes:        Output classes (2 = Real / Fake).
            classifier_dropout: Dropout probability in the classifier head.
        """
        super().__init__()

        # ── Spatial encoder (shared with video pipeline) ─────────────────
        self.cnn = CNNFeatureExtractor(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_cnn,
            output_dim=feature_dim,
        )

        # ── Classifier head ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(classifier_dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(classifier_dropout * 0.5),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(classifier_dropout * 0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, C, H, W).

        Returns:
            Logits of shape (batch, num_classes).
        """
        features = self.cnn(x)              # (batch, feature_dim)
        logits = self.classifier(features)   # (batch, num_classes)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities instead of raw logits."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count (trainable) parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

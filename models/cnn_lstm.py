"""
CNN + LSTM Deepfake Detector.

Combines spatial feature extraction (CNN) with temporal modelling
(LSTM) to classify video sequences as Real or Fake.
"""

import torch
import torch.nn as nn
from .cnn import CNNFeatureExtractor


class DeepfakeDetector(nn.Module):
    """
    Full deepfake detection model.

    Pipeline:  (batch, seq_len, C, H, W)
                       ↓
               CNN per frame  →  (batch, seq_len, feature_dim)
                       ↓
               LSTM             →  (batch, hidden_dim)
                       ↓
               Classifier       →  (batch, num_classes)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_cnn: bool = False,
        feature_dim: int = 512,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        num_classes: int = 2,
        classifier_dropout: float = 0.4,
    ):
        """
        Args:
            backbone:           CNN backbone name.
            pretrained:         Use pretrained CNN weights.
            freeze_cnn:         Freeze CNN backbone parameters.
            feature_dim:        CNN output / LSTM input dimension.
            lstm_hidden:        LSTM hidden state size.
            lstm_layers:        Number of stacked LSTM layers.
            lstm_dropout:       Dropout between LSTM layers.
            num_classes:        Output classes (2 = Real / Fake).
            classifier_dropout: Dropout in the classifier head.
        """
        super().__init__()

        # ── Spatial encoder ──────────────────────────────────────────────
        self.cnn = CNNFeatureExtractor(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_cnn,
            output_dim=feature_dim,
        )

        # ── Temporal encoder ─────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        # ── Classifier head ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Dropout(classifier_dropout),
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, C, H, W).

        Returns:
            Logits of shape (batch, num_classes).
        """
        batch, seq_len, C, H, W = x.shape

        # Flatten sequence into batch for CNN
        x = x.view(batch * seq_len, C, H, W)       # (B*T, C, H, W)
        features = self.cnn(x)                       # (B*T, feature_dim)

        # Reshape back to sequence
        features = features.view(batch, seq_len, -1) # (B, T, feature_dim)

        # Temporal modelling
        lstm_out, (h_n, _) = self.lstm(features)     # lstm_out: (B, T, hidden)

        # Use last hidden state from the final LSTM layer
        last_hidden = h_n[-1]                         # (B, hidden)

        # Classify
        logits = self.classifier(last_hidden)         # (B, num_classes)
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

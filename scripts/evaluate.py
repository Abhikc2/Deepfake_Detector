"""
Evaluation script for the CNN+LSTM Deepfake Detector.

Loads a trained checkpoint and evaluates it on a test dataset,
printing accuracy, precision, recall, F1 score, and confusion matrix.

Usage:
    python scripts/evaluate.py \
        --data_dir data/faces \
        --checkpoint checkpoints/best_model.pth \
        --batch_size 8
"""

import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cnn_lstm import DeepfakeDetector
from dataset.dataset_sequence import DeepfakeSequenceDataset, get_val_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate(model, loader, device):
    """Run evaluation and return all predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            logits = model(sequences)
            probs = torch.softmax(logits, dim=1)

            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()

            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def print_metrics(preds, labels, class_names=("Real", "Fake")):
    """Print comprehensive evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary", zero_division=0)
    recall = recall_score(labels, preds, average="binary", zero_division=0)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    cm = confusion_matrix(labels, preds)

    print("\n" + "═" * 50)
    print("        EVALUATION RESULTS")
    print("═" * 50)
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("─" * 50)
    print("  Confusion Matrix:")
    print(f"                Predicted")
    print(f"              {class_names[0]:>6}  {class_names[1]:>6}")
    print(f"  Actual {class_names[0]:>5}  {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"         {class_names[1]:>5}  {cm[1][0]:6d}  {cm[1][1]:6d}")
    print("─" * 50)
    print("\n  Classification Report:")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))
    print("═" * 50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Deepfake Detector")
    parser.add_argument("--data_dir", type=str, default="data/faces",
                        help="Root directory with real/ and fake/ subdirs.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="Path to model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=15)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Load checkpoint ──────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})

    logger.info("Loaded checkpoint: epoch %d, val_loss=%.4f, val_acc=%.4f",
                checkpoint.get("epoch", -1),
                checkpoint.get("val_loss", -1),
                checkpoint.get("val_acc", -1))

    # ── Build model ──────────────────────────────────────────────────────
    model = DeepfakeDetector(
        backbone=saved_args.get("backbone", "resnet18"),
        pretrained=False,  # We'll load weights from checkpoint
        feature_dim=512,
        lstm_hidden=256,
        lstm_layers=2,
        num_classes=2,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model weights loaded successfully.")

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = DeepfakeSequenceDataset(
        root_dir=args.data_dir,
        seq_length=args.seq_length,
        transform=get_val_transforms(args.image_size),
        image_size=args.image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info("Evaluation dataset: %d sequences", len(dataset))
    logger.info("Class distribution: %s", dataset.get_class_distribution())

    # ── Evaluate ─────────────────────────────────────────────────────────
    preds, labels, probs = evaluate(model, loader, device)
    print_metrics(preds, labels)


if __name__ == "__main__":
    main()

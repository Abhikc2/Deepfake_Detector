"""
Training script for the CNN+LSTM Deepfake Detector.

Usage:
    python scripts/train.py \
        --data_dir data/faces \
        --epochs 15 \
        --batch_size 16 \
        --lr 1e-4 \
        --seq_length 15
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cnn_lstm import DeepfakeDetector
from dataset.dataset_sequence import (
    DeepfakeSequenceDataset,
    get_train_transforms,
    get_val_transforms,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (sequences, labels) in enumerate(loader):
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients with LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        running_loss += loss.item() * sequences.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Validate the model. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        logits = model(sequences)
        loss = criterion(logits, labels)

        running_loss += loss.item() * sequences.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train the Deepfake Detector")
    parser.add_argument("--data_dir", type=str, default="data/faces",
                        help="Root directory with real/ and fake/ subdirs of face crops.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_length", type=int, default=15)
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data for validation.")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--freeze_cnn", action="store_true",
                        help="Freeze CNN backbone weights.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Dataset ──────────────────────────────────────────────────────────
    full_dataset = DeepfakeSequenceDataset(
        root_dir=args.data_dir,
        seq_length=args.seq_length,
        transform=get_train_transforms(args.image_size),
        image_size=args.image_size,
    )

    logger.info("Total sequences: %d", len(full_dataset))
    logger.info("Class distribution: %s", full_dataset.get_class_distribution())

    # Train / val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Override validation transforms (no augmentation)
    # Note: since we use random_split, we create a wrapper
    val_dataset.dataset = DeepfakeSequenceDataset(
        root_dir=args.data_dir,
        seq_length=args.seq_length,
        transform=get_val_transforms(args.image_size),
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info("Train: %d sequences | Val: %d sequences", train_size, val_size)

    # ── Model ────────────────────────────────────────────────────────────
    model = DeepfakeDetector(
        backbone=args.backbone,
        pretrained=True,
        freeze_cnn=args.freeze_cnn,
        feature_dim=512,
        lstm_hidden=256,
        lstm_layers=2,
        num_classes=2,
    ).to(device)

    logger.info(
        "Model: %s | Trainable params: %s",
        args.backbone,
        f"{model.get_num_params():,}",
    )

    # ── Training setup ───────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.1
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    # ── Training loop ────────────────────────────────────────────────────
    logger.info("═══ Starting training ═══")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %02d/%02d  │  "
            "Train Loss: %.4f  Acc: %.2f%%  │  "
            "Val Loss: %.4f  Acc: %.2f%%  │  "
            "Time: %.1fs",
            epoch, args.epochs,
            train_loss, train_acc * 100,
            val_loss, val_acc * 100,
            elapsed,
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)
            logger.info("  ★ Saved best model (val_loss=%.4f)", val_loss)

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, final_path)
    logger.info("═══ Training complete ═══  Best val_loss: %.4f", best_val_loss)


if __name__ == "__main__":
    main()

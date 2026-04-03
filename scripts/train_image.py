"""
Training script for the CNN-only Deepfake Image Classifier.

Trains on individual face crop images (no sequences / LSTM).

Usage (pre-split dataset like Celeb-DF-v2):
    python scripts/train_image.py \
        --train_dir Celeb_V2/Train \
        --val_dir Celeb_V2/Val \
        --epochs 20 \
        --batch_size 32 \
        --lr 1e-4

Usage (single directory with auto-split):
    python scripts/train_image.py \
        --data_dir data/faces \
        --val_split 0.2 \
        --epochs 20
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

from models.cnn_classifier import DeepfakeImageClassifier
from dataset.dataset_image import DeepfakeImageDataset
from dataset.dataset_sequence import get_train_transforms, get_val_transforms

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
) -> tuple:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
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

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train the CNN-only Deepfake Image Classifier",
    )
    # ── Data directories ─────────────────────────────────────────────────
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Single directory with real/ and fake/ subdirs "
                             "(auto-split into train/val). Ignored if --train_dir is set.")
    parser.add_argument("--train_dir", type=str, default=None,
                        help="Training directory with real/ and fake/ subdirs "
                             "(e.g. Celeb_V2/Train).")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Validation directory with real/ and fake/ subdirs "
                             "(e.g. Celeb_V2/Val).")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Optional test directory (evaluated after training).")
    # ── Training hyperparameters ─────────────────────────────────────────
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/image",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction for validation (only used with --data_dir).")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--freeze_cnn", action="store_true",
                        help="Freeze CNN backbone weights.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (0 = disabled).")
    args = parser.parse_args()

    # Validate data args
    if args.train_dir is None and args.data_dir is None:
        parser.error("Provide either --train_dir (+ optional --val_dir) or --data_dir.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Dataset ──────────────────────────────────────────────────────────
    if args.train_dir:
        # ── Mode 1: Pre-split directories (Celeb_V2 style) ──────────────
        train_dataset = DeepfakeImageDataset(
            root_dir=args.train_dir,
            transform=get_train_transforms(args.image_size),
            image_size=args.image_size,
        )
        logger.info("Train images: %d  |  Distribution: %s",
                     len(train_dataset), train_dataset.get_class_distribution())

        if args.val_dir:
            val_dataset = DeepfakeImageDataset(
                root_dir=args.val_dir,
                transform=get_val_transforms(args.image_size),
                image_size=args.image_size,
            )
            logger.info("Val images: %d  |  Distribution: %s",
                         len(val_dataset), val_dataset.get_class_distribution())
        else:
            # No val dir → split from train
            val_size = int(len(train_dataset) * args.val_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            logger.info("Auto-split: %d train / %d val", train_size, val_size)

    else:
        # ── Mode 2: Single directory with auto-split ─────────────────────
        full_dataset = DeepfakeImageDataset(
            root_dir=args.data_dir,
            transform=get_train_transforms(args.image_size),
            image_size=args.image_size,
        )
        logger.info("Total images: %d  |  Distribution: %s",
                     len(full_dataset), full_dataset.get_class_distribution())

        if len(full_dataset) == 0:
            logger.error("No images found in %s — nothing to train on.", args.data_dir)
            sys.exit(1)

        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Override validation transforms (no augmentation)
        val_dataset.dataset = DeepfakeImageDataset(
            root_dir=args.data_dir,
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

    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # ── Model ────────────────────────────────────────────────────────────
    model = DeepfakeImageClassifier(
        backbone=args.backbone,
        pretrained=True,
        freeze_cnn=args.freeze_cnn,
        feature_dim=512,
        num_classes=2,
    ).to(device)

    logger.info(
        "Model: %s (image-only) | Trainable params: %s",
        args.backbone,
        f"{model.get_num_params():,}",
    )

    # ── Training setup ───────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.1,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # ── Training loop ────────────────────────────────────────────────────
    logger.info("═══ Starting image training ═══")

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
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.checkpoint_dir, "best_image_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)
            logger.info("  ★ Saved best model (val_loss=%.4f)", val_loss)
        else:
            epochs_no_improve += 1

        # Early stopping
        if args.patience > 0 and epochs_no_improve >= args.patience:
            logger.info(
                "  ⏹ Early stopping triggered after %d epochs without improvement.",
                args.patience,
            )
            break

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "final_image_model.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, final_path)
    logger.info("═══ Training complete ═══  Best val_loss: %.4f", best_val_loss)


if __name__ == "__main__":
    main()

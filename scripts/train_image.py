"""
Training script for the CNN-only Deepfake Image Classifier.

Trains on individual face crop images (no sequences / LSTM).
Optimised for Google Colab (T4 GPU) with mixed-precision training.

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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cnn_classifier import DeepfakeImageClassifier
from dataset.dataset_image import (
    DeepfakeImageDataset,
    get_image_train_transforms,
    get_image_val_transforms,
)

# Metrics
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Backbone → native feature dim mapping ──────────────────────────────────
_BACKBONE_DIM = {
    "efficientnet_b2": 1408,
    "efficientnet_b0": 1280,
    "resnet18": 512,
}


def compute_class_weights(dataset, device):
    """
    Compute inverse-frequency class weights from a dataset or Subset.

    Returns a tensor of shape (num_classes,) on the given device.
    """
    if isinstance(dataset, Subset):
        labels = [dataset.dataset.samples[i][1] for i in dataset.indices]
    else:
        labels = [s[1] for s in dataset.samples]

    counts = np.bincount(labels)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(counts)  # normalise
    logger.info("Class counts: %s | Weights: %s", counts.tolist(),
                [f"{w:.3f}" for w in weights])
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool = True,
) -> tuple:
    """Train for one epoch with mixed-precision. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        # Scaled backward pass
        scaler.scale(loss).backward()

        # Gradient clipping (unscale first for correct norm)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        scaler.step(optimizer)
        scaler.update()

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
    use_amp: bool = True,
) -> dict:
    """
    Validate the model. Returns dict with loss, accuracy, and sklearn metrics.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # prob of "fake" class

    avg_loss = running_loss / total
    accuracy = correct / total

    # Compute sklearn metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }

    # AUC requires both classes present
    try:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["auc"] = 0.0

    return metrics


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
                        help="Directory to save model checkpoints (supports Google Drive paths).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay.")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction for validation (only used with --data_dir).")
    parser.add_argument("--backbone", type=str, default="efficientnet_b2",
                        choices=["resnet18", "efficientnet_b0", "efficientnet_b2"])
    parser.add_argument("--freeze_cnn", action="store_true",
                        help="Freeze CNN backbone weights.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (0 = disabled).")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed-precision training.")
    args = parser.parse_args()

    # Validate data args
    if args.train_dir is None and args.data_dir is None:
        parser.error("Provide either --train_dir (+ optional --val_dir) or --data_dir.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")
    logger.info("Using device: %s  |  AMP: %s", device, use_amp)

    # Auto-select feature_dim from backbone
    feature_dim = _BACKBONE_DIM.get(args.backbone, 512)

    # ── Dataset ──────────────────────────────────────────────────────────
    train_transforms = get_image_train_transforms(args.image_size)
    val_transforms = get_image_val_transforms(args.image_size)

    if args.train_dir:
        # ── Mode 1: Pre-split directories (Celeb_V2 style) ──────────────
        train_dataset = DeepfakeImageDataset(
            root_dir=args.train_dir,
            transform=train_transforms,
            image_size=args.image_size,
        )
        logger.info("Train images: %d  |  Distribution: %s",
                     len(train_dataset), train_dataset.get_class_distribution())

        if args.val_dir:
            val_dataset = DeepfakeImageDataset(
                root_dir=args.val_dir,
                transform=val_transforms,
                image_size=args.image_size,
            )
            logger.info("Val images: %d  |  Distribution: %s",
                         len(val_dataset), val_dataset.get_class_distribution())
        else:
            # No val dir → stratified split from train
            train_dataset, val_dataset = _stratified_split(
                train_dataset, args.val_split, train_transforms, val_transforms,
            )

    else:
        # ── Mode 2: Single directory with auto-split (data leakage fixed) ─
        full_dataset = DeepfakeImageDataset(
            root_dir=args.data_dir,
            transform=train_transforms,  # placeholder, overridden by Subset
            image_size=args.image_size,
        )
        logger.info("Total images: %d  |  Distribution: %s",
                     len(full_dataset), full_dataset.get_class_distribution())

        if len(full_dataset) == 0:
            logger.error("No images found in %s — nothing to train on.", args.data_dir)
            sys.exit(1)

        train_dataset, val_dataset = _stratified_split(
            full_dataset, args.val_split, train_transforms, val_transforms,
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
        feature_dim=feature_dim,
        num_classes=2,
    ).to(device)

    logger.info(
        "Model: %s (image-only) | feature_dim: %d | Trainable params: %s",
        args.backbone,
        feature_dim,
        f"{model.get_num_params():,}",
    )

    # ── Training setup ───────────────────────────────────────────────────
    class_weights = compute_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # ── Training loop ────────────────────────────────────────────────────
    logger.info("═══ Starting image training ═══")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp,
        )
        val_metrics = validate(model, val_loader, criterion, device, use_amp)

        scheduler.step(val_metrics["loss"])
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %02d/%02d  │  "
            "Train Loss: %.4f  Acc: %.2f%%  │  "
            "Val Loss: %.4f  Acc: %.2f%%  │  "
            "AUC: %.4f  F1: %.4f  Prec: %.4f  Rec: %.4f  │  "
            "LR: %.2e  Time: %.1fs",
            epoch, args.epochs,
            train_loss, train_acc * 100,
            val_metrics["loss"], val_metrics["accuracy"] * 100,
            val_metrics["auc"], val_metrics["f1"],
            val_metrics["precision"], val_metrics["recall"],
            current_lr, elapsed,
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.checkpoint_dir, "best_image_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_metrics": val_metrics,
                "args": vars(args),
            }, ckpt_path)
            logger.info("  ★ Saved best model (val_loss=%.4f, AUC=%.4f, F1=%.4f)",
                        val_metrics["loss"], val_metrics["auc"], val_metrics["f1"])
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
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "val_metrics": val_metrics,
        "args": vars(args),
    }, final_path)
    logger.info("═══ Training complete ═══  Best val_loss: %.4f", best_val_loss)


# ─── Helpers ─────────────────────────────────────────────────────────────────

class _TransformSubset(Subset):
    """A Subset that overrides the parent dataset's transform per-sample."""

    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        image_path, label = self.dataset.samples[self.indices[idx]]
        from PIL import Image as _Image
        img = _Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _stratified_split(dataset, val_fraction, train_transform, val_transform):
    """
    Split a DeepfakeImageDataset into train/val Subsets with separate transforms.

    Uses stratified sampling to maintain class balance. Avoids data leakage
    by using index-based Subsets instead of reassigning the parent dataset.
    """
    labels = np.array([s[1] for s in dataset.samples])
    indices = np.arange(len(dataset))

    # Per-class stratified split
    train_indices, val_indices = [], []
    for cls in np.unique(labels):
        cls_idx = indices[labels == cls]
        np.random.seed(42)
        np.random.shuffle(cls_idx)
        n_val = int(len(cls_idx) * val_fraction)
        val_indices.extend(cls_idx[:n_val])
        train_indices.extend(cls_idx[n_val:])

    logger.info("Stratified split: %d train / %d val", len(train_indices), len(val_indices))

    train_subset = _TransformSubset(dataset, train_indices, train_transform)
    val_subset = _TransformSubset(dataset, val_indices, val_transform)
    return train_subset, val_subset


if __name__ == "__main__":
    main()

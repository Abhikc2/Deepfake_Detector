"""
Evaluation script for the CNN-only Deepfake Image Classifier.

Loads a saved checkpoint and runs evaluation on a pre-split test directory.
Reports Accuracy, AUC, Precision, Recall, and F1-score.

Usage:
    python scripts/evaluate_image.py \
        --test_dir /path/to/test_dataset \
        --checkpoint_path checkpoints/image/best_image_model.pth
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cnn_classifier import DeepfakeImageClassifier
from dataset.dataset_image import DeepfakeImageDataset, get_image_val_transforms
from scripts.train_image import validate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Backbone mapping (same as train_image.py)
_BACKBONE_DIM = {
    "efficientnet_b2": 1408,
    "efficientnet_b0": 1280,
    "resnet18": 512,
}

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Trained Deepfake Image Classifier")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to test directory with real/ and fake/ subdirs.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the saved .pth model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 1. Load Checkpoint
    if not Path(args.checkpoint_path).exists():
        logger.error("Checkpoint not found at: %s", args.checkpoint_path)
        sys.exit(1)

    logger.info("Loading checkpoint: %s", args.checkpoint_path)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    
    # Auto-detect backbone from checkpoint args
    saved_args = checkpoint.get("args", {})
    backbone = saved_args.get("backbone", "efficientnet_b2")
    feature_dim = _BACKBONE_DIM.get(backbone, 1408)
    
    # 2. Build Model
    model = DeepfakeImageClassifier(
        backbone=backbone,
        pretrained=False, # We are loading trained weights below
        freeze_cnn=False,
        feature_dim=feature_dim,
        num_classes=2,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded weights for backbone: %s (Feature Dim: %d)", backbone, feature_dim)

    # 3. Load Dataset
    test_transforms = get_image_val_transforms(args.image_size)
    test_dataset = DeepfakeImageDataset(
        root_dir=args.test_dir,
        transform=test_transforms,
        image_size=args.image_size,
    )
    
    logger.info("Test images: %d | Distribution: %s", 
                len(test_dataset), test_dataset.get_class_distribution())

    if len(test_dataset) == 0:
        logger.error("No images found in test directory.")
        sys.exit(1)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 4. Evaluate using the validate function from train_image.py
    logger.info("Starting evaluation...")
    criterion = torch.nn.CrossEntropyLoss()
    metrics = validate(model, test_loader, criterion, device, use_amp=(device.type == "cuda"))

    # 5. Report Results
    print("\n" + "="*40)
    print("         EVALUATION RESULTS")
    print("="*40)
    print(f" Loss:      {metrics['loss']:.4f}")
    print(f" Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f" ROC-AUC:   {metrics['auc']:.4f}")
    print(f" Precision: {metrics['precision']:.4f}")
    print(f" Recall:    {metrics['recall']:.4f}")
    print(f" F1-Score:  {metrics['f1']:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()

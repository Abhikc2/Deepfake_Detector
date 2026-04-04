"""
Deepfake Image Dataset.

Loads individual pre-extracted face crops for training a CNN-only
image classifier. Each image is one sample (not grouped into sequences).

Includes robust augmentations that simulate social-media degradations
(JPEG compression, blur, noise) for real-world robustness.

Expected directory structure:
    root/
        real/
            video_001/
                face_00000.jpg
                face_00001.jpg
                ...
            image_002/
                face_00000.jpg
        fake/
            video_101/
                face_00000.jpg
                ...
"""

import io
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# Supported face crop extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ImageNet normalisation constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ─── Custom Augmentation Transforms ─────────────────────────────────────────

class JPEGCompression:
    """Simulate JPEG compression artifacts (social-media re-encoding)."""

    def __init__(self, quality_range: Tuple[int, int] = (30, 85), p: float = 0.3):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() > self.p:
            return img
        quality = np.random.randint(self.quality_range[0], self.quality_range[1] + 1)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(quality={self.quality_range}, p={self.p})"


class GaussianNoise:
    """Add random Gaussian noise to simulate sensor / upload noise."""

    def __init__(self, mean: float = 0.0, std_range: Tuple[float, float] = (0.01, 0.05), p: float = 0.2):
        self.mean = mean
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return tensor
        std = np.random.uniform(self.std_range[0], self.std_range[1])
        noise = torch.randn_like(tensor) * std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std_range}, p={self.p})"


# ─── Transform Factories ────────────────────────────────────────────────────

def get_image_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Training transforms with social-media-realistic augmentations.

    Includes JPEG compression artifacts, Gaussian blur, noise injection,
    color jitter, and random crops to improve real-world robustness.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        JPEGCompression(quality_range=(30, 85), p=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        GaussianNoise(std_range=(0.01, 0.05), p=0.2),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_image_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Validation / inference transforms — NO augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ─── Dataset ────────────────────────────────────────────────────────────────

class DeepfakeImageDataset(Dataset):
    """
    PyTorch Dataset that yields (image_tensor, label) pairs.

    Each sample is a single face crop image, transformed into a
    tensor of shape (C, H, W).
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224,
    ):
        """
        Args:
            root_dir:   Root directory containing 'real/' and 'fake/' subdirs,
                        each with per-source subdirectories of face crops.
            transform:  Torchvision transforms applied to each image.
            image_size: Resize target (used by default transforms).
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or get_image_val_transforms(image_size)

        # Build flat sample list: [(image_path, label), ...]
        self.samples: List[Tuple[str, int]] = []
        self._build_samples()

        logger.info(
            "Image dataset loaded: %d samples from %s", len(self.samples), root_dir
        )

    def _build_samples(self):
        """
        Scan directory tree and collect all individual face crops.

        Supports two layouts:
          1. Flat:   root/real/img001.jpg        (e.g. Celeb-DF-v2)
          2. Nested: root/real/video_001/face.jpg (preprocessing pipeline output)
        """
        label_map = {"real": 0, "fake": 1}

        for label_name, label_id in label_map.items():
            label_dir = self.root_dir / label_name
            if not label_dir.exists():
                logger.warning("Label directory not found: %s", label_dir)
                continue

            for entry in sorted(label_dir.iterdir()):
                if entry.is_file() and entry.suffix.lower() in _IMAGE_EXTENSIONS:
                    # Flat layout: image directly in real/ or fake/
                    self.samples.append((str(entry), label_id))

                elif entry.is_dir():
                    # Nested layout: subdirectory of face crops
                    for img_file in sorted(entry.iterdir()):
                        if img_file.suffix.lower() in _IMAGE_EXTENSIONS:
                            self.samples.append((str(img_file), label_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]

        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_distribution(self) -> dict:
        """Return count of samples per class."""
        counts = {"real": 0, "fake": 0}
        for _, label in self.samples:
            counts["real" if label == 0 else "fake"] += 1
        return counts

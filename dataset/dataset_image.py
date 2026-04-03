"""
Deepfake Image Dataset.

Loads individual pre-extracted face crops for training a CNN-only
image classifier. Each image is one sample (not grouped into sequences).

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

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .dataset_sequence import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)

# Supported face crop extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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
        self.transform = transform or get_val_transforms(image_size)

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

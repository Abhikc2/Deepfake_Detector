"""
Deepfake Sequence Dataset.

Loads pre-extracted face crops organized by video into fixed-length
sequences for training the CNN+LSTM model.

Expected directory structure:
    root/
        real/
            video_001/
                face_00000.jpg
                face_00001.jpg
                ...
            video_002/
                ...
        fake/
            video_101/
                face_00000.jpg
                ...
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

# ─── Default transforms ─────────────────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training transforms with data augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        ),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Validation / inference transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class DeepfakeSequenceDataset(Dataset):
    """
    PyTorch Dataset that yields (sequence_tensor, label) pairs.

    Each sample is a sequence of ``seq_length`` face images from one video,
    stacked into a tensor of shape (seq_length, C, H, W).
    """

    def __init__(
        self,
        root_dir: str,
        seq_length: int = 15,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224,
        overlap: int = 0,
    ):
        """
        Args:
            root_dir:   Root directory containing 'real/' and 'fake/' subdirs.
            seq_length: Number of frames per sequence.
            transform:  Torchvision transforms (applied per frame).
            image_size: Resize target for frames.
            overlap:    Number of overlapping frames between consecutive sequences
                        from the same video (0 = no overlap).
        """
        self.root_dir = Path(root_dir)
        self.seq_length = seq_length
        self.transform = transform or get_val_transforms(image_size)
        self.overlap = overlap

        # Build sample list: [(list_of_frame_paths, label), ...]
        self.samples: List[Tuple[List[str], int]] = []
        self._build_samples()

        logger.info(
            "Dataset loaded: %d sequences from %s", len(self.samples), root_dir
        )

    def _build_samples(self):
        """Scan directory tree and create sequence samples."""
        label_map = {"real": 0, "fake": 1}

        for label_name, label_id in label_map.items():
            label_dir = self.root_dir / label_name
            if not label_dir.exists():
                logger.warning("Label directory not found: %s", label_dir)
                continue

            # Each subdirectory = one video's face crops
            for video_dir in sorted(label_dir.iterdir()):
                if not video_dir.is_dir():
                    continue

                frames = sorted(
                    [
                        str(f)
                        for f in video_dir.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    ]
                )

                if len(frames) < self.seq_length:
                    # Pad by repeating last frame if not enough frames
                    if len(frames) == 0:
                        continue
                    while len(frames) < self.seq_length:
                        frames.append(frames[-1])
                    self.samples.append((frames[: self.seq_length], label_id))
                else:
                    # Slide window over the sequence
                    step = max(1, self.seq_length - self.overlap)
                    for start in range(0, len(frames) - self.seq_length + 1, step):
                        seq = frames[start : start + self.seq_length]
                        self.samples.append((seq, label_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        frame_paths, label = self.samples[idx]

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Stack: (seq_length, C, H, W)
        sequence = torch.stack(frames, dim=0)
        return sequence, label

    def get_class_distribution(self) -> dict:
        """Return count of samples per class."""
        counts = {"real": 0, "fake": 0}
        for _, label in self.samples:
            counts["real" if label == 0 else "fake"] += 1
        return counts

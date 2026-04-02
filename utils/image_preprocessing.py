"""
Image preprocessing utilities for deepfake image detection.

Provides functions to load, validate, and transform images into
model-ready tensors, reusing the same normalisation as the video pipeline.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# Same normalisation used across the entire project
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = 224


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from disk using OpenCV.

    Args:
        image_path: Path to the image file (JPG, PNG, JPEG).

    Returns:
        BGR numpy array (OpenCV format).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(
            f"Could not decode image (corrupted or unsupported format): {image_path}"
        )

    logger.info("Loaded image %s — shape %s", image_path.name, image.shape)
    return image


def get_inference_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    """
    Build the standard inference transform pipeline.

    Returns:
        A torchvision Compose transform that converts an RGB numpy array
        (or PIL Image) into a normalised (C, H, W) tensor.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image_tensor(
    image: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> torch.Tensor:
    """
    Convert a BGR numpy image to a normalised model-ready tensor.

    Steps:
        1. BGR → RGB conversion
        2. Resize to (image_size × image_size)
        3. Scale to [0, 1] and apply ImageNet normalisation
        4. Return tensor of shape (C, H, W)

    Args:
        image:      BGR numpy array (OpenCV format).
        image_size: Target spatial size (default 224).

    Returns:
        Float tensor of shape (3, image_size, image_size).
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = get_inference_transform(image_size)
    tensor = transform(rgb)
    return tensor

"""
Face extraction utility using MTCNN from facenet-pytorch.

Extracts and normalizes face regions from images or person crops.
"""

import logging
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class FaceExtractor:
    """
    Extracts faces from images using MTCNN.

    The MTCNN model is loaded lazily and cached for subsequent calls.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        min_face_size: int = 40,
        margin: int = 20,
        device: Optional[str] = None,
    ):
        """
        Args:
            target_size:   Output face size (width, height).
            min_face_size: Minimum detectable face size in pixels.
            margin:        Pixel margin around detected face box.
            device:        'cpu' or 'cuda'. None = auto-detect.
        """
        self.target_size = target_size
        self.min_face_size = min_face_size
        self.margin = margin
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._mtcnn = None

    # ── Lazy loader ──────────────────────────────────────────────────────
    def _load_mtcnn(self):
        from facenet_pytorch import MTCNN

        logger.info("Loading MTCNN on device=%s", self.device)
        self._mtcnn = MTCNN(
            image_size=self.target_size[0],
            margin=self.margin,
            min_face_size=self.min_face_size,
            select_largest=True,  # Pick the largest face per image
            post_process=False,   # Return raw pixel values (0-255 range)
            device=self.device,
        )
        logger.info("MTCNN loaded successfully.")

    @property
    def mtcnn(self):
        if self._mtcnn is None:
            self._load_mtcnn()
        return self._mtcnn

    # ── Public API ───────────────────────────────────────────────────────
    def extract_face(
        self,
        image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Extract the largest face from an image.

        Args:
            image: BGR numpy array (OpenCV format).

        Returns:
            Face image as a numpy array (RGB, target_size) or None if no face found.
        """
        # Convert BGR → RGB → PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Detect + align
        face_tensor = self.mtcnn(pil_image)

        if face_tensor is None:
            return None

        # tensor is (C, H, W) float, range depends on post_process flag
        face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return face_np

    def extract_face_tensor(
        self,
        image: np.ndarray,
    ) -> Optional[torch.Tensor]:
        """
        Extract face and return as a PyTorch tensor (C, H, W), float32, [0, 1].

        Returns:
            Tensor of shape (3, H, W) or None.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        face_tensor = self.mtcnn(pil_image)
        if face_tensor is None:
            return None

        # Normalize to [0, 1]
        face_tensor = face_tensor.float() / 255.0
        return face_tensor

    def extract_all_faces(
        self,
        image: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Extract all detected faces from an image.

        Returns:
            List of face images (RGB numpy arrays). May be empty.
        """
        from facenet_pytorch import MTCNN as _MTCNN

        # Use a temporary MTCNN configured to return all faces
        multi_mtcnn = _MTCNN(
            image_size=self.target_size[0],
            margin=self.margin,
            min_face_size=self.min_face_size,
            select_largest=False,
            keep_all=True,
            post_process=False,
            device=self.device,
        )

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        faces_tensor = multi_mtcnn(pil_image)
        if faces_tensor is None:
            return []

        faces = []
        if faces_tensor.dim() == 3:
            # Single face returned
            faces_tensor = faces_tensor.unsqueeze(0)

        for ft in faces_tensor:
            face_np = ft.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            faces.append(face_np)

        return faces

    def has_face(self, image: np.ndarray) -> bool:
        """Quick check: does the image contain a detectable face?"""
        return self.extract_face(image) is not None

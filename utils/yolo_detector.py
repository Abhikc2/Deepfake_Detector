"""
YOLO-based person detection utility.

Uses Ultralytics YOLOv8 to detect and crop person regions from frames,
filtering out irrelevant image areas before face extraction.
"""

import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class index for "person"
_PERSON_CLASS_ID = 0


class YOLOPersonDetector:
    """
    Detects persons in frames using YOLOv8.

    The model is loaded lazily on first call and cached for subsequent use.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: YOLOv8 weight file (e.g. 'yolov8n.pt', 'yolov8s.pt').
            confidence_threshold: Minimum confidence to keep a detection.
            device: Force device ('cpu', 'cuda', '0'). None = auto-detect.
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    # ── Lazy loader ──────────────────────────────────────────────────────
    def _load_model(self):
        """Load the YOLO model on first use."""
        from ultralytics import YOLO

        logger.info("Loading YOLO model: %s", self.model_name)
        self._model = YOLO(self.model_name)
        if self.device:
            self._model.to(self.device)
        logger.info("YOLO model loaded successfully.")

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    # ── Public API ───────────────────────────────────────────────────────
    def detect_persons(
        self,
        frame: np.ndarray,
        padding: float = 0.1,
    ) -> List[np.ndarray]:
        """
        Detect persons in a single frame and return cropped regions.

        Args:
            frame:   BGR numpy array (OpenCV format).
            padding: Fractional padding around the bounding box (0.1 = 10%).

        Returns:
            List of cropped person-region images (BGR numpy arrays).
        """
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            classes=[_PERSON_CLASS_ID],
            verbose=False,
        )

        crops = []
        h, w = frame.shape[:2]

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Apply padding
                pad_w = int((x2 - x1) * padding)
                pad_h = int((y2 - y1) * padding)
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(w, x2 + pad_w)
                y2 = min(h, y2 + pad_h)

                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)

        return crops

    def detect_persons_with_boxes(
        self,
        frame: np.ndarray,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Detect persons and return (crop, bbox, confidence) tuples.

        Returns:
            List of (cropped_image, (x1, y1, x2, y2), confidence).
        """
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            classes=[_PERSON_CLASS_ID],
            verbose=False,
        )

        detections = []
        h, w = frame.shape[:2]

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    detections.append((crop, (x1, y1, x2, y2), conf))

        return detections

    def has_person(self, frame: np.ndarray) -> bool:
        """Quick check: does the frame contain at least one person?"""
        return len(self.detect_persons(frame)) > 0

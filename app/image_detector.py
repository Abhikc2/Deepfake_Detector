"""
Image-based deepfake detection pipeline.

Handles the full workflow:
    Load image → Extract faces → Preprocess → Model inference → Result.

Reuses the existing FaceExtractor and CNN+LSTM model from the video
pipeline.  For single-image inference the face tensor is repeated to
fill the sequence length expected by the LSTM.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import cv2
import numpy as np
import torch

from utils.face_extractor import FaceExtractor
from utils.image_preprocessing import (
    load_image,
    preprocess_image_tensor,
    DEFAULT_IMAGE_SIZE,
)

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 15  # Must match the value used in the video pipeline


def _build_sequence(
    face_tensor: torch.Tensor,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Repeat a single-frame tensor to create a pseudo video sequence.

    Args:
        face_tensor:     Tensor of shape (C, H, W).
        sequence_length: Number of repetitions.
        device:          Target device.

    Returns:
        Tensor of shape (1, sequence_length, C, H, W).
    """
    # (C,H,W) → (seq_len, C, H, W) → (1, seq_len, C, H, W)
    sequence = face_tensor.unsqueeze(0).repeat(sequence_length, 1, 1, 1)
    return sequence.unsqueeze(0).to(device)


def _run_single_face_inference(
    model: torch.nn.Module,
    face_image: np.ndarray,
    device: torch.device,
    is_sequence_model: bool = True,
    sequence_length: int = SEQUENCE_LENGTH,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Dict[str, float]:
    """
    Run inference on a single face crop.

    Returns:
        Dict with 'real_prob' and 'fake_prob'.
    """
    tensor = preprocess_image_tensor(face_image, image_size=image_size)
    
    if is_sequence_model:
        x_input = _build_sequence(tensor, sequence_length, device)
    else:
        x_input = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        probs = model.predict_proba(x_input)

    return {
        "real_prob": probs[0][0].item(),
        "fake_prob": probs[0][1].item(),
    }


def detect_image(
    model: torch.nn.Module,
    image_path: Union[str, Path],
    device: torch.device,
    face_extractor: FaceExtractor,
    sequence_length: int = SEQUENCE_LENGTH,
    is_sequence_model: bool = True,
) -> Dict[str, Any]:
    """
    Full image deepfake detection pipeline.

    Pipeline:
        1. Load image
        2. Extract all faces (MTCNN via FaceExtractor)
        3. If no face found → fall back to full image
        4. Preprocess each face → build pseudo-sequence → model inference
        5. If multiple faces → average probabilities
        6. Return aggregated result

    Args:
        model:            Loaded DeepfakeDetector (CNN+LSTM) in eval mode.
        image_path:       Path to input image (JPG / PNG / JPEG).
        device:           torch device (cpu / cuda).
        face_extractor:   Initialised FaceExtractor instance.
        sequence_length:  Frames expected by the LSTM (default 15).

    Returns:
        Dict with keys:
            prediction   – "REAL" or "FAKE"
            confidence   – float (0-1)
            real_prob    – aggregated real probability
            fake_prob    – aggregated fake probability
            num_faces    – number of faces processed
            face_images  – list of face crops (RGB numpy arrays)
            used_full_image – bool, True if fallback was used
            timings      – dict of stage durations
            error        – str or None
    """
    result: Dict[str, Any] = {
        "prediction": None,
        "confidence": 0.0,
        "real_prob": 0.0,
        "fake_prob": 0.0,
        "num_faces": 0,
        "face_images": [],
        "used_full_image": False,
        "timings": {},
        "error": None,
    }

    # ── Step 1: Load image ───────────────────────────────────────────────
    t_start = time.time()
    try:
        bgr_image = load_image(image_path)
    except (FileNotFoundError, ValueError) as exc:
        result["error"] = str(exc)
        logger.error("Image load failed: %s", exc)
        return result
    result["timings"]["load"] = time.time() - t_start

    # ── Step 2: Extract faces ────────────────────────────────────────────
    t_face = time.time()
    try:
        faces: List[np.ndarray] = face_extractor.extract_all_faces(bgr_image)
    except Exception as exc:
        logger.warning("Face extraction error, falling back to full image: %s", exc)
        faces = []

    # Convert RGB faces (from MTCNN) back to BGR for preprocess_image_tensor
    faces = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in faces]

    if not faces:
        # Fallback: use the full image (already BGR)
        logger.info("No faces detected — using full image as input.")
        resized = cv2.resize(bgr_image, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
        faces = [resized]
        result["used_full_image"] = True

    # Store RGB copies for display (faces are now BGR for preprocessing)
    result["face_images"] = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in faces]
    result["num_faces"] = len(faces)
    result["timings"]["face_extraction"] = time.time() - t_face

    # ── Step 3: Inference per face ───────────────────────────────────────
    t_infer = time.time()
    all_real: List[float] = []
    all_fake: List[float] = []

    for face_img in faces:
        try:
            preds = _run_single_face_inference(
                model, face_img, device, is_sequence_model, sequence_length
            )
            all_real.append(preds["real_prob"])
            all_fake.append(preds["fake_prob"])
        except Exception as exc:
            logger.warning("Inference failed for a face crop: %s", exc)

    result["timings"]["inference"] = time.time() - t_infer

    if not all_real:
        result["error"] = "Inference failed for all faces."
        return result

    # ── Step 4: Aggregate results ────────────────────────────────────────
    avg_real = sum(all_real) / len(all_real)
    avg_fake = sum(all_fake) / len(all_fake)

    result["real_prob"] = avg_real
    result["fake_prob"] = avg_fake
    result["prediction"] = "REAL" if avg_real > avg_fake else "FAKE"
    result["confidence"] = max(avg_real, avg_fake)
    result["timings"]["total"] = time.time() - t_start

    logger.info(
        "Image detection complete — %s (%.1f%% confidence, %d face(s))",
        result["prediction"],
        result["confidence"] * 100,
        result["num_faces"],
    )
    return result

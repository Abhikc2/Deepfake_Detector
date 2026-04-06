"""
Image-based deepfake detection pipeline.

Uses the pretrained ViT (Vision Transformer) model from
HuggingFace: hamzenium/ViT-Deepfake-Classifier

The model was trained on images, so we send the full image directly — no face cropping needed.
Face extraction is only used to verify a face exists and for display.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Union

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from utils.face_extractor import FaceExtractor

logger = logging.getLogger(__name__)

MODEL_ID = "dima806/deepfake_vs_real_image_detection"


def _run_image_inference(
    model: Any,
    processor: Any,
    pil_image: Image.Image,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run model inference on a PIL image.

    Args:
        model:      Loaded HuggingFace Image Classification model.
        processor:  Loaded HuggingFace Image Processor.
        pil_image:  PIL Image (RGB).
        device:     torch device.

    Returns:
        Dict with 'real_prob' and 'fake_prob'.
    """
    inputs = processor(images=pil_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Use the model's own label mapping
    id2label = model.config.id2label
    probs_dict = {}
    for idx, label in id2label.items():
        probs_dict[label.lower()] = probabilities[0][int(idx)].item()

    real_prob = probs_dict.get("real", 0.0)
    fake_prob = probs_dict.get("fake", 0.0)

    return {"real_prob": real_prob, "fake_prob": fake_prob}


def detect_image(
    model: Any,
    processor: Any,
    image_path: Union[str, Path],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Direct image deepfake detection pipeline using the HuggingFace model.

    Pipeline:
        1. Load image natively
        2. Analyze the full image
        3. Return probabilities

    Args:
        model:            Loaded HuggingFace Image Classification model.
        processor:        Loaded HuggingFace Image Processor.
        image_path:       Path to input image (JPG / PNG / JPEG).
        device:           torch device.

    Returns:
        Dict with keys: prediction, confidence, real_prob, fake_prob, etc.
    """
    result: Dict[str, Any] = {
        "prediction": None,
        "confidence": 0.0,
        "real_prob": 0.0,
        "fake_prob": 0.0,
        "num_faces": 0,
        "face_images": [],
        "used_full_image": True,
        "timings": {},
        "error": None,
    }

    t_start = time.time()
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load as PIL Image (RGB) for the processor
        pil_image = Image.open(str(image_path)).convert("RGB")
    except (FileNotFoundError, ValueError) as exc:
        result["error"] = str(exc)
        logger.error("Image load failed: %s", exc)
        return result
    
    result["timings"]["load"] = time.time() - t_start

    # ── Step 2: Inference ────────────────────────────────────────────────
    t_infer = time.time()
    try:
        preds = _run_image_inference(model, processor, pil_image, device)
        result["real_prob"] = preds["real_prob"]
        result["fake_prob"] = preds["fake_prob"]

        result["prediction"] = "REAL" if result["real_prob"] > result["fake_prob"] else "FAKE"
        result["confidence"] = max(result["real_prob"], result["fake_prob"])
    except Exception as exc:
        result["error"] = f"Inference failed: {exc}"
        logger.error("Inference failed: %s", exc)
        return result
    
    result["timings"]["inference"] = time.time() - t_infer
    result["timings"]["total"] = time.time() - t_start

    logger.info(
        "Image detection complete — %s (%.1f%% confidence, evaluated on full image)",
        result["prediction"],
        result["confidence"] * 100,
    )
    return result

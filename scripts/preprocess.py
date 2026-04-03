"""
Preprocessing pipeline.

Processes raw video AND image files through the full pipeline:
  Video → Frames → YOLO (person detection) → MTCNN (face extraction) → Saved face crops
  Image →          YOLO (person detection) → MTCNN (face extraction) → Saved face crops

Usage:
    python scripts/preprocess.py \
        --input_dir data/raw \
        --output_dir data/faces \
        --sample_rate 5 \
        --max_frames 80 \
        --include_images \
        --max_faces_per_image 1
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.video_to_frames import extract_frames, validate_video_file
from utils.yolo_detector import YOLOPersonDetector
from utils.face_extractor import FaceExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Supported file extensions ───────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def process_video(
    video_path: str,
    output_dir: str,
    yolo_detector: YOLOPersonDetector,
    face_extractor: FaceExtractor,
    sample_rate: int = 5,
    max_frames: int = 80,
) -> int:
    """
    Process a single video: extract frames → detect persons → crop faces → save.

    Returns:
        Number of face crops successfully saved.
    """
    # Validate the video file
    is_valid, msg = validate_video_file(video_path)
    if not is_valid:
        logger.warning("Skipping %s: %s", video_path, msg)
        return 0

    # Extract frames in memory
    frames = extract_frames(
        video_path,
        output_dir=None,
        sample_rate=sample_rate,
        max_frames=max_frames,
    )

    os.makedirs(output_dir, exist_ok=True)
    face_count = 0

    for i, frame in enumerate(frames):
        # Step 1: YOLO person detection
        person_crops = yolo_detector.detect_persons(frame)

        if not person_crops:
            # Fall back to using the full frame
            person_crops = [frame]

        # Step 2: MTCNN face extraction from person crops
        for crop in person_crops:
            face = face_extractor.extract_face(crop)
            if face is not None:
                # Save face (RGB → BGR for OpenCV)
                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                face_path = os.path.join(output_dir, f"face_{face_count:05d}.jpg")
                cv2.imwrite(face_path, face_bgr)
                face_count += 1
                break  # One face per frame is enough

    return face_count


def process_image(
    image_path: str,
    output_dir: str,
    yolo_detector: YOLOPersonDetector,
    face_extractor: FaceExtractor,
    max_faces: int = 1,
) -> int:
    """
    Process a single image: detect persons → crop faces → save.

    Args:
        image_path:      Path to the input image file.
        output_dir:      Directory to save extracted face crops.
        yolo_detector:   Shared YOLOPersonDetector instance.
        face_extractor:  Shared FaceExtractor instance.
        max_faces:       Maximum number of faces to extract (1 = single, >1 = multi).

    Returns:
        Number of face crops successfully saved.
    """
    # Load and validate the image
    image = cv2.imread(image_path)
    if image is None:
        logger.warning("Skipping %s: could not read image (corrupted or unsupported).", image_path)
        return 0

    os.makedirs(output_dir, exist_ok=True)
    face_count = 0

    # Step 1: YOLO person detection (expects BGR input)
    person_crops = yolo_detector.detect_persons(image)

    if not person_crops:
        # Fall back to using the full image
        logger.debug("No person detected in %s — using full image.", image_path)
        person_crops = [image]

    # Step 2: MTCNN face extraction from person crops
    for crop in person_crops:
        if face_count >= max_faces:
            break

        if max_faces > 1:
            # Extract all faces from this crop
            faces = face_extractor.extract_all_faces(crop)
        else:
            # Extract only the largest face
            single = face_extractor.extract_face(crop)
            faces = [single] if single is not None else []

        for face in faces:
            if face_count >= max_faces:
                break
            # Save face (RGB → BGR for OpenCV)
            face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            face_path = os.path.join(output_dir, f"face_{face_count:05d}.jpg")
            cv2.imwrite(face_path, face_bgr)
            face_count += 1

    return face_count


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess videos and images for deepfake detection",
    )
    parser.add_argument("--input_dir", type=str, default="data/raw",
                        help="Root directory containing 'real/' and 'fake/' subdirectories.")
    parser.add_argument("--output_dir", type=str, default="data/faces",
                        help="Output root for extracted face crops.")
    parser.add_argument("--sample_rate", type=int, default=5,
                        help="Extract every Nth frame (videos only).")
    parser.add_argument("--max_frames", type=int, default=80,
                        help="Maximum frames per video.")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model weights.")
    parser.add_argument("--yolo_conf", type=float, default=0.5,
                        help="YOLO confidence threshold.")
    parser.add_argument("--include_images", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Include image files in preprocessing (default: True). "
                             "Use --no-include_images to disable.")
    parser.add_argument("--max_faces_per_image", type=int, default=1,
                        help="Maximum faces to extract per image (default: 1).")
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    # Initialize detectors (loaded lazily on first call)
    yolo = YOLOPersonDetector(
        model_name=args.yolo_model,
        confidence_threshold=args.yolo_conf,
    )
    face_ext = FaceExtractor(target_size=(224, 224))

    total_videos = 0
    total_images = 0
    total_faces = 0
    skipped = 0

    for label in ["real", "fake"]:
        label_dir = input_root / label
        if not label_dir.exists():
            logger.warning("Directory not found: %s — skipping.", label_dir)
            continue

        # Collect and classify all files
        all_files = sorted(label_dir.iterdir())
        video_files = [f for f in all_files if f.suffix.lower() in VIDEO_EXTENSIONS]
        image_files = (
            [f for f in all_files if f.suffix.lower() in IMAGE_EXTENSIONS]
            if args.include_images else []
        )

        logger.info(
            "Found %d videos, %d images in %s/%s",
            len(video_files), len(image_files), label_dir, label,
        )

        # ── Process videos ───────────────────────────────────────────────
        for vf in video_files:
            out_dir = output_root / label / vf.stem
            logger.info("  ▸ [VIDEO] %s", vf.name)

            try:
                n_faces = process_video(
                    str(vf),
                    str(out_dir),
                    yolo,
                    face_ext,
                    sample_rate=args.sample_rate,
                    max_frames=args.max_frames,
                )
                total_videos += 1
                total_faces += n_faces
                logger.info("    → %d faces extracted.", n_faces)
            except Exception as exc:
                logger.error("    ✗ Failed to process %s: %s", vf.name, exc)
                skipped += 1

        # ── Process images ───────────────────────────────────────────────
        for img_f in image_files:
            out_dir = output_root / label / img_f.stem
            logger.info("  ▸ [IMAGE] %s", img_f.name)

            try:
                n_faces = process_image(
                    str(img_f),
                    str(out_dir),
                    yolo,
                    face_ext,
                    max_faces=args.max_faces_per_image,
                )
                total_images += 1
                total_faces += n_faces
                logger.info("    → %d faces extracted.", n_faces)
            except Exception as exc:
                logger.error("    ✗ Failed to process %s: %s", img_f.name, exc)
                skipped += 1

        # ── Warn about unsupported files ─────────────────────────────────
        known_exts = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
        for uf in all_files:
            if uf.is_file() and uf.suffix.lower() not in known_exts:
                logger.warning("  ⚠ Unsupported file skipped: %s", uf.name)
                skipped += 1

    logger.info(
        "═══ DONE ═══  %d videos, %d images processed — %d total face crops (%d skipped).",
        total_videos, total_images, total_faces, skipped,
    )


if __name__ == "__main__":
    main()

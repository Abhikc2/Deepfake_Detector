"""
Preprocessing pipeline.

Processes raw video files through the full pipeline:
  Video → Frames → YOLO (person detection) → MTCNN (face extraction) → Saved face crops.

Usage:
    python scripts/preprocess.py \
        --input_dir data/raw \
        --output_dir data/faces \
        --sample_rate 5 \
        --max_frames 80
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


def main():
    parser = argparse.ArgumentParser(description="Preprocess videos for deepfake detection")
    parser.add_argument("--input_dir", type=str, default="data/raw",
                        help="Root directory containing 'real/' and 'fake/' subdirectories with videos.")
    parser.add_argument("--output_dir", type=str, default="data/faces",
                        help="Output root for extracted face crops.")
    parser.add_argument("--sample_rate", type=int, default=5,
                        help="Extract every Nth frame.")
    parser.add_argument("--max_frames", type=int, default=80,
                        help="Maximum frames per video.")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model weights.")
    parser.add_argument("--yolo_conf", type=float, default=0.5,
                        help="YOLO confidence threshold.")
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
    total_faces = 0

    for label in ["real", "fake"]:
        label_dir = input_root / label
        if not label_dir.exists():
            logger.warning("Directory not found: %s — skipping.", label_dir)
            continue

        video_files = sorted([
            f for f in label_dir.iterdir()
            if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        ])

        logger.info("Processing %d %s videos from %s", len(video_files), label, label_dir)

        for vf in video_files:
            video_name = vf.stem
            out_dir = output_root / label / video_name
            logger.info("  ▸ %s", vf.name)

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

    logger.info("═══ DONE ═══  %d videos processed, %d total face crops.", total_videos, total_faces)


if __name__ == "__main__":
    main()

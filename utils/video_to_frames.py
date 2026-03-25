"""
Video-to-frames extraction utility.

Handles video ingestion, validation, and frame sampling using OpenCV.
"""

import os
import cv2
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 200  # Maximum upload size in megabytes


def validate_video_file(video_path: str) -> Tuple[bool, str]:
    """
    Validate a video file for type and size constraints.

    Args:
        video_path: Path to the video file.

    Returns:
        Tuple of (is_valid, message).
    """
    path = Path(video_path)

    if not path.exists():
        return False, f"File not found: {video_path}"

    # Check extension
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, (
            f"File too large ({size_mb:.1f} MB). "
            f"Maximum: {MAX_FILE_SIZE_MB} MB"
        )

    # Try opening with OpenCV to verify it's a valid video
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False, "Cannot open video file — it may be corrupted."

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count < 1:
        return False, "Video contains no readable frames."

    return True, f"Valid video ({frame_count} frames, {size_mb:.1f} MB)"


def extract_frames(
    video_path: str,
    output_dir: Optional[str] = None,
    sample_rate: int = 5,
    max_frames: int = 100,
    target_size: Optional[Tuple[int, int]] = None,
) -> List:
    """
    Extract frames from a video file.

    Args:
        video_path:  Path to the input video.
        output_dir:  Directory to save extracted frames (None = return in memory).
        sample_rate: Extract every Nth frame (default: every 5th).
        max_frames:  Maximum number of frames to extract.
        target_size: Optional (width, height) to resize each frame.

    Returns:
        If output_dir is given  → list of saved file paths.
        If output_dir is None   → list of numpy frames (BGR).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frames = []
    frame_idx = 0
    saved_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        "Extracting frames from %s (%d total, sample_rate=%d, max=%d)",
        video_path, total_frames, sample_rate, max_frames,
    )

    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            if target_size is not None:
                frame = cv2.resize(frame, target_size)

            if output_dir:
                fname = f"frame_{saved_count:05d}.jpg"
                fpath = os.path.join(output_dir, fname)
                cv2.imwrite(fpath, frame)
                frames.append(fpath)
            else:
                frames.append(frame)

            saved_count += 1

        frame_idx += 1

    cap.release()
    logger.info("Extracted %d frames.", saved_count)
    return frames


def get_video_info(video_path: str) -> dict:
    """
    Return basic metadata about a video file.

    Returns dict with keys: fps, frame_count, width, height, duration_sec.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = (
        info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    )

    cap.release()
    return info

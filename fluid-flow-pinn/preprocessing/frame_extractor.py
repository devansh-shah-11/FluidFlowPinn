"""Extract (frame_t, frame_{t+1}) pairs from FDST video clips."""

import os
from pathlib import Path
from typing import Generator, List, Tuple

import cv2
import numpy as np


def extract_frame_pairs(
    video_path: str | Path,
    max_frames: int | None = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Yield consecutive (frame_t, frame_t+1) pairs as uint8 BGR numpy arrays.

    Streams frames one at a time — never loads the full video into RAM.

    Args:
        video_path: Path to a video file.
        max_frames: If set, stop after yielding this many pairs.

    Yields:
        Tuples of (frame_t, frame_t1), each shaped (H, W, 3) uint8 BGR.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    try:
        ret, prev_frame = cap.read()
        if not ret:
            return

        count = 0
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            yield prev_frame, curr_frame
            prev_frame = curr_frame
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        cap.release()


def get_video_metadata(video_path: str | Path) -> dict:
    """Return basic metadata for a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()


def list_fdst_videos(fdst_root: str | Path) -> List[Path]:
    """Return sorted list of all video files under the FDST data root.

    FDST is organized as:
        fdst/
          train/
            <scene_id>/
              video.avi  (or .mp4)
          test/
            <scene_id>/
              video.avi
    Falls back to a flat glob if that structure is not found.
    """
    root = Path(fdst_root)
    videos: List[Path] = []
    for ext in ("*.avi", "*.mp4", "*.AVI", "*.MP4"):
        videos.extend(root.rglob(ext))
    return sorted(videos)

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


def list_fdst_scenes(fdst_root: str | Path, split: str = "train") -> List[Path]:
    """Return sorted list of scene directories for a given FDST split.

    Actual FDST layout on disk:
        <fdst_root>/
          train_data/
            1/   2/   3/ ...   ← scene dirs with flat .jpg + .json files
          test_data/
            1/   2/   3/ ...

    Args:
        fdst_root: Root directory of the FDST dataset.
        split: "train" or "test".

    Returns:
        Sorted list of scene directory Paths.
    """
    root = Path(fdst_root)
    split_dir = root / f"{split}_data"
    if not split_dir.exists():
        # Fallback: try without the _data suffix
        split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"FDST split directory not found: {split_dir}")
    return sorted([d for d in split_dir.iterdir() if d.is_dir()])

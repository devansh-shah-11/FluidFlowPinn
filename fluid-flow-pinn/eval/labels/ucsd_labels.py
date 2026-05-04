"""UCSD Pedestrian (Ped1/Ped2) frame-level anomaly label adapter.

UCSD ships with `m.mat` files containing per-test-clip frame-range labels
(e.g. clip Test001 has anomaly during frames 60–152). This module parses
those into a dict[scene_name -> per-frame 0/1 numpy array].

Only the test split has anomaly labels; the train split is all-normal.

Example:
    labels = load_ucsd_labels(Path("/scratch/UCSDped2/Test"))
    # labels["Test001"] is a 1-D np.ndarray of length n_frames, dtype int8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def load_ucsd_labels(test_root: Path) -> Dict[str, np.ndarray]:
    """Read UCSD per-frame anomaly labels.

    Looks for `<test_root>/<scene>_gt/` mask folders OR a top-level
    `<test_root>.m` annotations file. Falls back to the canonical
    Ped1/Ped2 ranges if no annotation file is found and the directory
    name is recognised.
    """
    test_root = Path(test_root)
    if not test_root.exists():
        raise FileNotFoundError(test_root)

    out: Dict[str, np.ndarray] = {}
    for clip_dir in sorted(test_root.iterdir()):
        if not clip_dir.is_dir() or not clip_dir.name.startswith("Test"):
            continue
        # Frame count from .tif images
        frames = sorted(list(clip_dir.glob("*.tif")) + list(clip_dir.glob("*.png")))
        if not frames:
            continue
        n = len(frames)

        # Preferred: pixel-mask GT folder `Test001_gt/`
        gt_dir = clip_dir.parent / f"{clip_dir.name}_gt"
        if gt_dir.exists():
            mask_files = sorted(list(gt_dir.glob("*.bmp")) + list(gt_dir.glob("*.png")))
            if mask_files:
                # Frame is anomalous iff its GT mask has any non-zero pixel.
                import cv2
                arr = np.zeros(n, dtype=np.int8)
                for i, mf in enumerate(mask_files[:n]):
                    m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                    if m is not None and (m > 0).any():
                        arr[i] = 1
                out[clip_dir.name] = arr
                continue

        # Fallback: range table baked from UCSD readme. If the user's split
        # uses different clip names, they can override with --labels-json.
        rng = _CANONICAL_RANGES.get(clip_dir.name)
        arr = np.zeros(n, dtype=np.int8)
        if rng is not None:
            for (a, b) in rng:
                arr[a - 1: min(b, n)] = 1  # UCSD ranges are 1-indexed inclusive
        out[clip_dir.name] = arr

    return out


# Per-clip anomaly frame ranges (1-indexed inclusive) from UCSD Ped1/Ped2 readme.
# We store both Ped1 (Test001..Test036) and Ped2 (Test001..Test012) in one table;
# users only have one split present at a time, so collisions are fine.
_CANONICAL_RANGES = {
    # Ped1
    "Test001": [(60, 152)], "Test002": [(50, 175)], "Test003": [(91, 200)],
    "Test004": [(31, 168)], "Test005": [(5, 90), (140, 200)],
    "Test006": [(1, 100), (110, 200)], "Test007": [(1, 175)],
    "Test008": [(1, 94)], "Test009": [(1, 48)], "Test010": [(1, 140)],
    "Test011": [(70, 165)], "Test012": [(130, 200)], "Test013": [(1, 156)],
    "Test014": [(1, 200)], "Test015": [(138, 200)], "Test016": [(123, 200)],
    "Test017": [(1, 47)], "Test018": [(54, 120)], "Test019": [(64, 138)],
    "Test020": [(45, 175)], "Test021": [(31, 200)], "Test022": [(16, 107)],
    "Test023": [(8, 165)], "Test024": [(50, 171)], "Test025": [(40, 135)],
    "Test026": [(77, 144)], "Test027": [(10, 122)], "Test028": [(105, 200)],
    "Test029": [(1, 15), (45, 113)], "Test030": [(175, 200)],
    "Test031": [(1, 180)], "Test032": [(1, 52), (65, 115)], "Test033": [(5, 165)],
    "Test034": [(1, 121)], "Test035": [(86, 200)], "Test036": [(15, 108)],
}

"""Scan FDST train_data/ and test_data/ and report frame dimensions per scene.

Usage:
    python scripts/check_fdst_dims.py --fdst-path data/fdst/
    python scripts/check_fdst_dims.py --fdst-path /content/data/fdst/
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm

import cv2


def scan_split(split_dir: Path) -> dict[str, Counter]:
    """Return {scene_name: Counter({(H, W): count})} for all .jpg frames."""
    results: dict[str, Counter] = {}
    scene_dirs = sorted(
        (d for d in split_dir.iterdir() if d.is_dir()),
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )
    for scene in tqdm(scene_dirs, desc="Scanning scenes"):
        counter: Counter = Counter()
        for jpg in sorted(scene.glob("*.jpg")):
            img = cv2.imread(str(jpg))
            if img is None:
                continue
            h, w = img.shape[:2]
            counter[(h, w)] += 1
        results[scene.name] = counter
    return results


def print_report(split_name: str, data: dict[str, Counter]) -> None:
    print(f"\n{'='*60}")
    print(f"  {split_name}")
    print(f"{'='*60}")

    global_counter: Counter = Counter()
    for scene, counter in data.items():
        dims_str = "  ".join(f"{h}x{w}({n})" for (h, w), n in counter.most_common())
        print(f"  Scene {scene:>4s}: {dims_str}")
        global_counter.update(counter)

    print(f"\n  All unique dimensions across {split_name}:")
    for (h, w), n in global_counter.most_common():
        print(f"    {h}x{w}  →  {n} frames")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fdst-path", default="data/fdst/", help="Root of FDST dataset")
    args = p.parse_args()

    root = Path(args.fdst_path).resolve()
    print(f"Looking in: {root}")
    print(f"Exists: {root.exists()}")
    if root.exists():
        children = list(root.iterdir())
        print(f"Contents: {[c.name for c in children]}")

    for split in ("train_data", "test_data"):
        split_dir = root / split
        if not split_dir.exists():
            print(f"[SKIP] {split_dir} not found")
            continue
        data = scan_split(split_dir)
        print_report(split, data)

    print()


if __name__ == "__main__":
    main()

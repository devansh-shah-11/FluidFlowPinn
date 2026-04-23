"""Split UMN .avi into 3 scene .avi files using hardcoded frame indices."""

from pathlib import Path

import cv2

# Scene boundaries (inclusive start, exclusive end).
# scene3 end=-1 means "to the last frame".
UMN_SCENE_INDICES = {
    "scene1": (0, 641),
    "scene2": (641, 1001),
    "scene3": (1001, -1),
}


def split_umn(
    avi_path: str | Path,
    output_dir: str | Path,
    scene_indices: dict | None = None,
) -> dict[str, Path]:
    """Split a single UMN .avi file into one .avi per scene.

    Args:
        avi_path: Path to the original UMN video file.
        output_dir: Directory where scene files are written.
        scene_indices: Dict mapping scene name → (start_frame, end_frame).
                       end_frame=-1 means until the last frame.
                       Defaults to UMN_SCENE_INDICES.

    Returns:
        Dict mapping scene name → output Path for each written file.
    """
    if scene_indices is None:
        scene_indices = UMN_SCENE_INDICES

    avi_path = Path(avi_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open UMN video: {avi_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    output_paths: dict[str, Path] = {}

    for scene_name, (start, end) in scene_indices.items():
        if end == -1:
            end = total_frames

        out_path = output_dir / f"{scene_name}.avi"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(end - start):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()

        output_paths[scene_name] = out_path
        print(f"  {scene_name}: frames {start}–{end - 1} → {out_path}")

    cap.release()
    return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split UMN .avi into scene clips")
    parser.add_argument("avi_path", help="Path to original UMN .avi")
    parser.add_argument("output_dir", help="Directory for output scene files")
    args = parser.parse_args()

    paths = split_umn(args.avi_path, args.output_dir)
    for name, p in paths.items():
        print(f"{name}: {p}")

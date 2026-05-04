"""Score every frame of UMN or UCSD with the inference pipeline.

Output: a CSV at `results/<dataset>_<config_tag>.csv` with one row per frame:
    scene, frame, p_max, label

`compute_metrics.py` consumes this CSV to report AUROC / AUPRC / lead-time.

Examples:
    # UMN, full (α=β=γ=1) config
    python eval/run_anomaly.py --dataset umn --umn-path data/UMN/ \\
        --p-alpha 1 --p-beta 1 --p-gamma 1 --tag full

    # UCSD Ped2, density-only baseline
    python eval/run_anomaly.py --dataset ucsd --ucsd-path /scratch/UCSDped2/Test \\
        --p-alpha 1 --p-beta 0 --p-gamma 0 --tag density_only \\
        --lwcc-model DM-Count
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


# ── Frame iterators ─────────────────────────────────────────────────────────

def _iter_umn(umn_root: Path) -> Iterator[Tuple[str, int, np.ndarray]]:
    """Yield (scene_name, frame_idx, bgr_frame) for every UMN scene clip."""
    videos = sorted(umn_root.glob("scene*.avi"))
    if not videos:
        videos = sorted(umn_root.glob("*.avi"))
    if not videos:
        raise FileNotFoundError(f"No .avi files under {umn_root}")
    for vpath in videos:
        cap = cv2.VideoCapture(str(vpath))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield vpath.stem, i, frame
            i += 1
        cap.release()


def _iter_ucsd(ucsd_test_root: Path) -> Iterator[Tuple[str, int, np.ndarray]]:
    """Yield (clip_name, frame_idx, bgr_frame) for every UCSD test clip."""
    clips = sorted([d for d in ucsd_test_root.iterdir()
                    if d.is_dir() and d.name.startswith("Test") and not d.name.endswith("_gt")])
    if not clips:
        raise FileNotFoundError(f"No Test* dirs under {ucsd_test_root}")
    for clip in clips:
        frames = sorted(list(clip.glob("*.tif")) + list(clip.glob("*.png")))
        for i, fp in enumerate(frames):
            img = cv2.imread(str(fp))
            if img is None:
                continue
            # UCSD frames are grayscale .tif; cv2.imread returns 3-channel BGR.
            yield clip.name, i, img


# ── Label resolvers ─────────────────────────────────────────────────────────

def _umn_labels(umn_labels_json: Path) -> dict[str, dict]:
    with open(umn_labels_json) as f:
        return json.load(f)["scenes"]


def _label_for(dataset: str, scene: str, frame_idx: int,
               umn_table: dict | None, ucsd_table: dict | None) -> int:
    if dataset == "umn":
        rec = umn_table.get(scene)
        if rec is None:
            return 0
        if frame_idx < rec["onset_frame"]:
            return 0
        if rec["end_frame"] != -1 and frame_idx >= rec["end_frame"]:
            return 0
        return 1
    if dataset == "ucsd":
        arr = ucsd_table.get(scene)
        if arr is None or frame_idx >= len(arr):
            return 0
        return int(arr[frame_idx])
    raise ValueError(dataset)


# ── Main ────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from pipeline import Pipeline, default_pipeline_args

    pipe_args = default_pipeline_args(
        raft_variant=args.raft_variant,
        raft_iters=args.raft_iters,
        lwcc_model=args.lwcc_model,
        lwcc_weights=args.lwcc_weights,
        detector=args.detector,
        detector_conf=args.detector_conf,
        center_radius=args.center_radius,
        density_mask_tau=args.density_mask_tau,
        flow_spatial_sigma=args.flow_spatial_sigma,
        flow_ema_alpha=args.flow_ema_alpha,
        p_alpha=args.p_alpha, p_beta=args.p_beta, p_gamma=args.p_gamma,
        var_kernel=args.var_kernel,
        p_agg=args.p_agg, topk_frac=args.topk_frac,
    )
    pipe = Pipeline(pipe_args)

    # Pick frame iterator + label table
    umn_tbl = ucsd_tbl = None
    if args.dataset == "umn":
        umn_tbl = _umn_labels(Path(args.umn_labels_json))
        frame_iter = _iter_umn(Path(args.umn_path))
    else:
        from eval.labels.ucsd_labels import load_ucsd_labels
        ucsd_tbl = load_ucsd_labels(Path(args.ucsd_path))
        frame_iter = _iter_ucsd(Path(args.ucsd_path))

    # Pipeline keeps state across consecutive frames; reset between scenes.
    out_path = Path(args.out_csv) if args.out_csv else REPO_ROOT / "results" / f"{args.dataset}_{args.tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    last_scene: str | None = None
    inf_w = (args.width // 8) * 8
    inf_h = (args.height // 8) * 8

    with open(out_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["scene", "frame", "p_max", "label"])
        for scene, frame_idx, bgr in frame_iter:
            if scene != last_scene:
                pipe.prev_tensor = None
                pipe.u_ema = None
                last_scene = scene
                log.info("Scene: %s", scene)
            bgr_resized = cv2.resize(bgr, (inf_w, inf_h))
            res = pipe.process(bgr_resized)
            if res["p_max"] is None:
                continue  # first frame of each scene has no flow
            label = _label_for(args.dataset, scene, frame_idx, umn_tbl, ucsd_tbl)
            w.writerow([scene, frame_idx, f"{res['p_max']:.6f}", label])
            rows_written += 1
            if rows_written % 200 == 0:
                log.info("  wrote %d rows", rows_written)

    pipe.close()
    log.info("Done — %d rows → %s", rows_written, out_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score frames of a labeled video dataset")
    p.add_argument("--dataset", choices=("umn", "ucsd"), required=True)
    p.add_argument("--umn-path",  default=str(REPO_ROOT / "data" / "umn"))
    p.add_argument("--umn-labels-json",
                   default=str(REPO_ROOT / "eval" / "labels" / "umn_labels.json"))
    p.add_argument("--ucsd-path", default=None,
                   help="UCSD Test directory (e.g. /scratch/UCSDped2/Test)")
    p.add_argument("--out-csv", default=None)
    p.add_argument("--tag", default="run", help="Short label baked into the CSV filename")
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    # Pipeline knobs
    p.add_argument("--raft-variant", choices=("small", "large"), default="small")
    p.add_argument("--raft-iters", type=int, default=6)
    p.add_argument("--lwcc-model", default=None)
    p.add_argument("--lwcc-weights", default="SHA")
    p.add_argument("--detector", default=None)
    p.add_argument("--detector-conf", type=float, default=0.25)
    p.add_argument("--center-radius", type=int, default=1)
    p.add_argument("--density-mask-tau", type=float, default=0.01)
    p.add_argument("--flow-spatial-sigma", type=float, default=0.0)
    p.add_argument("--flow-ema-alpha", type=float, default=1.0)
    p.add_argument("--p-alpha", type=float, default=1.0)
    p.add_argument("--p-beta",  type=float, default=1.0)
    p.add_argument("--p-gamma", type=float, default=1.0)
    p.add_argument("--var-kernel", type=int, default=5)
    p.add_argument("--p-agg", choices=("max", "topk", "weighted_mean"), default="max")
    p.add_argument("--topk-frac", type=float, default=0.001)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dataset == "ucsd" and not args.ucsd_path:
        log.error("--ucsd-path required for --dataset ucsd")
        sys.exit(2)
    run(args)

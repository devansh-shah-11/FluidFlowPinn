"""Component-validation evaluator for the inference pipeline.

The project no longer trains an end-to-end PINN; the per-frame compute lives in
`pipeline.Pipeline`. This script validates the *components* against benchmarks
that already have ground truth:

    --component density   FDST test split (per-frame head count) using lwcc.
                          Reports count MAE / RMSE / MAPE.
    --component count     YOLO-only count vs FDST GT (no lwcc).
    --component flow      CrowdFlow synthetic GT flow EPE for the RAFT branch.

Block B (anomaly detection on UMN/UCSD) lives under `eval/run_anomaly.py`.

Examples:
    python evaluate.py --component density --fdst-path /scratch/dns5508/FDST_Data/ \\
                       --lwcc-model DM-Count
    python evaluate.py --component count   --fdst-path /scratch/dns5508/FDST_Data/ \\
                       --detector yolo26n.pt
    python evaluate.py --component flow    --crowdflow-path /scratch/.../CrowdFlow/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Density / count on FDST ─────────────────────────────────────────────────

def _eval_fdst(args: argparse.Namespace) -> None:
    """Iterate FDST test split, run Pipeline on each frame pair, report count MAE."""
    sys.path.insert(0, str(Path(__file__).parent))
    from pipeline import Pipeline, default_pipeline_args
    from preprocessing.dataset_loader import FDSTDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from utils.metrics import count_mae, count_mape, count_rmse

    fdst_root = Path(args.fdst_path)
    _MEAN = [0.485, 0.456, 0.406]; _STD = [0.229, 0.224, 0.225]
    xform = T.Compose([T.ToTensor(), T.Normalize(mean=_MEAN, std=_STD)])
    dataset = FDSTDataset(root=fdst_root, split="test", transform=xform)

    log.info("FDST test set: %d samples", len(dataset))

    pipe_args = default_pipeline_args(
        lwcc_model=args.lwcc_model if args.component == "density" else None,
        lwcc_weights=args.lwcc_weights,
        detector=args.detector if args.component == "count" else None,
        detector_conf=args.detector_conf,
        center_radius=args.center_radius,
        # Pure-component mode: don't gate by mask; we want raw rho counts.
        density_mask_tau=0.0,
    )
    pipe = Pipeline(pipe_args)

    pred_counts: list[float] = []
    gt_counts:   list[float] = []

    # Directly iterate underlying samples so we can read frame_t as a BGR image
    # for the pipeline (which expects BGR np.ndarray, not the normalised tensor).
    for i, sample in enumerate(dataset._samples):
        bgr = cv2.imread(sample["frame_t"])
        if bgr is None:
            continue
        # Need a "previous" frame for the pipeline, but for density we only
        # need rho — feed the same frame twice (u will be ~0, ignored).
        _ = pipe.process(bgr)            # primes prev_tensor
        out = pipe.process(bgr)          # rho available now
        rho = out["rho"]
        if rho is None:
            continue
        pred_counts.append(float(rho.sum()))

        # GT: density map shipped by FDSTDataset integrates to head count.
        item = dataset[i]
        gt_counts.append(float(item["density_map"].sum().item()))
        # Reset prev_tensor so each sample is independent.
        pipe.prev_tensor = None
        pipe.u_ema = None

        if (i + 1) % 50 == 0:
            log.info("  %d / %d", i + 1, len(dataset._samples))

    pipe.close()

    log.info("─" * 56)
    log.info("FDST  |  MAE=%.2f  RMSE=%.2f  MAPE=%.1f%%  (n=%d)",
             count_mae(pred_counts, gt_counts),
             count_rmse(pred_counts, gt_counts),
             count_mape(pred_counts, gt_counts),
             len(pred_counts))
    log.info("─" * 56)


# ── Flow EPE on CrowdFlow ───────────────────────────────────────────────────

def _eval_crowdflow(args: argparse.Namespace) -> None:
    """Run RAFT on CrowdFlow sequences and compare to GT optical flow."""
    sys.path.insert(0, str(Path(__file__).parent))
    from pipeline import Pipeline, default_pipeline_args
    from utils.metrics import flow_epe

    cf_root = Path(args.crowdflow_path)
    if not cf_root.exists():
        log.error("CrowdFlow path not found: %s", cf_root)
        sys.exit(1)

    # CrowdFlow layout: <root>/IM01/frame_*.png + <root>/IM01_gt/flow_*.flo
    # We accept any layout: walk for paired (frame_t, frame_t+1, flow.flo).
    pairs = _discover_crowdflow_pairs(cf_root)
    if not pairs:
        log.error("No (frame, frame+1, flow.flo) triples found under %s", cf_root)
        sys.exit(1)
    log.info("CrowdFlow: %d frame pairs with GT flow", len(pairs))

    pipe_args = default_pipeline_args()  # plain RAFT, no density
    pipe = Pipeline(pipe_args)

    epes: list[float] = []
    for i, (f0, f1, flo) in enumerate(pairs):
        bgr0 = cv2.imread(str(f0)); bgr1 = cv2.imread(str(f1))
        if bgr0 is None or bgr1 is None:
            continue
        # Resize to multiples of 8 for RAFT
        h, w = bgr0.shape[:2]; h8 = (h // 8) * 8; w8 = (w // 8) * 8
        bgr0 = cv2.resize(bgr0, (w8, h8)); bgr1 = cv2.resize(bgr1, (w8, h8))
        pipe.prev_tensor = None; pipe.u_ema = None
        _ = pipe.process(bgr0); out = pipe.process(bgr1)
        u_pred = out["u"]  # (2, h8/8, w8/8) — grid units
        if u_pred is None:
            continue
        gt = _read_flo(flo)  # (2, H, W) in pixel units
        # Upsample prediction from H/8 grid to GT resolution and convert to pixels.
        Hp, Wp = gt.shape[1], gt.shape[2]
        u_up = np.stack([
            cv2.resize(u_pred[0], (Wp, Hp), interpolation=cv2.INTER_LINEAR) * 8.0,
            cv2.resize(u_pred[1], (Wp, Hp), interpolation=cv2.INTER_LINEAR) * 8.0,
        ], axis=0)
        epes.append(flow_epe(u_up, gt))
        if (i + 1) % 20 == 0:
            log.info("  %d / %d  EPE=%.3f", i + 1, len(pairs), float(np.mean(epes)))

    pipe.close()
    log.info("─" * 56)
    log.info("CrowdFlow  |  mean EPE = %.3f px  (n=%d)", float(np.mean(epes)), len(epes))
    log.info("─" * 56)


def _discover_crowdflow_pairs(root: Path) -> list[tuple[Path, Path, Path]]:
    """Best-effort scan for (frame_t, frame_t+1, flow.flo) triples.

    Tolerates a few common layouts because CrowdFlow distributions vary.
    """
    pairs: list[tuple[Path, Path, Path]] = []
    seqs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    for seq in seqs:
        frames = sorted([p for p in seq.glob("*.png")] + [p for p in seq.glob("*.jpg")])
        flo_dir = seq.parent / f"{seq.name}_gt"
        if not flo_dir.exists():
            flo_dir = seq / "flow"
        if not flo_dir.exists():
            continue
        flos = sorted(flo_dir.glob("*.flo"))
        if len(frames) < 2 or len(flos) == 0:
            continue
        for i, flo in enumerate(flos):
            if i + 1 >= len(frames):
                break
            pairs.append((frames[i], frames[i + 1], flo))
    return pairs


def _read_flo(path: Path) -> np.ndarray:
    """Middlebury .flo reader. Returns (2, H, W) float32 (u, v) in pixels."""
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic[0] != 202021.25:
            raise IOError(f"Not a .flo file: {path}")
        w = int(np.fromfile(f, np.int32, count=1)[0])
        h = int(np.fromfile(f, np.int32, count=1)[0])
        data = np.fromfile(f, np.float32, count=2 * w * h).reshape(h, w, 2)
    return data.transpose(2, 0, 1).astype(np.float32)


# ── Entry point ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Component validation for the inference pipeline")
    p.add_argument("--component", choices=("density", "count", "flow"), required=True,
                   help="density: lwcc on FDST. count: YOLO on FDST. flow: RAFT on CrowdFlow.")
    p.add_argument("--fdst-path", default=None, help="FDST root (for density/count modes)")
    p.add_argument("--crowdflow-path", default=None, help="CrowdFlow root (for flow mode)")
    p.add_argument("--lwcc-model", default="DM-Count")
    p.add_argument("--lwcc-weights", default="SHA")
    p.add_argument("--detector", default="yolo26n.pt")
    p.add_argument("--detector-conf", type=float, default=0.25)
    p.add_argument("--center-radius", type=int, default=-1,
                   help="For count mode, default to full bbox so density sums to N people.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    import os
    os.chdir(Path(__file__).parent)
    if args.component in ("density", "count"):
        if not args.fdst_path:
            log.error("--fdst-path required for component=%s", args.component)
            sys.exit(2)
        _eval_fdst(args)
    elif args.component == "flow":
        if not args.crowdflow_path:
            log.error("--crowdflow-path required for component=flow")
            sys.exit(2)
        _eval_crowdflow(args)

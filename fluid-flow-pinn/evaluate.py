"""Evaluation script — run on FDST test split (and optionally ShanghaiTech/CrowdFlow later).

Usage:
    # Quantitative metrics on FDST test split
    python evaluate.py --checkpoint checkpoints/best.pt --fdst-path /scratch/dns5508/FDST_Data/

    # + save visualizations for the first N batches
    python evaluate.py \
    --checkpoint /scratch/dns5508/FluidFlowPinn/checkpoints/best.pt \
    --fdst-path /scratch/dns5508/FDST_Data/ \
    --vis-dir /scratch/dns5508/FluidFlowPinn/outputs/vis/ \
    --vis-batches 5

    python evaluate.py \
    --checkpoint /scratch/dns5508/FluidFlowPinn/checkpoints/best.pt \
    --fdst-path /scratch/dns5508/FDST_Data/ \
    --timeline \
    --timeline-scene 10 \
    --vis-dir /scratch/dns5508/FluidFlowPinn/outputs/vis/


    python evaluate.py --checkpoint checkpoints/best.pt --fdst-path /scratch/.../FDST_Data/ --vis-dir outputs/vis/ --vis-batches 5

    # Pressure timeline on a single scene folder (anomaly detection demo)
    python evaluate.py --checkpoint checkpoints/best.pt --fdst-path /scratch/.../FDST_Data/ --timeline --timeline-scene 10
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # no display available in SLURM — must be set before any other matplotlib import

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms as T

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(
    pred_counts: list[float],
    gt_counts: list[float],
) -> dict[str, float]:
    import numpy as np
    p = np.array(pred_counts)
    g = np.array(gt_counts)
    mae  = float(np.abs(p - g).mean())
    mse  = float(np.sqrt(((p - g) ** 2).mean()))
    # Mean Absolute Percentage Error (guard against zero GT)
    mape_mask = g > 0
    mape = float(np.abs((p[mape_mask] - g[mape_mask]) / g[mape_mask]).mean() * 100) if mape_mask.any() else float("nan")
    return {"MAE": mae, "RMSE": mse, "MAPE%": mape}


# ── Single-scene pressure timeline ────────────────────────────────────────────

@torch.no_grad()
def _pressure_timeline(
    model,
    scene_dir: Path,
    device: torch.device,
    fps: float,
    threshold: float,
    use_fp16: bool,
    save_path: Path | None,
) -> None:
    import cv2
    from utils.visualize import plot_pressure_timeline

    frame_files = sorted(scene_dir.glob("*.jpg"), key=lambda p: p.stem)
    if len(frame_files) < 2:
        log.error("Scene %s has fewer than 2 frames.", scene_dir)
        return

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]
    xform = T.Compose([T.ToTensor(), T.Normalize(mean=_MEAN, std=_STD)])

    def _load(path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return xform(img).unsqueeze(0).to(device)

    model.eval()
    scores = []
    for i in range(len(frame_files) - 1):
        ft  = _load(frame_files[i])
        ft1 = _load(frame_files[i + 1])
        with autocast(device_type=device.type, enabled=use_fp16):
            out = model(ft, ft1)
        p_max = out["P"].max().item()
        scores.append(p_max)
        if (i + 1) % 50 == 0:
            log.info("  frame %d / %d  max_P=%.4f", i + 1, len(frame_files) - 1, p_max)

    fig = plot_pressure_timeline(
        scores, fps=fps, threshold=threshold, save_path=save_path
    )
    log.info("Pressure timeline: %d frames, max=%.4f, mean=%.4f",
             len(scores), max(scores), sum(scores) / len(scores))
    if save_path:
        log.info("Saved timeline → %s", save_path)
    else:
        log.info("No --vis-dir specified; timeline not saved.")


# ── Main evaluation loop ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt.get("config", {})

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = cfg.get("model", {}).get("use_fp16", True) and device.type == "cuda"
    log.info("Device: %s  FP16: %s", device, use_fp16)

    # ── Build model ───────────────────────────────────────────────────────────
    repo_root = Path(__file__).parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from models.pinn import FluidFlowPINN
    model = FluidFlowPINN(
        use_grad_checkpoint=False,   # not needed for inference
        raft_frozen=True,
        pressure_window=cfg.get("model", {}).get("pressure_window", 5),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("Model loaded (epoch %d, val_loss=%.4f)",
             ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan")))

    fps       = float(cfg.get("preprocessing", {}).get("fps", 30))
    threshold = float(args.threshold)

    # ── Pressure timeline mode ────────────────────────────────────────────────
    if args.timeline:
        from preprocessing.dataset_loader import FDSTDataset
        fdst_root = Path(args.fdst_path)
        test_dir  = fdst_root / "test_data"
        if not test_dir.exists():
            test_dir = fdst_root / "test"

        scene_name = str(args.timeline_scene)
        scene_dir  = test_dir / scene_name
        if not scene_dir.exists():
            # list available scenes
            available = [d.name for d in sorted(test_dir.iterdir()) if d.is_dir()]
            log.error("Scene '%s' not found. Available: %s", scene_name, available)
            sys.exit(1)

        save_path = Path(args.vis_dir) / f"timeline_scene{scene_name}.png" if args.vis_dir else None
        _pressure_timeline(model, scene_dir, device, fps, threshold, use_fp16, save_path)
        return

    # ── Quantitative evaluation on FDST test split ────────────────────────────
    from preprocessing.dataset_loader import FDSTDataset

    target_h = cfg.get("preprocessing", {}).get("target_height", 720)
    target_w = cfg.get("preprocessing", {}).get("target_width",  1280)
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]
    xform = T.Compose([T.ToTensor(), T.Normalize(mean=_MEAN, std=_STD)])

    fdst_root = Path(args.fdst_path)
    dataset   = FDSTDataset(root=fdst_root, split="test", transform=xform)

    # Filter to target resolution (both frames)
    import cv2

    def _res_ok(path):
        img = cv2.imread(path)
        return img is not None and img.shape[:2] == (target_h, target_w)

    n_before = len(dataset._samples)
    dataset._samples = [s for s in dataset._samples
                        if _res_ok(s["frame_t"]) and _res_ok(s["frame_t1"])]
    # Rebuild scene_ranges
    dataset.scene_ranges = {}
    for i, s in enumerate(dataset._samples):
        sc = s["scene"]
        if sc not in dataset.scene_ranges:
            dataset.scene_ranges[sc] = (i, i)
        dataset.scene_ranges[sc] = (dataset.scene_ranges[sc][0], i + 1)
    log.info("Test set: %d / %d samples at %dx%d", len(dataset), n_before, target_h, target_w)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    pred_counts, gt_counts = [], []
    pressure_maxes = []
    vis_saved = 0

    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)
        from utils.visualize import visualize_batch

    for batch_idx, batch in enumerate(loader):
        ft  = batch["frame_t"].to(device, non_blocking=True)
        ft1 = batch["frame_t1"].to(device, non_blocking=True)
        gt  = batch["density_map"].sum(dim=(1, 2, 3)).to(device)

        with autocast(device_type=device.type, enabled=use_fp16):
            out = model(ft, ft1)

        pred = out["rho"].sum(dim=(1, 2, 3))
        pred_counts.extend(pred.cpu().tolist())
        gt_counts.extend(gt.cpu().tolist())
        pressure_maxes.extend(out["P"].amax(dim=(1, 2, 3)).cpu().tolist())

        # Save visualizations for the first --vis-batches batches
        if vis_dir and vis_saved < args.vis_batches:
            figs = visualize_batch(
                frames=ft.cpu(),
                rhos=out["rho"].cpu(),
                us=out["u"].cpu(),
                Ps=out["P"].cpu(),
                gt_densities=batch["density_map"],
                save_dir=vis_dir,
                prefix=f"batch{batch_idx:04d}",
            )
            import matplotlib.pyplot as plt
            for fig in figs:
                plt.close(fig)
            vis_saved += 1
            log.info("Saved visualizations for batch %d → %s", batch_idx, vis_dir)

        if (batch_idx + 1) % 20 == 0:
            log.info("  batch %d / %d", batch_idx + 1, len(loader))

    # ── Print metrics ─────────────────────────────────────────────────────────
    metrics = _compute_metrics(pred_counts, gt_counts)
    log.info("─" * 50)
    log.info("FDST test  |  MAE=%.2f  RMSE=%.2f  MAPE=%.1f%%",
             metrics["MAE"], metrics["RMSE"], metrics["MAPE%"])
    log.info("Pressure   |  mean_max=%.4f  overall_max=%.4f",
             sum(pressure_maxes) / len(pressure_maxes), max(pressure_maxes))
    log.info("─" * 50)

    # Also save a summary timeline of max pressure across the whole test set
    if vis_dir:
        from utils.visualize import plot_pressure_timeline
        import matplotlib.pyplot as plt
        fig = plot_pressure_timeline(
            pressure_maxes,
            fps=fps,
            threshold=threshold,
            save_path=vis_dir / "pressure_timeline_testset.png",
        )
        plt.close(fig)
        log.info("Saved pressure timeline → %s", vis_dir / "pressure_timeline_testset.png")


# ── Entry point ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fluid-flow PINN")
    p.add_argument("--checkpoint",      required=True,         help="Path to .pt checkpoint (best.pt)")
    p.add_argument("--fdst-path",       required=True,         help="Path to FDST dataset root")
    p.add_argument("--batch-size",      type=int, default=4,   help="Eval batch size")
    p.add_argument("--threshold",       type=float, default=0.5, help="Pressure alert threshold")
    p.add_argument("--vis-dir",         default=None,          help="Directory to save visualizations")
    p.add_argument("--vis-batches",     type=int, default=5,   help="Number of batches to visualize")
    p.add_argument("--timeline",        action="store_true",   help="Run pressure timeline on a single scene")
    p.add_argument("--timeline-scene",  default="10",          help="Scene folder name for --timeline mode")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    import os
    os.chdir(Path(__file__).parent)
    evaluate(args)

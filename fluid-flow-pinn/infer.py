"""Realtime inference + dashboard for the fluid-flow PINN.

Takes a video file (or single image) and produces:
  • An annotated MP4 with input frame + ρ overlay + P overlay + rolling P(t) plot.
  • A final pressure-timeline PNG.
  • Optionally, a live OpenCV window (`--display`) for laptop demos / webcam.

Modes:
  • Video file:   --source path/to/clip.mp4
  • Webcam:       --source 0     (any integer index → cv2.VideoCapture(int))
  • Single image: --source path/to/img.jpg   (only ρ is shown; P needs two frames)

Examples:
    # Headless: process a clip, save annotated mp4 + timeline png
    python infer.py --checkpoint checkpoints/best.pt --source clip.mp4 \\
        --out-dir outputs/infer_run/

    # With live window
    python infer.py --checkpoint checkpoints/best.pt --source clip.mp4 \\
        --out-dir outputs/infer_run/ --display

    # Webcam, live only (set out-dir if you also want the recording)
    python infer.py --checkpoint checkpoints/best.pt --source 0 --display
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torchvision import transforms as T


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ── Frame I/O ────────────────────────────────────────────────────────────────

class _Source:
    """Unifies video files, webcam ints, and single images behind one iterator."""

    def __init__(self, source: str | int):
        self.is_image = False
        self.cap: Optional[cv2.VideoCapture] = None
        self._image: Optional[np.ndarray] = None
        self.fps = 30.0
        self.width = 0
        self.height = 0

        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            idx = int(source)
            self.cap = cv2.VideoCapture(idx)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open webcam index {idx}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = -1  # unknown
            return

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"Could not read image {path}")
            self.is_image = True
            self._image = img
            self.height, self.width = img.shape[:2]
            self.total_frames = 1
            return

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None at end-of-stream."""
        if self.is_image:
            img, self._image = self._image, None
            return img
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


# ── Model wrapper ────────────────────────────────────────────────────────────

def _load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    """Load the PINN from a training checkpoint. Returns (model, cfg, use_fp16)."""
    repo_root = Path(__file__).parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from models.pinn import FluidFlowPINN

    log.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt.get("config", {}) or {}

    use_fp16 = cfg.get("model", {}).get("use_fp16", True) and device.type == "cuda"
    model = FluidFlowPINN(
        use_grad_checkpoint=False,
        raft_frozen=True,
        pressure_window=cfg.get("model", {}).get("pressure_window", 5),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("Model loaded (epoch %s, val_loss=%.4f) | device=%s fp16=%s",
             ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan")),
             device, use_fp16)
    return model, cfg, use_fp16


def _bgr_to_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """BGR uint8 (H,W,3) → ImageNet-normalised float tensor (1,3,H,W)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std  = torch.tensor(_STD).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0).to(device, non_blocking=True)


# ── Visualization helpers ────────────────────────────────────────────────────

def _heatmap_overlay(
    frame_bgr: np.ndarray,
    field: np.ndarray,
    cmap: int = cv2.COLORMAP_JET,
    alpha: float = 0.55,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Upsample `field` (H',W') to frame size, color-map, blend on top of frame."""
    h, w = frame_bgr.shape[:2]
    f = field.astype(np.float32)
    if vmin is None: vmin = float(f.min())
    if vmax is None: vmax = float(f.max())
    if vmax - vmin < 1e-8:
        norm = np.zeros_like(f, dtype=np.uint8)
    else:
        norm = np.clip((f - vmin) / (vmax - vmin), 0.0, 1.0)
        norm = (norm * 255).astype(np.uint8)
    norm = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(norm, cmap)
    return cv2.addWeighted(colored, alpha, frame_bgr, 1.0 - alpha, 0.0)


def _label(img: np.ndarray, text: str, y: int = 26) -> None:
    """Draw white-on-black label in top-left of img (in place)."""
    cv2.rectangle(img, (0, y - 22), (10 + 9 * len(text), y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)


def _render_timeline_strip(
    history: deque[float],
    threshold: float,
    width: int,
    height: int = 160,
) -> np.ndarray:
    """Render a small live P(t) plot as a BGR image of given size."""
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111)
    if history:
        xs = np.arange(len(history))
        ys = np.array(history, dtype=np.float32)
        ax.plot(xs, ys, color="steelblue", linewidth=1.2)
        ax.set_xlim(0, max(history.maxlen, len(history)))
        ymax = max(threshold * 2.0, float(ys.max()) * 1.1, 1e-3)
        ax.set_ylim(0, ymax)
    else:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0)
    ax.set_title("max P(t) — recent window", fontsize=9)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.4)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())  # (H, W, 4) RGBA
    plt.close(fig)
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    return cv2.resize(bgr, (width, height))


def _build_dashboard(
    frame_bgr: np.ndarray,
    rho_np: Optional[np.ndarray],
    P_np: Optional[np.ndarray],
    history: deque[float],
    threshold: float,
    fps_actual: float,
    frame_idx: int,
    p_max: Optional[float],
) -> np.ndarray:
    """Compose the live dashboard image (BGR uint8).

    Layout (rows × cols):
        ┌──────────┬──────────┐
        │   raw    │ ρ overlay│
        ├──────────┼──────────┤
        │ P overlay│  status  │
        ├──────────┴──────────┤
        │  P(t) live timeline │
        └─────────────────────┘
    """
    h, w = frame_bgr.shape[:2]

    # Pane size — shrink to keep the dashboard manageable on small screens.
    target_w = 640
    scale = target_w / w
    pane_w = target_w
    pane_h = int(h * scale)
    fr = cv2.resize(frame_bgr, (pane_w, pane_h))

    if rho_np is not None:
        rho_pane = _heatmap_overlay(fr, rho_np, cmap=cv2.COLORMAP_HOT,
                                    alpha=0.55, vmin=0.0)
        _label(rho_pane, f"rho  sum={rho_np.sum():.1f}")
    else:
        rho_pane = fr.copy()
        _label(rho_pane, "rho  (n/a)")

    if P_np is not None:
        P_pane = _heatmap_overlay(fr, P_np, cmap=cv2.COLORMAP_TURBO,
                                  alpha=0.55, vmin=0.0)
        _label(P_pane, f"P  max={P_np.max():.3f}")
    else:
        P_pane = fr.copy()
        _label(P_pane, "P  (need 2 frames)")

    # Status pane — text-only metrics
    status = np.zeros_like(fr)
    alarm = (p_max is not None) and (p_max > threshold)
    color_alarm = (60, 60, 220) if alarm else (60, 200, 60)
    status_text = "ALARM" if alarm else "OK"
    cv2.putText(status, status_text, (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color_alarm, 3, cv2.LINE_AA)
    lines = [
        f"frame      : {frame_idx}",
        f"infer fps  : {fps_actual:5.2f}",
        f"max P      : {p_max:.4f}" if p_max is not None else "max P      : --",
        f"threshold  : {threshold:.3f}",
        f"history    : {len(history)}/{history.maxlen}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(status, line, (16, 110 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (230, 230, 230), 1, cv2.LINE_AA)

    _label(fr, "input")
    top = np.hstack([fr, rho_pane])
    bot = np.hstack([P_pane, status])
    grid = np.vstack([top, bot])

    timeline = _render_timeline_strip(history, threshold,
                                      width=grid.shape[1], height=180)
    return np.vstack([grid, timeline])


# ── Main inference loop ──────────────────────────────────────────────────────

@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, use_fp16 = _load_model(Path(args.checkpoint), device)

    src = _Source(args.source if not args.source.isdigit() else int(args.source))
    log.info("Source: %s | %dx%d @ %.1f fps | frames=%s",
             args.source, src.width, src.height, src.fps,
             src.total_frames if src.total_frames > 0 else "stream")

    out_dir: Optional[Path] = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    history_len = max(60, int(src.fps * 10)) if src.fps > 0 else 300
    history: deque[float] = deque(maxlen=history_len)
    all_scores: list[float] = []
    threshold = float(args.threshold)

    # Inference resolution — divisible by 8 for the H/8 outputs to stay aligned.
    inf_w = (args.width  // 8) * 8
    inf_h = (args.height // 8) * 8

    # ── Single-image path: just ρ, then exit ────────────────────────────────
    if src.is_image:
        frame = src.read()
        assert frame is not None
        frame_resized = cv2.resize(frame, (inf_w, inf_h))
        ft = _bgr_to_tensor(frame_resized, device)
        with autocast(device_type=device.type, enabled=use_fp16):
            rho = model.density_branch(ft)
        rho_np = rho.squeeze().detach().float().cpu().numpy()
        dash = _build_dashboard(frame_resized, rho_np, None,
                                history, threshold, 0.0, 0, None)
        if out_dir:
            cv2.imwrite(str(out_dir / "image_dashboard.png"), dash)
            log.info("Saved %s", out_dir / "image_dashboard.png")
        if args.display:
            cv2.imshow("PINN dashboard", dash)
            log.info("Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # ── Video / webcam path ────────────────────────────────────────────────
    writer: Optional[cv2.VideoWriter] = None
    prev_tensor: Optional[torch.Tensor] = None
    prev_resized: Optional[np.ndarray] = None
    frame_idx = 0
    last_t = time.time()
    fps_actual = 0.0

    try:
        while True:
            frame = src.read()
            if frame is None:
                break
            frame_resized = cv2.resize(frame, (inf_w, inf_h))
            cur_tensor = _bgr_to_tensor(frame_resized, device)

            rho_np: Optional[np.ndarray] = None
            P_np:   Optional[np.ndarray] = None
            p_max:  Optional[float] = None

            if prev_tensor is not None:
                with autocast(device_type=device.type, enabled=use_fp16):
                    out = model(prev_tensor, cur_tensor)
                rho_np = out["rho"].squeeze().detach().float().cpu().numpy()
                P_np   = out["P"].squeeze().detach().float().cpu().numpy()
                p_max  = float(P_np.max())
                history.append(p_max)
                all_scores.append(p_max)
            # On the first frame we have no pair yet — render the dashboard
            # using the current frame and skip ρ/P.

            now = time.time()
            dt = now - last_t
            if dt > 0:
                fps_actual = 0.9 * fps_actual + 0.1 * (1.0 / dt) if fps_actual else (1.0 / dt)
            last_t = now

            display_frame = prev_resized if prev_resized is not None else frame_resized
            dash = _build_dashboard(
                display_frame, rho_np, P_np, history, threshold,
                fps_actual, frame_idx, p_max,
            )

            if out_dir and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(out_dir / "dashboard.mp4"),
                    fourcc, src.fps if src.fps > 0 else 30.0,
                    (dash.shape[1], dash.shape[0]),
                )
                log.info("Writing → %s  (%dx%d)",
                         out_dir / "dashboard.mp4", dash.shape[1], dash.shape[0])
            if writer is not None:
                writer.write(dash)

            if args.display:
                cv2.imshow("PINN dashboard", dash)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    log.info("Quit requested.")
                    break

            if (frame_idx + 1) % 50 == 0 and p_max is not None:
                log.info("  frame %5d | maxP=%.4f | fps=%.2f",
                         frame_idx + 1, p_max, fps_actual)

            prev_tensor = cur_tensor
            prev_resized = frame_resized
            frame_idx += 1

    finally:
        src.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    # ── Final timeline png ───────────────────────────────────────────────
    if out_dir and all_scores:
        from utils.visualize import plot_pressure_timeline
        timeline_path = out_dir / "pressure_timeline.png"
        fig = plot_pressure_timeline(
            all_scores, fps=src.fps or 30.0,
            threshold=threshold, save_path=timeline_path,
        )
        plt.close(fig)
        log.info("Saved timeline → %s", timeline_path)

    if all_scores:
        log.info("Done. frames=%d  meanP=%.4f  maxP=%.4f  alarms>%.2f: %d",
                 len(all_scores),
                 sum(all_scores) / len(all_scores), max(all_scores),
                 threshold, sum(1 for s in all_scores if s > threshold))
    else:
        log.info("Done. (no frame pairs processed)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime PINN inference dashboard")
    p.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    p.add_argument("--source",     required=True,
                   help="Video file path, image path, or webcam index (e.g. 0)")
    p.add_argument("--out-dir",    default=None,
                   help="Directory to save dashboard.mp4 + pressure_timeline.png")
    p.add_argument("--display",    action="store_true",
                   help="Show a live OpenCV window (requires a display)")
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="P alarm threshold (default 0.5)")
    p.add_argument("--width",      type=int, default=1280,
                   help="Inference width — frames are resized to this (must be /8)")
    p.add_argument("--height",     type=int, default=720,
                   help="Inference height — frames are resized to this (must be /8)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    import os
    os.chdir(Path(__file__).parent)
    run(args)

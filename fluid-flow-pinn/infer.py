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
    # Default: lwcc DM-Count density + torchvision RAFT, head-mask-gated variance
    python infer.py --source clip.mp4 --out-dir outputs/infer_run/ \\
        --density-mask-tau 0.01

    # Live window
    python infer.py --source clip.mp4 --display --density-mask-tau 0.01

    # Webcam
    python infer.py --source 0 --display --density-mask-tau 0.01

    # After a training run, use a full checkpoint instead
    python infer.py --checkpoint checkpoints/best.pt --source clip.mp4 \\
        --out-dir outputs/infer_run/ --density-mask-tau 0.01
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
import tempfile
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

def _load_model(
    checkpoint_path: Optional[Path],
    device: torch.device,
    raft_iters: int = 6,
) -> tuple:
    """Load RAFT flow branch only — density is handled by lwcc externally.

      • `checkpoint_path` given: extract RAFT weights from a full training
        checkpoint (best.pt) so we don't waste time loading CSRNet.
      • None: load torchvision-pretrained RAFT directly.

    Returns (flow_branch, cfg, use_fp16) — flow_branch is a RAFTFlow module.
    """
    repo_root = Path(__file__).parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from models.branch2_flow import load_raft

    use_fp16 = device.type == "cuda"

    if checkpoint_path is not None:
        log.info("Loading RAFT from training checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg  = ckpt.get("config", {}) or {}
        use_fp16 = cfg.get("model", {}).get("use_fp16", True) and device.type == "cuda"
        # Extract only the flow_branch sub-state so we skip loading CSRNet weights.
        full_sd = ckpt["model"]
        flow_sd = {k[len("flow_branch."):]: v
                   for k, v in full_sd.items() if k.startswith("flow_branch.")}
        flow_branch = load_raft(frozen=True, num_flow_updates=raft_iters).to(device)
        flow_branch.load_state_dict(flow_sd)
        flow_branch.eval()
        log.info("RAFT loaded from checkpoint (epoch %s) | device=%s fp16=%s iters=%d",
                 ckpt.get("epoch", "?"), device, use_fp16, raft_iters)
        return flow_branch, cfg, use_fp16

    cfg = {}
    log.info("Loading torchvision-pretrained RAFT | device=%s fp16=%s iters=%d",
             device, use_fp16, raft_iters)
    flow_branch = load_raft(frozen=True, num_flow_updates=raft_iters).to(device)
    flow_branch.eval()
    log.info("RAFT ready.")
    return flow_branch, cfg, use_fp16


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


def _local_var_np(u_np: np.ndarray, k: int = 5) -> np.ndarray:
    """Per-pixel local variance of a 2-channel flow field (matches PressureMap).

    u_np: (2, H, W). Returns (H, W).
    """
    ux, uy = u_np[0].astype(np.float32), u_np[1].astype(np.float32)
    kernel = (k, k)
    Ex_x  = cv2.blur(ux, kernel)
    Ex_y  = cv2.blur(uy, kernel)
    Ex2_x = cv2.blur(ux * ux, kernel)
    Ex2_y = cv2.blur(uy * uy, kernel)
    var = ((Ex2_x - Ex_x ** 2) + (Ex2_y - Ex_y ** 2)) / 2.0
    return np.clip(var, 0.0, None)


def _load_lwcc(model_name: str, model_weights: str, device: torch.device):
    """Load lwcc crowd-counting model; move it to `device`; return (model, get_count_fn, tmp_path)."""
    try:
        from lwcc.LWCC import load_model, get_count
    except ImportError:
        log.error("lwcc not installed. Run: pip install lwcc")
        sys.exit(1)
    log.info("Loading lwcc %s/%s …", model_name, model_weights)
    m = load_model(model_name=model_name, model_weights=model_weights)
    m.to(device)
    m.eval()
    log.info("lwcc ready on %s.", device)
    # Pre-allocate a persistent temp file so per-frame create/delete overhead is gone.
    tmp_f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_f.close()
    return m, get_count, tmp_f.name


def _lwcc_density_np(
    bgr: np.ndarray,
    lwcc_model,
    lwcc_get_count,
    model_name: str,
    target_hw: tuple[int, int],
    tmp_path: str,
) -> np.ndarray:
    """Run lwcc on a BGR frame and return density map resized to target_hw (H, W).

    lwcc's get_count loads images on CPU and calls model(imgs) without moving to GPU.
    We call model inference ourselves so the tensor stays on the model's device.
    """
    from lwcc.util.functions import load_image as lwcc_load_image
    cv2.imwrite(tmp_path, bgr)
    img_tensor, _ = lwcc_load_image(tmp_path, model_name, is_gray=False, resize_img=False)
    # img_tensor shape: (1, 3, H, W) on CPU — move to model device
    device = next(lwcc_model.parameters()).device
    img_tensor = img_tensor.to(device, non_blocking=True)
    with torch.no_grad():
        output = lwcc_model(img_tensor)  # (1, 1, H', W')
    density = output[0, 0].cpu().numpy()
    rho = np.array(density, dtype=np.float32)
    H, W = target_hw
    if rho.shape != (H, W):
        rho = cv2.resize(rho, (W, H), interpolation=cv2.INTER_LINEAR)
    return rho


def _head_mask(rho_np: np.ndarray, tau: float) -> np.ndarray:
    """Binary mask (float32, 0/1) where density exceeds tau."""
    return (rho_np > tau).astype(np.float32)


def _denoise_flow(
    u_np: np.ndarray,
    spatial_sigma: float,
) -> np.ndarray:
    """Spatial Gaussian smooth on u to attenuate RAFT pixel-level noise.

    A smooth low-pass — unlike hard gates, this introduces no new edges, so
    Var(u) over the smoothed field reflects real coherent motion, not the
    boundaries we drew. `spatial_sigma` is in flow-grid pixels (H/8 units);
    sigma=1.0 is a gentle 1-pixel blur. Pass 0 to disable.
    """
    if spatial_sigma <= 0.0 or u_np.size == 0:
        return u_np.astype(np.float32, copy=False)
    out = u_np.astype(np.float32, copy=True)
    out[0] = cv2.GaussianBlur(out[0], ksize=(0, 0), sigmaX=spatial_sigma)
    out[1] = cv2.GaussianBlur(out[1], ksize=(0, 0), sigmaX=spatial_sigma)
    return out



def _aggregate_pressure(
    P_np: np.ndarray,
    rho_np: np.ndarray,
    mode: str,
    topk_frac: float,
) -> float:
    """Reduce a P map to a single scalar danger score.

    Modes:
        max           — original; max(P). Brittle to single-pixel artifacts.
        topk          — mean of top-k fraction of pixels. Robust to spikes.
        weighted_mean — sum(rho * P) / sum(rho). "Pressure on an average
                        person" — naturally suppresses lone-pixel hot spots
                        in low-density regions because they get tiny weight.
    """
    if mode == "max":
        return float(P_np.max())
    if mode == "topk":
        flat = P_np.reshape(-1)
        k = max(1, int(round(flat.size * topk_frac)))
        # argpartition for the k largest, then mean
        idx = np.argpartition(flat, -k)[-k:]
        return float(flat[idx].mean())
    if mode == "weighted_mean":
        w = rho_np.astype(np.float32)
        denom = float(w.sum())
        if denom <= 1e-9:
            return float(P_np.mean())
        return float((w * P_np).sum() / denom)
    raise ValueError(f"unknown p_agg mode: {mode}")


def _temporal_ema(
    u_prev_ema: Optional[np.ndarray],
    u_cur: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Exponential moving average across frames: u_t = α·u + (1-α)·u_{t-1}.

    RAFT noise is roughly uncorrelated frame-to-frame; real motion is
    correlated. EMA suppresses the former without flattening the latter or
    creating edges. alpha=1.0 disables (no smoothing). alpha=0.5 averages
    the current frame with the running history equally.
    """
    if u_prev_ema is None or alpha >= 1.0:
        return u_cur.astype(np.float32, copy=True)
    return (alpha * u_cur + (1.0 - alpha) * u_prev_ema).astype(np.float32)


def _compute_pressure_np(
    rho_np: np.ndarray,
    u_np: np.ndarray,
    window: int,
    rho_alpha: float,
    per_capita: bool,
    per_capita_eps: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Recompute P from ρ and (denoised) u with configurable coupling.

    mask: optional float32 (H', W') binary array from _head_mask(). When given,
          flow vectors outside the mask are zeroed before Var(u) is computed so
          that clothing/background motion doesn't pollute the variance.

    Coupling modes:
      • per_capita=True : P = Var(u) / (rho + eps)  — flags per-capita turbulence
      • else            : P = (rho ** rho_alpha) * Var(u)
        - rho_alpha=1 reproduces the original P = rho * Var(u)
        - rho_alpha=0 gives P = Var(u) (decoupled from density)
    """
    if mask is not None:
        # resize mask to flow grid if needed, then gate flow
        mh, mw = u_np.shape[1], u_np.shape[2]
        m = mask if mask.shape == (mh, mw) else cv2.resize(mask, (mw, mh), interpolation=cv2.INTER_NEAREST)
        u_np = u_np * m[np.newaxis, :, :]

    var_u = _local_var_np(u_np, k=window)  # (Hf, Wf)
    h, w = rho_np.shape[-2:]
    if var_u.shape != (h, w):
        var_u = cv2.resize(var_u, (w, h), interpolation=cv2.INTER_LINEAR)
    if per_capita:
        return var_u / (rho_np.astype(np.float32) + per_capita_eps)
    if rho_alpha == 1.0:
        return rho_np.astype(np.float32) * var_u
    if rho_alpha == 0.0:
        return var_u
    return (rho_np.astype(np.float32) ** rho_alpha) * var_u


def _flow_arrows(
    frame_bgr: np.ndarray,
    u_np: np.ndarray,
    step: int = 24,
    scale: float = 1.0,
) -> np.ndarray:
    """Sparse arrow overlay showing flow direction. u_np: (2, H', W')."""
    out = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    # Upsample u to frame resolution
    ux = cv2.resize(u_np[0].astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR)
    uy = cv2.resize(u_np[1].astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR)
    # Account for the H/8 grid: each grid cell is 8 input pixels, so multiply
    # the (already-rescaled) flow by 8 to get input-pixel displacements.
    ux *= 8.0 * scale
    uy *= 8.0 * scale
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = float(ux[y, x]), float(uy[y, x])
            mag = (dx * dx + dy * dy) ** 0.5
            if mag < 1.0:
                continue
            cv2.arrowedLine(out, (x, y),
                            (int(x + dx), int(y + dy)),
                            (0, 255, 255), 1, tipLength=0.35)
    return out


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
    u_np: Optional[np.ndarray],
    P_np: Optional[np.ndarray],
    history: deque[float],
    threshold: float,
    fps_actual: float,
    frame_idx: int,
    p_max: Optional[float],
    pressure_window: int = 5,
    p_formula: str = "rho * Var(u)",
) -> np.ndarray:
    """Compose the live dashboard image (BGR uint8).

    Layout (3 rows × 2 cols):
        ┌──────────┬──────────┐
        │   input  │ rho      │
        ├──────────┼──────────┤
        │  |u| +   │ Var(u)   │
        │  arrows  │          │
        ├──────────┼──────────┤
        │   P      │ status   │
        ├──────────┴──────────┤
        │  P(t) live timeline │
        └─────────────────────┘
    """
    h, w = frame_bgr.shape[:2]

    target_w = 640
    pane_w = target_w
    pane_h = int(h * (target_w / w))
    fr = cv2.resize(frame_bgr, (pane_w, pane_h))

    # Row 1 — input + ρ
    input_pane = fr.copy()
    _label(input_pane, "input")

    if rho_np is not None:
        rho_pane = _heatmap_overlay(fr, rho_np, cmap=cv2.COLORMAP_HOT,
                                    alpha=0.55, vmin=0.0)
        _label(rho_pane, f"rho  sum={rho_np.sum():.1f}")
    else:
        rho_pane = fr.copy()
        _label(rho_pane, "rho  (n/a)")

    # Row 2 — |u| (with arrows) + Var(u)
    if u_np is not None:
        u_mag = np.sqrt(u_np[0] ** 2 + u_np[1] ** 2)
        u_pane = _heatmap_overlay(fr, u_mag, cmap=cv2.COLORMAP_COOL,
                                  alpha=0.45, vmin=0.0)
        u_pane = _flow_arrows(u_pane, u_np, step=24, scale=1.0)
        _label(u_pane, f"|u|  max={u_mag.max():.2f} px/frame")

        var_u = _local_var_np(u_np, k=pressure_window)
        var_pane = _heatmap_overlay(fr, var_u, cmap=cv2.COLORMAP_PLASMA,
                                    alpha=0.55, vmin=0.0)
        _label(var_pane, f"Var(u)  max={var_u.max():.2f}")
    else:
        u_pane = fr.copy(); _label(u_pane, "|u|  (need 2 frames)")
        var_pane = fr.copy(); _label(var_pane, "Var(u)  (need 2 frames)")

    # Row 3 — P + status
    if P_np is not None:
        P_pane = _heatmap_overlay(fr, P_np, cmap=cv2.COLORMAP_TURBO,
                                  alpha=0.55, vmin=0.0)
        _label(P_pane, f"P = {p_formula}  max={P_np.max():.3f}")
    else:
        P_pane = fr.copy()
        _label(P_pane, "P  (need 2 frames)")

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

    row1 = np.hstack([input_pane, rho_pane])
    row2 = np.hstack([u_pane,     var_pane])
    row3 = np.hstack([P_pane,     status])
    grid = np.vstack([row1, row2, row3])

    timeline = _render_timeline_strip(history, threshold,
                                      width=grid.shape[1], height=180)
    return np.vstack([grid, timeline])


# ── Main inference loop ──────────────────────────────────────────────────────

@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    model, cfg, use_fp16 = _load_model(ckpt, device, raft_iters=args.raft_iters)

    # lwcc is the density source — load once, reuse every frame
    lwcc_model, lwcc_get_count, lwcc_tmp = _load_lwcc(args.lwcc_model, args.lwcc_weights, device)

    src = _Source(args.source if not args.source.isdigit() else int(args.source))
    log.info("Source: %s | %dx%d @ %.1f fps | frames=%s",
             args.source, src.width, src.height, src.fps,
             src.total_frames if src.total_frames > 0 else "stream")

    out_dir: Optional[Path] = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.per_capita:
        p_formula = "Var(u[mask]) / (rho+eps)" if args.density_mask_tau > 0 else "Var(u) / (rho+eps)"
    elif args.rho_alpha == 1.0:
        p_formula = "rho * Var(u[mask])" if args.density_mask_tau > 0 else "rho * Var(u)"
    elif args.rho_alpha == 0.0:
        p_formula = "Var(u[mask])" if args.density_mask_tau > 0 else "Var(u)"
    else:
        suffix = "[mask]" if args.density_mask_tau > 0 else ""
        p_formula = f"rho^{args.rho_alpha:g} * Var(u{suffix})"

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
        rho_np = _lwcc_density_np(
            frame_resized, lwcc_model, lwcc_get_count,
            args.lwcc_model, target_hw=(inf_h // 8, inf_w // 8),
            tmp_path=lwcc_tmp,
        )
        dash = _build_dashboard(frame_resized, rho_np, None, None,
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
    u_ema: Optional[np.ndarray] = None
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
            u_np:   Optional[np.ndarray] = None
            P_np:   Optional[np.ndarray] = None
            p_max:  Optional[float] = None

            if prev_tensor is not None:
                with autocast(device_type=device.type, enabled=use_fp16):
                    u_t = model(prev_tensor, cur_tensor)
                u_np = u_t.squeeze(0).detach().float().cpu().numpy()  # (2, H', W')

                # lwcc density — primary source for rho and head mask
                rho_np = _lwcc_density_np(
                    frame_resized, lwcc_model, lwcc_get_count,
                    args.lwcc_model, target_hw=(u_np.shape[1], u_np.shape[2]),
                    tmp_path=lwcc_tmp,
                )
                head_mask = _head_mask(rho_np, args.density_mask_tau) \
                    if args.density_mask_tau > 0.0 else None

                # Smooth u to suppress RAFT noise without introducing edges.
                # Spatial Gaussian first, then temporal EMA across frames.
                u_np = _denoise_flow(u_np, spatial_sigma=args.flow_spatial_sigma)
                u_np = _temporal_ema(u_ema, u_np, alpha=args.flow_ema_alpha)
                u_ema = u_np  # carry forward for next frame

                P_np = _compute_pressure_np(
                    rho_np, u_np,
                    window=cfg.get("model", {}).get("pressure_window", 5),
                    rho_alpha=args.rho_alpha,
                    per_capita=args.per_capita,
                    per_capita_eps=args.per_capita_eps,
                    mask=head_mask,
                )

                p_max = _aggregate_pressure(
                    P_np, rho_np, mode=args.p_agg, topk_frac=args.topk_frac,
                )
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
                display_frame, rho_np, u_np, P_np, history, threshold,
                fps_actual, frame_idx, p_max, p_formula=p_formula,
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
        Path(lwcc_tmp).unlink(missing_ok=True)

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
    p.add_argument("--checkpoint", default=None,
                   help="Path to a full training checkpoint .pt (best.pt). "
                        "Omit to run with torchvision-pretrained RAFT + lwcc density.")
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
    # Fix #1 — decouple ρ and Var(u) in the danger score
    p.add_argument("--rho-alpha",   type=float, default=1.0,
                   help="Exponent on rho in P = rho^alpha * Var(u). "
                        "1.0 = original; 0.0 = Var(u) only; 0.25 = mild coupling.")
    p.add_argument("--per-capita",  action="store_true",
                   help="Use P = Var(u) / (rho + eps) — per-capita turbulence. "
                        "Overrides --rho-alpha.")
    p.add_argument("--per-capita-eps", type=float, default=1e-3,
                   help="Epsilon in per-capita denominator (default 1e-3).")
    # Aggregator — how to reduce P-map to a scalar danger score
    p.add_argument("--p-agg", choices=("max", "topk", "weighted_mean"),
                   default="max",
                   help="Reduce P map to a scalar: max (orig), topk (mean of "
                        "top fraction), weighted_mean (rho-weighted average).")
    p.add_argument("--topk-frac", type=float, default=0.001,
                   help="Fraction of pixels for --p-agg topk (default 0.001 = 0.1%%).")
    # RAFT speed vs accuracy
    p.add_argument("--raft-iters", type=int, default=6,
                   help="Number of RAFT flow update iterations (default 6). "
                        "Use 4 for max speed, 12 for max accuracy.")
    # RAFT noise suppression — smooth, edge-free
    p.add_argument("--flow-spatial-sigma", type=float, default=0.0,
                   help="Gaussian sigma (in H/8-grid pixels) for spatial smooth on u "
                        "before Var(u). 0 disables. Try 0.8 .. 1.5.")
    p.add_argument("--flow-ema-alpha",     type=float, default=1.0,
                   help="Temporal EMA on u: u_t = alpha*u + (1-alpha)*u_{t-1}. "
                        "1.0 disables. Try 0.4 .. 0.6.")
    # Head-mask-gated variance via lwcc density
    p.add_argument("--density-mask-tau", type=float, default=0.0,
                   help="Density threshold τ for head mask. Var(u) is computed only "
                        "where lwcc density > τ. 0 disables (default). Try 0.01.")
    p.add_argument("--lwcc-model",   default="DM-Count",
                   choices=("CSRNet", "SFANet", "Bay", "DM-Count"),
                   help="lwcc model for head masking (default: DM-Count).")
    p.add_argument("--lwcc-weights", default="SHA",
                   choices=("SHA", "SHB", "QNRF"),
                   help="lwcc pretrained weights (default: SHA).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    import os
    os.chdir(Path(__file__).parent)
    run(args)

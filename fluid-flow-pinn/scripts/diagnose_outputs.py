"""Diagnostic: dump lwcc density + RAFT flow + YOLO detection outputs for threshold tuning.

Produces a single PNG with 9 panels (3×3):

  Row 0:  Input frame t  |  Density ρ (lwcc)  |  ρ histogram with tau lines
  Row 1:  |u| unmasked   |  YOLO detections   |  YOLO mask at flow grid (selected conf)
  Row 2:  Var(u) masked  |  P masked          |  Conf sweep bar chart

The conf-sweep panel (bottom-right) shows max P and mask coverage at conf=0.1/0.25/0.5
— use it to pick --detector-conf for infer.py.

Usage:
    python scripts/diagnose_outputs.py \\
        --frame-t  path/to/frame_000.jpg \\
        --frame-t1 path/to/frame_001.jpg \\
        --detector yolo26n.pt

    # Tune confidence explicitly (default 0.25):
    python scripts/diagnose_outputs.py \\
        --frame-t img0.jpg --frame-t1 img1.jpg \\
        --detector yolo26n.pt --conf 0.4

    # lwcc-only mode (no YOLO — YOLO panels show rho>tau mask instead):
    python scripts/diagnose_outputs.py \\
        --frame-t img0.jpg --frame-t1 img1.jpg \\
        --tau 0.01
"""

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.branch2_flow import load_raft

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_tensor(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std  = torch.tensor(_STD).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0).to(device)


def _local_var(u: np.ndarray, k: int = 5) -> np.ndarray:
    ux, uy = u[0].astype(np.float32), u[1].astype(np.float32)
    Ex  = cv2.blur(ux, (k, k));  Ey  = cv2.blur(uy, (k, k))
    Ex2 = cv2.blur(ux * ux, (k, k)); Ey2 = cv2.blur(uy * uy, (k, k))
    return np.clip(((Ex2 - Ex**2) + (Ey2 - Ey**2)) / 2.0, 0, None)


def _stats(arr: np.ndarray, name: str) -> str:
    return (f"{name}: min={arr.min():.4f}  max={arr.max():.4f}  "
            f"mean={arr.mean():.4f}  sum={arr.sum():.2f}  "
            f"p50={np.percentile(arr, 50):.4f}  "
            f"p90={np.percentile(arr, 90):.4f}  "
            f"p99={np.percentile(arr, 99):.4f}")


def _lwcc_density(bgr: np.ndarray, model, model_name: str, tmp_path: str) -> tuple[float, np.ndarray]:
    cv2.imwrite(tmp_path, bgr)
    from lwcc.util.functions import load_image as lwcc_load_image
    img_tensor, _ = lwcc_load_image(tmp_path, model_name, is_gray=False, resize_img=False)
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device, non_blocking=True)
    with torch.no_grad():
        output = model(img_tensor)
    density = output[0, 0].cpu().numpy().astype(np.float32)
    return float(density.sum()), density


def _yolo_boxes(bgr: np.ndarray, yolo_model, conf: float) -> np.ndarray:
    """Return (N, 4) xyxy person boxes as numpy array."""
    results = yolo_model(bgr, verbose=False)
    boxes = results[0].boxes
    keep = (boxes.cls == 0) & (boxes.conf > conf)
    return boxes.xyxy[keep].cpu().numpy()


def _boxes_to_mask(boxes_xyxy: np.ndarray, src_hw: tuple, target_hw: tuple) -> np.ndarray:
    H_src, W_src = src_hw
    H_out, W_out = target_hw
    mask = np.zeros((H_out, W_out), dtype=np.float32)
    sx, sy = W_out / W_src, H_out / H_src
    for box in boxes_xyxy:
        x1 = max(0, int(box[0] * sx));  y1 = max(0, int(box[1] * sy))
        x2 = min(W_out, int(box[2] * sx)); y2 = min(H_out, int(box[3] * sy))
        mask[y1:y2, x1:x2] = 1.0
    return mask


def _draw_boxes(bgr: np.ndarray, boxes_xyxy: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    out = bgr.copy()
    for box in boxes_xyxy:
        cv2.rectangle(out, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── lwcc ─────────────────────────────────────────────────────────────────
    try:
        from lwcc.LWCC import load_model as lwcc_load_model, get_count as lwcc_get_count
    except ImportError:
        print("ERROR: lwcc not installed.  pip install lwcc")
        sys.exit(1)
    print(f"Loading lwcc {args.lwcc_model}/{args.lwcc_weights} …")
    lwcc_model = lwcc_load_model(model_name=args.lwcc_model, model_weights=args.lwcc_weights)
    lwcc_model.to(device).eval()

    tmp_f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_f.close()
    tmp_path = tmp_f.name

    # ── YOLO (optional) ───────────────────────────────────────────────────────
    yolo_model = None
    if args.detector:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("ERROR: ultralytics not installed.  pip install ultralytics")
            sys.exit(1)
        print(f"Loading YOLO detector: {args.detector} …")
        yolo_model = YOLO(args.detector)
        yolo_model.to(device)
        print("YOLO ready.")

    # ── RAFT ──────────────────────────────────────────────────────────────────
    flow_branch = load_raft(frozen=True).to(device).eval()

    # ── frames ────────────────────────────────────────────────────────────────
    bgr_t  = cv2.imread(args.frame_t)
    bgr_t1 = cv2.imread(args.frame_t1)
    if bgr_t  is None: raise FileNotFoundError(args.frame_t)
    if bgr_t1 is None: raise FileNotFoundError(args.frame_t1)
    H = (bgr_t.shape[0] // 8) * 8
    W = (bgr_t.shape[1] // 8) * 8
    bgr_t  = cv2.resize(bgr_t,  (W, H))
    bgr_t1 = cv2.resize(bgr_t1, (W, H))
    rgb_t  = cv2.cvtColor(bgr_t, cv2.COLOR_BGR2RGB)

    # ── lwcc density ──────────────────────────────────────────────────────────
    print("Running lwcc …")
    count, rho_np = _lwcc_density(bgr_t, lwcc_model, args.lwcc_model, tmp_path)
    print(f"lwcc count: {count:.1f}")
    print(_stats(rho_np, "rho"))

    # ── RAFT flow ─────────────────────────────────────────────────────────────
    use_fp16 = device.type == "cuda"
    with autocast(device_type=device.type, enabled=use_fp16):
        u = flow_branch(_to_tensor(bgr_t, device), _to_tensor(bgr_t1, device))
    u_np  = u.squeeze(0).float().cpu().numpy()   # (2, Hf, Wf)
    Hf, Wf = u_np.shape[1], u_np.shape[2]
    u_mag = np.sqrt(u_np[0]**2 + u_np[1]**2)
    var_u = _local_var(u_np)
    print(_stats(u_mag, "|u|"))
    print(_stats(var_u, "Var(u)"))

    rho_at_flow = cv2.resize(rho_np, (Wf, Hf), interpolation=cv2.INTER_LINEAR)

    # ── primary mask at selected conf / tau ───────────────────────────────────
    if yolo_model is not None:
        boxes_sel = _yolo_boxes(bgr_t, yolo_model, args.conf)
        head_mask = _boxes_to_mask(boxes_sel, (H, W), (Hf, Wf))
        mask_label = f"YOLO conf>{args.conf:.2f}  {len(boxes_sel)} persons  {100*head_mask.mean():.1f}% px"
    else:
        head_mask = (rho_at_flow > args.tau).astype(np.float32)
        boxes_sel = np.empty((0, 4))
        mask_label = f"lwcc τ={args.tau}  {100*head_mask.mean():.1f}% px"

    u_masked   = u_np * head_mask[np.newaxis, :, :]
    var_masked = _local_var(u_masked)
    P_masked   = rho_at_flow * var_masked

    print(_stats(var_masked, "Var(u) masked"))
    print(_stats(P_masked,   "P masked"))
    print(f"mask coverage: {100*head_mask.mean():.1f}%")

    # ── conf/tau sweep for bottom-right panel ─────────────────────────────────
    CONF_SWEEP = [0.1, 0.25, 0.5] if yolo_model is not None else [0.005, 0.01, 0.05]
    sweep_Pmax, sweep_coverage = [], []
    for c in CONF_SWEEP:
        if yolo_model is not None:
            m = _boxes_to_mask(_yolo_boxes(bgr_t, yolo_model, c), (H, W), (Hf, Wf))
        else:
            m = (rho_at_flow > c).astype(np.float32)
        P_m = rho_at_flow * _local_var(u_np * m[np.newaxis, :, :])
        sweep_Pmax.append(P_m.max())
        sweep_coverage.append(100 * m.mean())

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    detector_tag = args.detector if args.detector else f"lwcc τ={args.tau}"
    fig.suptitle(
        f"Diagnostic  |  detector: {detector_tag}  |  frame: {Path(args.frame_t).name}\n"
        f"lwcc count={count:.1f}  |u| max={u_mag.max():.3f}  {mask_label}",
        fontsize=11,
    )

    # [0,0] input frame
    axes[0, 0].imshow(rgb_t)
    axes[0, 0].set_title(f"Input frame t  ({W}×{H})")
    axes[0, 0].axis("off")

    # [0,1] density map
    im01 = axes[0, 1].imshow(rho_np, cmap="hot", interpolation="nearest")
    axes[0, 1].set_title(
        f"Density ρ — lwcc {args.lwcc_model}  ({rho_np.shape[1]}×{rho_np.shape[0]})\n"
        f"count={count:.1f}  max={rho_np.max():.4f}  p99={np.percentile(rho_np, 99):.4f}"
    )
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # [0,2] rho histogram with tau lines
    flat_rho = rho_np.flatten()
    axes[0, 2].hist(flat_rho, bins=120, log=True, color="tomato", edgecolor="none")
    for t_line in (0.001, 0.005, 0.01, 0.05, 0.1, 0.2):
        pct = 100.0 * (flat_rho > t_line).mean()
        lw = 2.0 if t_line == args.tau else 0.9
        axes[0, 2].axvline(t_line, linestyle="--", linewidth=lw,
                           label=f"τ={t_line} ({pct:.1f}%)")
    axes[0, 2].legend(fontsize=7)
    axes[0, 2].set_xlabel("ρ value")
    axes[0, 2].set_ylabel("pixel count (log)")
    axes[0, 2].set_title(f"ρ histogram  (bold = τ={args.tau})")

    # [1,0] |u| unmasked
    im10 = axes[1, 0].imshow(u_mag, cmap="cool", interpolation="nearest")
    axes[1, 0].set_title(
        f"|u| unmasked  ({Wf}×{Hf})\n"
        f"max={u_mag.max():.3f}  mean={u_mag.mean():.3f} px/frame"
    )
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # [1,1] YOLO detections on frame (or lwcc mask overlay)
    if yolo_model is not None:
        det_img = cv2.cvtColor(_draw_boxes(bgr_t, boxes_sel), cv2.COLOR_BGR2RGB)
        axes[1, 1].imshow(det_img)
        axes[1, 1].set_title(
            f"YOLO detections  conf>{args.conf:.2f}\n"
            f"{len(boxes_sel)} persons detected"
        )
    else:
        rho_vis = cv2.resize(rho_np, (W, H), interpolation=cv2.INTER_LINEAR)
        mask_vis = (rho_vis > args.tau).astype(np.uint8) * 255
        overlay = rgb_t.copy()
        overlay[mask_vis > 0] = (overlay[mask_vis > 0] * 0.4 +
                                  np.array([255, 80, 0]) * 0.6).astype(np.uint8)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title(f"lwcc mask overlay  τ={args.tau}\n"
                              f"{100*(mask_vis>0).mean():.1f}% pixels")
    axes[1, 1].axis("off")

    # [1,2] head mask at flow grid
    im12 = axes[1, 2].imshow(head_mask, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    axes[1, 2].set_title(f"Head mask at flow grid\n{mask_label}")
    plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # [2,0] Var(u) masked
    vmax_var = np.percentile(var_u, 99) + 1e-9
    im20 = axes[2, 0].imshow(var_masked, cmap="plasma", interpolation="nearest",
                              vmin=0, vmax=vmax_var)
    axes[2, 0].set_title(
        f"Var(u) MASKED\nmax={var_masked.max():.4f}  p99={np.percentile(var_masked, 99):.4f}"
    )
    plt.colorbar(im20, ax=axes[2, 0], fraction=0.046, pad=0.04)

    # [2,1] P masked
    vmax_P = np.percentile(P_masked, 99) + 1e-9
    im21 = axes[2, 1].imshow(P_masked, cmap="turbo", interpolation="nearest",
                              vmin=0, vmax=vmax_P)
    axes[2, 1].set_title(
        f"P = ρ · Var(u) MASKED\nmax={P_masked.max():.4f}  p99={np.percentile(P_masked, 99):.4f}"
    )
    plt.colorbar(im21, ax=axes[2, 1], fraction=0.046, pad=0.04)

    # [2,2] conf/tau sweep bar chart
    ax_sweep = axes[2, 2]
    xs = np.arange(len(CONF_SWEEP))
    bars = ax_sweep.bar(xs, sweep_Pmax, color=["#3a86ff", "#ff006e", "#fb5607"], alpha=0.8)
    ax_sweep.set_xticks(xs)
    ax_sweep.set_xticklabels([str(c) for c in CONF_SWEEP], fontsize=9)
    xlabel = "YOLO conf threshold" if yolo_model is not None else "lwcc τ threshold"
    ax_sweep.set_xlabel(xlabel, fontsize=9)
    ax_sweep.set_ylabel("max P (masked)", fontsize=9)
    ax_sweep.set_title(
        f"{'Conf' if yolo_model else 'τ'} sweep — max P & mask coverage\n"
        f"(bold bar = selected {'conf' if yolo_model else 'τ'})"
    )
    for bar, cov in zip(bars, sweep_coverage):
        ax_sweep.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                      f"{cov:.1f}%", ha="center", va="bottom", fontsize=8)
    # bold the selected threshold bar
    sel = args.conf if yolo_model else args.tau
    if sel in CONF_SWEEP:
        bars[CONF_SWEEP.index(sel)].set_edgecolor("black")
        bars[CONF_SWEEP.index(sel)].set_linewidth(2.5)

    plt.tight_layout()
    Path(tmp_path).unlink(missing_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose lwcc density + RAFT flow + YOLO outputs")
    p.add_argument("--frame-t",  required=True, help="Path to frame t  (jpg/png)")
    p.add_argument("--frame-t1", required=True, help="Path to frame t+1 (jpg/png)")
    p.add_argument("--out", default="outputs/diag/diagnostic.png",
                   help="Output PNG path (default: outputs/diag/diagnostic.png)")
    p.add_argument("--lwcc-model",   default="DM-Count",
                   choices=("CSRNet", "SFANet", "Bay", "DM-Count"))
    p.add_argument("--lwcc-weights", default="SHA",
                   choices=("SHA", "SHB", "QNRF"))
    p.add_argument("--tau", type=float, default=0.01,
                   help="lwcc density threshold (used when --detector is omitted, default 0.01)")
    p.add_argument("--detector", default=None,
                   help="Ultralytics YOLO model for person detection, e.g. 'yolo26n.pt'. "
                        "Omit to use lwcc mask only.")
    p.add_argument("--conf", type=float, default=0.25,
                   help="YOLO confidence threshold for primary mask (default 0.25). "
                        "Sweep panel always shows 0.1 / 0.25 / 0.5 for comparison.")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())

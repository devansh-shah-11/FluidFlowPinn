"""Diagnostic: dump raw density (lwcc) + flow (RAFT) outputs for threshold exploration.

Runs lwcc crowd counter + RAFT on a pair of frames and produces a single PNG with 6 panels:

  1. Input frame t
  2. Raw density map (rho) from lwcc — min/max/sum/percentiles in title
  3. Density histogram (log-scale) — use this to pick a rho threshold tau
  4. Flow magnitude |u| at coarse grid
  5. Local variance Var(u) — unmasked, shows the clothing-noise problem
  6. Head-mask preview: what rho > tau looks like at 3 tau values overlaid on frame

Usage:
    python scripts/diagnose_outputs.py \\
        --frame-t  /path/to/frame_000.jpg \\
        --frame-t1 /path/to/frame_001.jpg \\
        --out      outputs/diag/diagnostic.png

    # Model choice (default DM-Count/SHA — best overall):
    python scripts/diagnose_outputs.py \\
        --frame-t img0.jpg --frame-t1 img1.jpg \\
        --lwcc-model DM-Count --lwcc-weights SHA

    # If you only have one image, pass it twice — flow will be zero
    # but all density panels still render.
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

# ── repo root on path ─────────────────────────────────────────────────────────
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


def _lwcc_density(bgr: np.ndarray, model, model_name: str, get_count_fn) -> tuple[float, np.ndarray]:
    """Run lwcc on a BGR frame (in-memory) via a temp file.

    Returns (count, density_map_2d) where density_map_2d is a float32 numpy
    array at the model's native output resolution (~H/8 for CSRNet-based models).
    """
    # lwcc only accepts file paths — write to a temp PNG, read back, clean up.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    try:
        cv2.imwrite(tmp_path, bgr)
        count, density = get_count_fn(
            tmp_path,
            model_name=model_name,
            model=model,
            return_density=True,
            resize_img=False,   # we already handle our own resolution
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return float(count), np.array(density, dtype=np.float32)


# ── main ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── load lwcc model once ──────────────────────────────────────────────────
    try:
        from lwcc.LWCC import load_model as lwcc_load_model, get_count as lwcc_get_count
    except ImportError:
        print("ERROR: lwcc not installed. Run:  pip install lwcc")
        sys.exit(1)

    print(f"Loading lwcc model: {args.lwcc_model} / {args.lwcc_weights} …")
    lwcc_model = lwcc_load_model(
        model_name=args.lwcc_model,
        model_weights=args.lwcc_weights,
    )
    print("lwcc model ready.")

    # ── load RAFT once ────────────────────────────────────────────────────────
    flow_branch = load_raft(frozen=True).to(device).eval()

    # ── load frames ───────────────────────────────────────────────────────────
    bgr_t  = cv2.imread(args.frame_t)
    bgr_t1 = cv2.imread(args.frame_t1)
    if bgr_t is None:  raise FileNotFoundError(args.frame_t)
    if bgr_t1 is None: raise FileNotFoundError(args.frame_t1)

    # snap to multiples of 8 (RAFT requirement)
    H = (bgr_t.shape[0] // 8) * 8
    W = (bgr_t.shape[1] // 8) * 8
    bgr_t  = cv2.resize(bgr_t,  (W, H))
    bgr_t1 = cv2.resize(bgr_t1, (W, H))

    # ── density via lwcc ──────────────────────────────────────────────────────
    print("Running lwcc density estimation …")
    count, rho_np = _lwcc_density(bgr_t, lwcc_model, args.lwcc_model, lwcc_get_count)
    print(f"lwcc count estimate: {count:.1f}")
    print(_stats(rho_np, "rho  "))

    # ── flow via RAFT ─────────────────────────────────────────────────────────
    ft  = _to_tensor(bgr_t,  device)
    ft1 = _to_tensor(bgr_t1, device)
    use_fp16 = device.type == "cuda"
    with autocast(device_type=device.type, enabled=use_fp16):
        u = flow_branch(ft, ft1)
    u_np  = u.squeeze(0).float().cpu().numpy()   # (2, Hf, Wf)
    u_mag = np.sqrt(u_np[0]**2 + u_np[1]**2)
    var_u = _local_var(u_np)

    print(_stats(u_mag,  "|u|  "))
    print(_stats(var_u,  "Var(u)"))

    tau = args.tau

    # ── build head mask at flow grid resolution ───────────────────────────────
    # rho is at lwcc native res; downsample to RAFT coarse grid (H/8, W/8)
    Hf, Wf = u_np.shape[1], u_np.shape[2]
    rho_at_flow = cv2.resize(rho_np, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
    head_mask = (rho_at_flow > tau).astype(np.float32)          # 1 inside head, 0 outside

    # masked flow: zero out non-head pixels before computing variance
    u_masked = u_np * head_mask[np.newaxis, :, :]               # (2, Hf, Wf)
    var_u_masked = _local_var(u_masked)

    # masked pressure = rho * masked_var  (rho also at flow grid)
    P_unmasked = rho_at_flow * var_u
    P_masked   = rho_at_flow * var_u_masked

    print(_stats(var_u_masked, f"Var(u) τ={tau}"))
    print(_stats(P_masked,     f"P masked τ={tau}"))

    # rho is at lwcc native res; flow is at H/8.  Report both.
    print(f"\nrho shape  : {rho_np.shape}  (lwcc native output res)")
    print(f"u shape    : {u_np.shape}    (RAFT H/8 grid)")
    print(f"mask coverage: {100*head_mask.mean():.1f}% of flow-grid pixels above τ={tau}")

    # ── figure: 3 rows × 3 cols ───────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(
        f"Diagnostic  |  {args.lwcc_model}/{args.lwcc_weights}  τ={tau}  |  "
        f"frame: {Path(args.frame_t).name}\n"
        f"lwcc count={count:.1f}  rho max={rho_np.max():.4f}  "
        f"|u| max={u_mag.max():.3f}  mask={100*head_mask.mean():.1f}% pixels",
        fontsize=11,
    )

    rgb_t = cv2.cvtColor(bgr_t, cv2.COLOR_BGR2RGB)

    # Row 0 — input | density | histogram
    axes[0, 0].imshow(rgb_t)
    axes[0, 0].set_title(f"Input frame t  ({W}×{H})")
    axes[0, 0].axis("off")

    im01 = axes[0, 1].imshow(rho_np, cmap="hot", interpolation="nearest")
    axes[0, 1].set_title(
        f"Density ρ — lwcc {args.lwcc_model}  ({rho_np.shape[1]}×{rho_np.shape[0]})\n"
        f"count={count:.1f}  max={rho_np.max():.4f}  p99={np.percentile(rho_np, 99):.4f}"
    )
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    flat_rho = rho_np.flatten()
    axes[0, 2].hist(flat_rho, bins=120, log=True, color="tomato", edgecolor="none")
    for t_line in (0.001, 0.005, 0.01, 0.05, 0.1, 0.2):
        pct_above = 100.0 * (flat_rho > t_line).mean()
        lw = 2.0 if t_line == tau else 0.9
        axes[0, 2].axvline(t_line, linestyle="--", linewidth=lw,
                           label=f"τ={t_line}  ({pct_above:.1f}%)")
    axes[0, 2].legend(fontsize=7)
    axes[0, 2].set_xlabel("ρ value")
    axes[0, 2].set_ylabel("pixel count (log)")
    axes[0, 2].set_title(f"ρ histogram  (bold = selected τ={tau})")

    # Row 1 — flow magnitude | Var(u) unmasked | head mask
    im10 = axes[1, 0].imshow(u_mag, cmap="cool", interpolation="nearest")
    axes[1, 0].set_title(
        f"|u|  ({Wf}×{Hf})\n"
        f"max={u_mag.max():.3f}  mean={u_mag.mean():.3f} coarse-grid px/frame"
    )
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

    vmax_var = np.percentile(var_u, 99) + 1e-9
    im11 = axes[1, 1].imshow(var_u, cmap="plasma", interpolation="nearest",
                              vmin=0, vmax=vmax_var)
    axes[1, 1].set_title(
        f"Var(u) UNMASKED\nmax={var_u.max():.4f}  p99={np.percentile(var_u, 99):.4f}"
    )
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im12 = axes[1, 2].imshow(head_mask, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    axes[1, 2].set_title(
        f"Head mask  (ρ > τ={tau} at flow grid)\n"
        f"{100*head_mask.mean():.1f}% pixels active"
    )
    plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # Row 2 — Var(u) masked | P unmasked | P masked
    im20 = axes[2, 0].imshow(var_u_masked, cmap="plasma", interpolation="nearest",
                              vmin=0, vmax=vmax_var)
    axes[2, 0].set_title(
        f"Var(u) MASKED  τ={tau}\nmax={var_u_masked.max():.4f}  p99={np.percentile(var_u_masked, 99):.4f}"
    )
    plt.colorbar(im20, ax=axes[2, 0], fraction=0.046, pad=0.04)

    im21 = axes[2, 1].imshow(P_unmasked, cmap="turbo", interpolation="nearest",
                              vmin=0, vmax=np.percentile(P_unmasked, 99) + 1e-9)
    axes[2, 1].set_title(
        f"P = ρ · Var(u)  UNMASKED\nmax={P_unmasked.max():.4f}  p99={np.percentile(P_unmasked, 99):.4f}"
    )
    plt.colorbar(im21, ax=axes[2, 1], fraction=0.046, pad=0.04)

    im22 = axes[2, 2].imshow(P_masked, cmap="turbo", interpolation="nearest",
                              vmin=0, vmax=np.percentile(P_unmasked, 99) + 1e-9)
    axes[2, 2].set_title(
        f"P = ρ · Var(u)  MASKED  τ={tau}\nmax={P_masked.max():.4f}  p99={np.percentile(P_masked, 99):.4f}"
    )
    plt.colorbar(im22, ax=axes[2, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose lwcc density + RAFT flow outputs")
    p.add_argument("--frame-t",  required=True, help="Path to frame t  (jpg/png)")
    p.add_argument("--frame-t1", required=True, help="Path to frame t+1 (jpg/png)")
    p.add_argument("--out", default="outputs/diag/diagnostic.png",
                   help="Output PNG path (default: outputs/diag/diagnostic.png)")
    p.add_argument("--lwcc-model",   default="DM-Count",
                   choices=("CSRNet", "SFANet", "Bay", "DM-Count"),
                   help="lwcc model (default: DM-Count)")
    p.add_argument("--lwcc-weights", default="SHA",
                   choices=("SHA", "SHB", "QNRF"),
                   help="lwcc pretrained weights dataset (default: SHA)")
    p.add_argument("--tau", type=float, default=0.01,
                   help="Density threshold for head mask (default: 0.01)")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())

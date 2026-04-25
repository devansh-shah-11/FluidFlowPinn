"""Overlay ρ, u, P on frames and save figures."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # no display in SLURM — must be set before pyplot import

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _denorm_frame(tensor: torch.Tensor) -> np.ndarray:
    """(3,H,W) ImageNet-normalised tensor → (H,W,3) uint8 RGB."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.float().cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _upsample_map(map_hw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinear upsample a (H,W) numpy array to (target_h, target_w)."""
    t = torch.from_numpy(map_hw).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def visualize_sample(
    frame_t: torch.Tensor,
    rho: torch.Tensor,
    u: torch.Tensor,
    P: torch.Tensor,
    gt_density: Optional[torch.Tensor] = None,
    save_path: Optional[str | Path] = None,
    title: str = "",
) -> plt.Figure:
    """Plot a 2×3 grid showing what the model has learned for a single sample.

    Row 1: RGB frame | Predicted density ρ | GT density (or blank)
    Row 2: Flow magnitude |u| | Flow variance Var(u) | Pressure map P

    Args:
        frame_t:    (3, H, W) ImageNet-normalised input frame
        rho:        (1, Hd, Wd) predicted density map
        u:          (2, Hd, Wd) predicted flow field
        P:          (1, Hd, Wd) predicted pressure map
        gt_density: (1, Hd, Wd) ground-truth density map (optional)
        save_path:  if given, figure is saved here as PNG
        title:      optional suptitle string

    Returns:
        matplotlib Figure
    """
    H, W = frame_t.shape[-2:]

    rgb    = _denorm_frame(frame_t)
    rho_np = _to_numpy(rho.squeeze(0).squeeze(0))
    u_np   = _to_numpy(u)             # (2, Hd, Wd)
    P_np   = _to_numpy(P.squeeze(0).squeeze(0))

    # Upsample all maps to frame resolution for overlay
    rho_up = _upsample_map(rho_np, H, W)
    P_up   = _upsample_map(P_np,   H, W)

    # Flow magnitude and per-pixel variance
    u_mag = np.sqrt(u_np[0] ** 2 + u_np[1] ** 2)
    u_var = (
        (u_np[0] - u_np[0].mean()) ** 2 +
        (u_np[1] - u_np[1].mean()) ** 2
    ) / 2.0
    u_mag_up = _upsample_map(u_mag, H, W)
    u_var_up = _upsample_map(u_var, H, W)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    if title:
        fig.suptitle(title, fontsize=13)

    def _overlay(ax, bg, heatmap, label, cmap="jet", alpha=0.55):
        ax.imshow(bg)
        im = ax.imshow(heatmap, cmap=cmap, alpha=alpha)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.set_title(label)
        ax.axis("off")

    # Row 1
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input frame")
    axes[0, 0].axis("off")

    _overlay(axes[0, 1], rgb, rho_up,
             f"Predicted density ρ  (sum={rho_np.sum():.1f})", cmap="hot")

    if gt_density is not None:
        gt_np  = _to_numpy(gt_density.squeeze(0).squeeze(0))
        gt_up  = _upsample_map(gt_np, H, W)
        _overlay(axes[0, 2], rgb, gt_up,
                 f"GT density  (count={gt_np.sum():.1f})", cmap="hot")
    else:
        axes[0, 2].set_visible(False)

    # Row 2
    _overlay(axes[1, 0], rgb, u_mag_up,
             "Flow magnitude |u|  (px/frame)", cmap="cool")
    _overlay(axes[1, 1], rgb, u_var_up,
             "Flow variance Var(u)", cmap="plasma")
    _overlay(axes[1, 2], rgb, P_up,
             "Pressure P = ρ·Var(u)", cmap="RdYlGn_r")

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    return fig


def visualize_batch(
    frames: torch.Tensor,
    rhos: torch.Tensor,
    us: torch.Tensor,
    Ps: torch.Tensor,
    gt_densities: Optional[torch.Tensor] = None,
    save_dir: Optional[str | Path] = None,
    prefix: str = "sample",
) -> list[plt.Figure]:
    """Call visualize_sample for each item in a batch.

    Args:
        frames:       (B, 3, H, W)
        rhos:         (B, 1, Hd, Wd)
        us:           (B, 2, Hd, Wd)
        Ps:           (B, 1, Hd, Wd)
        gt_densities: (B, 1, Hd, Wd) or None
        save_dir:     directory to save PNGs; None = don't save
        prefix:       filename prefix, e.g. "epoch05_val"

    Returns:
        list of matplotlib Figures (one per batch item)
    """
    B = frames.shape[0]
    figs = []
    for i in range(B):
        gt = gt_densities[i] if gt_densities is not None else None
        save_path = (
            Path(save_dir) / f"{prefix}_{i:03d}.png"
            if save_dir is not None else None
        )
        fig = visualize_sample(
            frame_t=frames[i],
            rho=rhos[i],
            u=us[i],
            P=Ps[i],
            gt_density=gt,
            save_path=save_path,
            title=f"{prefix} [{i}]",
        )
        figs.append(fig)
    return figs


def plot_pressure_timeline(
    pressure_scores: list[float],
    fps: float = 30.0,
    threshold: float = 0.5,
    anomaly_frame: Optional[int] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot max-pressure over time for a video sequence — the anomaly detection signal.

    Args:
        pressure_scores: list of max(P) values, one per frame pair
        fps:             video frame rate (for x-axis in seconds)
        threshold:       alert threshold line
        anomaly_frame:   frame index where the anomaly visibly starts (for lead-time annotation)
        save_path:       save figure here if given
    """
    times = [i / fps for i in range(len(pressure_scores))]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, pressure_scores, color="steelblue", linewidth=1.5, label="max P(t)")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"threshold={threshold}")

    if anomaly_frame is not None:
        t_anomaly = anomaly_frame / fps
        ax.axvline(t_anomaly, color="orange", linestyle=":", linewidth=1.5,
                   label=f"visible anomaly @ {t_anomaly:.1f}s")

        # Annotate lead time — first frame where P > threshold before anomaly
        lead_frames = [i for i, p in enumerate(pressure_scores)
                       if i < anomaly_frame and p > threshold]
        if lead_frames:
            t_alert = lead_frames[0] / fps
            lead = t_anomaly - t_alert
            ax.axvline(t_alert, color="green", linestyle=":", linewidth=1.5,
                       label=f"first alert @ {t_alert:.1f}s  (lead={lead:.1f}s)")
            ax.annotate(
                f"lead = {lead:.1f}s",
                xy=(t_alert, threshold),
                xytext=(t_alert + 0.5, threshold * 1.1),
                arrowprops=dict(arrowstyle="->", color="green"),
                color="green",
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max pressure P")
    ax.set_title("Crowd pressure timeline — anomaly detection signal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    return fig

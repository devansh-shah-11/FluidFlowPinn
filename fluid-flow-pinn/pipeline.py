"""Per-frame inference pipeline shared by the live dashboard and eval scripts.

`infer.py` and the eval scripts under `eval/` both need to:
  1. Load RAFT (+ optional lwcc, + optional YOLO).
  2. For each frame pair, compute (rho, u, P, p_max, ang_var) using the
     hand-engineered formula  P = α·density + β·speed + γ·angVar.

Keeping that loop in one place avoids drift between the dashboard and the
metrics we report. `Pipeline.process(frame_bgr)` accepts the next frame, holds
the previous-frame state (tensor + EMA flow), and returns a result dict.

Construction takes the same argparse.Namespace `infer.py` already builds, so
eval scripts can build a minimal namespace with only the fields they need.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.amp import autocast

from infer import (
    _aggregate_pressure,
    _bgr_to_tensor,
    _boxes_to_density,
    _compute_pressure_np,
    _denoise_flow,
    _head_mask,
    _load_lwcc,
    _load_model,
    _load_yolo,
    _local_angular_var_np,
    _lwcc_density_np,
    _temporal_ema,
    _yolo_person_mask,
)


class Pipeline:
    """Stateful per-frame inference pipeline.

    Holds the model handles + previous-frame state. Call `process(frame_bgr)`
    once per frame in arrival order; the first call returns a result with
    `u`, `P`, `p_max`, `ang_var` set to None (no flow available from a single
    frame). Subsequent calls return the full dict.
    """

    def __init__(self, args: argparse.Namespace, device: Optional[torch.device] = None):
        self.args = args
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = Path(args.checkpoint) if getattr(args, "checkpoint", None) else None
        raft_w = Path(args.raft_weights) if getattr(args, "raft_weights", None) else None
        self.flow_model, self.cfg, self.use_fp16 = _load_model(
            ckpt, self.device,
            raft_iters=args.raft_iters,
            raft_weights=raft_w,
            raft_variant=args.raft_variant,
        )

        if getattr(args, "no_lwcc", False):
            args.lwcc_model = None
        self.lwcc_model, self.lwcc_get_count, self.lwcc_tmp = _load_lwcc(
            getattr(args, "lwcc_model", None),
            getattr(args, "lwcc_weights", "SHA"),
            self.device,
        )

        self.yolo_model = (
            _load_yolo(args.detector, self.device)
            if getattr(args, "detector", None) else None
        )

        # Per-frame state
        self.prev_tensor: Optional[torch.Tensor] = None
        self.u_ema: Optional[np.ndarray] = None
        self.frame_idx = 0

    @property
    def has_density(self) -> bool:
        return self.lwcc_model is not None or self.yolo_model is not None

    @torch.no_grad()
    def process(self, frame_bgr: np.ndarray) -> dict:
        """Compute one frame's outputs.

        Returns dict with keys:
          rho        — (Hf, Wf) float32 density map (or None)
          u          — (2, Hf, Wf) float32 flow (or None on first frame)
          P          — (Hf, Wf) float32 pressure map (or None on first frame)
          p_max      — scalar danger score (or None)
          ang_var    — (Hf, Wf) float32 angular variance (or None)
          head_mask  — (Hf, Wf) float32 0/1 mask (or None)
          yolo_boxes — (N, 4) source-pixel boxes (or None)
          frame_idx  — running counter
        """
        args = self.args
        cur_tensor = _bgr_to_tensor(frame_bgr, self.device)

        result: dict = {
            "rho": None, "u": None, "P": None, "p_max": None,
            "ang_var": None, "head_mask": None, "yolo_boxes": None,
            "frame_idx": self.frame_idx,
        }

        if self.prev_tensor is None:
            self.prev_tensor = cur_tensor
            self.frame_idx += 1
            return result

        with autocast(device_type=self.device.type, enabled=self.use_fp16):
            u_t = self.flow_model(self.prev_tensor, cur_tensor)
        u_np = u_t.squeeze(0).detach().float().cpu().numpy()
        target_hw = (u_np.shape[1], u_np.shape[2])

        rho_np: Optional[np.ndarray] = None
        head_mask: Optional[np.ndarray] = None
        yolo_boxes: Optional[np.ndarray] = None

        if self.yolo_model is not None:
            head_mask, yolo_boxes = _yolo_person_mask(
                frame_bgr, self.yolo_model,
                target_hw=target_hw,
                conf=args.detector_conf,
                center_radius=args.center_radius,
            )
            if self.lwcc_model is not None:
                rho_np = _lwcc_density_np(
                    frame_bgr, self.lwcc_model, self.lwcc_get_count,
                    args.lwcc_model, target_hw=target_hw,
                    tmp_path=self.lwcc_tmp,
                )
            else:
                rho_np = _boxes_to_density(
                    yolo_boxes, frame_bgr.shape[:2], target_hw,
                    center_radius=args.center_radius,
                )
        elif self.lwcc_model is not None:
            rho_np = _lwcc_density_np(
                frame_bgr, self.lwcc_model, self.lwcc_get_count,
                args.lwcc_model, target_hw=target_hw,
                tmp_path=self.lwcc_tmp,
            )
            head_mask = (
                _head_mask(rho_np, args.density_mask_tau)
                if args.density_mask_tau > 0.0 else None
            )

        u_np = _denoise_flow(u_np, spatial_sigma=args.flow_spatial_sigma)
        u_np = _temporal_ema(self.u_ema, u_np, alpha=args.flow_ema_alpha)
        self.u_ema = u_np

        P_np = _compute_pressure_np(
            rho_np, u_np,
            p_alpha=args.p_alpha,
            p_beta=args.p_beta,
            p_gamma=args.p_gamma,
            mask=head_mask,
            var_kernel=args.var_kernel,
        )

        # Always compute angular variance for visualization / inspection.
        _m_vis: Optional[np.ndarray] = None
        if head_mask is not None:
            mh, mw = u_np.shape[1], u_np.shape[2]
            _m_vis = head_mask if head_mask.shape == (mh, mw) else cv2.resize(
                head_mask, (mw, mh), interpolation=cv2.INTER_NEAREST)
        ang_var_np = _local_angular_var_np(
            u_np * (_m_vis[np.newaxis] if _m_vis is not None else 1.0),
            k=args.var_kernel, person_mask=_m_vis,
        )

        p_max = _aggregate_pressure(
            P_np, rho_np, mode=args.p_agg, topk_frac=args.topk_frac,
        )

        result.update({
            "rho": rho_np, "u": u_np, "P": P_np, "p_max": p_max,
            "ang_var": ang_var_np, "head_mask": head_mask, "yolo_boxes": yolo_boxes,
        })

        self.prev_tensor = cur_tensor
        self.frame_idx += 1
        return result

    def close(self) -> None:
        if self.lwcc_tmp is not None:
            Path(self.lwcc_tmp).unlink(missing_ok=True)


def default_pipeline_args(**overrides) -> argparse.Namespace:
    """Build a Namespace with the same defaults `infer.py` exposes.

    Eval scripts that don't want to surface every flag can call this and
    override only what they need:
        args = default_pipeline_args(detector="yolo26n.pt", p_gamma=0.0)
    """
    defaults = dict(
        checkpoint=None,
        raft_weights=None,
        raft_variant="small",
        raft_iters=6,
        flow_spatial_sigma=0.0,
        flow_ema_alpha=1.0,
        density_mask_tau=0.01,
        lwcc_model=None,
        lwcc_weights="SHA",
        no_lwcc=False,
        detector=None,
        detector_conf=0.25,
        center_radius=1,
        p_alpha=1.0,
        p_beta=1.0,
        p_gamma=1.0,
        var_kernel=5,
        p_agg="max",
        topk_frac=0.001,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)

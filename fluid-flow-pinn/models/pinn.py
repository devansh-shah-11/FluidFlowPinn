"""Full three-branch PINN: forward(frame_t, frame_t1) → {rho, u, P}."""

from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.branch1_density import CSRNet, load_csrnet
from models.branch2_flow import RAFTFlow, load_raft
from models.branch3_pressure import PressureMap


class FluidFlowPINN(nn.Module):
    """Three-branch Physics-Informed Neural Network for crowd pressure detection.

    Branches:
        1. CSRNet  → density map ρ  (B, 1, H/8, W/8)
        2. RAFT    → flow field u   (B, 2, H/8, W/8)
        3. Pressure → P = ρ · Var(u) (B, 1, H/8, W/8)

    FP16 is handled externally via torch.cuda.amp.autocast(). Gradient
    checkpointing on the CSRNet frontend is toggled by use_grad_checkpoint.

    Args:
        use_grad_checkpoint: enable gradient checkpointing on CSRNet frontend.
        raft_frozen:         freeze RAFT weights (recommended for T4 VRAM budget).
        pressure_window:     spatial window size for local variance in Branch 3.
        csrnet_weights:      optional path to CSRNet checkpoint.
        raft_weights:        optional path to RAFT checkpoint.
    """

    def __init__(
        self,
        use_grad_checkpoint: bool = True,
        raft_frozen: bool = True,
        pressure_window: int = 5,
        csrnet_weights: Optional[str | Path] = None,
        raft_weights: Optional[str | Path] = None,
    ) -> None:
        super().__init__()

        self.density_branch = load_csrnet(
            weights_path=csrnet_weights,
            use_grad_checkpoint=use_grad_checkpoint,
            pretrained_vgg=(csrnet_weights is None),
        )

        self.flow_branch = load_raft(
            weights_path=raft_weights,
            frozen=raft_frozen,
        )

        self.pressure_branch = PressureMap(window_size=pressure_window)

    def forward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            frame_t:  (B, 3, H, W) ImageNet-normalised RGB at time t
            frame_t1: (B, 3, H, W) ImageNet-normalised RGB at time t+1

        Returns:
            dict with:
                'rho': (B, 1, H/8, W/8) density map at t
                'rho_t1': (B, 1, H/8, W/8) density map at t+1
                'u':   (B, 2, H/8, W/8) flow field
                'P':   (B, 1, H/8, W/8) pressure map
        """
        # Branch 1: density at both time steps
        rho_t  = self.density_branch(frame_t)   # (B, 1, H/8, W/8)
        rho_t1 = self.density_branch(frame_t1)  # (B, 1, H/8, W/8)

        # Branch 2: optical flow (downsampled to H/8, W/8 inside RAFTFlow)
        u = self.flow_branch(frame_t, frame_t1)  # (B, 2, H/8, W/8)

        # Branch 3: pressure map
        P = self.pressure_branch(rho_t, u)       # (B, 1, H/8, W/8)

        return {"rho": rho_t, "rho_t1": rho_t1, "u": u, "P": P}

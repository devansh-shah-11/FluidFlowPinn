"""L_total = L_count + λ1·L_motion + λ2·‖R‖²."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.continuity_loss import ContinuityLoss


def _warp_density(rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Bilinear warp of rho_t forward by flow u → approximation of rho_t1.

    Args:
        rho: (B, 1, H, W) density map at time t
        u:   (B, 2, H, W) flow field (ux, uy) in pixel units at density resolution

    Returns:
        (B, 1, H, W) warped density
    """
    B, _, H, W = rho.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=rho.device, dtype=rho.dtype),
        torch.arange(W, device=rho.device, dtype=rho.dtype),
        indexing="ij",
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1) + u[:, 0]
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1) + u[:, 1]
    # Normalise coordinates to [-1, 1] for grid_sample
    grid_x = 2.0 * grid_x / (W - 1) - 1.0
    grid_y = 2.0 * grid_y / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, H, W, 2)
    return F.grid_sample(rho, grid, mode="bilinear", padding_mode="border", align_corners=True)


class TotalLoss(nn.Module):
    """Combined loss for the fluid-flow PINN.

    L_total = L_count + λ1·L_motion + λ2·‖R‖²

    L_count  : MSE between predicted density integral and GT head count.
    L_motion : When GT flow is provided (CrowdFlow): EPE against GT flow.
               Otherwise: density warp consistency — MAE between warp(rho_t, u)
               and rho_t1. Uses RAFT's predicted flow as a self-supervised signal
               so λ1 is always active (e.g. on FDST which has no GT flow).
    ‖R‖²     : squared continuity residual from ContinuityLoss.

    Args:
        lambda1: weight for the motion loss term.
        lambda2: weight for the physics (continuity) loss term.
        fps:     frames-per-second used for the temporal derivative in R.
    """

    def __init__(
        self,
        lambda1: float = 0.1,
        lambda2: float = 0.01,
        fps: float = 30.0,
    ) -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.continuity = ContinuityLoss(fps=fps)

    def forward(
        self,
        rho_t: torch.Tensor,
        rho_t1: torch.Tensor,
        u: torch.Tensor,
        gt_count: torch.Tensor,
        gt_flow: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            rho_t:    predicted density at t,   (B, 1, H, W)
            rho_t1:   predicted density at t+1, (B, 1, H, W)
            u:        predicted flow field,      (B, 2, H, W)
            gt_count: ground-truth head count,   (B,) or (B, 1)
            gt_flow:  ground-truth flow (optional, CrowdFlow only), (B, 2, H, W)

        Returns:
            dict with keys: 'total', 'count', 'motion', 'physics'
        """
        # L_count: MSE between predicted count (sum of density map) and GT count
        pred_count = rho_t.sum(dim=(1, 2, 3))          # (B,)
        gt_count = gt_count.view(-1).float()            # (B,)
        l_count = nn.functional.mse_loss(pred_count, gt_count)

        # L_motion: EPE when GT flow is available; else density warp consistency
        if gt_flow is not None:
            diff = u - gt_flow                          # (B, 2, H, W)
            epe  = diff.norm(p=2, dim=1)               # (B, H, W)
            l_motion = epe.mean()
        else:
            # Self-supervised: warp rho_t with RAFT flow and compare to rho_t1.
            # No detach — gradients flow into CSRNet (rho_t, rho_t1) so it learns
            # to produce density maps consistent with RAFT's frozen flow.
            rho_warped = _warp_density(rho_t, u)
            l_motion = F.l1_loss(rho_warped, rho_t1)

        # L_physics: ||R||² continuity residual
        l_physics = self.continuity(rho_t, rho_t1, u)

        total = l_count + self.lambda1 * l_motion + self.lambda2 * l_physics

        return {
            "total":   total,
            "count":   l_count,
            "motion":  l_motion,
            "physics": l_physics,
        }

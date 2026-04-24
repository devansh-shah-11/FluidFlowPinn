"""L_total = L_count + λ1·L_motion + λ2·‖R‖²."""

import torch
import torch.nn as nn

from losses.continuity_loss import ContinuityLoss


class TotalLoss(nn.Module):
    """Combined loss for the fluid-flow PINN.

    L_total = L_count + λ1·L_motion + λ2·‖R‖²

    L_count  : MSE between predicted density integral and GT head count.
    L_motion : EPE (endpoint error) against GT flow when available, else 0.
    ‖R‖²     : squared continuity residual from ContinuityLoss.

    Args:
        lambda1: weight for the motion (EPE) loss term.
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

        # L_motion: EPE against GT flow when available
        if gt_flow is not None:
            diff = u - gt_flow                          # (B, 2, H, W)
            epe  = diff.norm(p=2, dim=1)               # (B, H, W)
            l_motion = epe.mean()
        else:
            l_motion = rho_t.new_zeros(1).squeeze()

        # L_physics: ||R||² continuity residual
        l_physics = self.continuity(rho_t, rho_t1, u)

        total = l_count + self.lambda1 * l_motion + self.lambda2 * l_physics

        return {
            "total":   total,
            "count":   l_count,
            "motion":  l_motion,
            "physics": l_physics,
        }

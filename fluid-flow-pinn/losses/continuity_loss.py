"""Continuity residual R = ∂ρ/∂t + ∇·(ρu) via finite differences."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuityLoss(nn.Module):
    """nn.Module wrapper around continuity_loss for use in the training loop."""

    def __init__(self, fps: float = 30.0):
        super().__init__()
        self.fps = fps

    def forward(self, rho_t: torch.Tensor, rho_t1: torch.Tensor,
                u: torch.Tensor) -> torch.Tensor:
        return continuity_loss(rho_t, rho_t1, u, fps=self.fps)


def continuity_loss(rho_t: torch.Tensor, rho_t1: torch.Tensor,
                    u: torch.Tensor, fps: float = 30.0) -> torch.Tensor:
    """
    Compute the squared continuity residual ||R||² where
        R = ∂ρ/∂t + ∂(ρ·ux)/∂x + ∂(ρ·uy)/∂y

    Args:
        rho_t:  density map at time t,   shape (B, 1, H, W)
        rho_t1: density map at time t+1, shape (B, 1, H, W)
        u:      flow field (ux, uy),     shape (B, 2, H, W)
                expected to be downsampled to match rho resolution (1/8)
        fps:    frame rate used for temporal derivative

    Returns:
        Scalar: mean of ||R||² over the batch
    """
    dt = 1.0 / fps

    # ∂ρ/∂t  (forward difference in time)
    drho_dt = (rho_t1 - rho_t) / dt                        # (B, 1, H, W)

    ux = u[:, 0:1, :, :]                                    # (B, 1, H, W)
    uy = u[:, 1:2, :, :]                                    # (B, 1, H, W)

    # Use rho_t as the spatial density for the divergence term
    rho_ux = rho_t * ux                                     # (B, 1, H, W)
    rho_uy = rho_t * uy                                     # (B, 1, H, W)

    # ∇·(ρu) — full H×W output, exact at every pixel:
    # central differences in the interior, one-sided at boundaries.
    d_rho_ux_dx = _diff_x(rho_ux)                          # (B, 1, H, W)
    d_rho_uy_dy = _diff_y(rho_uy)                          # (B, 1, H, W)

    R = drho_dt + d_rho_ux_dx + d_rho_uy_dy                # (B, 1, H, W)

    return (R ** 2).mean()


def _diff_x(x: torch.Tensor) -> torch.Tensor:
    """∂/∂col: forward at left edge, backward at right edge, central interior.
    Output is the same shape as input."""
    out = torch.empty_like(x)
    out[:, :, :,  0]   = x[:, :, :,  1] - x[:, :, :,  0]        # forward
    out[:, :, :, -1]   = x[:, :, :, -1] - x[:, :, :, -2]        # backward
    out[:, :, :, 1:-1] = (x[:, :, :, 2:] - x[:, :, :, :-2]) / 2.0
    return out


def _diff_y(x: torch.Tensor) -> torch.Tensor:
    """∂/∂row: forward at top edge, backward at bottom edge, central interior.
    Output is the same shape as input."""
    out = torch.empty_like(x)
    out[:, :,  0, :]   = x[:, :,  1, :] - x[:, :,  0, :]        # forward
    out[:, :, -1, :]   = x[:, :, -1, :] - x[:, :, -2, :]        # backward
    out[:, :, 1:-1, :] = (x[:, :, 2:, :] - x[:, :, :-2, :]) / 2.0
    return out


# Keep old names as aliases so any existing imports don't break
_central_diff_x = _diff_x
_central_diff_y = _diff_y

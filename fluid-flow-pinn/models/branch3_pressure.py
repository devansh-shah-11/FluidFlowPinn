"""Branch 3: Pressure map — P = ρ · Var(u) over a local spatial window."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PressureMap(nn.Module):
    """Computes crowd pressure as P = ρ · local_var(u).

    Local variance of the flow field is estimated using a sliding window
    (uniform average pooling): Var(u) = E[u²] - E[u]².

    Input:
        rho: (B, 1, H, W)  density map from Branch 1
        u:   (B, 2, H, W)  flow field from Branch 2 (same spatial resolution)
    Output:
        P:   (B, 1, H, W)  pressure map
    """

    def __init__(self, window_size: int = 5) -> None:
        super().__init__()
        self.window_size = window_size
        self.padding = window_size // 2

    def forward(self, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rho: (B, 1, H, W) density map
            u:   (B, 2, H, W) flow field [ux, uy]
        Returns:
            P:   (B, 1, H, W) pressure map
        """
        var_u = _local_variance(u, self.window_size, self.padding)  # (B, 1, H, W)
        P = rho * var_u
        return P


def _local_variance(u: torch.Tensor, window_size: int, padding: int) -> torch.Tensor:
    """Compute per-pixel local variance of the 2-channel flow field.

    Var(u) = mean_over_channels( E[u²] - E[u]² ) over the local window.

    Returns shape (B, 1, H, W).
    """
    # Average pooling kernel — treats each channel independently
    ux = u[:, 0:1, :, :]  # (B, 1, H, W)
    uy = u[:, 1:2, :, :]  # (B, 1, H, W)

    var_x = _channel_local_var(ux, window_size, padding)  # (B, 1, H, W)
    var_y = _channel_local_var(uy, window_size, padding)  # (B, 1, H, W)

    return (var_x + var_y) / 2.0


def _channel_local_var(x: torch.Tensor, window_size: int, padding: int) -> torch.Tensor:
    """Local variance of a single-channel tensor via avg_pool2d."""
    E_x2 = F.avg_pool2d(x ** 2, kernel_size=window_size, stride=1, padding=padding)
    E_x  = F.avg_pool2d(x,      kernel_size=window_size, stride=1, padding=padding)
    # clamp to avoid tiny negatives from floating-point error
    return (E_x2 - E_x ** 2).clamp(min=0.0)

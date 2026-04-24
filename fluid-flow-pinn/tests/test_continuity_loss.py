"""Tests for losses/continuity_loss.py (Step 6)."""

import pytest
import torch
from losses.continuity_loss import continuity_loss, _diff_x, _diff_y

B, H, W = 2, 60, 80   # 1/8 of a 480×640 frame


# ── helper ────────────────────────────────────────────────────────────────────

def _zeros_like_rho():
    return torch.zeros(B, 1, H, W)

def _ones_like_rho(val=1.0):
    return torch.full((B, 1, H, W), val)

def _zeros_like_u():
    return torch.zeros(B, 2, H, W)


# ── output contract ───────────────────────────────────────────────────────────

def test_returns_scalar():
    loss = continuity_loss(_ones_like_rho(), _ones_like_rho(), _zeros_like_u())
    assert loss.shape == torch.Size([]), "loss must be a scalar tensor"


def test_returns_non_negative():
    rho_t  = torch.rand(B, 1, H, W)
    rho_t1 = torch.rand(B, 1, H, W)
    u      = torch.rand(B, 2, H, W)
    assert continuity_loss(rho_t, rho_t1, u).item() >= 0.0


def test_output_is_finite():
    rho_t  = torch.rand(B, 1, H, W)
    rho_t1 = torch.rand(B, 1, H, W)
    u      = torch.rand(B, 2, H, W) * 2 - 1
    assert torch.isfinite(continuity_loss(rho_t, rho_t1, u))


# ── physics correctness ───────────────────────────────────────────────────────

def test_zero_residual_static_zero_flow():
    """Constant density + zero flow → R = 0 everywhere."""
    rho = _ones_like_rho(5.0)
    u   = _zeros_like_u()
    loss = continuity_loss(rho, rho, u)
    assert loss.item() < 1e-10, f"Expected ~0, got {loss.item()}"


def test_zero_residual_uniform_flow_constant_density():
    """Uniform (spatially constant) flow + constant density: divergence term
    ∂(ρ·ux)/∂x = 0 and ∂ρ/∂t = 0, so R should be 0."""
    rho = _ones_like_rho(3.0)
    u   = torch.full((B, 2, H, W), 0.7)
    loss = continuity_loss(rho, rho, u)
    assert loss.item() < 1e-10, f"Expected ~0, got {loss.item()}"


def test_nonzero_residual_when_density_changes():
    """If density changes over time but flow is zero, ∂ρ/∂t ≠ 0 → R ≠ 0."""
    rho_t  = _ones_like_rho(1.0)
    rho_t1 = _ones_like_rho(2.0)   # density doubled
    u      = _zeros_like_u()
    loss = continuity_loss(rho_t, rho_t1, u)
    assert loss.item() > 0.0, "Loss should be positive when density changes"


def test_fps_scales_temporal_derivative():
    """Higher fps → smaller dt → larger ∂ρ/∂t → larger loss."""
    rho_t  = _ones_like_rho(1.0)
    rho_t1 = _ones_like_rho(2.0)
    u      = _zeros_like_u()
    loss_30  = continuity_loss(rho_t, rho_t1, u, fps=30.0)
    loss_10  = continuity_loss(rho_t, rho_t1, u, fps=10.0)
    assert loss_30.item() > loss_10.item(), \
        "Higher fps should produce larger temporal derivative and larger loss"


def test_loss_differentiable():
    """Gradients must flow through rho and u (needed for training)."""
    rho_t  = torch.rand(B, 1, H, W, requires_grad=True)
    rho_t1 = torch.rand(B, 1, H, W, requires_grad=True)
    u      = torch.rand(B, 2, H, W, requires_grad=True)
    loss = continuity_loss(rho_t, rho_t1, u)
    loss.backward()
    assert rho_t.grad  is not None
    assert rho_t1.grad is not None
    assert u.grad      is not None


# ── finite-difference helpers ─────────────────────────────────────────────────

def test_diff_x_linear():
    """For f(x) = c·x, _diff_x should equal c at every pixel including boundaries."""
    c = 3.0
    cols = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W).expand(B, 1, H, W)
    dx = _diff_x(c * cols)
    assert torch.allclose(dx, torch.full_like(dx, c), atol=1e-5), \
        f"Max deviation: {(dx - c).abs().max().item()}"


def test_diff_y_linear():
    """For f(y) = c·y, _diff_y should equal c at every pixel including boundaries."""
    c = 2.5
    rows = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1).expand(B, 1, H, W)
    dy = _diff_y(c * rows)
    assert torch.allclose(dy, torch.full_like(dy, c), atol=1e-5), \
        f"Max deviation: {(dy - c).abs().max().item()}"


def test_diff_output_shape():
    """Both diff helpers must preserve input shape."""
    x = torch.rand(B, 1, H, W)
    assert _diff_x(x).shape == (B, 1, H, W)
    assert _diff_y(x).shape == (B, 1, H, W)

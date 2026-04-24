"""Tests for models/branch3_pressure.py (Step 7)."""

import pytest
import torch
from models.branch3_pressure import PressureMap, _local_variance, _channel_local_var

B, H, W = 2, 60, 80  # 1/8 of a 480×640 frame


# ── output contract ───────────────────────────────────────────────────────────

def test_output_shape():
    pm = PressureMap(window_size=5)
    rho = torch.rand(B, 1, H, W)
    u   = torch.rand(B, 2, H, W)
    P   = pm(rho, u)
    assert P.shape == (B, 1, H, W), f"Expected ({B}, 1, {H}, {W}), got {P.shape}"


def test_output_non_negative():
    """Pressure must be non-negative (ρ ≥ 0, Var ≥ 0)."""
    pm  = PressureMap(window_size=5)
    rho = torch.rand(B, 1, H, W).abs()
    u   = torch.rand(B, 2, H, W)
    P   = pm(rho, u)
    assert (P >= 0).all(), "Pressure map contains negative values"


def test_output_finite():
    pm  = PressureMap(window_size=5)
    rho = torch.rand(B, 1, H, W)
    u   = torch.rand(B, 2, H, W) * 10
    P   = pm(rho, u)
    assert torch.isfinite(P).all(), "Pressure map contains non-finite values"


# ── physics correctness ───────────────────────────────────────────────────────

def test_zero_pressure_zero_density():
    """If ρ = 0, pressure must be 0 regardless of flow."""
    pm  = PressureMap(window_size=5)
    rho = torch.zeros(B, 1, H, W)
    u   = torch.rand(B, 2, H, W)
    P   = pm(rho, u)
    assert P.abs().max().item() == 0.0, "Zero density should yield zero pressure"


def test_zero_pressure_uniform_flow():
    """Spatially uniform flow has zero local variance → zero pressure in interior."""
    pm  = PressureMap(window_size=5)
    rho = torch.rand(B, 1, H, W).abs() + 0.1  # positive densities
    u   = torch.ones(B, 2, H, W) * 3.7         # perfectly uniform flow
    P   = pm(rho, u)
    # Border pixels see zero-padded windows so their variance is non-zero; skip them.
    p = 5 // 2  # half window
    P_interior = P[:, :, p:-p, p:-p]
    assert P_interior.abs().max().item() < 1e-5, \
        f"Uniform flow should give ~0 interior pressure, got max={P_interior.abs().max().item()}"


def test_higher_variance_higher_pressure():
    """Higher flow variance should yield higher pressure for the same density."""
    pm  = PressureMap(window_size=5)
    rho = torch.ones(B, 1, H, W)

    # Low-variance flow: nearly uniform
    u_low  = torch.ones(B, 2, H, W) + torch.randn(B, 2, H, W) * 0.01
    # High-variance flow: strongly random
    u_high = torch.randn(B, 2, H, W) * 5.0

    P_low  = pm(rho, u_low).mean().item()
    P_high = pm(rho, u_high).mean().item()
    assert P_high > P_low, \
        f"Expected P_high ({P_high:.4f}) > P_low ({P_low:.4f})"


def test_pressure_scales_with_density():
    """Doubling ρ should double P (linear in density)."""
    pm   = PressureMap(window_size=5)
    rho1 = torch.rand(B, 1, H, W).abs() + 0.5
    rho2 = rho1 * 2.0
    u    = torch.randn(B, 2, H, W)

    # Variance is the same for both; only rho differs
    P1 = pm(rho1, u)
    P2 = pm(rho2, u)
    ratio = (P2 / (P1 + 1e-8)).mean().item()
    assert abs(ratio - 2.0) < 0.01, \
        f"Expected pressure ratio ~2.0, got {ratio:.4f}"


# ── window size ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("window_size", [3, 5, 7, 11])
def test_various_window_sizes(window_size):
    pm  = PressureMap(window_size=window_size)
    rho = torch.rand(B, 1, H, W)
    u   = torch.rand(B, 2, H, W)
    P   = pm(rho, u)
    assert P.shape == (B, 1, H, W)
    assert torch.isfinite(P).all()


# ── differentiability ─────────────────────────────────────────────────────────

def test_pressure_differentiable_wrt_rho():
    pm  = PressureMap(window_size=5)
    rho = torch.rand(B, 1, H, W, requires_grad=True)
    u   = torch.rand(B, 2, H, W)
    P   = pm(rho, u)
    P.sum().backward()
    assert rho.grad is not None, "No gradient w.r.t. rho"


def test_pressure_differentiable_wrt_u():
    pm  = PressureMap(window_size=5)
    rho = torch.rand(B, 1, H, W)
    u   = torch.rand(B, 2, H, W, requires_grad=True)
    P   = pm(rho, u)
    P.sum().backward()
    assert u.grad is not None, "No gradient w.r.t. u"


# ── internal helper ───────────────────────────────────────────────────────────

def test_channel_local_var_non_negative():
    x   = torch.randn(B, 1, H, W) * 3
    var = _channel_local_var(x, window_size=5, padding=2)
    assert (var >= 0).all(), "Local variance must be non-negative"


def test_local_variance_shape():
    u   = torch.rand(B, 2, H, W)
    var = _local_variance(u, window_size=5, padding=2)
    assert var.shape == (B, 1, H, W), f"Expected ({B}, 1, {H}, {W}), got {var.shape}"

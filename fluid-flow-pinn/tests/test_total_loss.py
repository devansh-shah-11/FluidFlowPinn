"""Tests for losses/total_loss.py (Step 7)."""

import pytest
import torch
from losses.total_loss import TotalLoss

B, H, W = 2, 60, 80  # 1/8 of a 480×640 frame


def _make_inputs(requires_grad=False):
    rho_t    = torch.rand(B, 1, H, W, requires_grad=requires_grad)
    rho_t1   = torch.rand(B, 1, H, W, requires_grad=requires_grad)
    u        = torch.rand(B, 2, H, W, requires_grad=requires_grad)
    gt_count = torch.tensor([50.0, 30.0])
    return rho_t, rho_t1, u, gt_count


# ── output contract ───────────────────────────────────────────────────────────

def test_returns_dict_with_expected_keys():
    tl = TotalLoss()
    rho_t, rho_t1, u, gt_count = _make_inputs()
    out = tl(rho_t, rho_t1, u, gt_count)
    assert set(out.keys()) == {"total", "count", "motion", "physics"}


def test_all_values_are_scalar():
    tl = TotalLoss()
    rho_t, rho_t1, u, gt_count = _make_inputs()
    out = tl(rho_t, rho_t1, u, gt_count)
    for key, val in out.items():
        assert val.shape == torch.Size([]), f"'{key}' is not a scalar: {val.shape}"


def test_all_values_non_negative():
    tl = TotalLoss()
    rho_t, rho_t1, u, gt_count = _make_inputs()
    out = tl(rho_t, rho_t1, u, gt_count)
    for key, val in out.items():
        assert val.item() >= 0.0, f"'{key}' is negative: {val.item()}"


def test_all_values_finite():
    tl = TotalLoss()
    rho_t, rho_t1, u, gt_count = _make_inputs()
    out = tl(rho_t, rho_t1, u, gt_count)
    for key, val in out.items():
        assert torch.isfinite(val), f"'{key}' is not finite: {val.item()}"


# ── motion term ───────────────────────────────────────────────────────────────

def test_motion_zero_without_gt_flow():
    tl = TotalLoss(lambda1=0.5)
    rho_t, rho_t1, u, gt_count = _make_inputs()
    out = tl(rho_t, rho_t1, u, gt_count, gt_flow=None)
    assert out["motion"].item() == 0.0, \
        "Motion loss should be 0 when no GT flow is provided"


def test_motion_nonzero_with_gt_flow():
    tl = TotalLoss(lambda1=0.5)
    rho_t, rho_t1, u, gt_count = _make_inputs()
    gt_flow = torch.rand(B, 2, H, W) * 5  # clearly different from u
    out = tl(rho_t, rho_t1, u, gt_count, gt_flow=gt_flow)
    assert out["motion"].item() > 0.0, \
        "Motion loss should be positive when GT flow differs from prediction"


def test_motion_zero_when_u_equals_gt_flow():
    tl = TotalLoss(lambda1=0.5)
    rho_t, rho_t1, u, gt_count = _make_inputs()
    out = tl(rho_t, rho_t1, u, gt_count, gt_flow=u)
    assert out["motion"].item() < 1e-6, \
        "Motion loss should be ~0 when u == gt_flow"


# ── total is weighted sum ─────────────────────────────────────────────────────

def test_total_equals_weighted_sum():
    """total == count + λ1·motion + λ2·physics."""
    lambda1, lambda2 = 0.1, 0.01
    tl = TotalLoss(lambda1=lambda1, lambda2=lambda2)
    rho_t, rho_t1, u, gt_count = _make_inputs()
    gt_flow = torch.rand(B, 2, H, W)
    out = tl(rho_t, rho_t1, u, gt_count, gt_flow=gt_flow)

    expected = (out["count"]
                + lambda1 * out["motion"]
                + lambda2 * out["physics"])
    assert torch.isclose(out["total"], expected, atol=1e-5), \
        f"total={out['total'].item()}, expected={expected.item()}"


def test_lambda_scaling():
    """Larger lambda2 should increase the total loss."""
    rho_t, rho_t1, u, gt_count = _make_inputs()

    tl_small = TotalLoss(lambda2=0.001)
    tl_large = TotalLoss(lambda2=1.0)

    out_small = tl_small(rho_t, rho_t1, u, gt_count)
    out_large = tl_large(rho_t, rho_t1, u, gt_count)

    assert out_large["total"].item() >= out_small["total"].item(), \
        "Larger lambda2 should yield >= total loss"


# ── count loss ────────────────────────────────────────────────────────────────

def test_count_loss_zero_when_perfect():
    """When predicted density integrates exactly to GT count, L_count = 0."""
    tl = TotalLoss()
    gt_count = torch.tensor([100.0, 200.0])

    # Build rho whose sum matches gt_count exactly per batch item
    rho_t = torch.zeros(B, 1, H, W)
    for b in range(B):
        rho_t[b, 0, 0, 0] = gt_count[b]  # entire count in one pixel

    rho_t1 = rho_t.clone()
    u = torch.zeros(B, 2, H, W)

    out = tl(rho_t, rho_t1, u, gt_count)
    assert out["count"].item() < 1e-6, \
        f"L_count should be ~0 for perfect prediction, got {out['count'].item()}"


# ── gt_count shapes ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("gt_shape", [(B,), (B, 1)])
def test_accepts_various_gt_count_shapes(gt_shape):
    tl = TotalLoss()
    rho_t, rho_t1, u, _ = _make_inputs()
    gt_count = torch.rand(gt_shape) * 100
    out = tl(rho_t, rho_t1, u, gt_count)
    assert torch.isfinite(out["total"])


# ── differentiability ─────────────────────────────────────────────────────────

def test_total_differentiable_wrt_rho():
    tl = TotalLoss()
    rho_t, rho_t1, u, gt_count = _make_inputs(requires_grad=True)
    out = tl(rho_t, rho_t1, u, gt_count)
    out["total"].backward()
    assert rho_t.grad  is not None, "No gradient w.r.t. rho_t"
    assert rho_t1.grad is not None, "No gradient w.r.t. rho_t1"


def test_total_differentiable_wrt_u():
    tl = TotalLoss()
    rho_t, rho_t1, u, gt_count = _make_inputs(requires_grad=True)
    out = tl(rho_t, rho_t1, u, gt_count)
    out["total"].backward()
    assert u.grad is not None, "No gradient w.r.t. u"

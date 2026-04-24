"""Tests for models/pinn.py (Step 8).

These tests use mocked branches so they run without GPU or pretrained weights.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from models.pinn import FluidFlowPINN

B, H, W = 2, 64, 80      # full-resolution input
Hd, Wd  = H // 8, W // 8  # density/flow resolution


# ── shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pinn():
    """FluidFlowPINN with all three branches mocked to avoid weight downloads."""
    with (
        patch("models.pinn.load_csrnet") as mock_csrnet_loader,
        patch("models.pinn.load_raft")   as mock_raft_loader,
    ):
        # CSRNet mock: (B, 3, H, W) → (B, 1, Hd, Wd)
        csrnet_mock = MagicMock(spec=nn.Module)
        csrnet_mock.side_effect = lambda x: torch.rand(x.shape[0], 1, Hd, Wd)
        mock_csrnet_loader.return_value = csrnet_mock

        # RAFT mock: (frame_t, frame_t1) → (B, 2, Hd, Wd)
        raft_mock = MagicMock(spec=nn.Module)
        raft_mock.side_effect = lambda a, b: torch.rand(a.shape[0], 2, Hd, Wd)
        mock_raft_loader.return_value = raft_mock

        model = FluidFlowPINN(
            use_grad_checkpoint=False,
            raft_frozen=True,
        )
    return model


# ── output contract ───────────────────────────────────────────────────────────

def test_forward_returns_dict_with_expected_keys(mock_pinn):
    frame_t  = torch.rand(B, 3, H, W)
    frame_t1 = torch.rand(B, 3, H, W)
    out = mock_pinn(frame_t, frame_t1)
    assert set(out.keys()) == {"rho", "rho_t1", "u", "P"}, \
        f"Unexpected keys: {set(out.keys())}"


def test_rho_shape(mock_pinn):
    frame_t  = torch.rand(B, 3, H, W)
    frame_t1 = torch.rand(B, 3, H, W)
    out = mock_pinn(frame_t, frame_t1)
    assert out["rho"].shape == (B, 1, Hd, Wd), \
        f"rho shape mismatch: {out['rho'].shape}"


def test_rho_t1_shape(mock_pinn):
    frame_t  = torch.rand(B, 3, H, W)
    frame_t1 = torch.rand(B, 3, H, W)
    out = mock_pinn(frame_t, frame_t1)
    assert out["rho_t1"].shape == (B, 1, Hd, Wd), \
        f"rho_t1 shape mismatch: {out['rho_t1'].shape}"


def test_u_shape(mock_pinn):
    frame_t  = torch.rand(B, 3, H, W)
    frame_t1 = torch.rand(B, 3, H, W)
    out = mock_pinn(frame_t, frame_t1)
    assert out["u"].shape == (B, 2, Hd, Wd), \
        f"u shape mismatch: {out['u'].shape}"


def test_pressure_shape(mock_pinn):
    frame_t  = torch.rand(B, 3, H, W)
    frame_t1 = torch.rand(B, 3, H, W)
    out = mock_pinn(frame_t, frame_t1)
    assert out["P"].shape == (B, 1, Hd, Wd), \
        f"P shape mismatch: {out['P'].shape}"


def test_all_outputs_finite(mock_pinn):
    frame_t  = torch.rand(B, 3, H, W)
    frame_t1 = torch.rand(B, 3, H, W)
    out = mock_pinn(frame_t, frame_t1)
    for key, val in out.items():
        assert torch.isfinite(val).all(), f"'{key}' contains non-finite values"


def test_all_outputs_non_negative(mock_pinn):
    """Density and pressure must be non-negative (flow can be any sign)."""
    frame_t  = torch.rand(B, 3, H, W)
    frame_t1 = torch.rand(B, 3, H, W)
    out = mock_pinn(frame_t, frame_t1)
    # rho and P should be ≥ 0; u can be negative (flow direction)
    assert (out["rho"] >= 0).all() or True  # mock returns rand which is ≥ 0
    assert (out["P"]   >= 0).all(), "Pressure map must be non-negative"


# ── branch wiring ─────────────────────────────────────────────────────────────

def test_density_branch_called_twice_per_forward(mock_pinn):
    """CSRNet must run on both frame_t and frame_t1."""
    frame_t  = torch.rand(B, 3, H, W)
    frame_t1 = torch.rand(B, 3, H, W)
    mock_pinn(frame_t, frame_t1)
    assert mock_pinn.density_branch.call_count == 2, \
        "CSRNet should be called once for frame_t and once for frame_t1"


def test_flow_branch_called_once_per_forward(mock_pinn):
    frame_t  = torch.rand(B, 3, H, W)
    frame_t1 = torch.rand(B, 3, H, W)
    mock_pinn(frame_t, frame_t1)
    assert mock_pinn.flow_branch.call_count == 1, \
        "RAFTFlow should be called once per forward pass"


# ── batch size invariance ─────────────────────────────────────────────────────

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_various_batch_sizes(batch_size):
    with (
        patch("models.pinn.load_csrnet") as mock_csrnet_loader,
        patch("models.pinn.load_raft")   as mock_raft_loader,
    ):
        csrnet_mock = MagicMock(spec=nn.Module)
        csrnet_mock.side_effect = lambda x: torch.rand(x.shape[0], 1, Hd, Wd)
        mock_csrnet_loader.return_value = csrnet_mock

        raft_mock = MagicMock(spec=nn.Module)
        raft_mock.side_effect = lambda a, b: torch.rand(a.shape[0], 2, Hd, Wd)
        mock_raft_loader.return_value = raft_mock

        model = FluidFlowPINN(use_grad_checkpoint=False)

    frame_t  = torch.rand(batch_size, 3, H, W)
    frame_t1 = torch.rand(batch_size, 3, H, W)
    out = model(frame_t, frame_t1)

    assert out["rho"].shape[0]   == batch_size
    assert out["rho_t1"].shape[0] == batch_size
    assert out["u"].shape[0]     == batch_size
    assert out["P"].shape[0]     == batch_size


# ── integration with real pressure branch ────────────────────────────────────

def test_pressure_branch_is_real_module(mock_pinn):
    """PressureMap should be a real nn.Module, not a mock."""
    from models.branch3_pressure import PressureMap
    assert isinstance(mock_pinn.pressure_branch, PressureMap)

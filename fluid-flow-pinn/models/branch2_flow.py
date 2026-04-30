"""Branch 2: optical-flow estimator — u=(ux,uy) at (B,2,H/8,W/8).

Two interchangeable backends behind the same forward(frame_t, frame_t1) API:
  • RAFTFlow      — torchvision RAFT-Small, pairwise flow.
  • AllTrackerFlow — AllTracker (Harley et al. 2025), multi-frame anchor tracker.
                     Maintains a rolling window of recent frames and returns
                     the pairwise displacement between the last two frames,
                     derived from the anchor→t trajectories.

Both produce u: (B, 2, H/8, W/8) in coarse-grid pixels/frame, so downstream
PINN code (continuity loss, PressureMap) is unchanged.

Use load_flow_branch(backend=...) to pick a backend from config.
"""

from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RAFT (original) ──────────────────────────────────────────────────────────

class RAFTFlow(nn.Module):
    """Wraps torchvision pretrained RAFT-Small to produce crowd velocity fields.

    Input:  frame_t, frame_t1 — (B, 3, H, W) ImageNet-normalised RGB
    Output: u — (B, 2, H/8, W/8) velocity field (ux, uy) in pixels/frame
    """

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, frozen: bool = True, num_flow_updates: int = 12) -> None:
        super().__init__()
        self.num_flow_updates = num_flow_updates

        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        self.raft = raft_small(weights=Raft_Small_Weights.C_T_V2)

        if frozen:
            self._freeze()

    def set_trainable(self, trainable: bool) -> None:
        for p in self.raft.parameters():
            p.requires_grad_(trainable)

    def forward(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        H, W = frame_t.shape[-2:]
        img1 = self._denorm(frame_t)
        img2 = self._denorm(frame_t1)

        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h or pad_w:
            img1 = F.pad(img1, (0, pad_w, 0, pad_h))
            img2 = F.pad(img2, (0, pad_w, 0, pad_h))

        flow_predictions = self.raft(img1, img2, num_flow_updates=self.num_flow_updates)
        flow = flow_predictions[-1][:, :, :H, :W]  # crop padding, (B, 2, H, W)

        Hd, Wd = H // 8, W // 8
        u = F.interpolate(flow, size=(Hd, Wd), mode="bilinear", align_corners=False)
        u = u / 8.0  # rescale to coarse-grid units
        return u

    def _freeze(self) -> None:
        for p in self.raft.parameters():
            p.requires_grad_(False)
        self.raft.eval()

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._IMAGENET_MEAN.to(x.device)
        std  = self._IMAGENET_STD.to(x.device)
        return (x * std + mean).clamp(0.0, 1.0) * 255.0


def load_raft(
    weights_path: Optional[str | Path] = None,
    frozen: bool = True,
    num_flow_updates: int = 12,
) -> RAFTFlow:
    model = RAFTFlow(frozen=frozen, num_flow_updates=num_flow_updates)

    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"[RAFTFlow] Loaded weights from {weights_path}")

    return model


# ── AllTracker (new) ─────────────────────────────────────────────────────────

# AllTracker is a multi-frame point-tracker. Given a clip (B,T,3,H,W) it
# returns flow_anchor[t] = (frame0 → frame_t) for t=0..T-1. To match RAFT's
# pairwise contract we differentiate consecutive anchor flows:
#       u_pairwise(t) = flow_anchor[t] - flow_anchor[t-1]
# and downsample to the H/8 grid.
#
# For real-time use we keep a rolling deque of the last `window_len` frames.
# Until the buffer holds ≥2 frames, forward() returns zeros and `valid=False`
# (caller can treat that as "warmup"). The intended call pattern in train/
# infer is unchanged: forward(frame_t, frame_t1) — but here `frame_t` is
# *only used to seed the buffer on the very first call*; subsequent calls
# only need to supply the new frame (frame_t1). We keep the two-arg signature
# for drop-in compatibility.

class AllTrackerFlow(nn.Module):
    """Streaming AllTracker → pairwise flow on the H/8 grid.

    Input:  frame_t, frame_t1 — (B, 3, H, W) ImageNet-normalised RGB
            (B must be 1 in streaming mode; AllTracker's sliding window is
             a per-clip operation.)
    Output: u — (B, 2, H/8, W/8) velocity field (ux, uy) in pixels/frame
            (zeros during warmup; check `self.is_warm` to detect it).

    Args:
        window_len: AllTracker temporal window (default 16). Larger = more
                    context but more compute & VRAM per frame.
        inference_iters: refinement iters in AllTracker.forward_sliding (default 4).
        tiny: use the lighter `alltracker_tiny.pth` checkpoint.
        weights_path: optional local checkpoint .pth (else downloaded from HF).
        repo_path: optional path to a local clone of `aharley/alltracker`. If
                    not given, we expect the repo to be importable on PYTHONPATH.
    """

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    _HF_URL_FULL = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
    _HF_URL_TINY = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker_tiny.pth"

    def __init__(
        self,
        window_len: int = 16,
        inference_iters: int = 4,
        tiny: bool = False,
        weights_path: Optional[str | Path] = None,
        repo_path: Optional[str | Path] = None,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        self.window_len = int(window_len)
        self.inference_iters = int(inference_iters)
        self.tiny = bool(tiny)

        # Make the AllTracker repo importable. The user must clone
        # https://github.com/aharley/alltracker somewhere; either add it to
        # PYTHONPATH or pass repo_path.
        if repo_path is not None:
            import sys
            rp = str(Path(repo_path).expanduser().resolve())
            if rp not in sys.path:
                sys.path.insert(0, rp)

        try:
            from nets.alltracker import Net  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Could not import `nets.alltracker.Net`. Clone "
                "https://github.com/aharley/alltracker and either add it to "
                "PYTHONPATH or pass repo_path=... to AllTrackerFlow."
            ) from e

        if self.tiny:
            self.net = Net(self.window_len, use_basicencoder=True, no_split=True)
        else:
            self.net = Net(self.window_len)

        # Load weights
        url = self._HF_URL_TINY if self.tiny else self._HF_URL_FULL
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.net.load_state_dict(state, strict=True)
            print(f"[AllTrackerFlow] Loaded weights from {weights_path}")
        else:
            state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.net.load_state_dict(state, strict=True)
            print(f"[AllTrackerFlow] Loaded weights from {url}")

        if frozen:
            self._freeze()

        # Rolling buffer of the last `window_len` frames as raw [0,255] RGB
        # tensors (3, H, W) on the model's device. We keep raw, *not*
        # ImageNet-normalised, frames because AllTracker normalises internally.
        self._buf: deque[torch.Tensor] = deque(maxlen=self.window_len)
        self._last_seen_t: Optional[int] = None  # id() of the last frame_t seen, to detect resets

    @property
    def is_warm(self) -> bool:
        return len(self._buf) >= 2

    def reset(self) -> None:
        """Clear the rolling buffer (call between independent clips)."""
        self._buf.clear()
        self._last_seen_t = None

    def set_trainable(self, trainable: bool) -> None:
        for p in self.net.parameters():
            p.requires_grad_(trainable)

    def forward(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        """Streaming forward.

        On the first call, both `frame_t` and `frame_t1` are pushed into the
        buffer. On subsequent calls only `frame_t1` is pushed (we assume
        `frame_t` is the previous call's `frame_t1`). Output is the
        pairwise displacement (last-1 → last) on the H/8 grid.
        """
        assert frame_t.shape == frame_t1.shape, "frame_t/frame_t1 shape mismatch"
        B, C, H, W = frame_t1.shape
        assert C == 3, f"expected 3 channels, got {C}"
        if B != 1:
            raise NotImplementedError(
                "AllTrackerFlow streaming mode requires B=1. "
                "For multi-clip training, instantiate one buffer per clip."
            )

        # Seed buffer on first call of a new stream
        if not self._buf:
            self._buf.append(self._to_raw(frame_t)[0])
        self._buf.append(self._to_raw(frame_t1)[0])
        self._last_seen_t = id(frame_t1)

        Hd, Wd = H // 8, W // 8
        if not self.is_warm:
            # Should never happen (we just pushed ≥2), but defensive.
            return torch.zeros((B, 2, Hd, Wd), dtype=frame_t1.dtype,
                               device=frame_t1.device)

        # Build (1, T, 3, H, W) clip from the buffer. AllTracker padder needs
        # H,W divisible by 8; we ensure that by padding here and unpadding the
        # output, mirroring RAFTFlow's pattern.
        T = len(self._buf)
        clip = torch.stack(list(self._buf), dim=0).unsqueeze(0)  # (1, T, 3, H, W)
        clip = clip.to(device=frame_t1.device, dtype=torch.float32)

        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h or pad_w:
            clip = F.pad(clip.view(T, 3, H, W), (0, pad_w, 0, pad_h))
            clip = clip.view(1, T, 3, H + pad_h, W + pad_w)

        # forward_sliding returns (B, T, 2, H, W) flow_anchor[t] = frame0→t
        full_flows, _, _, _ = self.net.forward_sliding(
            clip, iters=self.inference_iters, sw=None, is_training=False,
        )
        # full_flows is on cpu in the multi-frame branch; move back if needed.
        full_flows = full_flows.to(frame_t1.device)
        # Crop padding
        full_flows = full_flows[:, :, :, :H, :W]  # (1, T, 2, H, W)

        # Pairwise = anchor[T-1] - anchor[T-2]
        flow_pair = full_flows[:, -1] - full_flows[:, -2]  # (1, 2, H, W)

        # Downsample to H/8 grid (area-avg keeps magnitude correct for a
        # uniform field; same as torchvision interpolate bilinear, then /8 for
        # coarse-grid units — matches the RAFT branch contract).
        u = F.interpolate(flow_pair, size=(Hd, Wd),
                          mode="bilinear", align_corners=False)
        u = u / 8.0
        return u

    def _freeze(self) -> None:
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

    def _to_raw(self, x_normed: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalised RGB (B,3,H,W) → raw [0,255] RGB (B,3,H,W)."""
        mean = self._IMAGENET_MEAN.to(x_normed.device, dtype=x_normed.dtype)
        std  = self._IMAGENET_STD.to(x_normed.device,  dtype=x_normed.dtype)
        return (x_normed * std + mean).clamp(0.0, 1.0) * 255.0


def load_alltracker(
    weights_path: Optional[str | Path] = None,
    repo_path: Optional[str | Path] = None,
    window_len: int = 16,
    inference_iters: int = 4,
    tiny: bool = False,
    frozen: bool = True,
) -> AllTrackerFlow:
    return AllTrackerFlow(
        window_len=window_len,
        inference_iters=inference_iters,
        tiny=tiny,
        weights_path=weights_path,
        repo_path=repo_path,
        frozen=frozen,
    )


# ── Unified loader ───────────────────────────────────────────────────────────

def load_flow_branch(
    backend: str = "raft",
    weights_path: Optional[str | Path] = None,
    frozen: bool = True,
    # RAFT options
    num_flow_updates: int = 12,
    # AllTracker options
    repo_path: Optional[str | Path] = None,
    window_len: int = 16,
    inference_iters: int = 4,
    tiny: bool = False,
) -> nn.Module:
    """Build a flow branch by name. Currently supports 'raft' or 'alltracker'."""
    backend = backend.lower()
    if backend == "raft":
        return load_raft(
            weights_path=weights_path,
            frozen=frozen,
            num_flow_updates=num_flow_updates,
        )
    if backend in ("alltracker", "all_tracker", "at"):
        return load_alltracker(
            weights_path=weights_path,
            repo_path=repo_path,
            window_len=window_len,
            inference_iters=inference_iters,
            tiny=tiny,
            frozen=frozen,
        )
    raise ValueError(f"Unknown flow backend: {backend!r}. Use 'raft' or 'alltracker'.")

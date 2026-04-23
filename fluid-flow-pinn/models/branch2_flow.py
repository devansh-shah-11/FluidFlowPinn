"""Branch 2: RAFT optical-flow estimator — u=(ux,uy) at (B,2,H/8,W/8)."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        flow_predictions = self.raft(img1, img2, num_flow_updates=self.num_flow_updates)
        flow = flow_predictions[-1]  # (B, 2, H, W)

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

"""Branch 2: RAFT optical-flow estimator — u=(ux,uy) at (B,2,H/8,W/8)."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RAFTFlow(nn.Module):
    """Wraps torchvision RAFT (small or large) to produce crowd velocity fields.

    Input:  frame_t, frame_t1 — (B, 3, H, W) ImageNet-normalised RGB
    Output: u — (B, 2, H/8, W/8) velocity field (ux, uy) in pixels/frame

    variant: "small" (default, lighter) or "large" (required for the
             crowdflow fine-tuned checkpoint which has larger channel dims).
    """

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, frozen: bool = True, num_flow_updates: int = 12,
                 variant: str = "small") -> None:
        super().__init__()
        self.num_flow_updates = num_flow_updates

        if variant == "large":
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            self.raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2)
        else:
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


def _remap_raft_keys(state: dict) -> dict:
    """Translate original-RAFT checkpoint keys to torchvision RAFT-Large keys.

    Original RAFT (Teed & Deng 2020) uses fnet/cnet/gru naming.
    torchvision wraps the same architecture under different attribute names.

    Mapping:
      fnet.conv1            → feature_encoder.convnormrelu.0
      fnet.conv2            → feature_encoder.conv
      fnet.layerX.Y.conv1   → feature_encoder.layerX.Y.convnormrelu1.0
      fnet.layerX.Y.conv2   → feature_encoder.layerX.Y.convnormrelu2.0
      fnet.layerX.0.downsample.0 → feature_encoder.layerX.0.downsample.0

      cnet.norm1            → context_encoder.convnormrelu.1
      cnet.conv1            → context_encoder.convnormrelu.0
      cnet.conv2            → context_encoder.conv
      cnet.layerX.Y.conv1   → context_encoder.layerX.Y.convnormrelu1.0
      cnet.layerX.Y.norm1   → context_encoder.layerX.Y.convnormrelu1.1
      cnet.layerX.Y.conv2   → context_encoder.layerX.Y.convnormrelu2.0
      cnet.layerX.Y.norm2   → context_encoder.layerX.Y.convnormrelu2.1
      cnet.layerX.0.norm3   → context_encoder.layerX.0.downsample.1   (BN after proj)
      cnet.layerX.0.downsample.* → context_encoder.layerX.0.downsample.*

      update_block.encoder.conv{c1,c2,f1,f2,} → update_block.motion_encoder.conv{corr1,corr2,flow1,flow2,}
      update_block.gru.conv{z,r,q}{1,2}       → update_block.recurrent_block.convgru{1,2}.conv{z,r,q}
      update_block.mask.0                      → mask_predictor.convrelu.0
      update_block.mask.2                      → mask_predictor.conv
    """
    import re
    out = {}
    for k, v in state.items():
        # ── feature encoder (fnet) ──────────────────────────────────────────
        if k == "fnet.conv1.weight" or k == "fnet.conv1.bias":
            k = k.replace("fnet.conv1", "feature_encoder.convnormrelu.0")
        elif k == "fnet.conv2.weight" or k == "fnet.conv2.bias":
            k = k.replace("fnet.conv2", "feature_encoder.conv")
        elif m := re.match(r"fnet\.(layer\d+\.\d+)\.conv1\.(.*)", k):
            k = f"feature_encoder.{m.group(1)}.convnormrelu1.0.{m.group(2)}"
        elif m := re.match(r"fnet\.(layer\d+\.\d+)\.conv2\.(.*)", k):
            k = f"feature_encoder.{m.group(1)}.convnormrelu2.0.{m.group(2)}"
        elif m := re.match(r"fnet\.(.*)", k):
            k = f"feature_encoder.{m.group(1)}"

        # ── context encoder (cnet) ──────────────────────────────────────────
        elif k == "cnet.norm1.weight" or k == "cnet.norm1.bias" or \
             k.startswith("cnet.norm1.running") or k.startswith("cnet.norm1.num"):
            k = k.replace("cnet.norm1", "context_encoder.convnormrelu.1")
        elif k == "cnet.conv1.weight" or k == "cnet.conv1.bias":
            k = k.replace("cnet.conv1", "context_encoder.convnormrelu.0")
        elif k == "cnet.conv2.weight" or k == "cnet.conv2.bias":
            k = k.replace("cnet.conv2", "context_encoder.conv")
        elif m := re.match(r"cnet\.(layer\d+\.\d+)\.conv1\.(.*)", k):
            k = f"context_encoder.{m.group(1)}.convnormrelu1.0.{m.group(2)}"
        elif m := re.match(r"cnet\.(layer\d+\.\d+)\.norm1\.(.*)", k):
            k = f"context_encoder.{m.group(1)}.convnormrelu1.1.{m.group(2)}"
        elif m := re.match(r"cnet\.(layer\d+\.\d+)\.conv2\.(.*)", k):
            k = f"context_encoder.{m.group(1)}.convnormrelu2.0.{m.group(2)}"
        elif m := re.match(r"cnet\.(layer\d+\.\d+)\.norm2\.(.*)", k):
            k = f"context_encoder.{m.group(1)}.convnormrelu2.1.{m.group(2)}"
        elif m := re.match(r"cnet\.(layer\d+\.0)\.norm3\.(.*)", k):
            # norm3 is the BN on the residual projection — maps to downsample.1
            k = f"context_encoder.{m.group(1)}.downsample.1.{m.group(2)}"
        elif m := re.match(r"cnet\.(.*)", k):
            k = f"context_encoder.{m.group(1)}"

        # ── update block motion encoder ─────────────────────────────────────
        elif k.startswith("update_block.encoder."):
            k = k.replace("update_block.encoder.convc1", "update_block.motion_encoder.convcorr1.0") \
                 .replace("update_block.encoder.convc2", "update_block.motion_encoder.convcorr2.0") \
                 .replace("update_block.encoder.convf1", "update_block.motion_encoder.convflow1.0") \
                 .replace("update_block.encoder.convf2", "update_block.motion_encoder.convflow2.0") \
                 .replace("update_block.encoder.conv",   "update_block.motion_encoder.conv.0")

        # ── GRU ─────────────────────────────────────────────────────────────
        elif k.startswith("update_block.gru."):
            k = k.replace("update_block.gru.convz1", "update_block.recurrent_block.convgru1.convz") \
                 .replace("update_block.gru.convr1", "update_block.recurrent_block.convgru1.convr") \
                 .replace("update_block.gru.convq1", "update_block.recurrent_block.convgru1.convq") \
                 .replace("update_block.gru.convz2", "update_block.recurrent_block.convgru2.convz") \
                 .replace("update_block.gru.convr2", "update_block.recurrent_block.convgru2.convr") \
                 .replace("update_block.gru.convq2", "update_block.recurrent_block.convgru2.convq")

        # ── mask predictor ───────────────────────────────────────────────────
        elif k.startswith("update_block.mask."):
            k = k.replace("update_block.mask.0", "mask_predictor.convrelu.0") \
                 .replace("update_block.mask.2", "mask_predictor.conv")

        out[k] = v
    return out


def load_raft(
    weights_path: Optional[str | Path] = None,
    frozen: bool = True,
    num_flow_updates: int = 12,
    variant: str = "small",
) -> RAFTFlow:
    model = RAFTFlow(frozen=frozen, num_flow_updates=num_flow_updates, variant=variant)

    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        # Strip DataParallel "module." prefix
        if any(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        # Remap original-RAFT naming convention → torchvision naming convention.
        # The crowdflow checkpoint uses the original RAFT paper names (fnet/cnet/gru),
        # torchvision uses (feature_encoder/context_encoder/recurrent_block).
        if any(k.startswith("fnet.") or k.startswith("cnet.") for k in state):
            state = _remap_raft_keys(state)
        # Wrap under "raft." to match RAFTFlow's attribute name
        state = {f"raft.{k}": v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[RAFTFlow] Warning: {len(missing)} missing keys (first: {missing[0]})")
        if unexpected:
            print(f"[RAFTFlow] Warning: {len(unexpected)} unexpected keys (first: {unexpected[0]})")
        print(f"[RAFTFlow] Loaded weights from {weights_path}")

    return model

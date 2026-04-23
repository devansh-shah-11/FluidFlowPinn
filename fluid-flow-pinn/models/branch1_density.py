"""Branch 1: CSRNet density estimator — outputs ρ (B, 1, H/8, W/8)."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torchvision import models


# ── VGG-16 frontend (shared backbone) ────────────────────────────────────────
# VGG-16 feature layer indices:
#   0-4  : conv1_1, relu, conv1_2, relu, pool1  → 64ch,  H/2
#   5-9  : conv2_1, relu, conv2_2, relu, pool2  → 128ch, H/4
#  10-16 : conv3_1, relu, conv3_2, relu, conv3_3, relu, pool3 → 256ch, H/8
# We need layers[:17] to reach pool3: 256 channels at 1/8 resolution.
_VGG16_FRONTEND_LAYERS = 17


# ── Dilated convolution backend ───────────────────────────────────────────────
# Six 3×3 dilated conv layers replace the remaining VGG pooling stages.
# Dilation rates follow the original CSRNet paper (Li et al., CVPR 2018).
_BACKEND_CFG = [
    # (out_channels, dilation)
    (512, 2),
    (512, 2),
    (512, 2),
    (256, 2),
    (128, 2),
    (64,  2),
]


class CSRNet(nn.Module):
    """CSRNet: dilated convolutional neural network for crowd counting.

    Architecture (Li et al., CVPR 2018):
      Frontend : VGG-16 conv1_1 … pool3  → (B, 512, H/8, W/8)
      Backend  : 6× dilated conv (rate=2) → (B, 64,  H/8, W/8)
      Head     : 1×1 conv                 → (B, 1,   H/8, W/8)  density map ρ

    Input : (B, 3, H, W)  ImageNet-normalised RGB
    Output: (B, 1, H/8, W/8)  density map ρ  (sum ≈ person count)
    """

    def __init__(self, use_grad_checkpoint: bool = False) -> None:
        super().__init__()
        self.use_grad_checkpoint = use_grad_checkpoint

        # Frontend: first 10 layers of VGG-16 (up to and including pool3)
        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(
            *list(vgg.features.children())[:_VGG16_FRONTEND_LAYERS]
        )

        # Backend: dilated convolutions
        layers = []
        in_ch = 256  # VGG-16 pool3 outputs 256 channels
        for out_ch, dilation in _BACKEND_CFG:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.backend = nn.Sequential(*layers)

        # Density head: 1×1 conv → single-channel density map
        self.density_head = nn.Conv2d(64, 1, kernel_size=1)

        self._init_weights(vgg)

    def _init_weights(self, vgg: models.VGG) -> None:
        """Copy pretrained VGG-16 weights into frontend; kaiming-init backend."""
        # Frontend — copy from pretrained VGG
        vgg_children = list(vgg.features.children())
        for dst, src in zip(self.frontend.children(), vgg_children[:_VGG16_FRONTEND_LAYERS]):
            if isinstance(dst, nn.Conv2d) and isinstance(src, nn.Conv2d):
                dst.weight.data.copy_(src.weight.data)
                dst.bias.data.copy_(src.bias.data)

        # Backend + head — kaiming uniform
        for m in list(self.backend.modules()) + list(self.density_head.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) ImageNet-normalised RGB frames.
        Returns:
            rho: (B, 1, H/8, W/8) density map.
        """
        if self.use_grad_checkpoint and self.training:
            x = checkpoint.checkpoint(self.frontend, x, use_reentrant=False)
        else:
            x = self.frontend(x)
        x = self.backend(x)
        return self.density_head(x)


# ── Weight loading ────────────────────────────────────────────────────────────

def load_csrnet(
    weights_path: Optional[str | Path] = None,
    use_grad_checkpoint: bool = False,
    pretrained_vgg: bool = True,
) -> CSRNet:
    """Build a CSRNet and optionally load weights.

    Args:
        weights_path: Path to a `.pth` checkpoint saved as a state_dict.
                      If None, the model is initialised with ImageNet VGG-16
                      weights in the frontend (downloaded automatically on
                      first call when pretrained_vgg=True).
        use_grad_checkpoint: Enable gradient checkpointing on the frontend
                             to reduce VRAM usage during training.
        pretrained_vgg: When weights_path is None, initialise the frontend
                        from the official torchvision VGG-16 pretrained weights.

    Returns:
        CSRNet instance with weights loaded.
    """
    if pretrained_vgg and weights_path is None:
        # Load ImageNet-pretrained VGG-16 just to copy its weights
        vgg_pretrained = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model = CSRNet(use_grad_checkpoint=use_grad_checkpoint)
        # Re-copy with pretrained weights (overrides kaiming init in frontend)
        vgg_children = list(vgg_pretrained.features.children())
        for dst, src in zip(
            model.frontend.children(),
            vgg_children[:_VGG16_FRONTEND_LAYERS],
        ):
            if isinstance(dst, nn.Conv2d) and isinstance(src, nn.Conv2d):
                dst.weight.data.copy_(src.weight.data)
                dst.bias.data.copy_(src.bias.data)
    else:
        model = CSRNet(use_grad_checkpoint=use_grad_checkpoint)

    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        # Handle checkpoints saved as {"model": state_dict, ...}
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"[CSRNet] Loaded weights from {weights_path}")

    return model

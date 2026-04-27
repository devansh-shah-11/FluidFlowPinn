"""Branch 1: CSRNet density estimator — outputs ρ (B, 1, H/8, W/8)."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torchvision import models


# ── VGG-16 frontend (shared backbone) ────────────────────────────────────────
# Canonical CSRNet (Li et al., CVPR 2018) uses VGG-16 up through conv4_3 —
# that's 10 conv layers + 3 max-pools, ending at 512 channels at 1/8 resolution.
# In torchvision's VGG-16, the conv4_3 ReLU sits at index 22, so we take [:23].
_VGG16_FRONTEND_LAYERS = 23


# ── Dilated convolution backend ───────────────────────────────────────────────
# Six 3×3 dilated conv layers replace the remaining VGG pooling stages.
# Dilation rates follow the original CSRNet paper.
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
      Frontend : VGG-16 conv1_1 … conv4_3 → (B, 512, H/8, W/8)
      Backend  : 6× dilated conv (rate=2) → (B,  64, H/8, W/8)
      Output   : 1×1 conv                 → (B,   1, H/8, W/8) density map ρ

    Negative values are clamped to zero at output (ReLU). This matches the
    original CSRNet calibration — pretrained ShanghaiTech checkpoints emit
    small positive values on heads and ≈0 elsewhere; using softplus instead
    would lift every "zero" pixel to ln(2)≈0.693 and inflate counts massively.

    Input : (B, 3, H, W)  ImageNet-normalised RGB
    Output: (B, 1, H/8, W/8)  density map ρ  (sum ≈ person count)
    """

    def __init__(self, use_grad_checkpoint: bool = False) -> None:
        super().__init__()
        self.use_grad_checkpoint = use_grad_checkpoint

        # Frontend: VGG-16 conv1_1 … conv4_3
        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(
            *list(vgg.features.children())[:_VGG16_FRONTEND_LAYERS]
        )

        # Backend: dilated convolutions. conv4_3 outputs 512 channels.
        layers = []
        in_ch = 512
        for out_ch, dilation in _BACKEND_CFG:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.backend = nn.Sequential(*layers)

        # Output: 1×1 conv → single-channel density map.
        # Named `output_layer` to match CommissarMa/CSRNet-pytorch checkpoints.
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._init_weights(vgg)

    def _init_weights(self, vgg: models.VGG) -> None:
        """Copy pretrained VGG-16 weights into frontend; kaiming-init backend."""
        vgg_children = list(vgg.features.children())
        for dst, src in zip(self.frontend.children(), vgg_children[:_VGG16_FRONTEND_LAYERS]):
            if isinstance(dst, nn.Conv2d) and isinstance(src, nn.Conv2d):
                dst.weight.data.copy_(src.weight.data)
                dst.bias.data.copy_(src.bias.data)

        for m in list(self.backend.modules()) + [self.output_layer]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) ImageNet-normalised RGB frames.
        Returns:
            rho: (B, 1, H/8, W/8) non-negative density map.
        """
        if self.use_grad_checkpoint and self.training:
            x = checkpoint.checkpoint(self.frontend, x, use_reentrant=False)
        else:
            x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return F.relu(x)

    def freeze(self) -> None:
        """Freeze all parameters — use after loading pretrained weights."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()


# ── Weight loading ────────────────────────────────────────────────────────────

def _remap_legacy_keys(state: dict) -> dict:
    """Map alternative key names found in third-party CSRNet checkpoints."""
    remapped = {}
    for k, v in state.items():
        # Legacy `density_head.*` from earlier versions of this repo
        if k.startswith("density_head."):
            k = k.replace("density_head.", "output_layer.", 1)
        # Some checkpoints save under a `module.` prefix (DataParallel)
        if k.startswith("module."):
            k = k[len("module."):]
        remapped[k] = v
    return remapped


def load_csrnet(
    weights_path: Optional[str | Path] = None,
    use_grad_checkpoint: bool = False,
    pretrained_vgg: bool = True,
    freeze: bool = False,
) -> CSRNet:
    """Build a CSRNet and optionally load weights.

    Args:
        weights_path: Path to a `.pth` checkpoint saved as a state_dict.
                      Compatible with CommissarMa/CSRNet-pytorch ShanghaiTech
                      checkpoints. If None, the model is initialised with
                      ImageNet VGG-16 weights in the frontend.
        use_grad_checkpoint: Enable gradient checkpointing on the frontend
                             to reduce VRAM usage during training.
        pretrained_vgg: When weights_path is None, initialise the frontend
                        from ImageNet VGG-16 weights.
        freeze: Freeze all CSRNet parameters after loading. Use this when
                you trust the pretrained weights and only want to learn the
                flow / pressure branches.

    Returns:
        CSRNet instance with weights loaded.
    """
    if pretrained_vgg and weights_path is None:
        vgg_pretrained = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model = CSRNet(use_grad_checkpoint=use_grad_checkpoint)
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
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for wrap_key in ("model", "state_dict", "model_state_dict"):
                if wrap_key in state and isinstance(state[wrap_key], dict):
                    state = state[wrap_key]
                    break
        state = _remap_legacy_keys(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[CSRNet] Loaded weights from {weights_path}")
        if missing:
            print(f"[CSRNet] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[CSRNet] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    if freeze:
        model.freeze()
        print("[CSRNet] Frozen — density branch will not be trained.")

    return model
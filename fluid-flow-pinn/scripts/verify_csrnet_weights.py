"""Verify a ShanghaiTech CSRNet checkpoint loads into our `models.branch1_density.CSRNet`.

Usage:
    python scripts/verify_csrnet_weights.py \
        --weights /scratch/dns5508/FluidFlowPinn/ShanghaiTechA_best.pth

What it does:
    1. Builds our CSRNet via `load_csrnet(weights_path=...)`.
    2. Reports missing / unexpected keys (should both be empty for a clean load).
    3. Runs a forward pass on a 224×224 random tensor to confirm output shape
       and non-negativity (softplus).
    4. Optionally runs on a real image (`--image path.jpg`) and prints sum(rho)
       — that's the predicted head count. Compare against your eyeballed count.

Exit code 0 = checkpoint loads and produces a sensible output; non-zero = problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to ShanghaiTech CSRNet .pth")
    p.add_argument("--image",   default=None,  help="Optional image to run a count on")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from models.branch1_density import load_csrnet

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[FAIL] checkpoint not found: {weights_path}", file=sys.stderr)
        return 2

    print(f"[1/3] Loading checkpoint via load_csrnet(...)")
    model = load_csrnet(weights_path=weights_path, pretrained_vgg=False, freeze=True)
    model.eval()

    print(f"[2/3] Forward pass on a 1x3x224x224 random tensor")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print(f"      output shape : {tuple(y.shape)}    (expected (1, 1, 28, 28))")
    print(f"      output min   : {y.min().item():.6f}")
    print(f"      output max   : {y.max().item():.6f}")
    print(f"      output sum   : {y.sum().item():.4f}")
    if y.shape != (1, 1, 28, 28):
        print("[FAIL] unexpected output shape", file=sys.stderr); return 3
    if (y < 0).any():
        print("[FAIL] negative density values — softplus broken?", file=sys.stderr); return 3

    if args.image:
        print(f"[3/3] Counting on real image: {args.image}")
        try:
            import cv2
        except ImportError:
            print("      (cv2 not installed — skipping image test)")
            return 0
        from torchvision import transforms as T

        img = cv2.imread(args.image)
        if img is None:
            print(f"[FAIL] could not read image {args.image}", file=sys.stderr); return 4
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to a multiple of 8 for clean H/8, W/8 output
        h, w = rgb.shape[:2]
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        rgb = cv2.resize(rgb, (w8, h8))

        xform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        t = xform(rgb).unsqueeze(0)
        with torch.no_grad():
            rho = model(t)
        count = rho.sum().item()
        print(f"      predicted count : {count:.2f}")
        print(f"      density min/max : {rho.min().item():.6f} / {rho.max().item():.6f}")
        print(f"      density shape   : {tuple(rho.shape)}  (image was {h8}x{w8})")
    else:
        print("[3/3] (skipped — no --image given)")

    print("\n[OK] CSRNet checkpoint loaded and produced a non-negative density map.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

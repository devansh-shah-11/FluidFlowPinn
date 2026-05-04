"""Run a single crowd-detection model on one image and save a visualization.

Usage:
    python scripts/detect_crowd.py --model yolo    --image path/to.jpg [--weights yolo26n.pt]
    python scripts/detect_crowd.py --model lwcc    --image path/to.jpg [--lwcc-model DM-Count --lwcc-weights SHA]
    python scripts/detect_crowd.py --model csrnet  --image path/to.jpg [--weights PartAmodel_best.pth]

Outputs:
    yolo   → image with bounding boxes drawn
    lwcc   → density heatmap overlay + count
    csrnet → density heatmap overlay + count (default visualization)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _overlay_heatmap(
    bgr: np.ndarray,
    density: np.ndarray,
    alpha: float = 0.5,
    tau: float = 0.0,
) -> np.ndarray:
    H, W = bgr.shape[:2]
    if density.shape != (H, W):
        density = cv2.resize(density, (W, H), interpolation=cv2.INTER_LINEAR)
    d = np.clip(density, 0.0, None)
    d = d / (d.max() + 1e-8)
    heat = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # Per-pixel blend: only tint where density is non-negligible. Gamma 0.5
    # lifts mid-range values so faint heads remain visible without flooding
    # the background with JET's blue floor.
    w = (d ** 0.5)[..., None] * alpha
    vis = (bgr.astype(np.float32) * (1.0 - w) + heat.astype(np.float32) * w).astype(np.uint8)
    if tau > 0.0:
        mask = (density > tau).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)
    return vis


def run_yolo(image_path: Path, weights: str, out_path: Path, conf: float) -> None:
    from ultralytics import YOLO

    model = YOLO(weights)
    model.to(_device())
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = model(bgr, verbose=False)
    boxes = results[0].boxes
    keep = (boxes.cls == 0) & (boxes.conf > conf)
    person_xyxy = boxes.xyxy[keep].cpu().numpy()
    confs = boxes.conf[keep].cpu().numpy()

    vis = bgr.copy()
    for (x1, y1, x2, y2), c in zip(person_xyxy, confs):
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(vis, f"{c:.2f}", (int(x1), max(0, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, f"persons: {len(person_xyxy)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)
    print(f"[yolo] {len(person_xyxy)} person(s) → {out_path}")


def run_lwcc(image_path: Path, model_name: str, model_weights: str, out_path: Path, tau: float = 0.0) -> None:
    from lwcc.LWCC import load_model
    from lwcc.util.functions import load_image as lwcc_load_image

    device = _device()
    model = load_model(model_name=model_name, model_weights=model_weights)
    model.to(device).eval()

    img_tensor, _ = lwcc_load_image(str(image_path), model_name, is_gray=False, resize_img=False)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
    density = output[0, 0].cpu().numpy().astype(np.float32)
    count = float(density.sum())

    bgr = cv2.imread(str(image_path))
    vis = _overlay_heatmap(bgr, density, tau=tau)
    label = f"count: {count:.1f}" + (f"  tau>{tau}" if tau > 0 else "")
    cv2.putText(vis, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)
    print(f"[lwcc:{model_name}/{model_weights}] count={count:.2f} → {out_path}")


def run_csrnet(image_path: Path, weights: str, out_path: Path, tau: float = 0.0) -> None:
    from models.branch1_density import load_csrnet
    from torchvision import transforms

    device = _device()
    model = load_csrnet(weights_path=weights, pretrained_vgg=False).to(device).eval()

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    x = tfm(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        rho = model(x)
    # CSRNet density map sums to count in the original 1/8-resolution; preserve
    # the count when upsampling for visualization.
    density = rho[0, 0].cpu().numpy().astype(np.float32)
    count = float(density.sum())

    vis = _overlay_heatmap(bgr, density, tau=tau)
    label = f"count: {count:.1f}" + (f"  tau>{tau}" if tau > 0 else "")
    cv2.putText(vis, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)
    print(f"[csrnet] count={count:.2f} → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yolo", "lwcc", "csrnet"])
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output image path (default: <image_stem>_<model>.jpg next to the input)")
    ap.add_argument("--weights", default=None,
                    help="yolo: .pt file (default yolo26n.pt). csrnet: .pth file (default PartAmodel_best.pth)")
    ap.add_argument("--lwcc-model", default="DM-Count",
                    help="lwcc model name (CSRNet, SFANet, Bay, DM-Count)")
    ap.add_argument("--lwcc-weights", default="SHA",
                    help="lwcc weights tag (SHA, SHB, QNRF)")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="Density threshold for lwcc/csrnet (matches infer.py --density-mask-tau). "
                         "When >0, draws contours where rho>tau on the heatmap.")
    args = ap.parse_args()

    if not args.image.exists():
        sys.exit(f"Image not found: {args.image}")

    out_path = args.out or args.image.with_name(f"{args.image.stem}_{args.model}.jpg")

    if args.model == "yolo":
        weights = args.weights or str(REPO_ROOT / "yolo26n.pt")
        run_yolo(args.image, weights, out_path, args.conf)
    elif args.model == "lwcc":
        run_lwcc(args.image, args.lwcc_model, args.lwcc_weights, out_path, tau=args.tau)
    elif args.model == "csrnet":
        weights = args.weights or str(REPO_ROOT.parent / "PartAmodel_best.pth")
        run_csrnet(args.image, weights, out_path, tau=args.tau)


if __name__ == "__main__":
    main()

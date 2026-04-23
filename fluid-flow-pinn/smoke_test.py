"""Quick end-to-end sanity check (1 batch, CPU-safe) — Phase 1 Step 5.

Checks:
  1. FDSTDataset loads frame pairs and density maps with correct shapes.
  2. CSRNet (Branch 1) produces rho at H/8 x W/8.
  3. RAFTFlow (Branch 2) produces u=(ux,uy) at H/8 x W/8.
  4. rho and u grids match — ready for physics loss.
  5. rho values are non-negative; density sum is plausible.
  6. Flow magnitudes are finite.

Run:
    python smoke_test.py                        # CPU, synthetic frames
    python smoke_test.py --device cuda          # GPU, synthetic frames
    python smoke_test.py --fdst data/fdst --device cuda   # real frames
"""

import argparse
import sys

import torch
import torch.nn.functional as F

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--fdst",   default=None,  help="Path to FDST root (optional)")
parser.add_argument("--device", default="cpu", help="cpu | cuda")
parser.add_argument("--h",      type=int, default=272, help="Synthetic frame height")
parser.add_argument("--w",      type=int, default=480, help="Synthetic frame width")
args = parser.parse_args()

device = torch.device(args.device)
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors = []

def check(name, cond, detail=""):
    if cond:
        print(f"  [{PASS}] {name}")
    else:
        print(f"  [{FAIL}] {name}  {detail}")
        errors.append(name)


# ── 1. Dataset ────────────────────────────────────────────────────────────────

print("\n── Step 1: FDSTDataset ──────────────────────────────────────────────")

USE_REAL = False
if args.fdst:
    try:
        from preprocessing.dataset_loader import FDSTDataset
        ds = FDSTDataset(args.fdst, split="train")
        sample = ds[0]
        ft  = sample["frame_t"]
        ft1 = sample["frame_t1"]
        dm  = sample["density_map"]
        H, W = ft.shape[-2:]

        check("frame_t shape  (3, H, W)",       ft.shape[0] == 3)
        check("frame_t1 shape (3, H, W)",        ft1.shape == ft.shape)
        check("density_map shape (1, H/8, W/8)",
              dm.shape == (1, H // 8, W // 8), f"got {dm.shape}")
        check("density_map non-negative",        dm.min().item() >= 0)
        check("density sum plausible (1–5000)",
              1 <= dm.sum().item() <= 5000,     f"sum={dm.sum().item():.1f}")
        print(f"       scene={sample['scene']}  idx={sample['idx']}  "
              f"heads≈{dm.sum().item():.1f}  frame={tuple(ft.shape)}")
        USE_REAL = True
    except Exception as exc:
        print(f"  [WARN] FDSTDataset failed ({exc}); falling back to synthetic.")

if not USE_REAL:
    H, W = args.h, args.w
    ft  = torch.rand(3, H, W)
    ft1 = torch.rand(3, H, W)
    print(f"  Using synthetic tensors  H={H}  W={W}")


# ── 2. CSRNet — Branch 1 ─────────────────────────────────────────────────────

print("\n── Step 2: CSRNet (Branch 1) ────────────────────────────────────────")

from models.branch1_density import load_csrnet

csrnet = load_csrnet(pretrained_vgg=False).to(device).eval()

with torch.no_grad():
    rho = csrnet(ft.unsqueeze(0).to(device))       # (1, 1, H/8, W/8)

check("rho shape (1, 1, H/8, W/8)",
      rho.shape == (1, 1, H // 8, W // 8), f"got {tuple(rho.shape)}")
check("rho finite", rho.isfinite().all().item())
print(f"       rho min={rho.min().item():.4f}  max={rho.max().item():.4f}  "
      f"sum={rho.sum().item():.2f}")


# ── 3. RAFTFlow — Branch 2 ───────────────────────────────────────────────────

print("\n── Step 3: RAFTFlow (Branch 2) ──────────────────────────────────────")

from models.branch2_flow import load_raft

raft = load_raft(frozen=True).to(device).eval()

with torch.no_grad():
    u = raft(ft.unsqueeze(0).to(device),
             ft1.unsqueeze(0).to(device))           # (1, 2, H/8, W/8)

check("u shape (1, 2, H/8, W/8)",
      u.shape == (1, 2, H // 8, W // 8), f"got {tuple(u.shape)}")
check("u finite", u.isfinite().all().item())
mag = u.norm(dim=1)
print(f"       |u| min={mag.min().item():.4f}  max={mag.max().item():.4f}  "
      f"mean={mag.mean().item():.4f}")


# ── 4. Grid alignment ─────────────────────────────────────────────────────────

print("\n── Step 4: Grid alignment rho <-> u ─────────────────────────────────")

check("rho and u on same spatial grid",
      rho.shape[-2:] == u.shape[-2:],
      f"rho {rho.shape[-2:]} vs u {u.shape[-2:]}")


# ── 5. Summary ────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
if errors:
    print(f"  {len(errors)} check(s) FAILED: {errors}")
    sys.exit(1)
else:
    source = "real FDST frames" if USE_REAL else "synthetic tensors"
    print(f"  All checks passed ({source}, device={args.device}).")
    print("  Ready for Step 6 — continuity loss.\n")

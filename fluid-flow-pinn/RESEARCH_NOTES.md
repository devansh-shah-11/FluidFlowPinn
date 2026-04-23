# Fluid-Flow PINN — Research & Implementation Notes

**Team:** Varad Suryavanshi, Sarvesh Bodke, Devansh Shah (NYU CS)
**Course:** Computer Vision, Spring 2026

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Core Idea — Why Treat Crowds as Fluid?](#2-core-idea--why-treat-crowds-as-fluid)
3. [Architecture Overview](#3-architecture-overview)
4. [Phase 1 — Perception](#4-phase-1--perception)
   - [Step 2: Preprocessing & Data Pipeline](#step-2-preprocessing--data-pipeline)
   - [Step 3: Branch 1 — CSRNet Density](#step-3-branch-1--csrnet-density)
5. [Datasets](#5-datasets)
6. [Design Decisions Log](#6-design-decisions-log)
7. [Open Questions](#7-open-questions)

---

## 1. Problem Statement

Crowd crushes (Itaewon 2022, Hillsborough 1989) kill people because dangerous pressure buildup is **invisible** until it is too late. Existing crowd analysis models (CSRNet, MCNN) predict *how many people* are in a frame but ignore the *forces* between them. A high-density static crowd is safe; the same density moving in conflicting directions is lethal.

**Goal:** Predict a spatial pressure map P(x, y, t) that spikes ≥ 5 seconds before a crush becomes visible, giving security staff an actionable lead time.

---

## 2. Core Idea — Why Treat Crowds as Fluid?

At high densities, individual pedestrian decisions become irrelevant — people are physically pushed by the crowd around them. This regime is well-described by **compressible fluid mechanics**. We borrow the **continuity equation** from fluid dynamics:

```
∂ρ/∂t + ∇·(ρu) = 0
```

Where:
- `ρ(x,y,t)` = crowd density (persons/m²) — estimated by CSRNet
- `u(x,y,t) = (ux, uy)` = crowd velocity field — estimated by RAFT optical flow
- The equation says: density changes where people are flowing in or out

In a crush, this equation **breaks down locally** — density is rising faster than the flow field can explain. The residual `R = ∂ρ/∂t + ∇·(ρu) ≠ 0` is a direct measure of dangerous compression. We use this residual as a physics-informed training loss.

**Pressure proxy:**
```
P(x,y,t) = ρ(x,y,t) · Var(u)
```
High density + high velocity variance = people being squeezed from multiple directions = danger.

**Why this beats pure deep learning:** A model trained only on labeled crush data will overfit to the appearance of panic (running, falling). Our physics loss teaches the model to detect the *forces* that cause panic, which appear earlier and are appearance-agnostic.

---

## 3. Architecture Overview

```
Video frames (t, t+1)
        │
        ▼
┌──────────────────┐
│  VGG-16 Frontend │  (shared feature extractor, gradient checkpointed)
└────────┬─────────┘
         │
   ┌─────┴──────┐
   ▼            ▼
Branch 1      Branch 2
CSRNet        RAFT
ρ (B,1,H/8,W/8)   u=(ux,uy) (B,2,H,W) → downsampled to H/8,W/8
   │            │
   └─────┬──────┘
         ▼
  Physics Loss Layer
  R = ∂ρ/∂t + ∇·(ρu)   ← finite differences
  L_physics = ‖R‖²
         │
         ▼
  Branch 3: Pressure map
  P(x,y,t) = ρ · Var(u)  over 5×5 spatial window
         │
         ▼
  Risk Alert  (P > threshold → alert with lead-time estimate)
```

**Total loss:**
```
L_total = L_count + λ1·L_motion + λ2·‖R‖²
```
- `L_count` = MSE(predicted density sum, GT head count) — from FDST annotations
- `L_motion` = EPE (endpoint error on CrowdFlow) — only when GT flow available
- `λ1 = 0.1`, `λ2 = 0.01` (starting values, tunable)

---

## 4. Phase 1 — Perception

### Step 2: Preprocessing & Data Pipeline

**Files:** `preprocessing/frame_extractor.py`, `preprocessing/umn_splitter.py`, `preprocessing/dataset_loader.py`

#### What we built

| Component | Purpose |
|---|---|
| `extract_frame_pairs()` | Streams consecutive `(frame_t, frame_t+1)` pairs from video without loading full file into RAM |
| `list_fdst_scenes()` | Discovers all 13 FDST scene directories under `train_data/` or `test_data/` |
| `split_umn()` | Cuts single UMN `.avi` into 3 scene clips at hardcoded boundaries (0–640, 641–1000, 1001–end) |
| `FDSTDataset` | PyTorch Dataset for FDST; builds Gaussian density maps from VIA bounding-box JSON annotations |
| `UMNDataset` | PyTorch Dataset for UMN; bicubic-upsamples 320×240 → 640×480 before returning tensors |
| `ShanghaiTechDataset` | PyTorch Dataset for ShanghaiTech; builds density maps from `.mat` point annotations |
| `SequentialSceneSampler` | Custom sampler that shuffles *scenes* between epochs but keeps frames within each scene in temporal order |

#### Key design decisions

**Why `SequentialSceneSampler` instead of `DataLoader(shuffle=True)`?**
The physics loss `∂ρ/∂t ≈ (ρ_t+1 - ρ_t) / Δt` is only meaningful between two frames from the same continuous video sequence. If we shuffle frame pairs independently, we would compute density differences between frames from entirely different scenes — the physics loss would be trained on nonsense and diverge. The sampler shuffles the order of scenes (for randomisation across epochs) but never breaks temporal order within a scene.

**Why Gaussian-smoothed density maps (σ=15 px)?**
Raw point/box annotations are delta functions — a single pixel with value 1. CNNs cannot learn from delta functions (the gradient signal is spatially too sparse). Gaussian smoothing spreads each head annotation into a small blob. The total integral is preserved (sum ≈ head count), but the spatial structure gives the model a learnable signal at every pixel near a head. σ=15 px is the value used in the original CSRNet paper for the FDST-scale images.

**Why bicubic upsample UMN from 320×240 → 640×480?**
RAFT optical flow quality degrades sharply at very low resolution. At 320×240, fine-grained motion vectors become unreliable. Bicubic upsampling before RAFT gives it enough spatial detail to produce usable flow. This is applied only to UMN — FDST frames are already 1080×1920 HD.

**FDST JSON format — VIA bounding boxes:**
The FDST dataset was annotated with the VGG Image Annotator (VIA) tool. Each `.json` stores bounding boxes around each head as `{"x", "y", "width", "height"}`. We extract the box center `(x + w/2, y + h/2)` as the head point to build the density map, matching the standard crowd counting convention.

#### Verified on Colab (2026-04-23)
- Train pairs: ~8,987 | Test pairs: ~5,987 across 13 scenes
- Batch shape: `frame_t (4, 3, 1080, 1920)`, `density (4, 1, 135, 240)` ✅
- Density map resolution `135×240 = 1080/8 × 1920/8` matches CSRNet output stride ✅
- Zero temporal ordering violations across 500 sampled pairs ✅

---

### Step 3: Branch 1 — CSRNet Density

**File:** `models/branch1_density.py`

#### What we built

`CSRNet` — a PyTorch `nn.Module` that takes `(B, 3, H, W)` RGB frames and outputs a density map `ρ` of shape `(B, 1, H/8, W/8)`.

#### Architecture details

**Why CSRNet?**
CSRNet (Li et al., CVPR 2018) is the established baseline for crowd density estimation. It achieves state-of-the-art MAE on ShanghaiTech and is specifically designed for the task we need: converting a crowd image into a spatial density map whose pixel values sum to the person count. It is also lightweight enough to fit alongside RAFT on a 16 GB T4.

**Why VGG-16 as the frontend?**
Three reasons:
1. **Proven feature extractor for crowds.** The original CSRNet paper uses VGG-16 and achieves the reported SOTA numbers. Using the same backbone lets us compare directly to published results.
2. **Transfer learning.** ImageNet-pretrained VGG-16 weights give the model strong low/mid-level visual features (edges, textures, object parts) from day one. Training from scratch on FDST's ~9,000 frames would severely underfit.
3. **Right output stride.** The first 10 VGG-16 layers (up to pool3) produce a feature map at 1/8 the input resolution — exactly what we need to match with RAFT's downsampled flow output. This resolution is a constraint from the physics loss: both `ρ` and `u` must be on the same spatial grid.

**Why only the first 10 VGG-16 layers?**
Layers 1–10 include conv1_1 through pool3 (three max-pool operations → stride 8). We stop here because:
- Further VGG pooling (pool4, pool5) would reduce spatial resolution to 1/32, making the density map too coarse to localise crowd pressure spatially.
- The dilated convolution backend replaces those deeper VGG layers with rate-2 dilations, which grow the receptive field without losing resolution.

**Why dilated convolutions in the backend?**
Standard convolutions with stride > 1 reduce spatial resolution. Dilated (atrous) convolutions increase the receptive field exponentially without downsampling — the filter "skips" pixels by the dilation rate. With rate=2 and a 3×3 kernel, the effective receptive field is 5×5 per layer. Six such layers give a receptive field large enough to see a full person cluster while keeping the output at H/8 resolution. This is the core insight of CSRNet over earlier crowd counting models.

**Why gradient checkpointing on the frontend?**
The VGG frontend is the memory bottleneck. At batch size 4 with 1080×1920 inputs, the intermediate activations of the frontend alone would exceed 16 GB VRAM. Gradient checkpointing trades compute for memory: instead of storing all intermediate activations for the backward pass, it recomputes them on demand. This roughly halves VRAM usage at the cost of ~30% longer backward pass — an acceptable trade-off on Colab T4.

**Weight initialisation:**
- Frontend: copied from `torchvision` ImageNet-pretrained VGG-16
- Backend + head: Kaiming uniform (designed for ReLU networks; prevents vanishing/exploding gradients at init)

#### Output

```
Input:  (B, 3, H,   W)
Output: (B, 1, H/8, W/8)   — density map ρ, values ≥ 0, sum ≈ person count
```

For FDST frames (1080×1920): output is `(B, 1, 135, 240)` — matches the ground-truth density maps built in Step 2.

---

## 5. Datasets

| Dataset | Role | Split | Status |
|---|---|---|---|
| FDST | Primary training | train: 9 scenes, 60 videos / test: 4 scenes, 40 videos | ✅ Working |
| UMN | Anomaly evaluation + lead-time | 3 scenes, 1 `.avi` | ⏳ Download pending |
| ShanghaiTech A/B | Density branch evaluation | test only | ⏳ Download pending |
| CrowdFlow | Flow branch evaluation | 5 synthetic sequences | Not started |
| Venice | Perspective validation | 4 sequences | Not started |

**FDST on-disk layout confirmed:**
```
FDST Dataset/
  train_data/
    2/  13/  16/  18/  21/ ...   ← 9 scene dirs
      001.jpg  001.json
      002.jpg  002.json
      ...  (150 frames per scene per video)
  test_data/
    10/  19/  20/  24/  100/ ...  ← 4 scene dirs
```

---

## 6. Design Decisions Log

| Decision | Alternative considered | Why we chose this |
|---|---|---|
| CSRNet for density | MCNN, DM-Count, BayesCrowd | CSRNet is the reference model in the crowd crushing literature; lightest that still achieves SOTA MAE |
| VGG-16 frontend | ResNet-50, MobileNet | CSRNet paper uses VGG-16; changing backbone would invalidate pretrained CSRNet weight compatibility |
| RAFT for optical flow | Farneback (classical), FlowNet | RAFT is current SOTA learned flow; far more accurate on crowd motion than classical methods |
| Continuity equation as physics loss | Momentum equation, energy equation | Continuity only requires `ρ` and `u` — no pressure terms needed as input, making it self-contained |
| σ=15 px Gaussian smoothing | Adaptive σ (geometry-aware), σ=4 | σ=15 is the CSRNet paper value for similar-scale datasets; geometry-aware requires camera calibration we don't have |
| 1/8 density resolution | 1/4, full resolution | 1/8 is the natural VGG-16 frontend output stride; going finer requires extra upsampling that adds noise |
| λ2=0.01 for physics loss | 0.1, 0.001 | Too high → physics loss dominates and density accuracy collapses; too low → physics constraint ignored |
| Batch size 4 | 8, 16 | T4 has 16 GB VRAM; at 1080×1920 inputs with FP16 + grad checkpoint, batch 4 is the safe maximum |

---

## 7. Open Questions

- [ ] Will the continuity residual `R` be meaningful at 1/8 resolution, or does downsampling smooth out the local compression signal?
- [ ] UMN is 30fps at 320×240 — after bicubic upsample to 640×480, will RAFT produce usable flow vectors or will the upsampled frames introduce artifacts?
- [ ] Should `λ1` and `λ2` be fixed or scheduled (e.g., ramp up physics loss weight as training progresses)?
- [ ] The pressure proxy `P = ρ · Var(u)` uses a 5×5 local window — is this window size physically motivated or empirical?
- [ ] ShanghaiTech Part A has images up to 3139 persons — will the Gaussian density maps with σ=15 px overlap too heavily at that density, creating a uniform blob instead of a meaningful map?

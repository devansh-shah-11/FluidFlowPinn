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
   - [Step 4: Branch 2 — RAFT Optical Flow](#step-4-branch-2--raft-optical-flow)
   - [Step 5: Smoke Test — End-to-End Branch 1 + 2](#step-5-smoke-test--end-to-end-branch-1--2)
5. [Phase 2 — Physics](#5-phase-2--physics)
   - [Step 6: Continuity Loss](#step-6-continuity-loss)
6. [Datasets](#6-datasets)
7. [Design Decisions Log](#7-design-decisions-log)
8. [Open Questions](#8-open-questions)

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

#### Bug fix — density map integral not preserved after downsampling (2026-04-23)

`F.interpolate(..., mode="bilinear")` computes a weighted **average** of the source pixels, which reduces the sum by the area ratio (64× for 8× downsampling). A density map encodes a count integral — the sum must equal the head count, not the per-pixel average. Fix applied to both `FDSTDataset` and `ShanghaiTechDataset`:

```python
density_t = F.interpolate(density_t, size=(dh, dw), mode="bilinear", align_corners=False)
density_t = density_t * (h * w) / (dh * dw)   # restore integral
```

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

**Why the first 17 VGG-16 layers?**
Layers 0–16 include conv1_1 through pool3 (three max-pool operations → stride 8). We stop here because:
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

### Step 4: Branch 2 — RAFT Optical Flow

**File:** `models/branch2_flow.py`

#### What we built

`RAFTFlow` — a PyTorch `nn.Module` that takes consecutive frame pairs `(frame_t, frame_t1)` at `(B, 3, H, W)` and outputs a velocity field `u = (ux, uy)` at `(B, 2, H/8, W/8)`.

#### Architecture details

**Why RAFT-Small (not RAFT-Large)?**
RAFT-Large requires ~5× more VRAM. At 1080×1920 inputs on a T4 (16 GB), running Large alongside CSRNet would OOM. RAFT-Small achieves ~1.5 px EPE on Sintel Clean — sufficient for crowd motion where displacements are typically 5–30 px/frame.

**Why frozen weights?**
FDST has no optical-flow ground truth. End-to-end fine-tuning would need `L_motion` from CrowdFlow (a small synthetic set), risking catastrophic forgetting of pretrained flow quality. Freezing produces reliable velocity estimates from day one. `set_trainable(True)` can be called if CrowdFlow GT is used.

**Why downsample flow to H/8?**
The physics loss `∂ρ/∂t + ∇·(ρu)` requires `ρ` and `u` on the same spatial grid. CSRNet outputs `ρ` at H/8 × W/8, so RAFT's full-resolution flow is bilinear-downsampled to match. Flow magnitudes are also scaled by 1/8 to keep units consistent at the coarse grid.

**ImageNet denormalisation:**
RAFT expects pixel values in [0, 255]. Our pipeline normalises frames to ImageNet mean/std, so the wrapper undoes this before calling RAFT and re-applies inside the forward pass transparently.

**`num_flow_updates = 12`:**
RAFT uses a recurrent GRU to iteratively refine the flow estimate. 12 iterations is the RAFT paper default for RAFT-Small. Fewer iterations trade accuracy for speed (useful for inference-time ablations).

#### Output

```
Input:  frame_t, frame_t1  — (B, 3, H, W)  ImageNet-normalised RGB
Output: u                  — (B, 2, H/8, W/8)  (ux, uy) in pixels/frame at coarse grid
```

For FDST frames (1080×1920): output is `(B, 2, 135, 240)` — matches the CSRNet density map grid for the physics loss.

---

### Step 5: Smoke Test — End-to-End Branch 1 + 2

**File:** `smoke_test.py`

#### What it checks

| Check | What failure would mean |
|---|---|
| FDSTDataset shapes + density sum | Dataset loader or annotation parsing broken |
| CSRNet output `(1, 1, H/8, W/8)` | Frontend stride or backend depth wrong |
| RAFTFlow output `(1, 2, H/8, W/8)` | Padding logic, RAFT import, or downsample broken |
| `rho.isfinite()` | NaN/Inf from uninitialised weights or bad forward pass |
| `u.isfinite()` | RAFT diverged on extreme inputs |
| `rho.shape[-2:] == u.shape[-2:]` | Grids misaligned — physics loss would be meaningless |

#### Usage

```bash
python smoke_test.py                              # CPU, synthetic 272x480 frames
python smoke_test.py --device cuda               # GPU, synthetic frames
python smoke_test.py --fdst data/fdst --device cuda  # real FDST frames
```

Falls back to synthetic random tensors automatically if FDST path is not provided or unavailable, so it can run in any environment.

---

## 5. Phase 2 — Physics

### Step 6: Continuity Loss

**File:** `losses/continuity_loss.py`

#### What we built

`continuity_loss(rho_t, rho_t1, u, fps)` — computes the squared continuity residual:

```
R = ∂ρ/∂t + ∂(ρ·ux)/∂x + ∂(ρ·uy)/∂y
L_physics = mean(R²)
```

`ContinuityLoss` — `nn.Module` wrapper for use in the training loop.

#### Boundary handling: options considered

Computing spatial derivatives `∂(ρu)/∂x` and `∂(ρu)/∂y` requires a choice at image boundaries where one-sided neighbors don't exist. Three options were evaluated:

**Option A — Interior only (no padding, discard boundary pixels)**
- Compute central differences without padding; output is `(H-2) × (W-2)`.
- Pro: Every computed value is mathematically exact (O(h²) error).
- Con: Systematically discards the 1-pixel border from the physics loss. Crowd crushes often happen at barriers and walls, which appear at frame edges — this would create a blind spot exactly where pressure is most dangerous.

**Option B — Replicate padding**
- Pad the border by replicating the edge pixel, then apply central differences everywhere; output is full `H × W`.
- Pro: Full spatial coverage.
- Con: At the boundary, the padded neighbor equals the edge value, so the central difference gives `(x[1] - x[0]) / 2` instead of the correct derivative. For a linear function this underestimates the derivative by 50%. The physics loss is **silently wrong at boundaries** — the worst kind of error.

**Option C — Mixed one-sided / central differences ✅ (chosen)**
- Forward difference at left/top edge, backward difference at right/bottom edge, central difference in the interior. Output is full `H × W`.
- Pro: Exact everywhere. Interior pixels have O(h²) error (central); boundary pixels have O(h) error (one-sided), which is acceptable at our spatial scales.
- Con: Slightly more code than Options A or B.

#### Why Option C

Option B is ruled out — a systematic 50% gradient underestimate at boundaries is worse than no boundary coverage at all. Between A and C, Option C is clearly better for this problem: crowd pressure builds up against physical barriers (walls, fences, stage barriers), which appear at the edges of fixed-camera footage. Option A's blind spot is exactly the wrong place to have one.

#### Implementation

```python
def _diff_x(x):
    out = torch.empty_like(x)
    out[:, :, :,  0]   = x[:, :, :,  1] - x[:, :, :,  0]         # forward
    out[:, :, :, -1]   = x[:, :, :, -1] - x[:, :, :, -2]         # backward
    out[:, :, :, 1:-1] = (x[:, :, :, 2:] - x[:, :, :, :-2]) / 2.0
    return out
```

The same pattern applies to `_diff_y` along rows.

#### Design decisions

| Decision | Alternative | Reason |
|---|---|---|
| Use `rho_t` (not average of `rho_t`, `rho_t1`) for divergence term | Average of both frames | `rho_t` is the density at the spatial snapshot we're differentiating; averaging would mix two time steps into a single spatial derivative, muddling the temporal/spatial separation |
| Forward difference for `∂ρ/∂t` | Backward, central | The pair `(rho_t, rho_t1)` is a forward step; forward difference is the natural choice and avoids needing `rho_{t-1}` |
| `mean(R²)` not `sum(R²)` | Sum | Mean is independent of spatial resolution — if we change the density map scale the loss magnitude stays comparable |

---

## 6. Datasets

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

## 7. Design Decisions Log

| Decision | Alternative considered | Why we chose this |
|---|---|---|
| CSRNet for density | MCNN, DM-Count, BayesCrowd | CSRNet is the reference model in the crowd crushing literature; lightest that still achieves SOTA MAE |
| VGG-16 frontend | ResNet-50, MobileNet | CSRNet paper uses VGG-16; changing backbone would invalidate pretrained CSRNet weight compatibility |
| RAFT for optical flow | Farneback (classical), FlowNet | RAFT is current SOTA learned flow; far more accurate on crowd motion than classical methods |
| RAFT-Small (not Large) | RAFT-Large | RAFT-Large would OOM on T4 alongside CSRNet at 1080×1920; Small is accurate enough for crowd motion (~5–30 px/frame displacements) |
| RAFT frozen by default | End-to-end fine-tuning | FDST has no flow GT; freezing avoids catastrophic forgetting; `set_trainable(True)` unlocks fine-tuning on CrowdFlow |
| Flow scaled by 1/8 after downsample | Keep raw pixel values | Downsampled flow represents motion at the coarse grid — dividing by 8 keeps units consistent with the density-grid pixel spacing |
| Continuity equation as physics loss | Momentum equation, energy equation | Continuity only requires `ρ` and `u` — no pressure terms needed as input, making it self-contained |
| σ=15 px Gaussian smoothing | Adaptive σ (geometry-aware), σ=4 | σ=15 is the CSRNet paper value for similar-scale datasets; geometry-aware requires camera calibration we don't have |
| 1/8 density resolution | 1/4, full resolution | 1/8 is the natural VGG-16 frontend output stride; going finer requires extra upsampling that adds noise |
| λ2=0.01 for physics loss | 0.1, 0.001 | Too high → physics loss dominates and density accuracy collapses; too low → physics constraint ignored |
| Batch size 4 | 8, 16 | T4 has 16 GB VRAM; at 1080×1920 inputs with FP16 + grad checkpoint, batch 4 is the safe maximum |
| Mixed one-sided/central differences for spatial derivatives | Replicate padding (Option B), interior only (Option A) | Option B silently underestimates boundary gradients by 50%; Option A creates a blind spot at frame edges where crowd pressure against barriers is most dangerous; Option C is exact everywhere |
| `mean(R²)` as physics loss scalar | `sum(R²)` | Mean is resolution-independent — loss magnitude stays comparable if density map scale changes |

---

## 8. Open Questions

- [ ] Will the continuity residual `R` be meaningful at 1/8 resolution, or does downsampling smooth out the local compression signal?
- [ ] UMN is 30fps at 320×240 — after bicubic upsample to 640×480, will RAFT produce usable flow vectors or will the upsampled frames introduce artifacts?
- [ ] Should `λ1` and `λ2` be fixed or scheduled (e.g., ramp up physics loss weight as training progresses)?
- [ ] The pressure proxy `P = ρ · Var(u)` uses a 5×5 local window — is this window size physically motivated or empirical?
- [ ] ShanghaiTech Part A has images up to 3139 persons — will the Gaussian density maps with σ=15 px overlap too heavily at that density, creating a uniform blob instead of a meaningful map?

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
   - [Step 7: Branch 3 — Pressure Map + Total Loss](#step-7-branch-3--pressure-map--total-loss)
   - [Step 8: Full PINN Model](#step-8-full-pinn-model)
6. [Phase 3 — Inference, Pretrained Weights, Live Dashboard](#6-phase-3--inference-pretrained-weights-live-dashboard)
   - [Step 9: First training run — diagnosing broken outputs](#step-9-first-training-run--diagnosing-broken-outputs)
   - [Step 10: Switching to pretrained ShanghaiTech CSRNet](#step-10-switching-to-pretrained-shanghaitech-csrnet)
   - [Step 11: Architectural fixes uncovered while wiring pretrained weights](#step-11-architectural-fixes-uncovered-while-wiring-pretrained-weights)
   - [Step 12: Inference-only pipeline (no training required)](#step-12-inference-only-pipeline-no-training-required)
   - [Step 13: Realtime dashboard — `infer.py`](#step-13-realtime-dashboard--inferpy)
   - [Step 14: Field findings on dense scenes — perspective bias](#step-14-field-findings-on-dense-scenes--perspective-bias)
7. [Datasets](#7-datasets)
8. [Design Decisions Log](#8-design-decisions-log)
9. [Open Questions](#9-open-questions)

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
- `L_motion` = EPE against GT flow (CrowdFlow) when available; else density warp consistency `L1(warp(rho_t, u), rho_t1)` — always active
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

**Why the first 23 VGG-16 layers (not 17)?**
*Note: the original implementation took layers `[:17]` (ending at pool3, 256 channels). This was an architectural bug — see [Step 11](#step-11-architectural-fixes-uncovered-while-wiring-pretrained-weights) for the fix and the canonical layer-index breakdown.*

Layers 0–22 include conv1_1 through conv4_3 (three max-pool operations + the conv4 block → stride 8 with 512 channels). We stop here because:
- Further VGG pooling (pool4, pool5) would reduce spatial resolution to 1/32, making the density map too coarse to localise crowd pressure spatially.
- The dilated convolution backend replaces those deeper VGG layers with rate-2 dilations, which grow the receptive field without losing resolution.
- The original CSRNet paper and the public reference implementation both end the frontend at conv4_3, producing 512-channel features at 1/8 resolution.

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

### Step 7: Branch 3 — Pressure Map + Total Loss

**Files:** `models/branch3_pressure.py`, `losses/total_loss.py`

#### What we built

`PressureMap` — an `nn.Module` that takes `ρ (B, 1, H, W)` and `u (B, 2, H, W)` and returns a pressure map `P (B, 1, H, W)`:

```
P(x, y, t) = ρ(x, y, t) · Var_local(u)
```

where `Var_local(u)` is the average of the per-channel local variances of `ux` and `uy` over a 5×5 sliding window:

```
Var_local(u) = (Var_window(ux) + Var_window(uy)) / 2
Var_window(f) = E[f²] - E[f]²    (computed via avg_pool2d)
```

`TotalLoss` — an `nn.Module` that combines all three loss terms:

```
L_total = L_count + λ1·L_motion + λ2·‖R‖²
```

- `L_count` = MSE(predicted density sum per image, GT head count)
- `L_motion` = EPE (endpoint error) against GT flow when available (CrowdFlow); otherwise **density warp consistency** — `L1(warp(rho_t, u), rho_t1)` using RAFT's predicted flow as a self-supervised signal (see below)
- `‖R‖²` = continuity residual from `ContinuityLoss`
- Returns a dict `{total, count, motion, physics}` for logging each term individually

#### Pressure map design

**Why `ρ · Var(u)`?**
A dense, slowly moving crowd (stadium standing area) has high `ρ` but low `Var(u)` — people all move the same direction. Pressure is low. A crush scenario has high `ρ` and high `Var(u)` — people are being pushed from multiple directions simultaneously. The product captures this interaction. Velocity magnitude alone is insufficient: a fast but uniform flow (e.g., an orderly evacuation) is not dangerous.

**Why local variance over a 5×5 window?**
Per-pixel variance would be zero for any smooth flow field — variance is a population statistic. A local window (5×5 at H/8 resolution ≈ 40×40 pixels at full resolution) captures the spatial spread of velocity within a human-scale neighbourhood. The window size is a hyperparameter; 5×5 was chosen as the smallest window that reliably captures multi-directional crowd motion in dense sequences.

**Boundary handling — zero-padding (deliberate choice):**
`avg_pool2d` uses zero-padding by default, which means border pixels see a mix of real values and artificial zeros in their local window. This inflates the apparent variance at image boundaries. We considered replicate padding (which would give zero variance for spatially uniform flow all the way to the border), but rejected it for two reasons:
1. The border artifact from zero-padding is physically harmless — it will produce slightly elevated pressure near frame edges, which is conservative (safer to over-alert than to miss).
2. Zero-padding is the simpler, more standard implementation. Replicate padding requires a manual pre-pad step and was introduced only to satisfy a test invariant that isn't physically required.

The test `test_zero_pressure_uniform_flow` checks interior pixels only (cropped by `window_size // 2`), which is where the physical invariant actually holds.

**Resolution filter checks both frames in a pair (bug fix, 2026-04-25):**
The training loop filters FDST samples to only those matching the target resolution (e.g. 720×1280). The original filter only checked `frame_t`, not `frame_t1`. A pair where `frame_t` is 720×1280 but `frame_t1` is a different resolution (possible at scene boundaries or in mixed-resolution scenes) would pass the filter and cause a shape mismatch in the batch collator or silently produce an incorrect temporal derivative. Fixed to require both frames match:

```python
kept = [s for s in dataset._samples if _res_ok(s["frame_t"]) and _res_ok(s["frame_t1"])]
```

**Why `L_count` uses the density sum (not pixel-wise MSE)?**
Our ground truth is a *count* per image, not a per-pixel density map for every training frame. ShanghaiTech provides density maps, but FDST annotations are head bounding boxes from which we construct Gaussian maps ourselves. Using the sum preserves the count constraint without assuming our Gaussian-smoothed maps are ground truth at every pixel. Pixel-wise MSE against self-constructed density maps would overfit to our σ=15 px choice.

**Why density warp consistency instead of `L_motion = 0` on FDST?**
FDST has no optical flow ground truth, so EPE against GT flow is unavailable. The original implementation set `L_motion = 0` in this case, making `λ1` completely inactive and wasting a training signal. The fix: use RAFT's *predicted* flow `u` as a self-supervised signal by warping `rho_t` forward and comparing to `rho_t1`:

```python
rho_warped = grid_sample(rho_t, grid_from_u)    # bilinear warp via u
l_motion   = L1(rho_warped, rho_t1)
```

This enforces that the density evolution is consistent with the optical flow — if RAFT says people moved right, the density at t+1 should match what you get by shifting density at t rightward. This directly tightens the coupling between `rho` and `u`, which is what makes the pressure map `P = ρ · Var(u)` reliable. When GT flow *is* available, EPE is used instead (the supervised signal is stronger).

The warp is implemented via `torch.nn.functional.grid_sample` with `padding_mode="border"` to handle pixels that flow outside the frame boundary.

#### Output

```
TotalLoss.forward(rho_t, rho_t1, u, gt_count, gt_flow=None)
→ {
    'total':   scalar  — backprop target
    'count':   scalar  — L_count alone
    'motion':  scalar  — L_motion alone (0 if no gt_flow)
    'physics': scalar  — ‖R‖² alone
  }
```

---

### Step 8: Full PINN Model

**File:** `models/pinn.py`

#### What we built

`FluidFlowPINN` — a single `nn.Module` that wires all three branches together:

```python
out = model(frame_t, frame_t1)
# out = {'rho': (B,1,H/8,W/8), 'rho_t1': (B,1,H/8,W/8), 'u': (B,2,H/8,W/8), 'P': (B,1,H/8,W/8)}
```

#### Architecture wiring

```
frame_t  ──► CSRNet ──► rho_t  ──┐
                                  ├──► PressureMap ──► P
frame_t1 ──► CSRNet ──► rho_t1   │
                                  │ (rho_t1 passed to TotalLoss for ∂ρ/∂t)
frame_t  ──┐
           ├──► RAFTFlow ──► u ──┘
frame_t1 ──┘
```

CSRNet runs twice per forward pass — once for `frame_t`, once for `frame_t1`. This is necessary because both density maps are needed: `rho_t` enters the pressure map and the continuity divergence term, while `rho_t1` enters the temporal derivative `∂ρ/∂t`. Sharing weights between the two calls is correct (it is the same model applied at different time steps) and costs zero extra parameters.

#### Memory design

Two VRAM-saving mechanisms are active by default:

| Mechanism | What it does | Where |
|---|---|---|
| Gradient checkpointing | Recomputes VGG frontend activations on backward instead of storing them | `CSRNet(use_grad_checkpoint=True)` |
| RAFT frozen | RAFT weights have `requires_grad=False`; no gradient storage for flow branch | `RAFTFlow(frozen=True)` |

FP16 is **not** managed inside `FluidFlowPINN`. It is applied externally via `torch.cuda.amp.autocast()` in the training loop. This is the standard PyTorch AMP pattern — the model itself stays in FP32 and AMP selectively casts operations.

**Why not freeze CSRNet too?** *(original reasoning, since superseded — see [Step 10](#step-10-switching-to-pretrained-shanghaitech-csrnet))*
CSRNet was originally planned to be fine-tuned on FDST to produce accurate density maps for our specific dataset and camera angles. The pretrained VGG-16 weights in the frontend gave a warm start, but the dilated backend and density head were randomly initialised and intended to be trained. RAFT was frozen because it already produces high-quality flow vectors without fine-tuning, and FDST has no flow ground truth to fine-tune against.

**Update — CSRNet is now frozen too in the MVP path.** The from-scratch training run produced uniform-noise ρ ([Step 9](#step-9-first-training-run--diagnosing-broken-outputs)). We switched to pretrained ShanghaiTech_A weights and freeze the entire density branch. With both branches frozen and PressureMap having zero learnable parameters, training is currently a no-op — the inference pipeline (`infer.py`) is the only execution path that runs day-to-day.

#### Output contract

```
{'rho':    (B, 1, H/8, W/8)   density map at t
 'rho_t1': (B, 1, H/8, W/8)  density map at t+1
 'u':      (B, 2, H/8, W/8)  flow field (ux, uy)
 'P':      (B, 1, H/8, W/8)  pressure map}
```

`rho_t1` is included in the output so the training loop can pass it directly to `TotalLoss` without re-running CSRNet.

---

## 6. Phase 3 — Inference, Pretrained Weights, Live Dashboard

This phase covers the work after the initial training run produced unusable outputs. The chronology matters: each step here was driven by a concrete failure observed in dashboards or evaluation, not by upfront design.

### Step 9: First training run — diagnosing broken outputs

**Symptoms observed in `outputs/vis/` after a full FDST training run:**

- `Predicted density ρ` panel showed values in `[-0.025, +0.025]` — uniform noise across the entire frame. Sum ≈ 20.6 vs. GT count 18.0 (matched only because a near-uniform field times the area integrates to ~anything).
- Negative ρ values — physically meaningless; nothing in the model prevented them.
- `Pressure timeline` had no dynamic range — baseline 1.0–1.5 with threshold at 0.5, so every frame fired ALARM. Var(u) values up to ~1400 dominated the signal; without a working ρ, P ≈ Var(u).

**Root-cause analysis of the count metric:**

`L_count = MSE(sum(ρ), GT_count)` is a **scalar** loss — it pushes the integral toward the right number but says nothing about *spatial structure*. A network that outputs a uniform constant `c = GT_count / (H·W)` everywhere achieves zero count loss while learning nothing about heads. With ~9k FDST frames and only this scalar signal driving the density head, that's the local minimum the model fell into.

The motion + physics losses are designed to add spatial structure on top of the count term, but they couple ρ to RAFT's frozen flow — so they only work *if ρ is already approximately correct*. A degenerate uniform ρ doesn't get pulled out of the bad minimum by either signal.

**Decision: stop trying to train CSRNet from scratch.** Switch to pretrained ShanghaiTech weights, where the spatial structure has already been learned on a much larger labeled dataset.

### Step 10: Switching to pretrained ShanghaiTech CSRNet

**Source chosen:** `CommissarMa/CSRNet-pytorch` ShanghaiTech Part_A weights — Part_A is the dense subset (avg ~500 heads/image, heavy occlusion) closest to the stampede regime we ultimately care about. Part_B (sparser street scenes) was rejected for the primary use case but kept as a fallback option for sparse footage.

**The reasoning about training under freeze:**

A natural question came up: *if we freeze CSRNet, what gets trained for pressure?* Answer: **nothing.** The three branches:

| Branch | Trainable? | Why |
|---|---|---|
| 1. CSRNet | frozen with pretrained weights | the only branch with real learnable params for our task |
| 2. RAFT | frozen by default | pretrained flow already strong; FDST has no flow GT |
| 3. PressureMap | **has zero parameters** | `P = ρ · local_var(u)` is a pure analytic operator |

`L_pressure` does not exist. There is no supervisory signal on P because we have no labeled anomaly data. With CSRNet frozen and RAFT frozen, the optimizer has nothing to update — `train.py` becomes a no-op. The correct workflow is therefore **inference-only**: load pretrained CSRNet + frozen RAFT, run `P = ρ · Var(u)` deterministically, and validate visually.

**`use_count_loss` flag added** to `TotalLoss`. When the density branch is frozen, the count loss is computed under `torch.no_grad()` so it appears in the metrics dict for logging but contributes no gradient (the optimizer wouldn't be able to use it anyway, and including it in the total inflates the reported number for no reason). Auto-disabled in `train.py` whenever `freeze_density=True`.

### Step 11: Architectural fixes uncovered while wiring pretrained weights

Loading the ShanghaiTech checkpoint exposed three architectural bugs in our CSRNet that had been silently degrading the from-scratch training run.

**Bug A — wrong VGG-16 frontend depth (the major one).**

Original code took `vgg.features[:17]`, ending at `pool3` with **256 output channels**. The CSRNet paper, and CommissarMa's reference, take the frontend through `conv4_3` — **512 output channels**, still at 1/8 spatial resolution. The backend's first dilated conv was therefore being fed half the expected feature dimensionality, with `in_channels=256` instead of 512. Even from-scratch training would have been crippled by this.

| | Old (broken) | Fixed |
|---|---|---|
| frontend slice | `vgg.features[:17]` | `vgg.features[:23]` |
| frontend out channels | 256 | 512 |
| backend `in_channels` | 256 | 512 |

Inspecting `torchvision.vgg16().features` confirmed the canonical layer indices:

```
features[16] : MaxPool2d   ← pool3 (1/8 res, 256ch)
features[17] : Conv2d 512  ← conv4_1
features[18] : ReLU
features[19] : Conv2d 512  ← conv4_2
features[20] : ReLU
features[21] : Conv2d 512  ← conv4_3
features[22] : ReLU        ← end of [:23]
features[23] : MaxPool2d   ← pool4 (1/16 — we don't want this)
```

**Bug B — output layer name mismatch.** Original code named the final 1×1 conv `density_head`. CommissarMa's checkpoints save it as `output_layer.*`. Renamed `density_head → output_layer` so ShanghaiTech `state_dict` keys load by name without manual remapping. Added `_remap_legacy_keys()` to also strip optional `module.` prefixes (DataParallel artifacts) and remap any older `density_head.*` keys saved before the rename.

**Bug C — softplus on output (introduced and reverted).**

To prevent negative ρ during from-scratch training, an initial fix wrapped the output in `F.softplus`. This worked for from-scratch training but **catastrophically miscalibrated the pretrained weights**:

- Pretrained CSRNet was trained with no output activation. It produces small positive values on heads and ≈0 (sometimes slightly negative) elsewhere.
- `softplus(0) = ln(2) ≈ 0.693`. Every "zero" pixel gets lifted to 0.693.
- On a 135×240 density map, that's `32400 × 0.693 ≈ 22460` of phantom count from background alone.

**Observed:** verification script reported predicted count = 22519 on a sparse scene with ~22 actual people — exactly the noise floor leaking through softplus.

**Resolution:** replaced `softplus` with `F.relu`. ReLU preserves the pretrained calibration (floor stays at 0) while still preventing negative output values for any future from-scratch run. Lesson: non-negativity activations need to match the activation the original weights were trained under, not be added prophylactically.

**Verification — bit-exact match against the reference:**

A diagnostic script (`scripts/diff_csrnet_against_ref.py`) loads the same checkpoint into our `CSRNet` and a verbatim copy of CommissarMa's reference, then runs the same input through both. Result on the ShanghaiTech_A `.pth`:

```
[keys] ours=34  ref=34  shared=34
[keys] all 34 shared params match exactly.
ours  shape=(1, 1, 28, 28)  sum=0.2515  min=0.000000  max=0.003155
ref   shape=(1, 1, 28, 28)  sum=0.0948  min=-0.002103  max=0.003155
max |ours - relu(ref)| = 0.00e+00     (architectures match)
```

Bit-exact. Architecture is correct after the three fixes; any further oddities are data/domain issues, not implementation bugs.

**Other small items resolved on the way:**

- `torch.load(..., weights_only=True)` is the PyTorch 2.6 default and rejects checkpoints with non-tensor metadata. Set `weights_only=False` in `load_csrnet` because the third-party ShanghaiTech `.pth` pickles optimizer state alongside the weights.
- The state_dict unwrap loop accepts `model`, `state_dict`, or `model_state_dict` wrapper keys to handle the various save conventions.

### Step 12: Inference-only pipeline (no training required)

Given that nothing is trainable when both upstream branches are frozen, the inference path doesn't need a `best.pt` training checkpoint. `infer.py` accepts **either** a full training checkpoint **or** a raw ShanghaiTech CSRNet `.pth`:

```bash
# Inference with pretrained weights only — no training step
python infer.py --csrnet-weights /path/PartAmodel_best.pth \
    --source clip.mp4 --out-dir outputs/infer_run/

# Or, if a training checkpoint exists
python infer.py --checkpoint checkpoints/best.pt \
    --source clip.mp4 --out-dir outputs/infer_run/
```

Internally, the `--csrnet-weights` path constructs a `FluidFlowPINN(csrnet_weights=..., freeze_density=True, raft_frozen=True)` directly. RAFT auto-downloads pretrained `C_T_V2` weights from torchvision the first time. PressureMap has no params. The whole model is deterministic at inference.

This is the **MVP path**: pretrained CSRNet + pretrained RAFT + analytic pressure → realtime dashboard, no labels needed, no gradient descent involved. Threshold calibration is a downstream concern (data-driven — pick from the percentile distribution of `max(P)` on a normal video — not a hand-picked guess).

### Step 13: Realtime dashboard — `infer.py`

**File:** `infer.py` (root of repo)

A single script with two modes that both write the same artifacts:

- **File output mode** (always on when `--out-dir` is set): writes `dashboard.mp4` (annotated frame-by-frame) and `pressure_timeline.png` (full P(t) curve).
- **Live window mode** (`--display` flag): also opens an OpenCV window showing the dashboard updating in realtime. Press `q` or `Esc` to quit. Headless-safe — falls back to file-only when `--display` is omitted, so it works on cluster nodes without X.

**Sources supported:** video file (`.mp4`/`.avi`/`.mov`/...), single image (`.jpg`/`.png` — only ρ, since P needs two frames), or webcam (`--source 0`).

**Dashboard layout (final, after Step 14 expansion):**

```
┌───────────┬───────────┐
│   input   │  rho      │   row 1 — what the scene contains
├───────────┼───────────┤
│ |u| with  │  Var(u)   │   row 2 — the motion factors
│ arrows    │           │
├───────────┼───────────┤
│   P       │  status   │   row 3 — combined output + ALARM/OK panel
├───────────┴───────────┤
│  P(t) live timeline   │   rolling sparkline + threshold line
└───────────────────────┘
```

Each colored panel has an embedded label with the salient scalar (e.g. `rho sum=353.7`, `P max=30.2`). `|u|` overlays sparse yellow flow arrows (every 24 px) so direction is visible, not just magnitude. The status panel surfaces frame number, infer-fps (so you know what "realtime" means on the current hardware), max P, threshold, and history depth.

**Helper utilities introduced:**

- `_local_var_np(u, k)` — numpy version of `PressureMap`'s local variance, used to render the Var(u) panel for *visualization*. The actual P field shown comes from the model's own PressureMap, not this helper — the helper exists so the diagnostic panel matches what's inside P.
- `_flow_arrows(frame, u, step, scale)` — sparse arrow overlay; scales flow by 8 to undo the H/8 → H/8-grid downsampling and the ÷8 in `RAFTFlow.forward`, so arrow lengths are in input-pixel units.
- `_render_timeline_strip` — renders a matplotlib strip into a BGR ndarray for the bottom of the dashboard each frame.

**Throughput:** ~2.2 fps observed on a Colab T4 with FP16 for 2320×1080 input. RAFT-Small is the bottleneck; CSRNet alone runs faster. A coarser `--width`/`--height` setting trades spatial resolution for higher fps if needed.

### Step 14: Field findings on dense scenes — perspective bias

**Scenario:** dashboard run on a real dense crowd video (station/festival, oblique camera angle with strong perspective). `maxP` ranged 26–102, mean 32; threshold of 0.5 was meaningless (every frame fired). The complaint after watching the dashboard wasn't "magnitude is too high" — it was **"pressure shows up at the wrong locations within the frame."**

**Spatial bias observed:** P peaks were concentrated in the *back* of the scene (where heads are small in pixel space), with very little response in the *foreground row* of obviously-crowded people closer to the camera.

**Why this happens — three stacking causes:**

1. **Perspective / scale mismatch in CSRNet.** ShanghaiTech_A is trained on images where heads occupy a roughly uniform pixel size (stadium framing, no strong perspective). On a perspective scene, near-camera heads are *too big* — they don't match the network's expected head-feature scale, so CSRNet under-fires on them. Far-camera heads are correctly scaled and CSRNet fires strongly. **ρ is structurally biased toward distance.**

2. **RAFT flow scales with screen-space velocity.** Same physical walking speed produces large `|u|` in the foreground (many pixels per frame) and small `|u|` in the back (few pixels per frame). But `Var(u)` measures local *disorder*, not magnitude. In the back, where many small heads are packed into a small image region with slightly different motion vectors, Var(u) is high. In the foreground, where one large person dominates the local window with one consistent flow vector, Var(u) is low.

3. **Fixed-size variance window (5×5 on H/8 grid = 40×40 input pixels).** That window covers roughly *one foreground head* but *ten background heads*. Variance is naturally higher when more independent motion sources fit inside the window. The formula is structurally biased toward dense distant crowds.

**Net effect:** `ρ_far · Var(u)_far ≫ ρ_near · Var(u)_near`. The bias is real and physical given the current formulation; it's not an implementation bug. Any single fix only addresses one of the three causes.

**What the dashboard expansion (Step 13's row 2) was for:** before this diagnosis, only ρ and P were visible. We could see "P fires in the back" but not which factor (ρ vs. Var(u)) was responsible. Adding the `|u|` and `Var(u)` panels lets us attribute every P spike: if Var(u) is the dominant cause, the fix needs to address motion measurement; if ρ is missing in the foreground, the fix needs to address perspective in the density branch.

**Three real options for fixing the bias** (deferred — none implemented yet):

1. **Perspective normalization map.** Multiply ρ by a per-pixel scale map (head-size-at-depth) so foreground heads get amplified to compensate for CSRNet under-firing. Standard "perspective density map" approach (referenced in CommissarMa's tutorial repo). Needs a hand-drawn perspective map per camera.

2. **Adaptive variance window.** Use a window size in *world-pixels*, not grid-cells — coarser in foreground, finer in back. Same effect as (1), but for the motion factor instead of density. Also needs a depth/perspective estimate.

3. **Density gating: zero out P where ρ is below a small threshold.** Doesn't fix the bias toward distance, but kills hallucinated pressure on textured background regions where ρ is small but nonzero. Cheap and effective for non-stadium footage with empty regions; doesn't help here because the entire dense scene has high ρ.

**Working theory for next iteration:** the pressure formula is correct as a relative score over time at a fixed location, but its *absolute* spatial distribution is unreliable on perspective scenes. Threshold calibration should therefore be **per-region** (or use a spatial normalization from a "normal" reference frame), not a single scalar across the whole image.

**Status:** dashboard expanded (Step 13 layout), perspective fix deferred. Next planned step is to run the same dashboard on a top-down camera angle to confirm whether the bias disappears when perspective is removed — if yes, we have direct evidence that perspective is the cause and option 1 is the right fix.

---

## 7. Datasets

| Dataset | Role | Split | Status |
|---|---|---|---|
| FDST | Primary training | train: 9 scenes, 60 videos / test: 4 scenes, 40 videos | ✅ Working |
| UMN | Anomaly evaluation + lead-time | 3 scenes, 1 `.avi` | ⏳ Download pending |
| ShanghaiTech A/B | Pretrained CSRNet weights (Part_A used for inference) | weights only | ✅ Part_A loaded |
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

## 8. Design Decisions Log

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
| `P = ρ · Var(u)` for pressure | `P = ρ · ‖u‖`, raw pressure via Navier-Stokes | Velocity magnitude misses directional conflict (orderly fast evacuation ≠ crush); `Var(u)` captures multi-directional compression; full NS pressure requires solving a Poisson equation |
| 5×5 local window for `Var(u)` | 3×3, 7×7, per-pixel | Per-pixel variance is always zero; 3×3 too small to capture crowd-scale motion spread; 5×5 ≈ 40×40 px at full resolution matches human body width at typical crowd densities |
| Zero-padding in `avg_pool2d` for pressure | Replicate padding | Zero-padding is standard; the border artifact is conservative (over-estimates pressure at edges, which is safer than under-estimating); replicate padding adds code without a physical justification |
| `L_count` = MSE(sum(ρ), GT count) | Pixel-wise MSE against density map | GT is a head count, not a validated per-pixel density map; count-loss avoids overfitting to our σ=15 Gaussian construction |
| Density warp consistency when no GT flow | `L_motion = 0` (original), dummy EPE against zero flow | Original implementation made λ1 permanently inactive on FDST. Warp consistency uses RAFT's predicted flow as self-supervision: `L1(warp(rho_t, u), rho_t1)` — trains CSRNet to produce density maps that evolve consistently with the flow, tightening the ρ–u coupling that the pressure map depends on. Setting to 0 wastes a training signal; EPE against zero would force RAFT toward no-motion. |
| Resolution filter checks both `frame_t` and `frame_t1` | Check only `frame_t` (original) | Original bug: pairs where `frame_t1` has a different resolution would pass the filter, causing shape mismatches or a broken temporal derivative `∂ρ/∂t`. Fixed to require both frames match the target resolution. |
| CSRNet run twice per forward (frame_t and frame_t1) | Run once, reuse rho_t as rho_t1 | Both time steps are needed for `∂ρ/∂t`; weight sharing is free (same model, different inputs); reusing rho_t would make the temporal derivative always zero |
| FP16 via external `autocast()`, not inside model | Cast tensors inside `forward()` | Standard PyTorch AMP pattern; keeps model portable and lets the training loop control precision scope |
| VGG-16 frontend through `conv4_3` (`features[:23]`, 512ch) | Stop at pool3 (`features[:17]`, 256ch) — original buggy state | Canonical CSRNet (Li et al., CVPR 2018) and CommissarMa's reference both go through conv4_3. Stopping at pool3 halved the feature dimensionality at the backend's input and almost certainly contributed to the failed from-scratch training run. Required for ShanghaiTech checkpoint compatibility. |
| Output-layer name `output_layer` (not `density_head`) | Custom name + key-remap loader | Matches CommissarMa/CSRNet-pytorch ShanghaiTech checkpoint keys directly; no remap needed. The `_remap_legacy_keys()` helper still handles the old `density_head.*` name and `module.*` (DataParallel) prefixes for any third-party checkpoint that uses them. |
| `F.relu` on density output | `F.softplus` (tried, reverted), no activation | Softplus floor is `ln(2)≈0.693`, which adds ~22000 of phantom count over a 135×240 map. Pretrained ShanghaiTech weights expect a linear output. ReLU preserves the pretrained calibration (floor stays at 0) while still preventing negative values for any future from-scratch run. Lesson: non-negativity activations must match the activation the original weights were trained under. |
| `torch.load(weights_only=False)` for CSRNet checkpoints | PyTorch 2.6 default `weights_only=True` | Third-party ShanghaiTech `.pth` files pickle non-tensor metadata (epoch, optimizer state). `weights_only=True` rejects them. Local files we control — safe to disable. |
| `freeze_density` flag on `FluidFlowPINN` + `--freeze-density` CLI | Always train CSRNet | When loading pretrained ShanghaiTech weights, fine-tuning on FDST risks destroying the spatial structure those weights already encode. Freezing isolates debugging — if P(t) still looks broken with a known-good ρ, the problem is in flow or pressure, not density. |
| `use_count_loss=False` when density is frozen | Always include `L_count` in total | When CSRNet is frozen, count loss is uninfluenceable — including it in the total just adds noise to the reported scalar without contributing a useful gradient. With `use_count_loss=False`, the term is computed under `torch.no_grad()` and stays in the metrics dict for logging only. Auto-disabled in `train.py` when `freeze_density=True`. |
| Inference-only path (no training step) for the MVP | Train CSRNet from scratch on FDST | The first training run produced uniform-noise ρ ([Step 9](#step-9-first-training-run--diagnosing-broken-outputs)). Pretrained ShanghaiTech_A handles dense crowds well enough to use directly; PressureMap has zero learnable params; RAFT works frozen. With both upstream branches frozen, `train.py` is a no-op anyway. Inference-only ships an MVP today; training resumes only when (a) labeled anomaly data enables a pressure loss, or (b) we decide to fine-tune RAFT on crowd motion. |
| ShanghaiTech Part_A weights as default | Part_B (sparser scenes), or train Part_A + Part_B ensemble | Part_A trains on dense crowds (~500 heads/image), which is exactly the regime where stampedes happen. Part_B (sparser, 9–578 heads) under-fires on dense scenes. Trade-off accepted: Part_A overcounts on out-of-distribution sparse scenes (cafeteria predicted ~124 vs. real ~22), but for stampede prediction the dense regime matters more. Part_B remains a documented fallback for sparse-only deployments. |
| `infer.py` accepts both `--checkpoint` and `--csrnet-weights` | Require a full training checkpoint | Cluster users (with the ShanghaiTech `.pth`) shouldn't have to run a meaningless `train.py` first. Mutually exclusive flags; exactly one must be specified. Keeps the inference path self-contained for the MVP. |
| Six-pane dashboard (input, ρ, \|u\|+arrows, Var(u), P, status) + timeline strip | Original four-pane (input, ρ, P, status) | The four-pane layout couldn't tell us *why* P fired in any given location — was ρ wrong, was Var(u) wrong, or both? The expanded layout shows every factor in the formula `P = ρ · Var(u)` plus the raw flow field. Diagnostic value paid for itself the first time we used it (Step 14 perspective bias would have been guesswork without the Var(u) panel). |

---

## 9. Open Questions

- [ ] Will the continuity residual `R` be meaningful at 1/8 resolution, or does downsampling smooth out the local compression signal?
- [ ] UMN is 30fps at 320×240 — after bicubic upsample to 640×480, will RAFT produce usable flow vectors or will the upsampled frames introduce artifacts?
- [ ] Should `λ1` and `λ2` be fixed or scheduled (e.g., ramp up physics loss weight as training progresses)?
- [x] The pressure proxy `P = ρ · Var(u)` uses a 5×5 local window — is this window size physically motivated or empirical? → Empirical. 5×5 at H/8 resolution ≈ 40×40 px at full resolution, roughly matching human body width at typical crowd densities. Physically motivated would require calibrated pixel-to-metre mapping (Venice dataset). Window size is exposed as a hyperparameter in `PressureMap(window_size=...)` and `configs/default.yaml`.
- [ ] Does the pressure map `P` need to be normalised before thresholding for the UMN lead-time evaluation, or can a fixed threshold be applied across scenes?
- [ ] The full PINN model runs CSRNet twice per forward pass (frame_t and frame_t1). At batch size 4 on T4 with FP16, does this double the activation memory footprint of the density branch, or does gradient checkpointing contain it?
- [ ] ShanghaiTech Part A has images up to 3139 persons — will the Gaussian density maps with σ=15 px overlap too heavily at that density, creating a uniform blob instead of a meaningful map?
- [ ] How do we calibrate the pressure threshold from data instead of guessing? Plan: run inference over a "normal" video, compute the percentile distribution of `max(P)`, set threshold at the 95–99th percentile. Per-camera or universal?
- [ ] Will the perspective bias from [Step 14](#step-14-field-findings-on-dense-scenes--perspective-bias) disappear on top-down camera angles, confirming that perspective is the dominant cause? (Test by running `infer.py` on a top-down clip and comparing the spatial distribution of P.)
- [ ] If perspective normalization is needed, can we estimate the perspective map automatically from a few minutes of pedestrian-tracking footage (heads-as-features) instead of requiring a hand-drawn calibration per camera?
- [ ] ShanghaiTech_A overcounts on sparse out-of-distribution scenes (cafeteria: 124 predicted vs. ~22 real, ~5.6×). For deployments that mix sparse and dense regimes, do we need both Part_A and Part_B weights, or a scene-classifier in front that picks?
- [ ] PressureMap has zero learnable parameters today. If we ever get labeled anomaly data, should the next learnable component sit (a) inside PressureMap as a refinement head over `[ρ, u, P]`, or (b) as a separate Branch 4 that consumes the deterministic P and predicts a calibrated risk score?
- [ ] RAFT was trained on Sintel/KITTI (cars, scenes). It may smooth flow across small heads in dense crowds, suppressing the per-head Var(u) signal. Worth fine-tuning RAFT on crowd footage with the warp-consistency + continuity losses (would unfreeze the only other parameter-bearing branch)?

# Fluid-Flow PINN — Claude Code Project Brief

## Who we are
Three NYU CS students building a computer vision course project:
- Varad Suryavanshi (vs3273@nyu.edu)
- Sarvesh Bodke (sb10583@nyu.edu)
- Devansh Shah (dns5508@nyu.edu)

---

## What we are building

A **Physics-Informed Neural Network (PINN)** that treats human crowds as a
compressible fluid to detect dangerous crowd pressure buildup **before** a
crush event becomes visible. The core idea: existing SOTA models (CSRNet,
MCNN) estimate density well but ignore the physical forces that govern crowd
disasters. We couple density + optical flow through the Navier-Stokes
continuity equation as a differentiable training constraint.

---

## Architecture — three branches, one physics loss

```
Video frames (t, t+1)
        │
        ▼
┌──────────────────┐
│  Shared backbone │  (lightweight CNN feature extractor)
└────────┬─────────┘
         │
   ┌─────┴─────┐
   ▼           ▼
Branch 1     Branch 2
CSRNet       RAFT
Density ρ    Flow u=(ux,uy)
   │           │
   └─────┬─────┘
         ▼
  Physics Loss Layer
  R = ∂ρ/∂t + ∇·(ρu)   ← continuity equation residual
  L_total = L_count + λ1·L_motion + λ2·‖R‖²
         │
         ▼
  Branch 3: Pressure map
  P(x,y,t) = ρ(x,y,t) · Var(u)
         │
         ▼
  Risk Alert + Lead-time estimate
```

---

## Datasets

| Dataset | Role | Format | Notes |
|---------|------|--------|-------|
| **FDST** | Primary training | Video (13 scenes, 394k annotations) | Strong left-to-right flow — ideal for continuity loss |
| **UMN** | Anomaly evaluation / lead-time | Single `.avi`, 3 scenes at 320×240, 30fps | Must split by scene using published frame indices |
| **ShanghaiTech A** | Density branch benchmark | Images, max 3139 persons/image | Evaluation only — no flow labels |
| **ShanghaiTech B** | Density branch benchmark | Images, fixed 768×1024 | Evaluation only |
| **CrowdFlow** | Flow branch diagnostic | 5 synthetic sequences, per-pixel GT flow | Evaluation only — synthetic, small |
| **Venice Video** | Geometry / perspective validation | 4 sequences, 167 annotated frames, known homography | Tests pixel-to-metric conversion |
| **Mall Dataset** | Lightweight density baseline | Sequential frames (images), MAT annotations | Early-stage training sanity check |
| **UCSD Pedestrian** | Anomaly baseline | Sequential frames, frame-level anomaly labels | Early-stage anomaly sanity check |

**Data root assumed:** `data/` directory at repo root.
Paths: `data/fdst/`, `data/umn/`, `data/shanghaitech/`, `data/crowdflow/`,
`data/venice/`, `data/mall/`, `data/ucsd/`

---

## Compute environment

- **GPU:** Tesla T4 via Google Colab (16 GB VRAM)
- **Python:** 3.10
- **CUDA:** 11.8
- **Key packages:** PyTorch 2.1, torchvision, OpenCV 4.9, NumPy, SciPy,
  Matplotlib 3.8
- **Model weights:** RAFT (princeton-vl/RAFT), CSRNet (leeyeehoo/CSRNet-pytorch)
- Use **gradient checkpointing** + **FP16 mixed precision** to stay within
  16 GB VRAM

---

## Repo structure to create

```
fluid-flow-pinn/
├── data/                        # datasets live here (not committed)
│   ├── fdst/
│   ├── umn/
│   ├── shanghaitech/
│   ├── crowdflow/
│   ├── venice/
│   ├── mall/
│   └── ucsd/
├── models/
│   ├── __init__.py
│   ├── branch1_density.py       # CSRNet wrapper → outputs density map ρ
│   ├── branch2_flow.py          # RAFT wrapper → outputs (ux, uy)
│   ├── branch3_pressure.py      # P = ρ · Var(u) computation
│   └── pinn.py                  # full three-branch model + physics loss
├── preprocessing/
│   ├── __init__.py
│   ├── frame_extractor.py       # video → frame pairs (t, t+1)
│   ├── umn_splitter.py          # cuts UMN .avi into 3 scene segments
│   └── dataset_loader.py        # PyTorch Dataset classes for each dataset
├── losses/
│   ├── __init__.py
│   ├── continuity_loss.py       # R = ∂ρ/∂t + ∇·(ρu), finite-difference
│   └── total_loss.py            # L_total = L_count + λ1·L_motion + λ2·‖R‖²
├── utils/
│   ├── __init__.py
│   ├── visualize.py             # overlay ρ, u, P on frames; save figures
│   └── metrics.py               # MAE, MSE, Physical Consistency Score (PCS)
├── configs/
│   └── default.yaml             # hyperparams: λ1, λ2, lr, batch_size, etc.
├── checkpoints/                 # saved model weights (not committed)
├── outputs/                     # visualisations, logs (not committed)
├── train.py                     # main training loop
├── evaluate.py                  # runs evaluation on test sets
├── smoke_test.py                # quick end-to-end sanity check (1 batch)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Implementation plan — do this in order

### Phase 1 — Perception (Part 1)

**Step 1: Repo scaffold**
- Create the full directory structure above
- Write `requirements.txt`, `.gitignore`, `README.md`
- Write `configs/default.yaml` with all hyperparameters

**Step 2: Preprocessing**
- `frame_extractor.py`: extract (frame_t, frame_{t+1}) pairs from FDST video
  clips; return as numpy arrays; handle variable resolutions across 13 scenes
- `umn_splitter.py`: split UMN `.avi` into 3 scene `.avi` files using
  hardcoded frame indices (scene 1: 0-640, scene 2: 641-1000,
  scene 3: 1001-end — verify against published indices)
- `dataset_loader.py`: implement `FDSTDataset`, `UMNDataset`,
  `ShanghaiTechDataset` as `torch.utils.data.Dataset` subclasses.
  Each `__getitem__` returns `{'frame_t': tensor, 'frame_t1': tensor,
  'density_map': tensor}`. Apply bicubic upsampling to UMN frames
  (320×240 → 640×480) before returning.

**Step 3: Branch 1 — CSRNet density**
- `models/branch1_density.py`: wrap CSRNet with dilated convolutions;
  load pretrained weights from `leeyeehoo/CSRNet-pytorch`;
  input: RGB frame (B, 3, H, W); output: density map ρ (B, 1, H/8, W/8)

**Step 4: Branch 2 — RAFT flow**
- `models/branch2_flow.py`: wrap RAFT; load pretrained weights;
  input: two consecutive frames (B, 3, H, W) each;
  output: flow field u = (ux, uy) as (B, 2, H, W)
  Note: upsample RAFT output to match CSRNet density map resolution

**Step 5: Smoke test Part 1**
- `smoke_test.py`: load 1 batch from FDSTDataset → run Branch 1 + Branch 2
  → print shapes of ρ and u → visualise with `utils/visualize.py` →
  save figure to `outputs/smoke_test.png`

---

### Phase 2 — Physics (Part 2)

**Step 6: Continuity loss**
- `losses/continuity_loss.py`: compute residual
  `R = ∂ρ/∂t + ∇·(ρu)` using finite differences.
  `∂ρ/∂t ≈ (ρ_t1 - ρ_t) / Δt` where Δt = 1/fps
  `∇·(ρu) = ∂(ρ·ux)/∂x + ∂(ρ·uy)/∂y` via central differences
  Return `‖R‖²` as the physics loss scalar

**Step 7: Total loss + Branch 3**
- `models/branch3_pressure.py`: P = ρ · Var(u) computed over a local
  spatial window (default 5×5 pixels)
- `losses/total_loss.py`: `L_total = L_count + λ1·L_motion + λ2·‖R‖²`
  where `L_count` = MSE(predicted density sum, GT count),
  `L_motion` = EPE (endpoint error) on CrowdFlow when GT flow available,
  else 0

**Step 8: Full PINN model**
- `models/pinn.py`: tie all three branches together; implement
  `forward(frame_t, frame_t1)` returning `{'rho': ..., 'u': ..., 'P': ...}`
  Use `torch.cuda.amp.autocast()` for FP16; enable gradient checkpointing
  on backbone

**Step 9: Training loop**
- `train.py`: standard PyTorch training loop; AdamW optimiser;
  cosine LR schedule; save checkpoints every epoch to `checkpoints/`;
  log MAE, MSE, PCS to stdout and optionally W&B

**Step 10: Evaluation**
- `evaluate.py`: run on ShanghaiTech (density MAE/MSE), CrowdFlow (EPE),
  UMN (pressure threshold → lead-time in seconds before visible panic onset)
- `utils/metrics.py`: implement MAE, MSE, EPE, and
  Physical Consistency Score (PCS = 1 - mean(|R|) / mean(|ρ|))

---

## Key implementation constraints

1. **Temporal pairs**: every batch item must be a consecutive frame pair
   (t, t+1) — never shuffle frames independently, always shuffle sequences
   then iterate within each sequence in order
2. **Resolution mismatch**: RAFT outputs full-resolution flow; CSRNet
   outputs 1/8 resolution density — upsample RAFT output using
   `F.interpolate(..., scale_factor=1/8, mode='bilinear')` to match
3. **UMN upsampling**: apply bicubic upsampling (320×240 → 640×480)
   BEFORE passing to RAFT — low resolution degrades flow quality
4. **Physics loss weighting**: start with λ1=0.1, λ2=0.01; these are
   tunable in `configs/default.yaml`
5. **Memory**: use gradient checkpointing on the shared backbone;
   use FP16 throughout; keep batch size ≤ 4 on T4

---

## What a good pressure map should look like

When the model works correctly on UMN:
- During normal dispersal: P(x,y,t) is low and uniform
- During panic onset (crowd acceleration + density spike): P spikes
  locally in the direction of motion, forming a visible "shockwave"
- Lead-time goal: P threshold exceeded ≥ 5 seconds before visible
  running/falling in the video

---

## Evaluation metrics to implement

| Metric | Formula | Purpose |
|--------|---------|---------|
| MAE | mean(|ρ_pred - ρ_gt|) | Density branch accuracy |
| MSE | mean((ρ_pred - ρ_gt)²) | Density branch accuracy |
| EPE | mean(‖u_pred - u_gt‖₂) | Flow branch accuracy (CrowdFlow only) |
| PCS | 1 - mean(|R|) / mean(|ρ|) | Physics constraint satisfaction |
| Lead-time (LT) | seconds between P > threshold and visible event | Anomaly detection power |

---

## Do NOT do

- Do not commit datasets or model weights to git (add to .gitignore)
- Do not load entire video into RAM — stream frames
- Do not train on CrowdFlow or ShanghaiTech (evaluation only)
- Do not hardcode absolute paths — use `configs/default.yaml` for all paths
- Do not skip the smoke test before moving to Phase 2

# Fluid-Flow PINN — Crowd Crush Early Warning

Physics-Informed Neural Network that treats human crowds as a compressible fluid
to detect dangerous pressure buildup **before** a crush event becomes visible.

**Team:** Varad Suryavanshi · Sarvesh Bodke · Devansh Shah (NYU CS)

---

## Core idea

Existing SOTA models (CSRNet, MCNN) estimate crowd density well but ignore the
physical forces governing crowd disasters. We couple density + optical flow
through the Navier-Stokes continuity equation as a differentiable training
constraint:

```
R = ∂ρ/∂t + ∇·(ρu)   (continuity residual — should be ≈ 0)

L_total = L_count + λ1·L_motion + λ2·‖R‖²

P(x,y,t) = ρ(x,y,t) · Var(u)   (pressure proxy)
```

---

## Architecture

```
Video frames (t, t+1)
        │
        ▼
┌──────────────────┐
│  Shared backbone │  (lightweight CNN)
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
  Physics Loss Layer  (continuity residual)
         │
         ▼
  Branch 3: Pressure map P = ρ · Var(u)
         │
         ▼
  Risk Alert + Lead-time estimate
```

---

## Quick start

```bash
pip install -r requirements.txt

# Sanity check (1 batch, no training)
python smoke_test.py

# Full training
python train.py --config configs/default.yaml

# Evaluation
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth
```

---

## Data layout

Place datasets under `data/` (not committed to git):

```
data/
├── fdst/          # Primary training — 13 scenes, 394k annotations
├── umn/           # Anomaly eval — single .avi, 3 scenes
├── shanghaitech/  # Density benchmark — part_A/ and part_B/
├── crowdflow/     # Flow diagnostic — 5 synthetic sequences
├── venice/        # Perspective validation
├── mall/          # Lightweight density baseline
└── ucsd/          # Anomaly baseline
```

---

## Key hyperparameters (`configs/default.yaml`)

| Param | Default | Notes |
|-------|---------|-------|
| `lambda1` | 0.1 | Weight for motion loss |
| `lambda2` | 0.01 | Weight for physics (continuity) loss |
| `lr` | 1e-4 | AdamW learning rate |
| `batch_size` | 4 | Max for T4 16 GB with FP16 |
| `fps` | 30 | Used for ∂ρ/∂t = (ρ_t1−ρ_t)/Δt |

---

## Evaluation metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| MAE | mean(|ρ_pred − ρ_gt|) | Density accuracy |
| MSE | mean((ρ_pred − ρ_gt)²) | Density accuracy |
| EPE | mean(‖u_pred − u_gt‖₂) | Flow accuracy (CrowdFlow only) |
| PCS | 1 − mean(|R|) / mean(|ρ|) | Physics constraint satisfaction |
| Lead-time | seconds before P > threshold precedes visible panic | Anomaly detection power |

---

## Compute

- GPU: Tesla T4 (16 GB VRAM) via Google Colab
- Python 3.10, CUDA 11.8, PyTorch 2.1
- FP16 mixed precision + gradient checkpointing throughout

"""Main training loop — Phase 2 Step 9.

Usage:
    python train.py                            # uses configs/default.yaml
    python train.py --config configs/default.yaml --epochs 50
    python train.py --wandb-project fluid-pinn --wandb-entity my-team
"""

from __future__ import annotations

import argparse
import heapq
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from losses.total_loss import TotalLoss
from models.pinn import FluidFlowPINN
from preprocessing.dataset_loader import FDSTDataset, SequentialSceneSampler
from torchvision import transforms as T

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config helpers ─────────────────────────────────────────────────────────────

def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _cfg(cfg: dict, *keys, default=None):
    """Nested key access with a default."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


# ── Checkpoint manager — keep top-K by validation loss ────────────────────────

class CheckpointManager:
    """Saves the best K checkpoints, deleting the worst when full.

    Heap stores (metric, path) tuples where *lower* metric is better.
    We negate the metric so the Python min-heap behaves as a max-heap,
    popping the worst (highest loss) checkpoint when we exceed k.
    """

    def __init__(self, ckpt_dir: Path, keep_top_k: int = 3) -> None:
        self.ckpt_dir = ckpt_dir
        self.keep_top_k = keep_top_k
        # Max-heap via negated loss: heap[0] is the worst (highest loss) entry
        self._heap: list[tuple[float, str]] = []  # (-loss, path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        state: dict,
        metric: float,
        tag: str,
    ) -> Path:
        """Save checkpoint; prune if we now exceed top-k.

        Args:
            state:  dict to torch.save
            metric: lower-is-better scalar (e.g. validation loss)
            tag:    filename stem, e.g. "epoch_005_loss_0.1234"

        Returns:
            Path of the saved checkpoint.
        """
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(state, path)
        log.info("Saved checkpoint: %s  (metric=%.6f)", path.name, metric)

        # Push negated so the heap root = worst (largest loss)
        heapq.heappush(self._heap, (-metric, str(path)))

        if len(self._heap) > self.keep_top_k:
            _, worst_path = heapq.heappop(self._heap)
            try:
                Path(worst_path).unlink(missing_ok=True)
                log.info("Pruned checkpoint: %s", Path(worst_path).name)
            except OSError as e:
                log.warning("Could not delete checkpoint %s: %s", worst_path, e)

        return path

    @property
    def best_metric(self) -> Optional[float]:
        if not self._heap:
            return None
        return -max(v for v, _ in self._heap)  # heap root is worst; best = min


# ── W&B wrapper — no-ops when wandb is disabled ────────────────────────────────

class _WandbLogger:
    def __init__(self, project: Optional[str], entity: Optional[str], config: dict) -> None:
        self._enabled = project is not None
        if self._enabled:
            try:
                import wandb
                wandb.init(
                    project=project,
                    entity=entity,
                    config=config,
                    resume="allow",
                )
                self._wandb = wandb
                log.info("W&B run: %s", wandb.run.url)
            except ImportError:
                log.warning("wandb not installed — disabling W&B logging.")
                self._enabled = False

    def log(self, metrics: dict, step: int) -> None:
        if self._enabled:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._enabled:
            self._wandb.finish()


# ── Training utilities ─────────────────────────────────────────────────────────

def _make_loader(
    cfg: dict,
    split: str,
    seed: int,
    epoch: int = 0,
) -> DataLoader:
    # split="train" → data/fdst/train_data/; split="test" → data/fdst/test_data/
    # Both are sub-splits of the same FDST dataset; validation uses the test split.
    data_root = Path(_cfg(cfg, "data", "fdst", default="data/fdst/"))
    batch_size = _cfg(cfg, "training", "batch_size", default=4)
    num_workers = min(
        _cfg(cfg, "training", "num_workers", default=4),
        os.cpu_count() or 1,
    )

    # Resize every frame to a fixed spatial size so the default collator can
    # stack variable-resolution FDST scenes into a batch.
    target_h = _cfg(cfg, "preprocessing", "target_height", default=1080)
    target_w = _cfg(cfg, "preprocessing", "target_width",  default=1920)
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]
    frame_transform = T.Compose([
        T.ToTensor(),
        T.Resize((target_h, target_w), antialias=True),
        T.Normalize(mean=_MEAN, std=_STD),
    ])

    dataset = FDSTDataset(root=data_root, split=split, transform=frame_transform)
    if len(dataset) == 0:
        raise RuntimeError(
            f"FDSTDataset for split='{split}' is empty. "
            f"Check that data exists at: {data_root / f'{split}_data'}"
        )

    if split == "train":
        sampler = SequentialSceneSampler(dataset, shuffle=True)
        sampler.set_epoch(epoch)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        # Val/test: deterministic, no shuffle
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )


@torch.no_grad()
def _validate(
    model: FluidFlowPINN,
    loader: DataLoader,
    criterion: TotalLoss,
    device: torch.device,
    use_fp16: bool,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"total": 0.0, "count": 0.0, "motion": 0.0, "physics": 0.0}
    mae_sum = 0.0
    n_batches = 0

    for batch in loader:
        frame_t  = batch["frame_t"].to(device, non_blocking=True)
        frame_t1 = batch["frame_t1"].to(device, non_blocking=True)
        gt_count = batch["density_map"].sum(dim=(1, 2, 3)).to(device, non_blocking=True)
        gt_flow  = batch.get("flow")
        if gt_flow is not None:
            gt_flow = gt_flow.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_fp16):
            out = model(frame_t, frame_t1)
            losses = criterion(
                rho_t=out["rho"],
                rho_t1=out["rho_t1"],
                u=out["u"],
                gt_count=gt_count,
                gt_flow=gt_flow,
            )

        for k in totals:
            totals[k] += losses[k].item()

        pred_count = out["rho"].sum(dim=(1, 2, 3))
        mae_sum += (pred_count - gt_count).abs().mean().item()
        n_batches += 1

    if n_batches == 0:
        return {k: float("inf") for k in totals}

    metrics = {f"val/{k}_loss": v / n_batches for k, v in totals.items()}
    metrics["val/mae"] = mae_sum / n_batches
    return metrics


# ── Main training loop ─────────────────────────────────────────────────────────

def train(cfg: dict, args: argparse.Namespace) -> None:
    # ── Seed ──────────────────────────────────────────────────────────────────
    seed = _cfg(cfg, "training", "seed", default=42)
    torch.manual_seed(seed)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    use_fp16 = _cfg(cfg, "model", "use_fp16", default=True) and device.type == "cuda"

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FluidFlowPINN(
        use_grad_checkpoint=_cfg(cfg, "model", "grad_checkpoint", default=True),
        raft_frozen=True,
        pressure_window=_cfg(cfg, "model", "pressure_window", default=5),
        csrnet_weights=_cfg(cfg, "model", "csrnet_weights"),
        raft_weights=_cfg(cfg, "model", "raft_weights"),
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = TotalLoss(
        lambda1=_cfg(cfg, "training", "lambda1", default=0.1),
        lambda2=_cfg(cfg, "training", "lambda2", default=0.01),
        fps=float(_cfg(cfg, "preprocessing", "fps", default=30)),
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr           = _cfg(cfg, "training", "lr", default=1e-4)
    weight_decay = _cfg(cfg, "training", "weight_decay", default=1e-4)
    # Only pass parameters that require grad (RAFT is frozen)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ── LR schedule: linear warmup → cosine decay ─────────────────────────────
    epochs       = _cfg(cfg, "training", "epochs", default=100)
    warmup_epochs = _cfg(cfg, "training", "warmup_epochs", default=5)
    min_lr       = _cfg(cfg, "training", "min_lr", default=1e-6)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=min_lr / lr,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # ── AMP scaler ────────────────────────────────────────────────────────────
    scaler = GradScaler(device="cuda", enabled=use_fp16)

    # ── Checkpoint manager ────────────────────────────────────────────────────
    ckpt_dir  = Path(_cfg(cfg, "output", "checkpoints", default="checkpoints/"))
    keep_top_k = _cfg(cfg, "training", "keep_top_k", default=3)
    ckpt_mgr  = CheckpointManager(ckpt_dir, keep_top_k=keep_top_k)

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_project = args.wandb_project or _cfg(cfg, "training", "wandb_project")
    wandb_entity  = args.wandb_entity  or _cfg(cfg, "training", "wandb_entity")
    logger = _WandbLogger(wandb_project, wandb_entity, config=cfg)

    # ── Data loaders ──────────────────────────────────────────────────────────
    batch_size  = _cfg(cfg, "training", "batch_size", default=4)
    num_workers = _cfg(cfg, "training", "num_workers", default=4)

    # ── Training state ────────────────────────────────────────────────────────
    start_epoch  = 1
    global_step  = 0
    best_val_loss = float("inf")

    # Optionally resume from a checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch  = ckpt["epoch"] + 1
        global_step  = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log.info("Resumed from %s  (epoch %d)", args.resume, ckpt["epoch"])

    log.info(
        "Training for %d epochs, batch_size=%d, lr=%.2e, λ1=%.3f, λ2=%.4f",
        epochs, batch_size, lr,
        _cfg(cfg, "training", "lambda1", default=0.1),
        _cfg(cfg, "training", "lambda2", default=0.01),
    )

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        # Build loaders fresh each epoch so SequentialSceneSampler shuffles
        train_loader = _make_loader(cfg, "train", seed=seed, epoch=epoch)
        val_loader   = _make_loader(cfg, "test",  seed=seed, epoch=epoch)

        model.train()
        epoch_totals: dict[str, float] = {
            "total": 0.0, "count": 0.0, "motion": 0.0, "physics": 0.0
        }
        epoch_mae = 0.0
        n_batches  = 0
        t0 = time.time()

        for batch in train_loader:
            frame_t  = batch["frame_t"].to(device, non_blocking=True)
            frame_t1 = batch["frame_t1"].to(device, non_blocking=True)
            gt_count = batch["density_map"].sum(dim=(1, 2, 3)).to(device, non_blocking=True)
            gt_flow  = batch.get("flow")
            if gt_flow is not None:
                gt_flow = gt_flow.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_fp16):
                out = model(frame_t, frame_t1)
                losses = criterion(
                    rho_t=out["rho"],
                    rho_t1=out["rho_t1"],
                    u=out["u"],
                    gt_count=gt_count,
                    gt_flow=gt_flow,
                )

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Accumulate metrics
            for k in epoch_totals:
                epoch_totals[k] += losses[k].item()

            with torch.no_grad():
                pred_count = out["rho"].sum(dim=(1, 2, 3))
                epoch_mae += (pred_count - gt_count).abs().mean().item()

            n_batches  += 1
            global_step += 1

            # Step-level W&B log every 50 steps
            if global_step % 50 == 0:
                step_metrics = {
                    "train/total_loss":   losses["total"].item(),
                    "train/count_loss":   losses["count"].item(),
                    "train/motion_loss":  losses["motion"].item(),
                    "train/physics_loss": losses["physics"].item(),
                    "train/lr":           scheduler.get_last_lr()[0],
                }
                logger.log(step_metrics, step=global_step)
                log.info(
                    "  step %6d | loss=%.4f  count=%.4f  motion=%.4f  phys=%.6f  lr=%.2e",
                    global_step,
                    losses["total"].item(),
                    losses["count"].item(),
                    losses["motion"].item(),
                    losses["physics"].item(),
                    scheduler.get_last_lr()[0],
                )

        scheduler.step()

        # ── Epoch-level metrics ──────────────────────────────────────────────
        if n_batches == 0:
            log.warning("Epoch %d: no training batches — skipping.", epoch)
            continue

        epoch_metrics = {
            "train/epoch_total_loss":   epoch_totals["total"]   / n_batches,
            "train/epoch_count_loss":   epoch_totals["count"]   / n_batches,
            "train/epoch_motion_loss":  epoch_totals["motion"]  / n_batches,
            "train/epoch_physics_loss": epoch_totals["physics"] / n_batches,
            "train/epoch_mae":          epoch_mae / n_batches,
            "epoch":                    epoch,
        }

        # ── Validation ──────────────────────────────────────────────────────
        val_metrics = _validate(model, val_loader, criterion, device, use_fp16)
        val_loss    = val_metrics["val/total_loss"]

        all_metrics = {**epoch_metrics, **val_metrics}
        logger.log(all_metrics, step=global_step)

        elapsed = time.time() - t0
        log.info(
            "Epoch %3d/%d | train_loss=%.4f  val_loss=%.4f  "
            "val_mae=%.2f  lr=%.2e  time=%.1fs",
            epoch, epochs,
            epoch_metrics["train/epoch_total_loss"],
            val_loss,
            val_metrics.get("val/mae", float("nan")),
            scheduler.get_last_lr()[0],
            elapsed,
        )

        # ── Checkpoint ──────────────────────────────────────────────────────
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        state = {
            "epoch":         epoch,
            "global_step":   global_step,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "scaler":        scaler.state_dict(),
            "val_loss":      val_loss,
            "best_val_loss": best_val_loss,
            "config":        cfg,
        }
        tag = f"epoch_{epoch:04d}_valloss_{val_loss:.6f}"
        ckpt_mgr.save(state, metric=val_loss, tag=tag)

        if is_best:
            best_path = ckpt_dir / "best.pt"
            torch.save(state, best_path)
            log.info("New best checkpoint saved → %s", best_path)

    logger.finish()
    log.info("Training complete. Best val loss: %.6f", best_val_loss)


# ── Entry point ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train fluid-flow PINN")
    p.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    p.add_argument("--epochs",       type=int,   default=None, help="Override training.epochs")
    p.add_argument("--batch-size",   type=int,   default=None, help="Override training.batch_size")
    p.add_argument("--lr",           type=float, default=None, help="Override training.lr")
    p.add_argument("--fdst-path",                default=None, help="Override data.fdst path (e.g. /data/fdst/)")
    p.add_argument("--wandb-project",            default=None, help="W&B project name")
    p.add_argument("--wandb-entity",             default=None, help="W&B entity / team")
    p.add_argument("--resume",                   default=None, help="Path to checkpoint to resume from")
    p.add_argument("--no-fp16", action="store_true",           help="Disable FP16 mixed precision")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    cfg = _load_config(args.config)

    # CLI overrides
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg.setdefault("training", {})["lr"] = args.lr
    if args.fdst_path is not None:
        cfg.setdefault("data", {})["fdst"] = args.fdst_path
    if args.no_fp16:
        cfg.setdefault("model", {})["use_fp16"] = False

    train(cfg, args)


if __name__ == "__main__":
    main()

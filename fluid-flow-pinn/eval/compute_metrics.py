"""Read a per-frame CSV from `run_anomaly.py` and print the metric table.

Outputs:
  * Overall AUROC, AUPRC, best-F1, threshold, lead-time-to-onset
  * Per-scene rows (so the user sees if one bad scene drags the mean down)
  * `results/<csv_stem>_roc.png`, `results/<csv_stem>_pr.png`, `results/<csv_stem>_timeline.png`

Examples:
    python eval/compute_metrics.py results/umn_full.csv
    python eval/compute_metrics.py results/umn_full.csv --fps 30
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


def _load_csv(path: Path) -> dict[str, dict]:
    """Group rows by scene → {frames, scores, labels}."""
    out: dict[str, dict] = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            scene = row["scene"]
            d = out.setdefault(scene, {"frames": [], "scores": [], "labels": []})
            d["frames"].append(int(row["frame"]))
            d["scores"].append(float(row["p_max"]))
            d["labels"].append(int(row["label"]))
    for d in out.values():
        d["frames"] = np.asarray(d["frames"], dtype=np.int64)
        d["scores"] = np.asarray(d["scores"], dtype=np.float64)
        d["labels"] = np.asarray(d["labels"], dtype=np.int32)
    return out


def _onset_frame(scene: str, dataset_hint: str | None, umn_table: dict | None) -> int | None:
    if dataset_hint == "umn" and umn_table is not None and scene in umn_table:
        return int(umn_table[scene]["onset_frame"])
    return None


def main(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from utils.metrics import anomaly_metrics
    from sklearn.metrics import precision_recall_curve, roc_curve

    csv_path = Path(args.csv)
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    by_scene = _load_csv(csv_path)
    log.info("Loaded %d scenes from %s", len(by_scene), csv_path)

    # Optional UMN onset table for lead-time
    umn_table = None
    if args.umn_labels_json and Path(args.umn_labels_json).exists():
        with open(args.umn_labels_json) as f:
            umn_table = json.load(f)["scenes"]

    dataset_hint = "umn" if "umn" in csv_path.stem else ("ucsd" if "ucsd" in csv_path.stem else None)

    # ── Per-scene metrics ─────────────────────────────────────────────────
    print(f"\nPer-scene metrics ({csv_path.name}):")
    print(f"  {'scene':<15} {'AUROC':>7} {'AUPRC':>7} {'F1*':>6} {'τ*':>7} {'lead(s)':>8}  pos/neg")
    for scene, d in by_scene.items():
        m = anomaly_metrics(
            d["scores"], d["labels"], fps=args.fps,
            onset_frame=_onset_frame(scene, dataset_hint, umn_table),
        )
        print(f"  {scene:<15} {m['auroc']:>7.3f} {m['auprc']:>7.3f} "
              f"{m['f1_best']:>6.3f} {m['threshold_best']:>7.3f} "
              f"{m['lead_time_s']:>8.2f}  {m['n_pos']}/{m['n_neg']}")

    # ── Pooled metrics ────────────────────────────────────────────────────
    all_scores = np.concatenate([d["scores"] for d in by_scene.values()])
    all_labels = np.concatenate([d["labels"] for d in by_scene.values()])
    pooled = anomaly_metrics(all_scores, all_labels, fps=args.fps)
    print("\nPooled across scenes:")
    print(f"  AUROC = {pooled['auroc']:.3f}")
    print(f"  AUPRC = {pooled['auprc']:.3f}")
    print(f"  F1*   = {pooled['f1_best']:.3f}  @ τ={pooled['threshold_best']:.3f}")
    print(f"  pos/neg = {pooled['n_pos']}/{pooled['n_neg']}\n")

    # ── Plots ─────────────────────────────────────────────────────────────
    out_dir = csv_path.parent
    stem = csv_path.stem

    if pooled["n_pos"] > 0 and pooled["n_neg"] > 0:
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        prec, rec, _ = precision_recall_curve(all_labels, all_scores)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(fpr, tpr, lw=2, label=f"AUROC={pooled['auroc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC — {stem}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / f"{stem}_roc.png", dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(rec, prec, lw=2, label=f"AUPRC={pooled['auprc']:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"PR — {stem}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / f"{stem}_pr.png", dpi=130); plt.close(fig)

        log.info("Saved ROC + PR curves under %s", out_dir)

    # Per-scene timeline plot (helpful for sanity inspection)
    n = len(by_scene)
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 2.5 * rows), squeeze=False)
    for ax, (scene, d) in zip(axes.flat, by_scene.items()):
        ax.plot(d["frames"], d["scores"], lw=1, color="steelblue", label="p_max")
        # Shade anomaly region
        in_anom = d["labels"] > 0
        if in_anom.any():
            ax.fill_between(d["frames"], 0, d["scores"].max() * 1.05,
                             where=in_anom, color="red", alpha=0.15, label="anomaly")
        if pooled["threshold_best"] == pooled["threshold_best"]:  # not nan
            ax.axhline(pooled["threshold_best"], color="green",
                       linestyle="--", lw=1, label=f"τ*={pooled['threshold_best']:.2f}")
        ax.set_title(scene, fontsize=9); ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(by_scene):]:
        ax.axis("off")
    fig.suptitle(f"Per-scene p_max(t) — {stem}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_timeline.png", dpi=130)
    plt.close(fig)

    log.info("Saved per-scene timeline → %s", out_dir / f"{stem}_timeline.png")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute AUROC/AUPRC/lead-time from a per-frame CSV")
    p.add_argument("csv", help="Path to a CSV produced by run_anomaly.py")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--umn-labels-json",
                   default=str(REPO_ROOT / "eval" / "labels" / "umn_labels.json"))
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())

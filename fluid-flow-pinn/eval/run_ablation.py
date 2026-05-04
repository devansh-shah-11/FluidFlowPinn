"""Run the four (α, β, γ) configs on a labeled dataset and aggregate.

Output: `results/ablation_<dataset>.md` — one Markdown table ready to paste
into a slide. Internally calls `run_anomaly.py` for each config and
`compute_metrics.py` to score it.

Examples:
    python eval/run_ablation.py --dataset umn --umn-path data/UMN/
    python eval/run_ablation.py --dataset ucsd --ucsd-path /scratch/UCSDped2/Test \\
        --lwcc-model DM-Count
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
RESULTS = REPO_ROOT / "results"

CONFIGS = [
    ("density_only",   1.0, 0.0, 0.0),
    ("speed_only",     0.0, 1.0, 0.0),
    ("density_speed",  1.0, 1.0, 0.0),
    ("full",           1.0, 1.0, 1.0),
]


def _run_one(args: argparse.Namespace, tag: str, a: float, b: float, g: float) -> Path:
    csv_path = RESULTS / f"{args.dataset}_{tag}.csv"
    if csv_path.exists() and not args.force:
        log.info("[skip] %s already exists (use --force to regenerate)", csv_path)
        return csv_path

    cmd = [
        sys.executable, str(REPO_ROOT / "eval" / "run_anomaly.py"),
        "--dataset", args.dataset,
        "--tag", tag,
        "--p-alpha", str(a), "--p-beta", str(b), "--p-gamma", str(g),
    ]
    if args.dataset == "umn":
        cmd += ["--umn-path", args.umn_path]
    else:
        cmd += ["--ucsd-path", args.ucsd_path]
    if args.lwcc_model:
        cmd += ["--lwcc-model", args.lwcc_model]
    if args.detector:
        cmd += ["--detector", args.detector]
    log.info("→ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return csv_path


def _score(csv_path: Path, fps: float) -> dict:
    """Reuse compute_metrics' pooled metrics via direct call."""
    sys.path.insert(0, str(REPO_ROOT))
    from utils.metrics import anomaly_metrics
    import numpy as np

    scores: list[float] = []; labels: list[int] = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            scores.append(float(row["p_max"]))
            labels.append(int(row["label"]))
    return anomaly_metrics(np.asarray(scores), np.asarray(labels), fps=fps)


def main(args: argparse.Namespace) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, float, float, float, dict]] = []
    for tag, a, b, g in CONFIGS:
        csv_path = _run_one(args, tag, a, b, g)
        m = _score(csv_path, fps=args.fps)
        rows.append((tag, a, b, g, m))

    # Build markdown table
    out_md = RESULTS / f"ablation_{args.dataset}.md"
    with open(out_md, "w") as f:
        f.write(f"# Ablation: {args.dataset.upper()}\n\n")
        f.write("| Config | α | β | γ | AUROC | AUPRC | F1\\* | τ\\* |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for tag, a, b, g, m in rows:
            f.write(f"| {tag} | {a} | {b} | {g} | "
                    f"{m['auroc']:.3f} | {m['auprc']:.3f} | "
                    f"{m['f1_best']:.3f} | {m['threshold_best']:.3f} |\n")
        f.write("\n_F1\\* and τ\\* are at the F1-maximizing threshold (pooled across scenes)._\n")
    log.info("Saved ablation table → %s", out_md)
    print()
    print(open(out_md).read())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run (α,β,γ) ablation on a labeled dataset")
    p.add_argument("--dataset", choices=("umn", "ucsd"), required=True)
    p.add_argument("--umn-path",  default=str(REPO_ROOT / "data" / "umn"))
    p.add_argument("--ucsd-path", default=None)
    p.add_argument("--lwcc-model", default=None,
                   help="lwcc model for the density term (e.g. DM-Count). "
                        "Required for the density_only / density_speed / full configs to be meaningful.")
    p.add_argument("--detector",   default=None,
                   help="YOLO detector to gate the variance window with a person mask.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--force", action="store_true",
                   help="Re-run even if results/<dataset>_<tag>.csv already exists")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dataset == "ucsd" and not args.ucsd_path:
        log.error("--ucsd-path required for --dataset ucsd")
        sys.exit(2)
    main(args)

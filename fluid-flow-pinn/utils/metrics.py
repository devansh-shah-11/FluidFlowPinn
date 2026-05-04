"""Quantitative metrics for the evaluation framework.

Three groups, all pure functions over numpy arrays / lists:

  * Density / count : `density_mae`, `density_rmse`, `count_mae`, `count_rmse`
  * Optical flow    : `flow_epe`
  * Anomaly score   : `anomaly_metrics` (AUROC, AUPRC, F1@best τ, lead-time)
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


# ── Density / count ──────────────────────────────────────────────────────────

def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def density_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-pixel L1 between predicted and ground-truth density maps."""
    p, g = _to_np(pred), _to_np(gt)
    return float(np.abs(p - g).mean())


def density_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _to_np(pred), _to_np(gt)
    return float(np.sqrt(((p - g) ** 2).mean()))


def count_mae(pred_counts: Iterable[float], gt_counts: Iterable[float]) -> float:
    """MAE between integrated counts (the standard crowd-counting metric)."""
    p = np.asarray(list(pred_counts), dtype=np.float64)
    g = np.asarray(list(gt_counts),   dtype=np.float64)
    return float(np.abs(p - g).mean())


def count_rmse(pred_counts: Iterable[float], gt_counts: Iterable[float]) -> float:
    p = np.asarray(list(pred_counts), dtype=np.float64)
    g = np.asarray(list(gt_counts),   dtype=np.float64)
    return float(np.sqrt(((p - g) ** 2).mean()))


def count_mape(pred_counts: Iterable[float], gt_counts: Iterable[float]) -> float:
    p = np.asarray(list(pred_counts), dtype=np.float64)
    g = np.asarray(list(gt_counts),   dtype=np.float64)
    mask = g > 0
    if not mask.any():
        return float("nan")
    return float(np.abs((p[mask] - g[mask]) / g[mask]).mean() * 100.0)


# ── Optical flow ─────────────────────────────────────────────────────────────

def flow_epe(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """End-Point Error: mean L2 distance between predicted and GT flow.

    Both arrays shape (2, H, W). `mask` (H, W) selects valid pixels — useful
    for synthetic datasets like CrowdFlow where occlusions are excluded.
    """
    p, g = _to_np(pred), _to_np(gt)
    assert p.shape == g.shape and p.shape[0] == 2, f"bad shapes {p.shape} {g.shape}"
    epe = np.sqrt(((p - g) ** 2).sum(axis=0))  # (H, W)
    if mask is not None:
        m = _to_np(mask) > 0
        if not m.any():
            return float("nan")
        return float(epe[m].mean())
    return float(epe.mean())


# ── Anomaly detection ────────────────────────────────────────────────────────

def anomaly_metrics(
    scores: Iterable[float],
    labels: Iterable[int],
    fps: float = 30.0,
    onset_frame: Optional[int] = None,
) -> dict:
    """AUROC / AUPRC / best-F1 / lead-time for a single sequence (or pooled set).

    Args:
      scores: per-frame anomaly score (e.g. `p_max`).
      labels: per-frame binary label (1 = anomaly/panic frame).
      fps:    frame rate, used to convert lead-time frames → seconds.
      onset_frame: frame index where the anomaly *first* becomes positive.
                   If given, lead_time_s = (onset_frame - first_alarm) / fps.
                   Positive = the detector raised the alarm BEFORE the labels
                   flipped → early warning. Negative = late detection.
                   If None, falls back to the first label==1 index.

    Returns dict with keys:
      auroc, auprc, f1_best, threshold_best, lead_time_s, n_pos, n_neg
    """
    s = np.asarray(list(scores), dtype=np.float64)
    y = np.asarray(list(labels), dtype=np.int32)
    if s.shape != y.shape:
        raise ValueError(f"scores/labels shape mismatch: {s.shape} vs {y.shape}")

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    out = {"n_pos": n_pos, "n_neg": n_neg,
           "auroc": float("nan"), "auprc": float("nan"),
           "f1_best": float("nan"), "threshold_best": float("nan"),
           "lead_time_s": float("nan")}

    if n_pos == 0 or n_neg == 0:
        return out

    try:
        from sklearn.metrics import (
            average_precision_score,
            precision_recall_curve,
            roc_auc_score,
        )
    except ImportError as e:
        raise ImportError("sklearn is required for anomaly_metrics — pip install scikit-learn") from e

    out["auroc"] = float(roc_auc_score(y, s))
    out["auprc"] = float(average_precision_score(y, s))

    # Best F1 over the precision-recall sweep (avoids picking τ by hand)
    precision, recall, thresholds = precision_recall_curve(y, s)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    # precision_recall_curve appends a (P=1, R=0) endpoint with no threshold.
    valid = np.arange(len(thresholds))
    if valid.size:
        best_i = int(np.nanargmax(f1[: len(thresholds)]))
        out["f1_best"] = float(f1[best_i])
        out["threshold_best"] = float(thresholds[best_i])

        # Lead time: first index where score crosses the best threshold,
        # measured against the labelled onset frame.
        first_alarm = np.argmax(s >= thresholds[best_i])  # 0 if never triggers OR triggers at 0
        triggered = bool((s >= thresholds[best_i]).any())
        if triggered:
            ref = onset_frame if onset_frame is not None else int(np.argmax(y > 0))
            out["lead_time_s"] = float((ref - int(first_alarm)) / max(fps, 1e-6))

    return out

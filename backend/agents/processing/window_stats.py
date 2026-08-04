"""Overlap-aware statistics for windowed evaluation (C4).

When windows overlap (stride < window) or are simply dense in time, adjacent
windows are strongly autocorrelated — two windows one second apart share almost
all of their samples. Treating each window as an independent observation makes
confidence intervals anti-conservative (too tight) and significance tests
over-confident.

The fix is to resample at the level of the **stop event**, not the window. Every
window carries an ``event_id`` (the upcoming stop for positives, a per-block id
for negatives); the block bootstrap resamples those groups with replacement so
the effective sample size is the number of distinct events, not windows.

HARD RULE (from the improvement plan): window-size / overlapping-window results
may not be reported with naive per-window CIs — use ``block_bootstrap_f1_ci``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def n_effective(groups: Sequence) -> int:
    """The honest sample size: number of distinct event blocks, not windows."""
    return len(set(groups))


def block_bootstrap_metric_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence,
    *,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = f1_score,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Block-bootstrap CI for a metric, resampling by ``groups`` (event blocks).

    Returns (point_estimate, ci_low, ci_high). The point estimate is the metric
    on the full data; the interval comes from resampling whole groups with
    replacement so autocorrelated windows within a group move together.
    """
    y_true = np.asarray(y_true, dtype=int)
    # NOTE: do NOT cast y_pred — it may be continuous scores (for AUC). An int
    # cast would collapse 0.9 -> 0 and silently destroy the metric.
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups, dtype=object)
    point = float(metric_fn(y_true, y_pred))

    # index rows by group
    by_group: Dict[object, List[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    unique_groups = list(by_group.keys())
    if len(unique_groups) < 2:
        return (point, point, point)

    # Stratify groups by whether they carry any positive sample. Under heavy
    # class imbalance (e.g. 1% positives) a plain group resample almost never
    # draws the few positive event-groups, collapsing the metric to its
    # one-class value. Resampling each stratum to its own size preserves the
    # class structure while still moving whole (autocorrelated) groups together.
    pos_groups = [g for g in unique_groups if int(np.any(y_true[by_group[g]] == 1))]
    neg_groups = [g for g in unique_groups if g not in set(pos_groups)]

    rng = np.random.RandomState(seed)
    stats: List[float] = []
    for _ in range(n_boot):
        idx: List[int] = []
        for stratum in (pos_groups, neg_groups):
            if not stratum:
                continue
            chosen = rng.choice(len(stratum), size=len(stratum), replace=True)
            for gi in chosen:
                idx.extend(by_group[stratum[gi]])
        idx_arr = np.asarray(idx, dtype=int)
        stats.append(float(metric_fn(y_true[idx_arr], y_pred[idx_arr])))
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (point, lo, hi)


def assert_non_overlapping(window_starts: Sequence[float], window_s: float) -> None:
    """Guard: headline metrics must use non-overlapping windows. Raise if any
    two consecutive windows overlap (start gap < window_s)."""
    s = sorted(float(x) for x in window_starts)
    for a, b in zip(s, s[1:]):
        if b - a < window_s - 1e-9:
            raise ValueError(
                f"Overlapping windows detected (gap {b - a:.3f}s < window {window_s:.3f}s). "
                f"Headline metrics require non-overlapping windows; use overlap only for "
                f"training augmentation and block_bootstrap_metric_ci for any CI."
            )

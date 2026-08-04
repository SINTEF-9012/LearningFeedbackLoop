"""Metric computation and cross-phase comparison."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricSet:
    """Standard binary-classification metrics for one phase."""

    phase: str = ""
    operation: str = ""
    n_samples: int = 0
    threshold: float = 0.5

    # Confusion matrix
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    # Derived
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    specificity: float = 0.0
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0

    # Rank-based
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    avg_precision: float = 0.0

    # Score distributions
    mean_score_positive: float = 0.0
    mean_score_negative: float = 0.0
    score_separation: float = 0.0  # mean_pos - mean_neg

    # Bootstrap 95% confidence intervals (added post-critique)
    f1_ci: Optional[List[float]] = None      # [lower, upper]
    auc_roc_ci: Optional[List[float]] = None
    precision_ci: Optional[List[float]] = None
    recall_ci: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Clean up None CIs
        for key in ["f1_ci", "auc_roc_ci", "precision_ci", "recall_ci"]:
            if d.get(key) is None:
                d[key] = []
        return d


@dataclass
class ComparisonReport:
    """Side-by-side comparison: baseline (test) vs feedback (eval)."""

    test_metrics: MetricSet = field(default_factory=MetricSet)
    eval_metrics: MetricSet = field(default_factory=MetricSet)

    # Absolute deltas (eval - test)
    delta_precision: float = 0.0
    delta_recall: float = 0.0
    delta_f1: float = 0.0
    delta_auc_roc: float = 0.0
    delta_auc_pr: float = 0.0
    delta_balanced_accuracy: float = 0.0
    delta_score_separation: float = 0.0

    # Relative (%) improvement
    pct_f1_improvement: float = 0.0
    pct_auc_roc_improvement: float = 0.0
    pct_precision_improvement: float = 0.0

    # Feedback-specific
    n_feedback_events: int = 0
    n_confirms: int = 0
    n_dismissals: int = 0
    n_missed_event_feedback: int = 0  # feedback from missed-event path
    feedback_accuracy: float = 0.0  # correct feedback / total feedback

    # Counterfactual analysis (eval only)
    mean_counterfactual_delta: float = 0.0  # avg(actual - counterfactual)
    pct_improved_by_feedback: float = 0.0   # % samples where feedback helped
    n_predictions_flipped: int = 0           # binary predictions changed by feedback

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test": self.test_metrics.to_dict(),
            "eval": self.eval_metrics.to_dict(),
            "deltas": {
                "precision": self.delta_precision,
                "recall": self.delta_recall,
                "f1": self.delta_f1,
                "auc_roc": self.delta_auc_roc,
                "auc_pr": self.delta_auc_pr,
                "balanced_accuracy": self.delta_balanced_accuracy,
                "score_separation": self.delta_score_separation,
            },
            "pct_improvements": {
                "f1": self.pct_f1_improvement,
                "auc_roc": self.pct_auc_roc_improvement,
                "precision": self.pct_precision_improvement,
            },
            "feedback_stats": {
                "n_events": self.n_feedback_events,
                "n_confirms": self.n_confirms,
                "n_dismissals": self.n_dismissals,
                "accuracy": self.feedback_accuracy,
                "n_missed_event_feedback": self.n_missed_event_feedback,
            },
            "counterfactual": {
                "mean_delta": self.mean_counterfactual_delta,
                "pct_improved": self.pct_improved_by_feedback,
                "n_predictions_flipped": self.n_predictions_flipped,
            },
        }

    def summary_text(self) -> str:
        lines = [
            "=" * 70,
            "  FEEDBACK IMPACT COMPARISON",
            "=" * 70,
            "",
            f"  {'Metric':<28} {'Test (baseline)':>15} {'Eval (feedback)':>15} {'Delta':>10}",
            f"  {'-'*28} {'-'*15} {'-'*15} {'-'*10}",
        ]

        def _fmt_ci(ms, attr):
            ci = getattr(ms, attr, None)
            if ci and len(ci) == 2:
                return f" [{ci[0]:.3f}, {ci[1]:.3f}]"
            return ""

        for name, attr in [
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1 Score", "f1"),
            ("Balanced Accuracy", "balanced_accuracy"),
            ("AUC-ROC", "auc_roc"),
            ("AUC-PR", "auc_pr"),
            ("Score Separation", "score_separation"),
        ]:
            tv = getattr(self.test_metrics, attr)
            ev = getattr(self.eval_metrics, attr)
            d = ev - tv
            ci_str = _fmt_ci(self.eval_metrics, f"{attr}_ci")
            lines.append(
                f"  {name:<28} {tv:>15.4f} {ev:>15.4f}{ci_str} {d:>+10.4f}"
            )

        lines.extend([
            "",
            f"  Feedback events:   {self.n_feedback_events}"
            f"  ({self.n_confirms} confirms, {self.n_dismissals} dismissals,"
            f" {self.n_missed_event_feedback} from missed-event path)",
            f"  Feedback accuracy: {self.feedback_accuracy:.2%}",
            "",
            # B1: the counterfactual (same eval events, feedback on vs off) is the
            # CAUSAL claim. The Test->Eval delta above is cross-operation and is
            # context only — it confounds "feedback helped" with "eval op differs".
            "  >>> FEEDBACK CAUSAL EFFECT (primary; same-operation counterfactual):",
            f"      Predictions flipped by feedback:   {self.n_predictions_flipped}",
            f"      Mean score delta (on vs off):      {self.mean_counterfactual_delta:+.4f}",
            f"      Samples improved by feedback:      {self.pct_improved_by_feedback:.1%}",
            "      (Test->Eval columns above are cross-operation context, NOT the causal claim.)",
            "",
            "=" * 70,
        ])
        return "\n".join(lines)


# =====================================================================
# Computation
# =====================================================================


def compute_metrics(phase_result, n_bootstrap: int = 1000) -> MetricSet:
    """Compute MetricSet from a PhaseResult.

    Uses per-sample significance_score and predicted_positive vs label.
    Includes bootstrap 95% confidence intervals on key metrics.
    """
    from .evaluator import PhaseResult

    ms = MetricSet(
        phase=phase_result.phase,
        operation=phase_result.operation,
        n_samples=phase_result.n_samples,
        threshold=phase_result.threshold,
    )

    y_true = []
    y_scores = []
    y_pred = []

    for sr in phase_result.sample_results:
        is_pos = 1 if sr.label == "pre_stoppage" else 0
        y_true.append(is_pos)
        y_scores.append(sr.significance_score)
        y_pred.append(1 if sr.predicted_positive else 0)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.array(y_pred)

    # Confusion matrix
    ms.tp = int(((y_pred == 1) & (y_true == 1)).sum())
    ms.fp = int(((y_pred == 1) & (y_true == 0)).sum())
    ms.tn = int(((y_pred == 0) & (y_true == 0)).sum())
    ms.fn = int(((y_pred == 0) & (y_true == 1)).sum())

    # Derived metrics
    ms.precision = ms.tp / (ms.tp + ms.fp) if (ms.tp + ms.fp) > 0 else 0.0
    ms.recall = ms.tp / (ms.tp + ms.fn) if (ms.tp + ms.fn) > 0 else 0.0
    ms.f1 = (
        2 * ms.precision * ms.recall / (ms.precision + ms.recall)
        if (ms.precision + ms.recall) > 0
        else 0.0
    )
    ms.specificity = ms.tn / (ms.tn + ms.fp) if (ms.tn + ms.fp) > 0 else 0.0
    ms.accuracy = (ms.tp + ms.tn) / max(len(y_true), 1)
    ms.balanced_accuracy = (ms.recall + ms.specificity) / 2.0

    # AUC scores
    ms.auc_roc = _compute_auc_roc(y_true, y_scores)
    ms.auc_pr = _compute_auc_pr(y_true, y_scores)
    ms.avg_precision = ms.auc_pr  # alias

    # Score distributions
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    ms.mean_score_positive = float(np.mean(y_scores[pos_mask])) if pos_mask.any() else 0.0
    ms.mean_score_negative = float(np.mean(y_scores[neg_mask])) if neg_mask.any() else 0.0
    ms.score_separation = ms.mean_score_positive - ms.mean_score_negative

    # Bootstrap 95% confidence intervals
    if len(y_true) >= 10 and n_bootstrap > 0:
        ms.f1_ci = _bootstrap_ci(y_true, y_scores, y_pred, "f1", phase_result.threshold, n_bootstrap)
        ms.auc_roc_ci = _bootstrap_ci(y_true, y_scores, y_pred, "auc_roc", phase_result.threshold, n_bootstrap)
        ms.precision_ci = _bootstrap_ci(y_true, y_scores, y_pred, "precision", phase_result.threshold, n_bootstrap)
        ms.recall_ci = _bootstrap_ci(y_true, y_scores, y_pred, "recall", phase_result.threshold, n_bootstrap)

    return ms


def compare_phases(test_result, eval_result) -> ComparisonReport:
    """Build a ComparisonReport from two PhaseResults."""
    test_metrics = compute_metrics(test_result)
    eval_metrics = compute_metrics(eval_result)

    report = ComparisonReport(
        test_metrics=test_metrics,
        eval_metrics=eval_metrics,
    )

    # Absolute deltas
    report.delta_precision = eval_metrics.precision - test_metrics.precision
    report.delta_recall = eval_metrics.recall - test_metrics.recall
    report.delta_f1 = eval_metrics.f1 - test_metrics.f1
    report.delta_auc_roc = eval_metrics.auc_roc - test_metrics.auc_roc
    report.delta_auc_pr = eval_metrics.auc_pr - test_metrics.auc_pr
    report.delta_balanced_accuracy = eval_metrics.balanced_accuracy - test_metrics.balanced_accuracy
    report.delta_score_separation = eval_metrics.score_separation - test_metrics.score_separation

    # Relative improvements
    report.pct_f1_improvement = _pct_change(test_metrics.f1, eval_metrics.f1)
    report.pct_auc_roc_improvement = _pct_change(test_metrics.auc_roc, eval_metrics.auc_roc)
    report.pct_precision_improvement = _pct_change(test_metrics.precision, eval_metrics.precision)

    # Feedback stats
    feedback_samples = [s for s in eval_result.sample_results if s.feedback_given]
    report.n_feedback_events = len(feedback_samples)
    report.n_confirms = sum(1 for s in feedback_samples if s.feedback_action == "CONFIRM")
    report.n_dismissals = sum(1 for s in feedback_samples if s.feedback_action == "DISMISS")
    report.n_missed_event_feedback = sum(
        1 for s in feedback_samples if getattr(s, "feedback_source", "") == "missed_event"
    )

    # Feedback accuracy: did the feedback match the label?
    correct = 0
    for s in feedback_samples:
        label_positive = s.label == "pre_stoppage"
        if s.feedback_action == "CONFIRM" and label_positive:
            correct += 1
        elif s.feedback_action == "DISMISS" and not label_positive:
            correct += 1
    report.feedback_accuracy = correct / max(len(feedback_samples), 1)

    # Counterfactual analysis — now covers ALL eval samples
    cf_deltas = []
    improved = 0
    for s in eval_result.sample_results:
        if s.counterfactual_score is not None:
            is_pos = s.label == "pre_stoppage"
            delta = s.significance_score - s.counterfactual_score
            cf_deltas.append(delta)
            # "improved" means: positive sample scored higher, or negative scored lower
            if (is_pos and delta > 0) or (not is_pos and delta < 0):
                improved += 1
    report.mean_counterfactual_delta = float(np.mean(cf_deltas)) if cf_deltas else 0.0
    report.pct_improved_by_feedback = improved / max(len(cf_deltas), 1)
    report.n_predictions_flipped = getattr(eval_result, "n_predictions_flipped", 0)

    return report


# =====================================================================
# AUC helpers (no sklearn dependency)
# =====================================================================


def _compute_auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute AUC-ROC using the trapezoidal rule."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_scores))
    except ImportError:
        pass

    # Manual fallback
    if len(np.unique(y_true)) < 2:
        return 0.5

    desc_idx = np.argsort(-y_scores)
    y_sorted = y_true[desc_idx]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_prev = 0.0
    fpr_prev = 0.0
    auc = 0.0
    tp = 0
    fp = 0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev = tpr
        fpr_prev = fpr

    return float(auc)


def _compute_auc_pr(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute AUC-PR using the trapezoidal rule."""
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_scores))
    except ImportError:
        pass

    # Manual fallback
    if len(np.unique(y_true)) < 2:
        return 0.0

    desc_idx = np.argsort(-y_scores)
    y_sorted = y_true[desc_idx]

    n_pos = y_true.sum()
    if n_pos == 0:
        return 0.0

    tp = 0
    auc = 0.0
    prev_recall = 0.0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        precision = tp / (i + 1)
        recall = tp / n_pos
        if recall > prev_recall:
            auc += precision * (recall - prev_recall)
            prev_recall = recall

    return float(auc)


def _pct_change(baseline: float, new: float) -> float:
    if baseline == 0:
        return 0.0
    return (new - baseline) / abs(baseline) * 100.0


# =====================================================================
# Bootstrap confidence intervals
# =====================================================================


def _bootstrap_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    threshold: float,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> List[float]:
    """Compute bootstrap 95% CI for a metric.

    Resamples (y_true, y_scores) with replacement n_bootstrap times,
    computes the metric each time, and returns [lower, upper].
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_values: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        bt = y_true[idx]
        bs = y_scores[idx]
        bp = (bs >= threshold).astype(int)

        # Skip degenerate bootstrap samples (only one class)
        if len(np.unique(bt)) < 2:
            continue

        if metric == "f1":
            tp = int(((bp == 1) & (bt == 1)).sum())
            fp = int(((bp == 1) & (bt == 0)).sum())
            fn = int(((bp == 0) & (bt == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        elif metric == "auc_roc":
            val = _compute_auc_roc(bt, bs)
        elif metric == "precision":
            tp = int(((bp == 1) & (bt == 1)).sum())
            fp = int(((bp == 1) & (bt == 0)).sum())
            val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif metric == "recall":
            tp = int(((bp == 1) & (bt == 1)).sum())
            fn = int(((bp == 0) & (bt == 1)).sum())
            val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            continue

        boot_values.append(val)

    if not boot_values:
        return [0.0, 0.0]

    lower = float(np.percentile(boot_values, 100 * alpha / 2))
    upper = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))
    return [lower, upper]


# =====================================================================
# Rotation (paired) statistics
# =====================================================================


def compute_rotation_statistics(rotation_results: List[Dict]) -> Dict[str, Any]:
    """Compute paired statistics across rotation results.

    For each metric delta across rotations, computes:
    - mean, std, min, max
    - Sign test p-value (are improvements consistent in direction?)
    - Whether result is significant at alpha=0.05

    Parameters
    ----------
    rotation_results : list of dict
        Each dict has "comparison" key with "deltas" sub-dict.

    Returns
    -------
    Dict with per-metric statistics and overall conclusion.
    """
    metrics = ["f1", "auc_roc", "auc_pr", "precision", "recall", "balanced_accuracy"]
    stats: Dict[str, Any] = {}

    for metric in metrics:
        deltas = []
        for r in rotation_results:
            d = r.get("comparison", {}).get("deltas", {}).get(metric, 0.0)
            deltas.append(d)

        if not deltas:
            continue

        arr = np.array(deltas)
        n_pos = int((arr > 0).sum())
        n_neg = int((arr < 0).sum())
        n_total = n_pos + n_neg  # exclude zeros

        # Sign test: under H0, n_pos ~ Binomial(n_total, 0.5)
        # p-value = P(X >= n_pos) for one-sided (improvement)
        sign_p = _binomial_pvalue(n_pos, n_total)

        stats[metric] = {
            "mean_delta": float(np.mean(arr)),
            "std_delta": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min_delta": float(np.min(arr)),
            "max_delta": float(np.max(arr)),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "sign_test_p": sign_p,
            "significant_at_005": sign_p < 0.05,
            "deltas": [float(d) for d in deltas],
        }

    # Overall conclusion
    n_sig = sum(1 for s in stats.values() if s.get("significant_at_005", False))
    stats["_summary"] = {
        "n_rotations": len(rotation_results),
        "n_metrics_significant": n_sig,
        "conclusion": (
            "Feedback impact is statistically significant"
            if n_sig >= 3
            else "Feedback impact is NOT statistically significant (insufficient rotations or inconsistent direction)"
        ),
    }

    return stats


def _binomial_pvalue(k: int, n: int) -> float:
    """One-sided p-value for sign test: P(X >= k) under Binomial(n, 0.5)."""
    if n == 0:
        return 1.0

    try:
        from scipy.stats import binom
        return float(1 - binom.cdf(k - 1, n, 0.5))
    except ImportError:
        pass

    # Manual fallback: exact binomial
    from math import comb
    p = 0.0
    for i in range(k, n + 1):
        p += comb(n, i) * (0.5 ** n)
    return p

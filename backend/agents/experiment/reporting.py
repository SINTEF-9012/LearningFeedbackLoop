"""Reporting: save JSON results and generate comparison plots."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import PATTERN_KEYS, ExperimentConfig
from .evaluator import PhaseResult
from .metrics import ComparisonReport, MetricSet

logger = logging.getLogger(__name__)


# =====================================================================
# JSON persistence
# =====================================================================


def save_results(
    config: ExperimentConfig,
    comparison: ComparisonReport,
    test_result: PhaseResult,
    eval_result: PhaseResult,
    train_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save full experiment results to JSON."""
    config.ensure_dirs()
    out = config.run_dir / "experiment_results.json"

    data = {
        "config": {
            "train_ops": config.train_ops,
            "test_op": config.test_op,
            "eval_op": config.eval_op,
            "eval_variant": config.eval_variant,
            "noise_rate": config.noise_rate,
            "feedback_every_n": config.feedback_every_n,
            "prediction_gap_s": config.prediction_gap_s,
            "features_csv": str(config.features_csv),
            "min_discrimination_ratio": config.min_discrimination_ratio,
            "negative_sampling_enabled": config.negative_sampling_enabled,
            "negative_sampling_rate": config.negative_sampling_rate,
            "store_threshold": config.store_threshold,
            "alert_threshold": config.alert_threshold,
            "critical_threshold": config.critical_threshold,
            "weight_protective_pattern": config.weight_protective_pattern,
        },
        "comparison": comparison.to_dict(),
        "test_phase": test_result.to_dict(),
        "eval_phase": eval_result.to_dict(),
    }
    if train_meta:
        data["train_phase"] = train_meta

    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("Results saved to %s", out)
    return out


# =====================================================================
# Plot generation
# =====================================================================


def generate_plots(
    config: ExperimentConfig,
    comparison: ComparisonReport,
    test_result: PhaseResult,
    eval_result: PhaseResult,
) -> List[Path]:
    """Generate all comparison plots. Returns list of saved file paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
        return []

    config.ensure_dirs()
    plots_dir = config.run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    generated: List[Path] = []

    generated.append(_plot_roc_overlay(plots_dir, comparison, test_result, eval_result))
    generated.append(_plot_score_distributions(plots_dir, test_result, eval_result))
    generated.append(_plot_prior_evolution(plots_dir, eval_result))
    generated.append(_plot_confusion_matrices(plots_dir, comparison))
    generated.append(_plot_metric_comparison_bar(plots_dir, comparison))

    if any(s.counterfactual_score is not None for s in eval_result.sample_results):
        generated.append(_plot_counterfactual(plots_dir, eval_result))

    plt.close("all")
    logger.info("Generated %d plots in %s", len(generated), plots_dir)
    return [p for p in generated if p is not None]


# ----- Individual plot functions ------------------------------------


def _plot_roc_overlay(
    plots_dir: Path,
    comparison: ComparisonReport,
    test_result: PhaseResult,
    eval_result: PhaseResult,
) -> Path:
    """ROC curves for both phases overlaid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))

    for result, label, color in [
        (test_result, "Test (no feedback)", "#2196F3"),
        (eval_result, "Eval (with feedback)", "#4CAF50"),
    ]:
        y_true, y_scores = _extract_labels_scores(result)
        fpr, tpr = _roc_curve(y_true, y_scores)
        auc = comparison.test_metrics.auc_roc if result.phase == "test" else comparison.eval_metrics.auc_roc
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison: Baseline vs Feedback")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    path = plots_dir / "01_roc_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_score_distributions(
    plots_dir: Path,
    test_result: PhaseResult,
    eval_result: PhaseResult,
) -> Path:
    """Score distributions (histograms) for both phases."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, result, title in [
        (axes[0], test_result, "Test (no feedback)"),
        (axes[1], eval_result, "Eval (with feedback)"),
    ]:
        bins = np.linspace(0, 1, 40)
        if result.scores_positive:
            ax.hist(result.scores_positive, bins=bins, alpha=0.6,
                    color="#F44336", label=f"pre-stoppage (n={len(result.scores_positive)})")
        if result.scores_negative:
            ax.hist(result.scores_negative, bins=bins, alpha=0.6,
                    color="#2196F3", label=f"normal (n={len(result.scores_negative)})")
        ax.axvline(result.threshold, color="black", linestyle="--",
                   linewidth=1.5, label=f"threshold={result.threshold:.2f}")
        ax.set_xlabel("Significance Score")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Score Distributions by Label", fontweight="bold")
    path = plots_dir / "02_score_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_prior_evolution(plots_dir: Path, eval_result: PhaseResult) -> Path:
    """Prior evolution during the feedback phase."""
    import matplotlib.pyplot as plt

    history = eval_result.prior_history
    if len(history) < 2:
        logger.info("Skipping prior evolution plot (insufficient data)")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Insufficient feedback events for prior evolution",
                ha="center", va="center", transform=ax.transAxes)
        path = plots_dir / "03_prior_evolution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(history)))

    colors = {
        "SPINDLE_POWER_SURGE": "#FF9800",
        "VIBRATION_REGIME_SHIFT": "#9C27B0",
        "FEED_OVERRIDE_DROP": "#2196F3",
        "SENSOR_DECORRELATION": "#4CAF50",
        "SPINDLE_LOAD_RAMP": "#795548",
        "FEED_STALL": "#607D8B",
        "POWER_ASYMMETRY": "#E91E63",
        "ENERGY_ACCUMULATION": "#00BCD4",
        "VARIANCE_EXPLOSION": "#F44336",
        "TREND_REVERSAL": "#CDDC39",
        "AUTOCORRELATION_BREAK": "#9E9E9E",
    }

    short_labels = {
        "SPINDLE_POWER_SURGE": "Power Surge",
        "VIBRATION_REGIME_SHIFT": "Vibration Shift",
        "FEED_OVERRIDE_DROP": "Feed Override Drop",
        "SENSOR_DECORRELATION": "Decorrelation",
        "SPINDLE_LOAD_RAMP": "Spindle Load Ramp",
        "FEED_STALL": "Feed Stall",
        "POWER_ASYMMETRY": "Power Asymmetry",
        "ENERGY_ACCUMULATION": "Energy Ramp",
        "VARIANCE_EXPLOSION": "Variance Explosion",
        "TREND_REVERSAL": "Trend Reversal",
        "AUTOCORRELATION_BREAK": "Autocorr Break",
    }

    for pk in PATTERN_KEYS:
        values = [snap.get(pk, 0.5) for snap in history]
        c = colors.get(pk, "#888888")
        lbl = short_labels.get(pk, pk)
        ax.plot(x, values, color=c, linewidth=1.8, label=lbl,
                marker="o", markersize=3, alpha=0.8)

    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.4, label="Neutral (0.5)")
    ax.set_xlabel("Feedback Events")
    ax.set_ylabel("Pattern Prior")
    ax.set_title("Prior Evolution During Feedback Phase")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    path = plots_dir / "03_prior_evolution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_confusion_matrices(plots_dir: Path, comparison: ComparisonReport) -> Path:
    """Side-by-side confusion matrices."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, ms, title in [
        (axes[0], comparison.test_metrics, "Test (baseline)"),
        (axes[1], comparison.eval_metrics, "Eval (feedback)"),
    ]:
        cm = np.array([[ms.tn, ms.fp], [ms.fn, ms.tp]])
        im = ax.imshow(cm, cmap="Blues", aspect="auto")

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=16, fontweight="bold", color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Normal", "Pred Stoppage"])
        ax.set_yticklabels(["True Normal", "True Stoppage"])
        ax.set_title(f"{title}\nF1={ms.f1:.3f}  Prec={ms.precision:.3f}  Rec={ms.recall:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Confusion Matrices", fontweight="bold")
    path = plots_dir / "04_confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_metric_comparison_bar(plots_dir: Path, comparison: ComparisonReport) -> Path:
    """Bar chart comparing key metrics between phases, with 95% CIs."""
    import matplotlib.pyplot as plt

    metrics_names = ["Precision", "Recall", "F1", "AUC-ROC", "AUC-PR", "Bal. Acc."]
    test_vals = [
        comparison.test_metrics.precision,
        comparison.test_metrics.recall,
        comparison.test_metrics.f1,
        comparison.test_metrics.auc_roc,
        comparison.test_metrics.auc_pr,
        comparison.test_metrics.balanced_accuracy,
    ]
    eval_vals = [
        comparison.eval_metrics.precision,
        comparison.eval_metrics.recall,
        comparison.eval_metrics.f1,
        comparison.eval_metrics.auc_roc,
        comparison.eval_metrics.auc_pr,
        comparison.eval_metrics.balanced_accuracy,
    ]

    # Compute error bars from CIs
    def _ci_to_err(ci, val):
        if ci and len(ci) == 2:
            return [max(0.0, val - ci[0]), max(0.0, ci[1] - val)]
        return [0.0, 0.0]

    test_ci_attrs = ["precision_ci", "recall_ci", "f1_ci", "auc_roc_ci", None, None]
    eval_ci_attrs = ["precision_ci", "recall_ci", "f1_ci", "auc_roc_ci", None, None]

    test_err_lo = []
    test_err_hi = []
    eval_err_lo = []
    eval_err_hi = []
    for i, attr in enumerate(test_ci_attrs):
        if attr:
            t_ci = getattr(comparison.test_metrics, attr, None)
            e_ci = getattr(comparison.eval_metrics, attr, None)
            t_e = _ci_to_err(t_ci, test_vals[i])
            e_e = _ci_to_err(e_ci, eval_vals[i])
        else:
            t_e = [0, 0]
            e_e = [0, 0]
        test_err_lo.append(t_e[0])
        test_err_hi.append(t_e[1])
        eval_err_lo.append(e_e[0])
        eval_err_hi.append(e_e[1])

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(
        x - width / 2, test_vals, width, label="Test (baseline)",
        color="#2196F3", alpha=0.8,
        yerr=[test_err_lo, test_err_hi], capsize=3, error_kw={"linewidth": 1},
    )
    bars2 = ax.bar(
        x + width / 2, eval_vals, width, label="Eval (feedback)",
        color="#4CAF50", alpha=0.8,
        yerr=[eval_err_lo, eval_err_hi], capsize=3, error_kw={"linewidth": 1},
    )

    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison: Baseline vs Feedback (with 95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8,
                )

    path = plots_dir / "05_metric_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_counterfactual(plots_dir: Path, eval_result: PhaseResult) -> Path:
    """Counterfactual analysis: actual vs initial-prior scores for ALL eval samples."""
    import matplotlib.pyplot as plt

    actual_scores = []
    cf_scores = []
    labels = []
    flipped = []

    for s in eval_result.sample_results:
        if s.counterfactual_score is not None:
            actual_scores.append(s.significance_score)
            cf_scores.append(s.counterfactual_score)
            labels.append(s.label)
            flipped.append(getattr(s, "prediction_flipped", False))

    if not actual_scores:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.text(0.5, 0.5, "No counterfactual data", ha="center", va="center",
                transform=ax.transAxes)
        path = plots_dir / "06_counterfactual.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    fig, ax = plt.subplots(figsize=(7, 7))

    _display_labels = {"pre_stoppage": "pre-stoppage", "normal": "normal"}
    for lbl, color, marker in [("pre_stoppage", "#F44336", "^"), ("normal", "#2196F3", "o")]:
        mask = [l == lbl for l in labels]
        xs = [cf_scores[i] for i in range(len(mask)) if mask[i]]
        ys = [actual_scores[i] for i in range(len(mask)) if mask[i]]
        ax.scatter(xs, ys, c=color, marker=marker, alpha=0.5,
                   s=20, label=_display_labels.get(lbl, lbl))

    # Highlight prediction flips with larger markers
    flip_x = [cf_scores[i] for i in range(len(flipped)) if flipped[i]]
    flip_y = [actual_scores[i] for i in range(len(flipped)) if flipped[i]]
    if flip_x:
        ax.scatter(flip_x, flip_y, facecolors="none", edgecolors="black",
                   s=100, linewidths=1.5, label=f"Prediction flipped (n={len(flip_x)})")

    # Diagonal (no change) + threshold lines
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="No change")
    ax.axhline(eval_result.threshold, color="grey", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.axvline(eval_result.threshold, color="grey", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Score WITHOUT feedback (counterfactual)")
    ax.set_ylabel("Score WITH feedback (actual)")
    ax.set_title("Counterfactual Analysis (all eval samples)")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    path = plots_dir / "06_counterfactual.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# =====================================================================
# Helpers
# =====================================================================


def _extract_labels_scores(result: PhaseResult):
    y_true = []
    y_scores = []
    for sr in result.sample_results:
        y_true.append(1 if sr.label == "pre_stoppage" else 0)
        y_scores.append(sr.significance_score)
    return np.array(y_true), np.array(y_scores)


def _roc_curve(y_true: np.ndarray, y_scores: np.ndarray):
    """Return (fpr, tpr) arrays for plotting."""
    try:
        from sklearn.metrics import roc_curve as sk_roc_curve
        fpr, tpr, _ = sk_roc_curve(y_true, y_scores)
        return fpr, tpr
    except ImportError:
        pass

    # Manual fallback
    thresholds = np.unique(np.concatenate([[0.0], y_scores, [1.0]]))
    thresholds = np.sort(thresholds)[::-1]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1])

    fpr_list = []
    tpr_list = []
    for t in thresholds:
        pred = (y_scores >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    return np.array(fpr_list), np.array(tpr_list)

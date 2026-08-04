"""Phase 1 — Train classical models on normal-only data."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import PATTERN_KEYS, LEAKY_COLUMNS, METADATA_COLUMNS, ExperimentConfig
from .pattern_registry import get_registry as _get_pattern_registry

logger = logging.getLogger(__name__)

_PROTECTIVE_MIN_FIRE_RATE = 0.02


def _build_initial_priors_payload(calibrated_pattern_thresholds: Dict[str, Any]) -> Dict[str, Any]:
    """Build the baseline prior payload for all tracked pattern keys."""
    return {
        "pattern_priors": {pk: 0.5 for pk in PATTERN_KEYS},
        "feedback_counts": {pk: {"confirm": 0, "dismiss": 0} for pk in PATTERN_KEYS},
        "calibrated_pattern_thresholds": calibrated_pattern_thresholds,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _safe_fire_rate(n_fire: int, n_total: int) -> float:
    return n_fire / max(1, n_total)


def _discrimination_score(
    rate_event: float,
    rate_normal: float,
    n_event: int,
    n_normal: int,
) -> float:
    eps = 1.0 / max(1, n_event + n_normal)
    return float(math.log((rate_event + eps) / (rate_normal + eps)))


def _classify_pattern_polarity(
    ratio: float,
    rate_normal: float,
    rate_event: float,
    min_ratio: float,
    min_fire_rate: float = _PROTECTIVE_MIN_FIRE_RATE,
) -> str:
    if rate_normal <= 0.0 and rate_event <= 0.0:
        return "uninformative"
    if min_ratio <= 0.0:
        if rate_event > rate_normal:
            return "fault_supporting"
        if rate_normal > rate_event and rate_normal >= min_fire_rate:
            return "protective"
        return "uninformative"

    protective_cutoff = 1.0 / min_ratio
    if rate_event > rate_normal and ratio >= min_ratio:
        return "fault_supporting"
    if rate_normal > rate_event and rate_normal >= min_fire_rate and ratio <= protective_cutoff:
        return "protective"
    return "uninformative"


# =====================================================================
# Pattern threshold columns and their comparison direction
# =====================================================================

# Each entry: (config_attr, csv_column, direction)
# direction: "above" means pattern fires when value > threshold
#            "below" means pattern fires when value < threshold
#            "band"  means pattern fires when 0 < |value| < threshold
PATTERN_COLUMNS = {
    # === BUILT-IN (original 4) ===
    "SPINDLE_POWER_SURGE": [
        ("pattern_power_spindle_delta_max", "power_spindle_delta_max", "above"),
        ("pattern_power_y_delta_max", "power_y_delta_max", "above"),
    ],
    "VIBRATION_REGIME_SHIFT": [
        ("pattern_vib_severity_x_delta_max", "vib_severity_x_delta_max", "above"),
        ("pattern_chatter_freq_x_slope_abs", "chatter_freq_x_slope", "above_abs"),
    ],
    "FEED_OVERRIDE_DROP": [
        ("pattern_feed_override_delta_mean", "feed_override_delta_mean", "below"),
        ("pattern_feed_override_min", "feed_override_min", "band_low"),
    ],
    "SENSOR_DECORRELATION": [
        ("pattern_corr_spindle_power_vib_x_low", "corr_spindle_power_vib_x", "band"),
    ],
    # === DOMAIN-EXPERT (new — calibrated from data) ===
    "SPINDLE_LOAD_RAMP": [
        ("pattern_power_spindle_slope", "power_spindle_slope", "above"),
    ],
    "FEED_STALL": [
        ("pattern_feed_actual_range_ratio", "feed_actual_range", "above"),
    ],
    "POWER_ASYMMETRY": [
        ("pattern_power_xy_asymmetry", "power_x_mean", "above"),  # checked via custom detector
    ],
    "ENERGY_ACCUMULATION": [
        ("pattern_energy_total_slope", "energy_total_slope", "above"),
    ],
    # === TIME-SERIES DERIVED ===
    "VARIANCE_EXPLOSION": [
        ("pattern_vib_severity_x_std", "vib_severity_x_std", "above"),
        ("pattern_power_spindle_std", "power_spindle_std", "above"),
    ],
    "TREND_REVERSAL": [
        ("pattern_power_spindle_slope_tr", "power_spindle_slope", "above"),
    ],
    "AUTOCORRELATION_BREAK": [
        ("pattern_vib_severity_x_iqr", "vib_severity_x_iqr", "above"),
    ],
}


@dataclass
class TrainResult:
    """Output of Phase 1 training."""

    model_path: str = ""
    priors_path: str = ""
    meta_path: str = ""

    n_train_samples: int = 0
    n_normal_used: int = 0
    n_pre_stoppage_held: int = 0
    feature_count: int = 0
    train_ops: List[str] = field(default_factory=list)

    # Calibration on the full train set (both labels)
    calibration: Dict[str, Any] = field(default_factory=dict)

    # Score distributions on training data
    normal_score_stats: Dict[str, float] = field(default_factory=dict)
    pre_stoppage_score_stats: Dict[str, float] = field(default_factory=dict)

    # Data-driven pattern thresholds
    calibrated_pattern_thresholds: Dict[str, Any] = field(default_factory=dict)

    # Supervised model
    supervised_model_path: str = ""
    supervised_feature_cols: List[str] = field(default_factory=list)
    supervised_metrics: Dict[str, Any] = field(default_factory=dict)

    # Tool-level priors
    tool_priors_path: str = ""
    tool_priors: Dict[str, float] = field(default_factory=dict)

    train_duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _score_distribution(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {}
    arr = np.array(scores)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
    }


def train_phase(
    train_df: pd.DataFrame,
    config: ExperimentConfig,
) -> TrainResult:
    """Phase 1: train SeedModel on normal-only rows, calibrate, save.

    Steps:
    1. Filter to label==normal for one-class training.
    2. Extract 28-feature vectors via features_from_dict.
    3. Train SeedModel (IsolationForest + LOF ensemble).
    4. Score ALL training rows (normal + pre_stoppage) for calibration.
    5. Pick a decision threshold using Youden's J statistic.
    6. Save model, neutral priors, and metadata.
    """
    from backend.agents.processing.classical_models import (
        FEATURE_NAMES,
        SeedModel,
        SeedModelConfig,
        features_from_dict,
        batch_features_from_df,
    )
    from backend.agents.processing.breakage_detector import (
        BreakageFeatureExtractor,
        _COL_MAP,
    )

    t0 = time.time()
    result = TrainResult(train_ops=list(config.train_ops))

    config.ensure_dirs()

    # --- 1. Prepare feature matrices ----------------------------------
    normal_df = train_df[train_df["label"] == "normal"]
    pre_stoppage_df = train_df[train_df["label"] == "pre_stoppage"]

    logger.info(
        "Phase 1 training: %d normal, %d pre_stoppage rows from %s",
        len(normal_df),
        len(pre_stoppage_df),
        config.train_ops,
    )

    def _rows_to_matrix(df: pd.DataFrame) -> np.ndarray:
        """Vectorized feature extraction — 10-100× faster than iterrows."""
        return batch_features_from_df(df, col_map=_COL_MAP)

    X_normal = _rows_to_matrix(normal_df)
    result.n_normal_used = len(X_normal)
    result.feature_count = X_normal.shape[1] if X_normal.ndim == 2 else 0

    # --- 2. Train SeedModel -------------------------------------------
    logger.info("Training SeedModel on %d normal samples (%d features)...",
                X_normal.shape[0], X_normal.shape[1])
    model = SeedModel(config=SeedModelConfig(random_state=config.random_seed))
    train_info = model.train(X_normal)
    logger.info("Training complete: %s", train_info)

    # --- 3. Score all training rows for calibration -------------------
    normal_scores: List[float] = list(model.score_batch(X_normal))

    pre_stoppage_scores: List[float] = []
    if len(pre_stoppage_df) > 0:
        X_pre_stoppage = _rows_to_matrix(pre_stoppage_df)
        result.n_pre_stoppage_held = len(X_pre_stoppage)
        pre_stoppage_scores = list(model.score_batch(X_pre_stoppage))

    result.normal_score_stats = _score_distribution(normal_scores)
    result.pre_stoppage_score_stats = _score_distribution(pre_stoppage_scores)

    logger.info(
        "Score distributions — normal: mean=%.3f std=%.3f, "
        "pre_stoppage: mean=%.3f std=%.3f",
        result.normal_score_stats.get("mean", 0),
        result.normal_score_stats.get("std", 0),
        result.pre_stoppage_score_stats.get("mean", 0),
        result.pre_stoppage_score_stats.get("std", 0),
    )

    # --- 3b. Calibrate pattern detection thresholds from data ----------
    if config.calibrate_patterns_from_data:
        calibrated = calibrate_pattern_thresholds(
            train_df, config,
            percentile=config.pattern_calibration_normal_percentile,
        )
        result.calibrated_pattern_thresholds = calibrated
        # Apply the calibrated thresholds to config for use in eval
        _apply_calibrated_thresholds(config, calibrated)
        logger.info("Pattern thresholds calibrated from training data")
        for pat, info in calibrated.items():
            logger.info("  %s: %s", pat, {k: round(v, 3) if isinstance(v, float) else v for k, v in info.items()})

    # --- 4. Calibrate threshold (leave-one-op-out within training) ---
    calibration = _calibrate_threshold_loocv(
        train_df, model, config,
        _rows_to_matrix, BreakageFeatureExtractor, features_from_dict,
    )
    result.calibration = calibration
    logger.info(
        "Calibrated threshold: %.3f  (mean Youden's J=%.3f across folds)",
        calibration.get("threshold", 0),
        calibration.get("mean_youden_j", 0),
    )

    # --- 5. Save model ------------------------------------------------
    model.save(config.seed_model_path)
    result.model_path = str(config.seed_model_path)
    logger.info("Model saved to %s", config.seed_model_path)

    # --- 6. Save neutral priors (Beta(1,1) = 0.5) --------------------
    priors = _build_initial_priors_payload(result.calibrated_pattern_thresholds)
    with open(config.baseline_priors_path, "w") as f:
        json.dump(priors, f, indent=2)
    result.priors_path = str(config.baseline_priors_path)
    logger.info("Baseline priors saved to %s", config.baseline_priors_path)

    # --- 7. Save metadata ---------------------------------------------
    result.n_train_samples = len(train_df)
    result.train_duration_s = time.time() - t0

    meta = result.to_dict()
    meta["feature_names"] = list(FEATURE_NAMES)
    meta["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["window_spec"] = {
        "window_size_s": config.window_size_s,
        "sample_rate_hz": config.sample_rate_hz,
        "window_entries": config.window_entries,
    }
    with open(config.train_meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    result.meta_path = str(config.train_meta_path)

    # --- 8. Train supervised model (both classes, safe features) -------
    if config.use_supervised_model:
        sup_result = _train_supervised_model(train_df, config)
        result.supervised_model_path = sup_result["model_path"]
        result.supervised_feature_cols = sup_result["feature_cols"]
        result.supervised_metrics = sup_result["metrics"]
        logger.info(
            "Supervised %s trained: %d features, train AUC=%.3f",
            config.supervised_model_type,
            len(sup_result["feature_cols"]),
            sup_result["metrics"].get("train_auc_roc", 0),
        )

        # --- 8b. Auto-mine patterns from feature importances ----------
        _mine_patterns_from_importances(
            train_df, sup_result["metrics"].get("top_10_features", []),
            config,
        )

    # --- 9. Compute tool-level priors ---------------------------------
    if config.use_tool_priors:
        tool_priors = _compute_tool_priors(train_df, config)
        result.tool_priors = tool_priors
        result.tool_priors_path = str(config.tool_priors_path)
        logger.info(
            "Tool priors computed for %d tools (range: %.2f - %.2f)",
            len(tool_priors),
            min(tool_priors.values()) if tool_priors else 0,
            max(tool_priors.values()) if tool_priors else 0,
        )

    logger.info("Phase 1 complete in %.1fs", result.train_duration_s)
    return result


# =====================================================================
# Supervised model training
# =====================================================================


def _get_safe_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric feature columns excluding leaky and metadata columns."""
    exclude = set(LEAKY_COLUMNS) | set(METADATA_COLUMNS)
    numeric_cols = set(df.select_dtypes(include="number").columns)
    cols = [c for c in df.columns if c not in exclude and c in numeric_cols]
    # A3: belt-and-suspenders — fail loudly if anything forbidden slipped through.
    from .config import assert_features_safe
    assert_features_safe(cols, where="_get_safe_feature_columns")
    return cols


def _train_supervised_model(
    train_df: pd.DataFrame,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """Train a supervised classifier on labeled training data.

    Uses ALL safe features (excluding leaky event_* and metadata columns)
    from the CSV — ~485 features vs the 28-feature SeedModel.
    This gives the supervised model access to the full sensor feature space.

    The model is trained on both normal and pre_stoppage samples with
    class_weight='balanced' to handle imbalance.
    """
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    feature_cols = _get_safe_feature_columns(train_df)
    X = train_df[feature_cols].fillna(0).values
    y = (train_df["label"] == "pre_stoppage").astype(int).values

    logger.info(
        "Training supervised %s on %d samples (%d features, %d positive, %d negative)",
        config.supervised_model_type,
        len(X), len(feature_cols),
        int(y.sum()), int((1 - y).sum()),
    )

    if config.supervised_model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=config.supervised_n_estimators,
            max_depth=config.supervised_max_depth,
            random_state=config.random_seed,
        )
    else:  # random_forest (default)
        clf = RandomForestClassifier(
            n_estimators=config.supervised_n_estimators,
            max_depth=config.supervised_max_depth,
            random_state=config.random_seed,
            class_weight="balanced",
        )

    clf.fit(X, y)

    # Training metrics (for sanity check, not for reporting)
    y_proba = clf.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, y_proba)
    train_acc = clf.score(X, y)

    # Save model + feature column names + window spec
    model_data = {
        "model": clf,
        "feature_cols": feature_cols,
        "model_type": config.supervised_model_type,
        "window_spec": {
            "window_size_s": config.window_size_s,
            "sample_rate_hz": config.sample_rate_hz,
            "window_entries": config.window_entries,
        },
    }
    joblib.dump(model_data, config.supervised_model_path)
    logger.info("Supervised model saved to %s", config.supervised_model_path)

    # Top features
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    top_features = [(feature_cols[i], float(importances[i])) for i in top_idx]

    return {
        "model_path": str(config.supervised_model_path),
        "feature_cols": feature_cols,
        "metrics": {
            "train_auc_roc": float(train_auc),
            "train_accuracy": float(train_acc),
            "n_features": len(feature_cols),
            "n_samples": len(X),
            "top_10_features": top_features,
        },
    }


# =====================================================================
# Tool-level priors (lightweight knowledge graph)
# =====================================================================


def _compute_tool_priors(
    train_df: pd.DataFrame,
    config: ExperimentConfig,
) -> Dict[str, float]:
    """Compute tool-level stop-rate priors from labeled training data.

    For each tool_number, the prior is:
        prior = (n_pre_stoppage + 1) / (n_total + 2)   [Laplace smoothing]

    This represents the historical probability of a stop event given
    this specific tool — simulating what a knowledge graph would store
    as  Tool --has_property--> stop_rate.

    Higher priors mean the tool has a history of stops → system is
    more suspicious of alerts involving this tool.
    """
    tool_priors: Dict[str, float] = {}

    if "tool_number" not in train_df.columns:
        logger.info("No tool_number column — skipping tool priors")
        return tool_priors

    for tool_num, group in train_df.groupby("tool_number"):
        n_total = len(group)
        n_pre_stoppage = int((group["label"] == "pre_stoppage").sum())
        # Laplace smoothing: Beta(1+n_pos, 1+n_neg)
        prior = (n_pre_stoppage + 1) / (n_total + 2)
        tool_priors[str(int(tool_num))] = round(prior, 4)

    # Save tool priors as JSON
    tool_data = {
        "tool_priors": tool_priors,
        "feedback_counts": {t: {"confirm": 0, "dismiss": 0} for t in tool_priors},
        "source": "training_data",
        "n_tools": len(tool_priors),
    }
    with open(config.tool_priors_path, "w") as f:
        json.dump(tool_data, f, indent=2)
    logger.info("Tool priors saved to %s", config.tool_priors_path)

    return tool_priors


def _calibrate_threshold(
    normal_scores: List[float],
    pre_stoppage_scores: List[float],
    n_thresholds: int = 200,
) -> Dict[str, Any]:
    """Find optimal threshold via Youden's J = sensitivity + specificity - 1.

    Returns a dict with threshold, sensitivity, specificity, youden_j,
    and the full curve data for later plotting.
    """
    if not normal_scores or not pre_stoppage_scores:
        return {
            "threshold": 0.5,
            "youden_j": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "note": "insufficient data for calibration",
        }

    normal = np.array(normal_scores)
    pre_stoppage = np.array(pre_stoppage_scores)

    thresholds = np.linspace(0, 1, n_thresholds)
    best_j = -1.0
    best_thresh = 0.5
    best_sens = 0.0
    best_spec = 0.0

    curve_data: List[Dict[str, float]] = []

    for t in thresholds:
        tp = int((pre_stoppage >= t).sum())
        fn = int((pre_stoppage < t).sum())
        tn = int((normal < t).sum())
        fp = int((normal >= t).sum())

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0

        curve_data.append({
            "threshold": float(t),
            "sensitivity": sens,
            "specificity": spec,
            "youden_j": j,
        })

        if j > best_j:
            best_j = j
            best_thresh = float(t)
            best_sens = sens
            best_spec = spec

    return {
        "threshold": best_thresh,
        "youden_j": best_j,
        "sensitivity": best_sens,
        "specificity": best_spec,
        "n_normal": len(normal_scores),
        "n_pre_stoppage": len(pre_stoppage_scores),
        "curve": curve_data,
    }


def _calibrate_threshold_loocv(
    train_df: pd.DataFrame,
    final_model,
    config,
    rows_to_matrix_fn,
    BreakageFeatureExtractor,
    features_from_dict,
    n_thresholds: int = 200,
) -> Dict[str, Any]:
    """Calibrate threshold using leave-one-operation-out cross-validation.

    This avoids the optimistic bias of calibrating on the same data the
    model was trained on. For each training operation:
      1. Train a temporary model on the OTHER training operations' normal data.
      2. Score ALL rows from the held-out operation.
      3. Find the best Youden's-J threshold on those held-out scores.
    Average the per-fold thresholds.

    If only one training operation exists, falls back to the direct method
    with a documented warning.
    """
    from backend.agents.processing.classical_models import SeedModel, SeedModelConfig

    ops = sorted(train_df["operation_id"].unique())

    if len(ops) < 2:
        logger.warning(
            "Only 1 training operation (%s); falling back to direct threshold "
            "calibration (known optimistic bias)", ops,
        )
        X_all = rows_to_matrix_fn(train_df)
        all_scores = final_model.score_batch(X_all)
        all_labels = train_df["label"].values
        normal_scores = [float(s) for s, lbl in zip(all_scores, all_labels) if lbl == "normal"]
        pre_stoppage_scores = [float(s) for s, lbl in zip(all_scores, all_labels) if lbl != "normal"]
        result = _calibrate_threshold(normal_scores, pre_stoppage_scores, n_thresholds)
        result["method"] = "direct_single_op"
        return result

    fold_thresholds: List[float] = []
    fold_js: List[float] = []
    fold_details: List[Dict[str, Any]] = []

    for held_out_op in ops:
        other_ops = [o for o in ops if o != held_out_op]

        # Train temporary model on other ops' normal data
        other_normal = train_df[
            (train_df["operation_id"].isin(other_ops)) & (train_df["label"] == "normal")
        ]
        if len(other_normal) < 5:
            logger.warning("Skipping fold %s: too few normal samples in other ops", held_out_op)
            continue

        X_other_normal = rows_to_matrix_fn(other_normal)
        temp_model = SeedModel(config=SeedModelConfig(random_state=config.random_seed))
        temp_model.train(X_other_normal)

        # Score held-out operation (batch)
        held_out = train_df[train_df["operation_id"] == held_out_op]
        X_held = rows_to_matrix_fn(held_out)
        all_scores = temp_model.score_batch(X_held)
        held_labels = held_out["label"].values
        normal_scores = [float(s) for s, lbl in zip(all_scores, held_labels) if lbl == "normal"]
        pre_stoppage_scores = [float(s) for s, lbl in zip(all_scores, held_labels) if lbl != "normal"]

        if not normal_scores or not pre_stoppage_scores:
            logger.warning(
                "Skipping fold %s: missing labels (normal=%d, pre_stoppage=%d)",
                held_out_op, len(normal_scores), len(pre_stoppage_scores),
            )
            continue

        fold_result = _calibrate_threshold(normal_scores, pre_stoppage_scores, n_thresholds)
        fold_thresholds.append(fold_result["threshold"])
        fold_js.append(fold_result["youden_j"])
        fold_details.append({
            "held_out_op": held_out_op,
            "threshold": fold_result["threshold"],
            "youden_j": fold_result["youden_j"],
            "sensitivity": fold_result["sensitivity"],
            "specificity": fold_result["specificity"],
            "n_normal": fold_result["n_normal"],
            "n_pre_stoppage": fold_result["n_pre_stoppage"],
        })
        logger.info(
            "LOOCV fold %s: threshold=%.3f, J=%.3f (normal=%d, pre_stoppage=%d)",
            held_out_op, fold_result["threshold"], fold_result["youden_j"],
            len(normal_scores), len(pre_stoppage_scores),
        )

    if not fold_thresholds:
        logger.warning("All LOOCV folds failed; using fallback threshold 0.5")
        return {
            "threshold": 0.5,
            "mean_youden_j": 0.0,
            "method": "loocv_failed_fallback",
            "folds": fold_details,
        }

    avg_threshold = float(np.mean(fold_thresholds))
    mean_j = float(np.mean(fold_js))

    logger.info(
        "LOOCV calibration: avg threshold=%.3f (std=%.3f), mean J=%.3f across %d folds",
        avg_threshold, float(np.std(fold_thresholds)), mean_j, len(fold_thresholds),
    )

    return {
        "threshold": avg_threshold,
        "mean_youden_j": mean_j,
        "std_threshold": float(np.std(fold_thresholds)),
        "method": "loocv",
        "n_folds": len(fold_thresholds),
        "folds": fold_details,
    }


# =====================================================================
# Data-driven pattern threshold calibration
# =====================================================================


def calibrate_pattern_thresholds(
    train_df: pd.DataFrame,
    config: ExperimentConfig,
    percentile: float = 95.0,
) -> Dict[str, Any]:
    """Compute pattern detection thresholds from the NORMAL training samples.

    For each pattern, we take the percentile of the relevant column's
    distribution among normal-only samples. This ensures the pattern fires
    rarely on normal data (at most ~5% at p95) — making any firing on
    test/eval data a meaningful signal.

    For "below" direction columns (feed_override_delta_mean), we use the
    complementary percentile (e.g., p5 = 100 - p95).

    Returns a dict keyed by pattern name with the calibrated thresholds
    and diagnostics (fire rates on normal vs pre_stoppage).
    """
    normal_df = train_df[train_df["label"] == "normal"]
    pre_stoppage_df = train_df[train_df["label"] == "pre_stoppage"]

    results: Dict[str, Any] = {}

    for pattern_name, col_specs in PATTERN_COLUMNS.items():
        pattern_info: Dict[str, Any] = {"thresholds": {}, "fire_rates": {}}

        for config_attr, csv_col, direction in col_specs:
            if csv_col not in train_df.columns:
                logger.warning("Column %s not found for pattern %s", csv_col, pattern_name)
                continue

            normal_vals = normal_df[csv_col].dropna()
            if len(normal_vals) < 5:
                logger.warning("Too few normal values for %s", csv_col)
                continue

            # Calculate threshold from normal distribution
            if direction == "above":
                # Pattern fires when value > threshold
                # Set threshold so only ~(100-percentile)% of normals fire
                thresh = float(np.percentile(normal_vals, percentile))
                # Count fire rates
                n_fire_normal = int((normal_df[csv_col] > thresh).sum())
                n_fire_pb = int((pre_stoppage_df[csv_col] > thresh).sum()) if len(pre_stoppage_df) > 0 else 0

            elif direction == "above_abs":
                # Pattern fires when |value| > threshold
                abs_vals = normal_vals.abs()
                thresh = float(np.percentile(abs_vals, percentile))
                n_fire_normal = int((normal_df[csv_col].abs() > thresh).sum())
                n_fire_pb = int((pre_stoppage_df[csv_col].abs() > thresh).sum()) if len(pre_stoppage_df) > 0 else 0

            elif direction == "below":
                # Pattern fires when value < threshold (lower = more anomalous)
                # Use p5 (or 100-percentile) of normal
                lower_pct = 100.0 - percentile
                thresh = float(np.percentile(normal_vals, lower_pct))
                n_fire_normal = int((normal_df[csv_col] < thresh).sum())
                n_fire_pb = int((pre_stoppage_df[csv_col] < thresh).sum()) if len(pre_stoppage_df) > 0 else 0

            elif direction == "band_low":
                # Pattern fires when 0 < value < threshold (low value = anomaly)
                # Use p5 of normal as threshold: values below this are unusual
                lower_pct = 100.0 - percentile
                nonzero = normal_vals[normal_vals > 0]
                if len(nonzero) < 5:
                    thresh = 50.0  # fallback
                else:
                    thresh = float(np.percentile(nonzero, lower_pct))
                n_fire_normal = int(((normal_df[csv_col] > 0) & (normal_df[csv_col] < thresh)).sum())
                n_fire_pb = int(((pre_stoppage_df[csv_col] > 0) & (pre_stoppage_df[csv_col] < thresh)).sum()) if len(pre_stoppage_df) > 0 else 0

            elif direction == "band":
                # Pattern fires when 0 < |value| < threshold (low correlation = anomaly)
                # Use p5 of |normal values where |v| > 0|
                abs_vals = normal_vals.abs()
                nonzero = abs_vals[abs_vals > 0]
                lower_pct = 100.0 - percentile
                if len(nonzero) < 5:
                    thresh = 0.3  # fallback
                else:
                    thresh = float(np.percentile(nonzero, lower_pct))
                n_fire_normal = int(((normal_df[csv_col].abs() > 0) & (normal_df[csv_col].abs() < thresh)).sum())
                n_fire_pb = int(((pre_stoppage_df[csv_col].abs() > 0) & (pre_stoppage_df[csv_col].abs() < thresh)).sum()) if len(pre_stoppage_df) > 0 else 0
            else:
                continue

            # Store calibrated threshold + diagnostics
            n_normal = len(normal_df)
            n_pb = len(pre_stoppage_df)
            rate_normal = _safe_fire_rate(n_fire_normal, n_normal)
            rate_pb = _safe_fire_rate(n_fire_pb, n_pb)
            ratio_val = round(
                rate_pb / max(0.01, rate_normal),
                2,
            ) if n_pb > 0 and n_normal > 0 else 0.0
            score_val = round(
                _discrimination_score(rate_pb, rate_normal, n_pb, n_normal),
                4,
            ) if n_pb > 0 and n_normal > 0 else 0.0
            polarity = _classify_pattern_polarity(
                ratio_val,
                rate_normal,
                rate_pb,
                config.min_discrimination_ratio,
            )

            pattern_info["thresholds"][config_attr] = {
                "value": thresh,
                "direction": direction,
                "csv_column": csv_col,
                "polarity": polarity,
                "discrimination_score": score_val,
            }
            pattern_info["fire_rates"][csv_col] = {
                "normal": f"{n_fire_normal}/{n_normal} ({100*n_fire_normal/max(1,n_normal):.0f}%)",
                "pre_stoppage": f"{n_fire_pb}/{n_pb} ({100*n_fire_pb/max(1,n_pb):.0f}%)" if n_pb > 0 else "N/A",
                "fire_rate_normal": round(rate_normal, 4),
                "fire_rate_pre_stoppage": round(rate_pb, 4),
                "discrimination_ratio": ratio_val,
                "discrimination_score": score_val,
                "polarity": polarity,
            }

            logger.info(
                "  %s.%s: threshold=%.3f (p%.0f of normal), "
                "fire normal=%d/%d (%.0f%%), fire pre_stoppage=%d/%d (%.0f%%)",
                pattern_name, csv_col, thresh, percentile,
                n_fire_normal, n_normal, 100 * n_fire_normal / max(1, n_normal),
                n_fire_pb, n_pb, 100 * n_fire_pb / max(1, n_pb),
            )

            # Fix 4b: auto-disable anti-discriminative patterns ---------------
            if polarity == "uninformative":
                # Set threshold to an impossible value so pattern never fires
                if direction in ("above", "above_abs"):
                    pattern_info["thresholds"][config_attr]["value"] = float("inf")
                elif direction in ("below",):
                    pattern_info["thresholds"][config_attr]["value"] = float("-inf")
                elif direction in ("band", "band_low"):
                    pattern_info["thresholds"][config_attr]["value"] = -1.0  # impossible for band
                pattern_info["thresholds"][config_attr]["disabled"] = True
                pattern_info["thresholds"][config_attr]["disabled_reason"] = (
                    f"pattern classified as uninformative (discrimination_ratio={ratio_val:.2f})"
                )
                logger.warning(
                    "  DISABLED %s.%s: discrimination_ratio=%.2f, score=%.4f (uninformative)",
                    pattern_name, csv_col, ratio_val, score_val,
                )
            elif polarity == "protective":
                pattern_info["thresholds"][config_attr]["disabled"] = False
                logger.info(
                    "  PROTECTIVE %s.%s: discrimination_ratio=%.2f, score=%.4f (normal-supporting)",
                    pattern_name, csv_col, ratio_val, score_val,
                )

        results[pattern_name] = pattern_info

    # Sync discrimination ratios with the pattern registry so that
    # disabled patterns are also disabled in the registry detectors
    # (used by evaluator's _detect_patterns).
    registry = _get_pattern_registry()
    disc_ratios: Dict[str, float] = {}
    disc_scores: Dict[str, float] = {}
    polarity_by_name: Dict[str, str] = {}
    fire_rates_normal: Dict[str, float] = {}
    fire_rates_event: Dict[str, float] = {}
    for pname, pinfo in results.items():
        sub_stats = [
            fr
            for fr in pinfo.get("fire_rates", {}).values()
            if isinstance(fr, dict) and "discrimination_ratio" in fr
        ]
        if not sub_stats:
            continue
        best_stat = max(
            sub_stats,
            key=lambda fr: abs(float(fr.get("discrimination_score", 0.0))),
        )
        disc_ratios[pname] = float(best_stat.get("discrimination_ratio", 0.0))
        disc_scores[pname] = float(best_stat.get("discrimination_score", 0.0))
        polarity_by_name[pname] = str(best_stat.get("polarity", "fault_supporting"))
        fire_rates_normal[pname] = float(best_stat.get("fire_rate_normal", 0.0))
        fire_rates_event[pname] = float(best_stat.get("fire_rate_pre_stoppage", 0.0))
        pinfo["polarity"] = polarity_by_name[pname]
        pinfo["discrimination_score"] = disc_scores[pname]
        pinfo["discrimination_ratio"] = disc_ratios[pname]
    if disc_ratios and config.min_discrimination_ratio > 0:
        buckets = registry.classify_patterns(
            config.min_discrimination_ratio,
            disc_ratios,
            polarities=polarity_by_name,
            discrimination_scores=disc_scores,
            fire_rate_normal=fire_rates_normal,
            fire_rate_event=fire_rates_event,
        )
        protective = buckets.get("protective", [])
        uninformative = buckets.get("uninformative", [])
        if protective or uninformative:
            logger.info(
                "Registry classification: %d protective, %d uninformative patterns",
                len(protective),
                len(uninformative),
            )

    return results


def _apply_calibrated_thresholds(
    config: ExperimentConfig,
    calibrated: Dict[str, Any],
) -> None:
    """Write calibrated threshold values back into the config object."""
    for _pattern_name, info in calibrated.items():
        for config_attr, details in info.get("thresholds", {}).items():
            if hasattr(config, config_attr):
                setattr(config, config_attr, details["value"])
                logger.debug("Config.%s = %.3f (calibrated)", config_attr, details["value"])


# =====================================================================
# Feature-importance-driven pattern mining
# =====================================================================


def _mine_patterns_from_importances(
    train_df: pd.DataFrame,
    top_features: List[tuple],
    config: ExperimentConfig,
    min_importance: float = 0.02,
    max_patterns: int = 5,
) -> int:
    """Auto-generate candidate patterns from feature importances.

    Examines the top features from the supervised model and creates
    threshold-based detectors for features that:
      1. Have importance ≥ min_importance
      2. Are not already covered by an existing registry pattern
      3. Show meaningful separation between normal and pre_stoppage

    The generated patterns use ``source="feature_importance"`` and are
    registered with the singleton PatternRegistry so the evaluator
    picks them up automatically.

    Returns the number of patterns created.
    """
    registry = _get_pattern_registry()
    existing_cols = set()
    for pdef in registry.list_patterns():
        existing_cols.update(pdef.columns)

    normal_df = train_df[train_df["label"] == "normal"]
    pre_df = train_df[train_df["label"] == "pre_stoppage"]
    created = 0

    for feat_name, importance in top_features:
        if created >= max_patterns:
            break
        if importance < min_importance:
            continue
        if feat_name in existing_cols:
            continue
        if feat_name in LEAKY_COLUMNS or feat_name in METADATA_COLUMNS:
            continue
        if feat_name not in train_df.columns:
            continue

        normal_vals = normal_df[feat_name].dropna()
        pre_vals = pre_df[feat_name].dropna()
        if len(normal_vals) < 5 or len(pre_vals) < 5:
            continue

        # Check separation: mean difference relative to pooled std
        pooled_std = float(np.sqrt(
            (normal_vals.var() * len(normal_vals) + pre_vals.var() * len(pre_vals))
            / (len(normal_vals) + len(pre_vals))
        ))
        if pooled_std < 1e-9:
            continue
        cohens_d = abs(float(pre_vals.mean() - normal_vals.mean())) / pooled_std
        if cohens_d < 0.5:
            # Weak separation — not worth creating a pattern
            continue

        # Determine direction and threshold
        if float(pre_vals.mean()) > float(normal_vals.mean()):
            direction = "above"
            thresh = float(np.percentile(normal_vals, config.pattern_calibration_normal_percentile))
            detector_fn = _make_above_detector(feat_name, thresh)
        else:
            direction = "below"
            pct = 100.0 - config.pattern_calibration_normal_percentile
            thresh = float(np.percentile(normal_vals, pct))
            detector_fn = _make_below_detector(feat_name, thresh)

        # Check discrimination ratio
        if direction == "above":
            fire_normal = (normal_vals > thresh).sum()
            fire_pre = (pre_vals > thresh).sum()
        else:
            fire_normal = (normal_vals < thresh).sum()
            fire_pre = (pre_vals < thresh).sum()
        rate_normal = fire_normal / max(1, len(normal_vals))
        rate_pre = fire_pre / max(1, len(pre_vals))
        disc_ratio = rate_pre / max(0.01, rate_normal)
        if disc_ratio < config.min_discrimination_ratio:
            continue

        # Create and register the pattern
        pattern_key = f"IMP_{feat_name.upper()}"
        from .pattern_registry import PatternDefinition
        pdef = PatternDefinition(
            name=pattern_key,
            description=(
                f"Auto-mined from feature importance (d={cohens_d:.2f}, "
                f"disc={disc_ratio:.1f}x): {feat_name} {direction} {thresh:.3f}"
            ),
            category="mined",
            severity=min(0.90, 0.5 + cohens_d * 0.15),
            detector=detector_fn,
            columns=[feat_name],
            default_prior=0.5,
            source="feature_importance",
            discrimination_ratio=disc_ratio,
        )
        registry.register(pdef)
        created += 1
        logger.info(
            "Mined pattern %s: %s %s %.3f (d=%.2f, disc=%.1fx, imp=%.3f)",
            pattern_key, feat_name, direction, thresh,
            cohens_d, disc_ratio, importance,
        )

    if created:
        logger.info("Feature importance mining: created %d new patterns", created)
    return created


def _make_above_detector(col: str, threshold: float):
    """Create a closure detector that fires when col > threshold."""
    def _detector(features: Dict[str, float], _thresholds: Dict[str, Any]) -> bool:
        return features.get(col, 0) > threshold
    return _detector


def _make_below_detector(col: str, threshold: float):
    """Create a closure detector that fires when col < threshold."""
    def _detector(features: Dict[str, float], _thresholds: Dict[str, Any]) -> bool:
        val = features.get(col, 0)
        return val != 0 and val < threshold
    return _detector

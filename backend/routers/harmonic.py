"""
Harmonic Context Model — Training API router.

POST /harmonic/train   — Train (or retrain) the harmonic model
                         on a specified dataset (casedata / site_a_line2 /
                         pair_raw / custom).
GET  /harmonic/status   — Query model status, metrics, and config.

Tag: [HARMONIC_CONTEXT_V1]
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from ..json_utils import json_safe

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/harmonic", tags=["harmonic"])

_CASEDATA_DIR = Path(os.environ.get("HARMONIC_CASEDATA_DIR", "data/casedata"))
_SITE_A_LINE2_DIR = Path(os.environ.get("HARMONIC_SITE_A_LINE2_DIR", "data/Site_a_line2"))
_FEATURES_DIR = Path(os.environ.get("HARMONIC_FEATURES_DIR", "data/breakage_patterns"))
_PAIR_RAW_DIR = Path(os.environ.get("HARMONIC_PAIR_RAW_DIR", "data/breakage_patterns/splits"))
_BOOL_TRUE = {"1", "true", "yes", "on"}


# ── Request / Response schemas ────────────────────────────────────────────


class TrainRequest(BaseModel):
    """Request body for POST /harmonic/train."""

    dataset: str = Field(
        "casedata",
        description="Dataset preset name: 'casedata', 'stoppage_1hz', 'site_a_line2', 'pair_raw', 'pair_casedata', 'pair_lfl', or 'custom'",
    )
    data_dir: Optional[str] = Field(
        None,
        description="Override path to the dataset directory",
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional overrides merged into the HarmonicContextConfig",
    )
    random_seed: Optional[int] = Field(
        None,
        description="Optional training seed; when provided it is written into the harmonic config and overrides config_overrides.random_seed.",
    )
    model_save_path: Optional[str] = Field(
        None,
        description="Optional explicit checkpoint output path for the trained model.",
    )
    checkpoint_suffix: Optional[str] = Field(
        None,
        description="Optional suffix appended to the resolved checkpoint filename. If omitted, an explicit random_seed defaults to a seed-based experiment suffix.",
    )
    replace_checkpoint: bool = Field(
        False,
        description="When true, keep writing to the canonical checkpoint path instead of deriving an experiment checkpoint path.",
    )


class TrainResponse(BaseModel):
    """Response body for POST /harmonic/train (immediate)."""

    status: str  # "started" | "completed" | "error"
    message: str
    task_id: Optional[str] = None


class HarmonicRetrainRequest(BaseModel):
    dataset: Optional[str] = Field(
        None,
        description="Optional harmonic preset name to retrain from feedback buffer",
    )
    scorer_kind: Optional[str] = Field(
        None,
        description="Optional scorer family selector: context or pair",
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional config overrides applied before retraining",
    )
    random_seed: Optional[int] = Field(
        None,
        description="Optional retraining seed; when provided it is written into the harmonic config and overrides config_overrides.random_seed.",
    )
    model_save_path: Optional[str] = Field(
        None,
        description="Optional explicit checkpoint output path for the retrained model.",
    )
    checkpoint_suffix: Optional[str] = Field(
        None,
        description="Optional suffix appended to the resolved checkpoint filename. If omitted, an explicit random_seed defaults to a seed-based experiment suffix.",
    )
    replace_checkpoint: bool = Field(
        False,
        description="When true, keep writing to the canonical checkpoint path instead of deriving an experiment checkpoint path.",
    )


class HarmonicRetrainResponse(BaseModel):
    success: bool = False
    message: str = ""
    bucket_key: str = ""
    dataset_name: str = ""
    scorer_kind: str = "context"
    n_samples_used: int = 0
    n_confirmed: int = 0
    n_dismissed: int = 0
    model_path: str = ""
    best_val_loss: Optional[float] = None
    best_val_acc: Optional[float] = None
    duration_s: float = 0.0
    training_result: Dict[str, Any] = Field(default_factory=dict)


class HarmonicRetrainBucketStatus(BaseModel):
    dataset_name: str = ""
    scorer_kind: str = "context"
    total_feedback: int = 0
    since_last_retrain: int = 0
    retrain_threshold: int = 20
    buffer_size: int = 0
    confirmed_in_buffer: int = 0
    dismissed_in_buffer: int = 0
    should_retrain: bool = False
    retrain_count: int = 0
    last_retrain: Optional[str] = None
    model_save_path: str = ""


class HarmonicRetrainStatusResponse(BaseModel):
    total_feedback: int = 0
    active_bucket: Optional[str] = None
    buckets: Dict[str, HarmonicRetrainBucketStatus] = Field(default_factory=dict)


class HarmonicFeedbackSeedRequest(BaseModel):
    dataset: str = Field(
        "pair_lfl",
        description="Seed target preset: casedata, pair_raw, pair_casedata, or pair_lfl",
    )
    scorer_kind: Optional[str] = Field(
        None,
        description="Optional scorer kind override. Defaults to pair for pair_* presets and context otherwise.",
    )
    confirmed: int = Field(
        12,
        ge=0,
        le=500,
        description="Number of confirmed synthetic feedback samples to add.",
    )
    dismissed: int = Field(
        8,
        ge=0,
        le=500,
        description="Number of dismissed synthetic feedback samples to add.",
    )
    clear_existing: bool = Field(
        True,
        description="When true, clear the target in-memory feedback bucket before seeding.",
    )
    session_prefix: str = Field(
        "harmonic-seed",
        description="Prefix used when generating synthetic session and memory identifiers.",
    )
    operation_prefix: str = Field(
        "SEED-OP",
        description="Prefix used when generating synthetic operation identifiers.",
    )


class HarmonicFeedbackSeedResponse(BaseModel):
    enabled: bool = True
    bucket_key: str = ""
    dataset_name: str = ""
    scorer_kind: str = "context"
    added_confirmed: int = 0
    added_dismissed: int = 0
    cleared_existing: bool = False
    removed_buffer_size: int = 0
    removed_total_feedback: int = 0
    total_feedback: int = 0
    buffer_size: int = 0
    confirmed_in_buffer: int = 0
    dismissed_in_buffer: int = 0
    should_retrain: bool = False


class StatusResponse(BaseModel):
    """Response body for GET /harmonic/status."""

    available: bool
    torch_installed: bool
    model_loaded: bool
    dataset_name: str = ""
    scorer_kind: str = ""
    n_harm_features: int = 0
    n_params: int = 0
    harmonic_mode: str = ""
    cnn_window: int = 0
    decision_threshold: float = 0.5
    trained_at: Optional[str] = None
    training_metrics: Dict[str, Any] = Field(default_factory=dict)
    model_save_path: str = ""
    model_path_exists: bool = False
    checkpoint_statuses: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class HarmonicEvaluationRequest(BaseModel):
    """Request body for POST /harmonic/evaluate."""

    dataset: str = Field(
        "pair_casedata",
        description="Dataset preset name to evaluate. Runtime-style evaluation is currently implemented for pair scorers.",
    )
    data_dir: Optional[str] = Field(
        None,
        description="Optional dataset directory override used when reconstructing the evaluation frame.",
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional config overrides merged into the selected harmonic preset.",
    )
    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Decision threshold used when turning runtime scores into positive or negative predictions.",
    )
    context_mode: str = Field(
        "last_row",
        description="How to derive the pair model context vector per window: 'last_row' matches runtime, 'mean_window' is diagnostic.",
    )
    max_windows: Optional[int] = Field(
        None,
        ge=1,
        description="Optional cap on the number of labelled windows to score.",
    )


class HarmonicEvaluationResponse(BaseModel):
    success: bool = False
    dataset_name: str = ""
    scorer_kind: str = ""
    evaluation_mode: str = "runtime_pair_window"
    model_path: str = ""
    model_loaded: bool = False
    threshold: float = 0.5
    context_mode: str = "last_row"
    decision_threshold: float = 0.5
    applied_threshold: Optional[float] = None
    n_windows: int = 0
    n_positive: int = 0
    n_negative: int = 0
    n_skipped: int = 0
    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    recommended_threshold: Optional[float] = None
    recommended_balanced_accuracy: Optional[float] = None
    recommended_accuracy: Optional[float] = None
    confusion: Dict[str, int] = Field(default_factory=dict)
    score_summary: Dict[str, Any] = Field(default_factory=dict)
    training_metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# ── In-progress tracking ─────────────────────────────────────────────────

_training_in_progress = False
_last_train_result: Optional[Dict[str, Any]] = None


def _training_config_overrides(
    config_overrides: Optional[Dict[str, Any]],
    *,
    random_seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    overrides = dict(config_overrides or {})
    if random_seed is not None:
        overrides["random_seed"] = int(random_seed)
    return overrides or None


def _training_writes_experiment_checkpoint(
    *,
    random_seed: Optional[int] = None,
    model_save_path: Optional[str] = None,
    checkpoint_suffix: Optional[str] = None,
    replace_checkpoint: bool = False,
) -> bool:
    if replace_checkpoint:
        return False
    return bool(str(model_save_path or "").strip() or str(checkpoint_suffix or "").strip() or random_seed is not None)


def _harmonic_dev_seed_enabled() -> bool:
    raw = str(os.environ.get("HARMONIC_ENABLE_DEV_SEED", "") or "").strip().lower()
    return raw in _BOOL_TRUE


def _build_seed_context_metrics(sample_index: int, was_significant: bool) -> Dict[str, float]:
    base = 1.6 if was_significant else 0.7
    return {
        "Vibration_Harmonic_1_X_Amplitude": base + 0.04 * sample_index,
        "Vibration_Harmonic_2_X_Amplitude": base * 0.62 + 0.02 * sample_index,
        "Vibration_Harmonic_1_Y_Amplitude": base * 0.83 + 0.03 * sample_index,
        "spindle_speed_mean": 6000.0 + 10.0 * (sample_index % 5),
        "feed_rate_mean": 720.0 + 4.0 * (sample_index % 7),
    }


def _build_seed_pair_metrics(dataset_name: str, sample_index: int, was_significant: bool) -> Dict[str, float]:
    spindle_speed = 6000.0 + 30.0 * (sample_index % 5)
    feed_rate = 720.0 + 6.0 * (sample_index % 7)
    amp_scale = 1.9 if was_significant else 0.75
    metrics: Dict[str, float] = {
        "tool_diameter": 10.0,
        "num_teeth": 4.0,
        "spindle_speed_mean": spindle_speed,
        "feed_per_tooth": 0.03,
        "feed_rate_mean": feed_rate,
    }
    spindle_hz = spindle_speed / 60.0

    if dataset_name == "pair_raw":
        for channel in (1, 2):
            channel_scale = 1.0 if channel == 1 else 0.92
            for peak_idx in range(5):
                harmonic = 1.1 + float(peak_idx)
                metrics[f"Accel_FFT_Acc{channel}_range1_Frequencies_{peak_idx}"] = spindle_hz * harmonic
                metrics[f"Accel_FFT_Acc{channel}_range1_Amplitudes_{peak_idx}"] = (
                    amp_scale * channel_scale * (1.0 + 0.08 * peak_idx) + 0.01 * sample_index
                )
        return metrics

    for axis, axis_scale in (("X", 1.0), ("Y", 0.9)):
        for peak_num in range(1, 6):
            harmonic = 1.05 + float(peak_num)
            metrics[f"Vibration_Peak_{peak_num}_{axis}_Frequency"] = spindle_hz * harmonic
            metrics[f"Vibration_Peak_{peak_num}_{axis}_Amplitude"] = (
                amp_scale * axis_scale * (1.0 + 0.06 * peak_num) + 0.01 * sample_index
            )
    return metrics


def _build_seed_feedback_kwargs(
    *,
    dataset_name: str,
    scorer_kind: str,
    sample_index: int,
    was_significant: bool,
    session_prefix: str,
    operation_prefix: str,
) -> Dict[str, Any]:
    if scorer_kind == "pair":
        raw_metrics = _build_seed_pair_metrics(dataset_name, sample_index, was_significant)
    else:
        raw_metrics = _build_seed_context_metrics(sample_index, was_significant)

    operation_id = f"{operation_prefix}-{sample_index % 4:02d}"
    sign = "pos" if was_significant else "neg"
    return {
        "was_significant": was_significant,
        "raw_metrics": raw_metrics,
        "harmonic_context": None,
        "harmonic_runtime": {
            "scorer_kind": scorer_kind,
            "dataset": dataset_name,
        },
        "cutting_context": None,
        "source": f"harmonic_dev_seed:{dataset_name}",
        "casedata": {"operation_id": operation_id},
        "memory_id": f"{session_prefix}-{dataset_name}-{sign}-{sample_index:03d}",
        "session_id": f"{session_prefix}:{dataset_name}",
    }


def _apply_training_checkpoint_target(
    config: Any,
    *,
    random_seed: Optional[int] = None,
    model_save_path: Optional[str] = None,
    checkpoint_suffix: Optional[str] = None,
    replace_checkpoint: bool = False,
) -> str:
    from ..agents.processing.harmonic_config import resolve_training_model_save_path

    resolved_path = resolve_training_model_save_path(
        getattr(config, "model_save_path", ""),
        model_save_path=model_save_path,
        checkpoint_suffix=checkpoint_suffix,
        random_seed=random_seed,
        replace_checkpoint=replace_checkpoint,
    )
    if resolved_path:
        setattr(config, "model_save_path", resolved_path)
    return str(getattr(config, "model_save_path", "") or "")


def _refresh_runtime_harmonic_scorers(config: Any) -> None:
    try:
        from ..inference_streamer import clear_harmonic_scorer_cache

        clear_harmonic_scorer_cache()
    except Exception:
        logger.debug("Could not clear harmonic inference cache", exc_info=True)

    try:
        from ..agents.memory.orchestrator import get_orchestrator
        from ..agents.processing.harmonic_runtime import ensure_harmonic_scorer, harmonic_torch_available

        orchestrator = get_orchestrator()
        if orchestrator is None:
            return
        orchestrator.config.harmonic_config = config
        if harmonic_torch_available(config):
            orchestrator.harmonic_scorer = ensure_harmonic_scorer(config)
    except Exception:
        logger.debug("Could not refresh orchestrator harmonic scorer", exc_info=True)


def _build_stored_harmonic_explanation(
    metadata: Dict[str, Any],
    *,
    dataset_name: str,
    top_k: int,
) -> Optional[Dict[str, Any]]:
    from ..agents.processing.harmonic_explain import build_harmonic_explanation

    harmonic_context = metadata.get("harmonic_context")
    if not isinstance(harmonic_context, dict):
        return None

    labels = harmonic_context.get("feature_labels")
    values = harmonic_context.get("feature_values")
    weights = harmonic_context.get("context_weights")
    has_payload = any(
        isinstance(candidate, list) and bool(candidate)
        for candidate in (labels, values, weights)
    )
    if not has_payload:
        return None

    external_signals = metadata.get("external_signals")
    if not isinstance(external_signals, dict):
        external_signals = {}

    return build_harmonic_explanation(
        {
            "harmonic_context_score": external_signals.get("harmonic_context_score"),
            "model_source": external_signals.get("harmonic_context_source")
            or harmonic_context.get("source", ""),
            "context_weights": weights,
        },
        values if isinstance(values, list) else None,
        labels if isinstance(labels, list) else None,
        dataset_name=dataset_name,
        top_k=top_k,
    )


def _load_training_config_and_dataframe(
    dataset: str,
    data_dir: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> tuple[Any, Any]:
    from ..agents.processing.harmonic_config import (
        HarmonicContextConfig,
        casedata_stoppage_preset,
        pair_casedata_preset,
        pair_lfl_preset,
        pair_raw_preset,
        stoppage_1hz_preset,
        site_a_line2_breakage_preset,
    )

    overrides = config_overrides or {}
    dataset_key = str(dataset or "").strip().lower()
    if dataset_key == "casedata":
        config = casedata_stoppage_preset(**overrides)
        df = _load_casedata(
            data_dir,
            allow_unlabelled_fallback=bool(config.allow_unlabelled_fallback),
        )
    elif dataset_key == "stoppage_1hz":
        config = stoppage_1hz_preset(**overrides)
        df = _load_stoppage_1hz(data_dir)
    elif dataset_key == "site_a_line2":
        config = site_a_line2_breakage_preset(**overrides)
        df = _load_site_a_line2(data_dir)
    elif dataset_key == "pair_raw":
        config = pair_raw_preset(**overrides)
        df = _load_pair_raw(data_dir)
    elif dataset_key == "pair_casedata":
        config = pair_casedata_preset(**overrides)
        df = _load_pair_casedata(
            data_dir,
            positive_labels=config.positive_labels,
        )
    elif dataset_key == "pair_lfl":
        config = pair_lfl_preset(**overrides)
        df = _load_pair_casedata(
            data_dir,
            positive_labels=config.positive_labels,
        )
    elif dataset_key == "custom":
        config = HarmonicContextConfig(**overrides)
        if not data_dir:
            raise ValueError("data_dir required for custom dataset")
        df = _load_csv(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return config, df


def _binary_threshold_metrics(
    labels: Any,
    scores: Any,
    threshold: float,
) -> Dict[str, Any]:
    import numpy as np

    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    preds = (scores_arr >= float(threshold)).astype(int)

    tp = int(np.logical_and(preds == 1, labels_arr == 1).sum())
    tn = int(np.logical_and(preds == 0, labels_arr == 0).sum())
    fp = int(np.logical_and(preds == 1, labels_arr == 0).sum())
    fn = int(np.logical_and(preds == 0, labels_arr == 1).sum())

    pos_total = tp + fn
    neg_total = tn + fp
    tpr = tp / pos_total if pos_total else 0.0
    tnr = tn / neg_total if neg_total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    accuracy = float((preds == labels_arr).mean()) if len(labels_arr) else 0.0
    balanced_accuracy = float((tpr + tnr) / 2.0)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "confusion": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
    }


def _recommended_runtime_threshold(labels: Any, scores: Any) -> Dict[str, float]:
    import numpy as np

    score_values = sorted({float(val) for val in np.asarray(scores, dtype=float).tolist()})
    candidates = {0.0, 0.5, 1.0}
    candidates.update(score_values)
    candidates.update(
        (left + right) / 2.0
        for left, right in zip(score_values, score_values[1:])
    )

    best: Optional[Dict[str, float]] = None
    for threshold in sorted(candidates):
        metrics = _binary_threshold_metrics(labels, scores, float(threshold))
        candidate = {
            "threshold": float(threshold),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "accuracy": float(metrics["accuracy"]),
        }
        if best is None or (
            candidate["balanced_accuracy"],
            candidate["accuracy"],
            candidate["threshold"],
        ) > (
            best["balanced_accuracy"],
            best["accuracy"],
            best["threshold"],
        ):
            best = candidate

    return best or {
        "threshold": 0.5,
        "balanced_accuracy": 0.0,
        "accuracy": 0.0,
    }


def _summarize_scores_by_label(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    import numpy as np

    summary: Dict[str, Any] = {}
    labels = sorted({str(row["label"]) for row in rows})
    for label in labels:
        label_scores = np.asarray(
            [float(row["score"]) for row in rows if str(row["label"]) == label],
            dtype=float,
        )
        if label_scores.size == 0:
            continue
        summary[label] = {
            "count": int(label_scores.size),
            "mean": float(label_scores.mean()),
            "min": float(label_scores.min()),
            "p25": float(np.quantile(label_scores, 0.25)),
            "p50": float(np.quantile(label_scores, 0.5)),
            "p75": float(np.quantile(label_scores, 0.75)),
            "max": float(label_scores.max()),
        }
    return summary


def _persist_runtime_threshold(config: Any, threshold: float) -> float:
    numeric_threshold = float(threshold)
    setattr(config, "decision_threshold", numeric_threshold)
    training_metrics = dict(getattr(config, "training_metrics", {}) or {})
    training_metrics["decision_threshold"] = numeric_threshold
    setattr(config, "training_metrics", json_safe(training_metrics))

    scorer_kind = str(getattr(config, "scorer_kind", "context") or "context")
    if scorer_kind == "pair":
        from ..agents.processing.harmonic_pair_model import HarmonicPairScorer

        scorer = HarmonicPairScorer(config=config)
        model_path = Path(str(getattr(config, "model_save_path", "") or ""))
        if not scorer.load(model_path):
            raise FileNotFoundError(f"Could not load pair checkpoint for threshold persistence: {model_path}")
        scorer.config.decision_threshold = numeric_threshold
        scorer.config.training_metrics = json_safe(training_metrics)
        scorer.save(model_path)
        config.decision_threshold = numeric_threshold
        config.training_metrics = json_safe(training_metrics)
        return numeric_threshold

    return numeric_threshold


def _evaluate_pair_runtime(
    config: Any,
    df: Any,
    *,
    threshold: float = 0.5,
    context_mode: str = "last_row",
    max_windows: Optional[int] = None,
) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    from ..agents.processing.harmonic_features import (
        extract_context_params,
        resolve_spindle_speed_source_column,
        runtime_context_normalize,
        runtime_context_param_stats,
    )
    from ..agents.processing.harmonic_pair_model import HarmonicPairScorer
    from ..agents.processing.harmonic_peak_pairs import (
        discover_peak_pair_columns,
        extract_peak_pairs_from_df,
    )

    dataset_name = str(getattr(config, "dataset_name", "") or "")
    scorer_kind = str(getattr(config, "scorer_kind", "context") or "context")
    result: Dict[str, Any] = {
        "success": False,
        "dataset_name": dataset_name,
        "scorer_kind": scorer_kind,
        "evaluation_mode": "runtime_pair_window",
        "model_path": str(getattr(config, "model_save_path", "") or ""),
        "model_loaded": False,
        "threshold": float(threshold),
        "context_mode": str(context_mode),
        "decision_threshold": float(getattr(config, "decision_threshold", 0.5) or 0.5),
        "n_windows": 0,
        "n_positive": 0,
        "n_negative": 0,
        "n_skipped": 0,
        "training_metrics": json_safe(getattr(config, "training_metrics", {}) or {}),
        "score_summary": {},
        "confusion": {},
    }

    if scorer_kind != "pair":
        result["error"] = "Runtime evaluation is currently implemented only for pair scorers"
        return result
    if context_mode not in {"last_row", "mean_window"}:
        result["error"] = f"Unsupported context_mode: {context_mode}"
        return result
    if df is None or len(df) == 0:
        result["error"] = "No data loaded"
        return result

    model_path = Path(result["model_path"])
    if not model_path.is_file():
        result["error"] = f"Model checkpoint not found: {model_path}"
        return result

    scorer = HarmonicPairScorer(config=config)
    if not scorer.load(model_path):
        result["error"] = f"Could not load pair checkpoint: {model_path}"
        return result

    result["model_loaded"] = True
    result["training_metrics"] = json_safe(getattr(scorer.config, "training_metrics", {}) or {})
    result["decision_threshold"] = float(getattr(scorer.config, "decision_threshold", result["decision_threshold"]) or result["decision_threshold"])

    label_col = str(getattr(scorer.config, "target_label", "label") or "label")
    if label_col not in df.columns:
        result["error"] = f"Label column '{label_col}' not in evaluation frame"
        return result

    specs = discover_peak_pair_columns(
        list(df.columns),
        frequency_patterns=list(getattr(scorer.config, "pair_frequency_column_patterns", []) or []),
        amplitude_patterns=list(getattr(scorer.config, "pair_amplitude_column_patterns", []) or []),
        k_peaks=int(getattr(scorer.config, "k_peaks", 5)),
    )
    if not specs:
        result["error"] = "No FFT peak frequency/amplitude column pairs found in evaluation frame"
        return result

    ctx_sources = (
        {key: value.get("source_column", key) for key, value in scorer.config.context_param_stats.items()}
        if getattr(scorer.config, "context_param_stats", None)
        else getattr(scorer.config, "context_param_sources", {})
    )
    runtime_stats = runtime_context_param_stats(scorer.config)
    normalize_context = runtime_context_normalize(scorer.config)
    spindle_speed_col = resolve_spindle_speed_source_column(scorer.config)

    if "operation_id" in df.columns:
        groups = df.groupby("operation_id", sort=False)
    else:
        groups = [(dataset_name or "pair_dataset", df)]

    rows: list[Dict[str, Any]] = []
    skipped = 0
    for operation_id, group in groups:
        if max_windows is not None and len(rows) >= int(max_windows):
            break
        group = group.reset_index(drop=True)
        pair_tensor = extract_peak_pairs_from_df(
            group,
            specs,
            spindle_speed_col=spindle_speed_col,
            k_peaks=int(getattr(scorer.config, "k_peaks", 5)),
            f_max_rel=float(getattr(scorer.config, "f_max_rel", 12.0)),
        )
        if pair_tensor.shape[0] == 0 or pair_tensor.shape[1] == 0:
            skipped += 1
            continue

        if context_mode == "mean_window":
            ctx_source: Dict[str, Any] = group.iloc[-1].to_dict()
            for param_key in getattr(scorer.config, "context_param_keys", []) or []:
                source_col = ctx_sources.get(param_key, param_key)
                if source_col in group.columns:
                    values = pd.to_numeric(group[source_col], errors="coerce")
                    if values.notna().any():
                        ctx_source[source_col] = float(values.mean())
        else:
            ctx_source = group.iloc[-1].to_dict()

        ctx_vec = extract_context_params(
            ctx_source,
            getattr(scorer.config, "context_param_keys", []) or [],
            ctx_sources,
            runtime_stats,
            normalize=normalize_context,
        )
        score_result = scorer.score(pair_tensor, ctx_vec)
        score = score_result.get("harmonic_context_score")
        if not isinstance(score, (int, float)) or not np.isfinite(float(score)):
            skipped += 1
            continue

        rows.append({
            "operation_id": str(operation_id),
            "label": str(group[label_col].iloc[0]),
            "score": float(score),
        })

    result["n_skipped"] = int(skipped)
    if not rows:
        result["error"] = "No labelled windows could be scored with runtime semantics"
        return result

    positive_labels = {str(label) for label in getattr(scorer.config, "positive_labels", []) or []}
    labels = np.asarray([1 if row["label"] in positive_labels else 0 for row in rows], dtype=int)
    scores = np.asarray([row["score"] for row in rows], dtype=float)

    result["n_windows"] = int(len(rows))
    result["n_positive"] = int(labels.sum())
    result["n_negative"] = int(len(labels) - labels.sum())
    result["score_summary"] = _summarize_scores_by_label(rows)
    result.update(_binary_threshold_metrics(labels, scores, float(threshold)))

    recommendation = _recommended_runtime_threshold(labels, scores)
    result["recommended_threshold"] = float(recommendation["threshold"])
    result["recommended_balanced_accuracy"] = float(recommendation["balanced_accuracy"])
    result["recommended_accuracy"] = float(recommendation["accuracy"])
    result["success"] = True
    return json_safe(result)


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.get("/status", response_model=StatusResponse)
async def harmonic_status(
    dataset: Optional[str] = Query(
        default=None,
        description="Optional preset selector: casedata, stoppage_1hz, site_a_line2, raw_accelerometer, pair_raw, pair_casedata, pair_lfl, or default",
    )
):
    """Query the harmonic context model status."""
    try:
        from ..agents.processing.harmonic_config import (
            HarmonicContextConfig,
            casedata_stoppage_preset,
            pair_casedata_preset,
            pair_lfl_preset,
            pair_raw_preset,
            raw_accelerometer_preset,
            stoppage_1hz_preset,
            site_a_line2_breakage_preset,
        )
        from ..agents.processing.harmonic_runtime import build_harmonic_scorer, harmonic_torch_available

        dataset_key = str(dataset or "default").strip().lower()
        config_factories = {
            "default": HarmonicContextConfig,
            "casedata": casedata_stoppage_preset,
            "stoppage_1hz": stoppage_1hz_preset,
            "site_a_line2": site_a_line2_breakage_preset,
            "raw_accelerometer": raw_accelerometer_preset,
            "pair_casedata": pair_casedata_preset,
            "pair_lfl": pair_lfl_preset,
            "pair_raw": pair_raw_preset,
        }
        config_factory = config_factories.get(dataset_key, HarmonicContextConfig)
        config = config_factory()

        def _checkpoint_status(checkpoint_config: Any) -> Dict[str, Any]:
            model_path = Path(str(getattr(checkpoint_config, "model_save_path", "") or ""))
            scorer_kind = str(getattr(checkpoint_config, "scorer_kind", "context") or "context")
            status: Dict[str, Any] = {
                "dataset_name": str(getattr(checkpoint_config, "dataset_name", "")),
                "scorer_kind": scorer_kind,
                "model_save_path": str(model_path),
                "model_path_exists": model_path.is_file(),
                "decision_threshold": float(getattr(checkpoint_config, "decision_threshold", 0.5) or 0.5),
            }

            torch_ok = harmonic_torch_available(checkpoint_config)
            status["torch_installed"] = bool(torch_ok)
            if not torch_ok:
                status.update({
                    "available": False,
                    "model_loaded": False,
                })
                return status

            scorer = build_harmonic_scorer(checkpoint_config)
            model_loaded = bool(scorer._ensure_model())
            status.update(scorer.get_model_info())
            status["model_loaded"] = model_loaded
            status.setdefault("scorer_kind", scorer_kind)
            status["model_path_exists"] = model_path.is_file()
            return status

        checkpoint_statuses: Dict[str, Dict[str, Any]] = {}
        for name, factory in config_factories.items():
            if name == "default":
                continue
            try:
                checkpoint_statuses[name] = _checkpoint_status(factory())
            except Exception as exc:
                logger.debug("Could not inspect harmonic checkpoint %s: %s", name, exc, exc_info=True)
                checkpoint_statuses[name] = {
                    "dataset_name": name,
                    "scorer_kind": "pair" if name.startswith("pair_") else "context",
                    "available": False,
                    "torch_installed": False,
                    "model_loaded": False,
                    "model_save_path": "",
                    "model_path_exists": False,
                    "decision_threshold": 0.5,
                    "error": str(exc),
                }

        if not harmonic_torch_available(config):
            return StatusResponse(
                available=False,
                torch_installed=False,
                model_loaded=False,
                scorer_kind=str(getattr(config, "scorer_kind", "context") or "context"),
                model_save_path=str(getattr(config, "model_save_path", "") or ""),
                model_path_exists=Path(str(getattr(config, "model_save_path", "") or "")).is_file(),
                checkpoint_statuses=checkpoint_statuses,
            )

        info = _checkpoint_status(config)
        info["checkpoint_statuses"] = checkpoint_statuses
        return StatusResponse(**info)
    except Exception as e:
        logger.warning("Harmonic status check failed: %s", e)
        return StatusResponse(available=False, torch_installed=False, model_loaded=False)


@router.get("/train/result")
async def harmonic_train_result():
    """Get the result of the last training run."""
    if _last_train_result is None:
        return {"status": "no_training_run"}
    return json_safe(_last_train_result)


@router.post("/evaluate", response_model=HarmonicEvaluationResponse)
async def harmonic_evaluate(req: HarmonicEvaluationRequest):
    """Evaluate a saved harmonic checkpoint with runtime-style scoring semantics."""
    try:
        config, df = _load_training_config_and_dataframe(
            req.dataset,
            req.data_dir,
            req.config_overrides,
        )
        result = _evaluate_pair_runtime(
            config,
            df,
            threshold=req.threshold,
            context_mode=req.context_mode,
            max_windows=req.max_windows,
        )
        return HarmonicEvaluationResponse(**result)
    except Exception as exc:
        logger.warning("Harmonic evaluation failed: %s", exc, exc_info=True)
        return HarmonicEvaluationResponse(
            success=False,
            dataset_name=req.dataset,
            threshold=req.threshold,
            context_mode=req.context_mode,
            error=str(exc),
        )


@router.post("/retrain", response_model=HarmonicRetrainResponse)
async def harmonic_retrain(req: HarmonicRetrainRequest):
    """Retrain a harmonic model from accumulated operator-feedback samples."""
    from ..agents.memory.orchestrator import get_orchestrator
    from ..agents.processing.harmonic_feedback_retrainer import get_harmonic_feedback_retrainer

    orchestrator = get_orchestrator()
    retrainer = get_harmonic_feedback_retrainer(
        model_confidence_path=getattr(orchestrator.scorer, "_model_confidence_path", None),
    )
    result = retrainer.retrain(
        dataset_name=req.dataset,
        scorer_kind=req.scorer_kind,
        config_overrides=_training_config_overrides(
            req.config_overrides,
            random_seed=req.random_seed,
        ),
        random_seed=req.random_seed,
        model_save_path=req.model_save_path,
        checkpoint_suffix=req.checkpoint_suffix,
        replace_checkpoint=req.replace_checkpoint,
    )

    if result.success and result.config is not None and not _training_writes_experiment_checkpoint(
        random_seed=req.random_seed,
        model_save_path=req.model_save_path,
        checkpoint_suffix=req.checkpoint_suffix,
        replace_checkpoint=req.replace_checkpoint,
    ):
        _refresh_runtime_harmonic_scorers(result.config)

    return HarmonicRetrainResponse(
        success=result.success,
        message=result.message,
        bucket_key=result.bucket_key,
        dataset_name=result.dataset_name,
        scorer_kind=result.scorer_kind,
        n_samples_used=result.n_samples_used,
        n_confirmed=result.n_confirmed,
        n_dismissed=result.n_dismissed,
        model_path=result.model_path,
        best_val_loss=result.best_val_loss,
        best_val_acc=result.best_val_acc,
        duration_s=result.duration_s,
        training_result=result.training_result,
    )


@router.get("/retrain/status", response_model=HarmonicRetrainStatusResponse)
async def harmonic_retrain_status(
    dataset: Optional[str] = Query(default=None),
    scorer_kind: Optional[str] = Query(default=None),
):
    """Inspect harmonic feedback buffers and retrain readiness per preset."""
    from ..agents.memory.orchestrator import get_orchestrator
    from ..agents.processing.harmonic_feedback_retrainer import get_harmonic_feedback_retrainer

    orchestrator = get_orchestrator()
    retrainer = get_harmonic_feedback_retrainer(
        model_confidence_path=getattr(orchestrator.scorer, "_model_confidence_path", None),
    )
    status = retrainer.get_status(dataset_name=dataset, scorer_kind=scorer_kind)
    return HarmonicRetrainStatusResponse(**status)


@router.post("/dev/seed-feedback", response_model=HarmonicFeedbackSeedResponse)
async def harmonic_seed_feedback(req: HarmonicFeedbackSeedRequest):
    """Populate an in-memory harmonic feedback bucket for local smoke testing."""
    if not _harmonic_dev_seed_enabled():
        raise HTTPException(
            status_code=403,
            detail="Harmonic dev feedback seeding is disabled. Set HARMONIC_ENABLE_DEV_SEED=1 to enable this route.",
        )

    from ..agents.memory.orchestrator import get_orchestrator
    from ..agents.processing.harmonic_feedback_retrainer import (
        _config_for_feedback_bucket,
        get_harmonic_feedback_retrainer,
    )

    orchestrator = get_orchestrator()
    retrainer = get_harmonic_feedback_retrainer(
        model_confidence_path=getattr(orchestrator.scorer, "_model_confidence_path", None),
    )

    resolved_dataset, resolved_kind, _ = _config_for_feedback_bucket(req.dataset, req.scorer_kind)
    removed = {"removed_buffer_size": 0, "removed_total_feedback": 0}
    if req.clear_existing:
        removed = retrainer.reset_feedback(dataset_name=resolved_dataset, scorer_kind=resolved_kind)

    total_samples = int(req.confirmed) + int(req.dismissed)
    if total_samples <= 0:
        raise HTTPException(status_code=400, detail="confirmed + dismissed must be greater than zero")

    for sample_index in range(int(req.confirmed)):
        retrainer.record_feedback(
            **_build_seed_feedback_kwargs(
                dataset_name=resolved_dataset,
                scorer_kind=resolved_kind,
                sample_index=sample_index,
                was_significant=True,
                session_prefix=req.session_prefix,
                operation_prefix=req.operation_prefix,
            )
        )
    for sample_index in range(int(req.dismissed)):
        retrainer.record_feedback(
            **_build_seed_feedback_kwargs(
                dataset_name=resolved_dataset,
                scorer_kind=resolved_kind,
                sample_index=int(req.confirmed) + sample_index,
                was_significant=False,
                session_prefix=req.session_prefix,
                operation_prefix=req.operation_prefix,
            )
        )

    status = retrainer.get_status(dataset_name=resolved_dataset, scorer_kind=resolved_kind)
    bucket_key = status.get("active_bucket") or f"{resolved_kind}:{resolved_dataset}"
    bucket = (status.get("buckets") or {}).get(bucket_key, {})

    return HarmonicFeedbackSeedResponse(
        enabled=True,
        bucket_key=bucket_key,
        dataset_name=resolved_dataset,
        scorer_kind=resolved_kind,
        added_confirmed=int(req.confirmed),
        added_dismissed=int(req.dismissed),
        cleared_existing=bool(req.clear_existing),
        removed_buffer_size=int(removed.get("removed_buffer_size", 0) or 0),
        removed_total_feedback=int(removed.get("removed_total_feedback", 0) or 0),
        total_feedback=int(status.get("total_feedback", 0) or 0),
        buffer_size=int(bucket.get("buffer_size", 0) or 0),
        confirmed_in_buffer=int(bucket.get("confirmed_in_buffer", 0) or 0),
        dismissed_in_buffer=int(bucket.get("dismissed_in_buffer", 0) or 0),
        should_retrain=bool(bucket.get("should_retrain", False)),
    )


# ── /harmonic/explain/{memory_id} — per-harmonic attribution ─────────────


@router.get("/explain/{memory_id}")
async def harmonic_explain(memory_id: str, top_k: int = 5):
    """Return per-harmonic contribution for a stored memory.

    Agent O (2026-04-24). Loads the memory via the orchestrator, pulls
    ``raw_metrics`` + ``cutting_context`` from its metadata, runs the
    harmonic scorer, and returns score + per-harmonic weight/value/
    contribution alongside the top-k most influential features.

    Always returns 200; use the ``available`` + ``reason`` fields to
    decide whether to render the explanation UI.
    """
    from ..agents.memory.orchestrator import get_orchestrator
    from ..agents.processing.harmonic_explain import build_harmonic_explanation

    try:
        orchestrator = get_orchestrator()
    except Exception as exc:  # pragma: no cover - defensive
        return build_harmonic_explanation(
            None, None, None, available=False,
            reason=f"orchestrator unavailable: {exc}",
        )

    memory = None
    try:
        memory = orchestrator.get_memory(memory_id)
    except Exception:
        memory = None
    if memory is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    meta = getattr(memory, "metadata", {}) or {}
    stored_explanation = _build_stored_harmonic_explanation(
        meta,
        dataset_name="",
        top_k=int(top_k),
    )

    scorer = getattr(orchestrator, "harmonic_scorer", None)
    if scorer is None or not getattr(scorer, "is_available", lambda: False)():
        if stored_explanation is not None:
            return stored_explanation
        return build_harmonic_explanation(
            None, None, None, available=False,
            reason="harmonic scorer not loaded",
        )

    # Pull raw_metrics + cutting_context out of the memory (saved by the
    # orchestrator when the memory was created).
    raw_metrics = meta.get("raw_metrics") or {}
    cutting_context = meta.get("cutting_context") or {}
    dataset_name = getattr(getattr(scorer, "config", None), "dataset_name", "")

    if not raw_metrics:
        if stored_explanation is not None:
            stored_explanation["dataset"] = dataset_name
            return stored_explanation
        return build_harmonic_explanation(
            None, None, None, available=False,
            reason="memory has no raw_metrics; re-process with classical features enabled",
        )

    try:
        import numpy as np
        import pandas as pd
        from ..agents.processing.harmonic_features import (
            extract_context_params,
            extract_harmonic_matrix_from_df,
        )
        from ..agents.processing.harmonic_peak_pairs import (
            discover_peak_pair_columns,
            extract_peak_pairs_from_df,
        )

        cfg = scorer.config
        scorer_kind = str(getattr(cfg, "scorer_kind", "context") or "context").strip().lower()

        row_df = pd.DataFrame([{k: v for k, v in raw_metrics.items() if isinstance(v, (int, float))}])
        score_input = None
        if scorer_kind == "pair":
            pair_specs = discover_peak_pair_columns(
                list(row_df.columns),
                frequency_patterns=list(getattr(cfg, "pair_frequency_column_patterns", []) or []),
                amplitude_patterns=list(getattr(cfg, "pair_amplitude_column_patterns", []) or []),
                k_peaks=int(getattr(cfg, "k_peaks", 5)),
            )
            if not pair_specs:
                if stored_explanation is not None:
                    stored_explanation["dataset"] = getattr(cfg, "dataset_name", "")
                    return stored_explanation
                return build_harmonic_explanation(
                    None, None, None, available=False,
                    reason="no pair FFT columns present in memory raw_metrics",
                    dataset_name=getattr(cfg, "dataset_name", ""),
                )

            from ..agents.processing.harmonic_features import resolve_spindle_speed_source_column

            spindle_speed_col = resolve_spindle_speed_source_column(cfg)
            p_mat = extract_peak_pairs_from_df(
                row_df,
                pair_specs,
                spindle_speed_col=spindle_speed_col,
                k_peaks=int(getattr(cfg, "k_peaks", 5)),
                f_max_rel=float(getattr(cfg, "f_max_rel", 12.0)),
            )
            if p_mat.shape[0] == 0 or p_mat.shape[1] == 0:
                if stored_explanation is not None:
                    stored_explanation["dataset"] = getattr(cfg, "dataset_name", "")
                    return stored_explanation
                return build_harmonic_explanation(
                    None, None, None, available=False,
                    reason="no valid pair features present in memory raw_metrics",
                    dataset_name=getattr(cfg, "dataset_name", ""),
                )
            score_input = p_mat
            harmonic_row = np.asarray(p_mat[0, :, :, 1], dtype=float).reshape(-1).tolist()
        else:
            # Build harmonic feature row (F,) — single-event pre_extracted path.
            h_mat = extract_harmonic_matrix_from_df(row_df, cfg.harmonic_columns)
            if h_mat.shape[0] == 0 or h_mat.shape[1] == 0:
                if stored_explanation is not None:
                    stored_explanation["dataset"] = getattr(cfg, "dataset_name", "")
                    return stored_explanation
                return build_harmonic_explanation(
                    None, None, None, available=False,
                    reason="no harmonic columns present in memory raw_metrics",
                    dataset_name=getattr(cfg, "dataset_name", ""),
                )
            score_input = h_mat
            harmonic_row = np.asarray(h_mat[0], dtype=float).tolist()

        # Build context vector.
        ctx_source: Dict[str, Any] = {}
        if isinstance(cutting_context, dict):
            ctx_source.update(cutting_context)
        # raw_metrics may also contain the context keys (spindle_speed etc.)
        for k, v in raw_metrics.items():
            ctx_source.setdefault(k, v)
        from ..agents.processing.harmonic_features import runtime_context_normalize, runtime_context_param_stats

        runtime_stats = runtime_context_param_stats(cfg)
        normalize_context = runtime_context_normalize(cfg)
        ctx_sources = (
            {k: v.get("source_column", k) for k, v in cfg.context_param_stats.items()}
            if cfg.context_param_stats else cfg.context_param_sources
        )
        ctx_vec = extract_context_params(
            ctx_source,
            cfg.context_param_keys,
            ctx_sources,
            runtime_stats,
            normalize=normalize_context,
        )

        score_result = scorer.score(score_input, ctx_vec)
        try:
            labels = scorer.get_feature_labels()
        except Exception:
            stored_harmonic_context = meta.get("harmonic_context")
            labels = (
                stored_harmonic_context.get("feature_labels")
                if isinstance(stored_harmonic_context, dict)
                else None
            )

        return build_harmonic_explanation(
            score_result,
            harmonic_row,
            labels,
            dataset_name=getattr(cfg, "dataset_name", ""),
            top_k=int(top_k),
        )
    except Exception as exc:
        logger.warning("harmonic_explain failed: %s", exc, exc_info=True)
        return build_harmonic_explanation(
            None, None, None, available=False,
            reason=f"explain failed: {exc}",
        )


@router.post("/train", response_model=TrainResponse)
async def harmonic_train(req: TrainRequest, background_tasks: BackgroundTasks):
    """Train (or retrain) the harmonic context-weighted CNN.

    Runs training asynchronously in the background.  Poll
    ``GET /harmonic/train/result`` for completion.
    """
    global _training_in_progress

    if _training_in_progress:
        raise HTTPException(status_code=409, detail="Training already in progress")

    try:
        from ..agents.processing.harmonic_model import TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            raise HTTPException(
                status_code=400,
                detail="PyTorch not installed — cannot train harmonic model",
            )
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="Harmonic model module not available",
        )

    _training_in_progress = True
    background_tasks.add_task(_run_training, req)

    return TrainResponse(
        status="started",
        message=f"Training started for dataset '{req.dataset}'",
    )


# ── Background training task ─────────────────────────────────────────────


def _run_training(req: TrainRequest) -> None:
    """Execute training in a background thread.

    Loads the appropriate dataset, selects the preset config, trains the
    model, and stores the result.
    """
    global _training_in_progress, _last_train_result
    import time

    try:
        from ..agents.processing.harmonic_pair_trainer import HarmonicPairTrainer
        from ..agents.processing.harmonic_trainer import HarmonicContextTrainer

        config, df = _load_training_config_and_dataframe(
            req.dataset,
            req.data_dir,
            _training_config_overrides(
                req.config_overrides,
                random_seed=req.random_seed,
            ),
        )
        _apply_training_checkpoint_target(
            config,
            random_seed=req.random_seed,
            model_save_path=req.model_save_path,
            checkpoint_suffix=req.checkpoint_suffix,
            replace_checkpoint=req.replace_checkpoint,
        )

        if df is None or len(df) == 0:
            _last_train_result = {
                "status": "error",
                "message": "No data loaded",
                "dataset": req.dataset,
            }
            return

        logger.info(
            "Starting harmonic training: dataset=%s, rows=%d", req.dataset, len(df)
        )

        if getattr(config, "scorer_kind", "context") == "pair":
            trainer = HarmonicPairTrainer(config)
        else:
            trainer = HarmonicContextTrainer(config)
        result = trainer.train_from_dataframe(df)

        if result.success and not _training_writes_experiment_checkpoint(
            random_seed=req.random_seed,
            model_save_path=req.model_save_path,
            checkpoint_suffix=req.checkpoint_suffix,
            replace_checkpoint=req.replace_checkpoint,
        ):
            _refresh_runtime_harmonic_scorers(config)

        runtime_evaluation = None
        if result.success and getattr(config, "scorer_kind", "context") == "pair":
            try:
                runtime_evaluation = _evaluate_pair_runtime(
                    config,
                    df,
                    threshold=0.5,
                    context_mode="last_row",
                )
            except Exception as exc:
                logger.warning("Pair runtime evaluation failed: %s", exc, exc_info=True)
                runtime_evaluation = {
                    "success": False,
                    "dataset_name": str(getattr(config, "dataset_name", "") or ""),
                    "scorer_kind": "pair",
                    "evaluation_mode": "runtime_pair_window",
                    "model_path": str(getattr(config, "model_save_path", "") or ""),
                    "threshold": 0.5,
                    "context_mode": "last_row",
                    "error": f"runtime evaluation failed: {exc}",
                }
            else:
                recommended_threshold = runtime_evaluation.get("recommended_threshold")
                if isinstance(recommended_threshold, (int, float)) and math.isfinite(float(recommended_threshold)):
                    try:
                        applied_threshold = _persist_runtime_threshold(config, float(recommended_threshold))
                        runtime_evaluation["applied_threshold"] = float(applied_threshold)
                    except Exception as exc:
                        logger.warning("Could not persist pair runtime threshold: %s", exc, exc_info=True)
                        runtime_evaluation["threshold_persist_error"] = str(exc)

        _last_train_result = {
            "status": "completed" if result.success else "error",
            "dataset": req.dataset,
            "random_seed": getattr(config, "random_seed", None),
            "runtime_checkpoint_activated": not _training_writes_experiment_checkpoint(
                random_seed=req.random_seed,
                model_save_path=req.model_save_path,
                checkpoint_suffix=req.checkpoint_suffix,
                replace_checkpoint=req.replace_checkpoint,
            ),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            **result.to_dict(),
        }
        if runtime_evaluation is not None:
            _last_train_result["runtime_evaluation"] = json_safe(runtime_evaluation)

        logger.info("Harmonic training completed: %s", _last_train_result["status"])

    except Exception as e:
        logger.exception("Harmonic training failed")
        _last_train_result = {
            "status": "error",
            "message": str(e),
            "dataset": req.dataset,
        }
    finally:
        _training_in_progress = False


# ── Dataset loaders ───────────────────────────────────────────────────────



def _casedata_feature_candidates(data_dir: Optional[str] = None) -> list[Path]:
    candidates: list[Path] = []
    search_roots = [Path(data_dir)] if data_dir else []
    search_roots.extend([_FEATURES_DIR, _CASEDATA_DIR])
    for root in search_roots:
        candidates.extend([
            root / "stoppage_features.csv",
            root / "breakage_features.csv",
        ])
    return candidates


def _find_casedata_feature_path(data_dir: Optional[str] = None) -> Optional[Path]:
    return next((path for path in _casedata_feature_candidates(data_dir) if path.exists()), None)


def _resolve_dataset_loader(data_dir: Optional[str] = None) -> Any:
    from ..agents.processing.dataset_loader import DatasetLoader

    candidate_roots: list[Path] = []
    if data_dir:
        supplied = Path(data_dir)
        candidate_roots.append(supplied if supplied.is_dir() else supplied.parent)
    candidate_roots.append(_CASEDATA_DIR)

    seen: set[Path] = set()
    for root in candidate_roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        try:
            loader = DatasetLoader(root)
            if loader.list_operations():
                return loader
        except Exception:
            logger.debug("Could not initialize DatasetLoader at %s", root, exc_info=True)

    search_roots = [str(path) for path in candidate_roots]
    raise FileNotFoundError(f"No raw casedata operations found in {search_roots}")


def _load_casedata(
    data_dir: Optional[str] = None,
    *,
    allow_unlabelled_fallback: bool = False,
) -> Any:
    """Load labelled casedata stoppage/breakage features CSV."""
    import pandas as pd

    # Prefer stoppage-named exports for the stoppage preset, but keep the
    # legacy breakage filename for backward compatibility with older artifacts.
    candidates = _casedata_feature_candidates(data_dir)

    for p in candidates:
        if p and p.exists():
            logger.info("Loading casedata features from %s", p)
            return pd.read_csv(p)

    if not allow_unlabelled_fallback:
        search_roots = [str(p) for p in candidates if p is not None]
        raise FileNotFoundError(
            "No labelled casedata features file found. Expected stoppage_features.csv or "
            "breakage_features.csv "
            f"in one of: {search_roots}. Raw casedata fallback is disabled because "
            "it cannot infer ground-truth labels safely. Pass "
            "allow_unlabelled_fallback=true only for unsupervised inspection."
        )

    # Fallback: use DatasetLoader to build features from raw operation CSVs
    try:
        root = Path(data_dir) if data_dir else _CASEDATA_DIR
        from ..agents.processing.dataset_loader import DatasetLoader

        loader = DatasetLoader(root)
        ops = loader.list_operations()
        if not ops:
            logger.warning("No casedata operations found in %s", root)
            return pd.DataFrame()

        rows = []
        for op in ops:
            try:
                feat_rows = loader.extract_bulk_features(
                    op.operation_id,
                    window_seconds=10,
                    stride_seconds=5,
                    max_windows=5000,
                )
            except Exception as op_exc:
                logger.debug("Feature extraction failed for %s: %s", op.operation_id, op_exc)
                continue

            for f in feat_rows:
                # Keep operation boundary for leakage-safe splitting in trainer.
                f["operation_id"] = op.operation_id

                # Raw fallback is only for explicit exploratory runs; labels remain
                # synthetic and should not be used for supervised evaluation.
                f.setdefault("label", 0)
                rows.append(f)

        if not rows:
            logger.warning("No casedata features could be extracted from %s", root)
            return pd.DataFrame()

        logger.info("Extracted %d fallback casedata feature rows from raw CSVs", len(rows))
        return pd.DataFrame(rows)
    except Exception as e:
        logger.warning("Could not load casedata: %s", e)
        return pd.DataFrame()


def _load_pair_casedata(
    data_dir: Optional[str] = None,
    *,
    positive_labels: Optional[list[str]] = None,
) -> Any:
    """Reconstruct raw casedata windows for pair-input training.

    The labelled stoppage_features.csv artifact preserves exact sample-window
    metadata, but it strips the raw Vibration_Peak_* columns needed by the pair
    model. Rebuild those windows from raw vibration and machine-state CSVs and
    keep each labelled sample isolated under its own operation_id so the trainer
    does not slide across discontinuous windows.
    """
    import pandas as pd

    from ..agents.processing.tool_lookup import resolve_machine_family, resolve_tool_context

    feature_path = _find_casedata_feature_path(data_dir)
    if feature_path is None:
        search_roots = [str(path) for path in _casedata_feature_candidates(data_dir)]
        raise FileNotFoundError(
            "No labelled casedata features file found for pair_casedata. Expected "
            "stoppage_features.csv or breakage_features.csv in one of: "
            f"{search_roots}"
        )

    samples = pd.read_csv(feature_path)
    required_cols = {"operation_id", "label", "event_timestamp", "window_seconds"}
    missing = required_cols - set(samples.columns)
    if missing:
        raise ValueError(
            f"pair_casedata requires labelled sample metadata columns: {sorted(missing)}"
        )

    samples = samples.copy()
    if "sample_id" not in samples.columns:
        samples["sample_id"] = [f"sample_{idx}" for idx in range(len(samples))]

    active_positive_labels = [str(label) for label in (positive_labels or ["pre_stoppage"])]
    allowed_labels = set(active_positive_labels) | {"normal"}
    samples = samples[samples["label"].isin(allowed_labels)].copy()
    if samples.empty:
        logger.warning("No pair_casedata samples found with labels in %s", sorted(allowed_labels))
        return pd.DataFrame()

    samples["event_timestamp"] = pd.to_datetime(samples["event_timestamp"], utc=True, errors="coerce")
    samples["window_seconds"] = pd.to_numeric(samples["window_seconds"], errors="coerce")
    samples["gap_seconds"] = pd.to_numeric(samples.get("gap_seconds", 0.0), errors="coerce").fillna(0.0)
    samples["trim_seconds_removed"] = pd.to_numeric(
        samples.get("trim_seconds_removed", 0.0),
        errors="coerce",
    ).fillna(0.0)
    samples = samples.dropna(subset=["event_timestamp", "window_seconds", "operation_id", "sample_id"])
    if samples.empty:
        logger.warning("No valid pair_casedata sample windows remained after timestamp parsing")
        return pd.DataFrame()

    loader = _resolve_dataset_loader(data_dir)
    operations_by_id: dict[str, list[Any]] = {}
    for operation in loader.list_operations():
        operations_by_id.setdefault(str(operation.operation_id), []).append(operation)

    if "case_dir" not in samples.columns:
        samples["case_dir"] = pd.NA

    missing_case_mask = samples["case_dir"].isna() | samples["case_dir"].fillna("").astype(str).str.strip().eq("")
    if missing_case_mask.any():
        time_bounds_cache: dict[tuple[str, str], Optional[tuple[Any, Any]]] = {}

        def _operation_time_bounds(operation: Any) -> Optional[tuple[Any, Any]]:
            cache_key = (str(operation.case_dir), str(operation.operation_id))
            if cache_key in time_bounds_cache:
                return time_bounds_cache[cache_key]

            for friendly in ("vibration", "machine_state", "axis_power", "energy"):
                file_path = operation.channel_files.get(friendly)
                if file_path is None:
                    continue
                try:
                    ts_frame = pd.read_csv(file_path, usecols=["timestamp"])
                except Exception:
                    continue
                parsed = pd.to_datetime(ts_frame.get("timestamp"), utc=True, errors="coerce").dropna()
                if parsed.empty:
                    continue
                bounds = (parsed.min(), parsed.max())
                time_bounds_cache[cache_key] = bounds
                return bounds

            time_bounds_cache[cache_key] = None
            return None

        resolved_case_dirs = 0
        unresolved_samples: list[str] = []
        for idx, sample in samples.loc[missing_case_mask, ["sample_id", "operation_id", "event_timestamp"]].iterrows():
            operation_id = str(sample["operation_id"])
            candidates = operations_by_id.get(operation_id, [])
            if len(candidates) == 1:
                samples.at[idx, "case_dir"] = str(candidates[0].case_dir)
                resolved_case_dirs += 1
                continue

            matching_candidates = []
            event_timestamp = sample["event_timestamp"]
            if pd.notna(event_timestamp):
                for operation in candidates:
                    bounds = _operation_time_bounds(operation)
                    if bounds is None:
                        continue
                    if bounds[0] <= event_timestamp <= bounds[1]:
                        matching_candidates.append(operation)

            if len(matching_candidates) == 1:
                samples.at[idx, "case_dir"] = str(matching_candidates[0].case_dir)
                resolved_case_dirs += 1
                continue

            candidate_cases = sorted({str(operation.case_dir) for operation in candidates})
            sample_id = str(sample["sample_id"])
            if not candidates:
                unresolved_samples.append(f"{sample_id}:{operation_id}:missing_raw_operation")
            elif len(matching_candidates) > 1:
                unresolved_samples.append(
                    f"{sample_id}:{operation_id}:ambiguous_timestamp:{candidate_cases}"
                )
            else:
                unresolved_samples.append(
                    f"{sample_id}:{operation_id}:missing_case_dir:{candidate_cases}"
                )

        if unresolved_samples:
            preview = ", ".join(unresolved_samples[:5])
            raise ValueError(
                "pair_casedata feature rows require case_dir when operation_id is duplicated across case studies. "
                f"Could not resolve {len(unresolved_samples)} sample(s) from {feature_path}: {preview}"
            )

        if resolved_case_dirs > 0:
            logger.info(
                "Resolved %d pair_casedata sample case_dir values from raw casedata timestamps",
                resolved_case_dirs,
            )

    joined_cache: dict[tuple[Optional[str], str], Any] = {}
    windows = []
    skipped = 0

    def _load_joined_operation(case_dir: Optional[str], operation_id: str) -> Any:
        cache_key = (case_dir, operation_id)
        if cache_key in joined_cache:
            return joined_cache[cache_key]

        op = loader.get_operation(operation_id, case=case_dir) if case_dir else loader.get_operation(operation_id)
        vib_path = op.channel_files.get("vibration")
        ms_path = op.channel_files.get("machine_state")
        if vib_path is None or ms_path is None:
            raise ValueError(f"Operation {operation_id} is missing vibration or machine_state CSVs")

        vib_header = pd.read_csv(vib_path, nrows=0)
        peak_cols = [
            str(col)
            for col in vib_header.columns
            if str(col).startswith("Vibration_Peak_")
            and (str(col).endswith("_Amplitude") or str(col).endswith("_Frequency"))
        ]
        if not peak_cols:
            raise ValueError(f"Operation {operation_id} has no Vibration_Peak_* columns")

        vib = pd.read_csv(vib_path, usecols=["timestamp", *peak_cols])
        ms = pd.read_csv(
            ms_path,
            usecols=["timestamp", "Spindle_Speed_Actual", "Feed_Rate_Actual", "Tool_Number"],
        )
        vib["timestamp"] = pd.to_datetime(vib["timestamp"], utc=True, errors="coerce")
        ms["timestamp"] = pd.to_datetime(ms["timestamp"], utc=True, errors="coerce")
        vib = vib.dropna(subset=["timestamp"]).sort_values("timestamp")
        ms = ms.dropna(subset=["timestamp"]).sort_values("timestamp")

        merged = pd.merge_asof(
            vib,
            ms.rename(columns={"timestamp": "machine_state_timestamp"}),
            left_on="timestamp",
            right_on="machine_state_timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=120),
        )
        merged = merged.dropna(subset=["machine_state_timestamp"]).copy()
        merged["spindle_speed_mean"] = pd.to_numeric(merged["Spindle_Speed_Actual"], errors="coerce")
        merged["feed_rate_mean"] = pd.to_numeric(merged["Feed_Rate_Actual"], errors="coerce")
        machine_id = str(getattr(op, "case_dir", "") or case_dir or "").strip()
        machine_family = resolve_machine_family(machine_id) if machine_id else ""
        merged["machine_family"] = machine_family or pd.NA
        merged["tool_number"] = pd.to_numeric(merged["Tool_Number"], errors="coerce")
        merged["tool_id"] = pd.NA
        merged["tool_diameter"] = float("nan")
        merged["num_teeth"] = float("nan")
        if machine_family:
            for raw_tool_number in sorted({int(value) for value in merged["tool_number"].dropna().tolist()}):
                tool_context = resolve_tool_context(
                    machine_family,
                    raw_tool_number,
                    machine_id=machine_id or None,
                )
                if not isinstance(tool_context, dict):
                    continue
                tool_mask = merged["tool_number"] == raw_tool_number
                tool_id = tool_context.get("tool_id")
                if tool_id is not None:
                    merged.loc[tool_mask, "tool_id"] = str(tool_id)
                tool_diameter = tool_context.get("tool_diameter")
                if tool_diameter is not None:
                    merged.loc[tool_mask, "tool_diameter"] = float(tool_diameter)
                num_teeth = tool_context.get("num_teeth")
                if num_teeth is not None:
                    merged.loc[tool_mask, "num_teeth"] = float(num_teeth)
        merged["feed_per_tooth"] = float("nan")
        valid_feed_mask = (
            pd.to_numeric(merged["num_teeth"], errors="coerce").gt(0)
            & merged["spindle_speed_mean"].gt(0)
            & merged["feed_rate_mean"].notna()
        )
        merged.loc[valid_feed_mask, "feed_per_tooth"] = (
            merged.loc[valid_feed_mask, "feed_rate_mean"]
            / (
                merged.loc[valid_feed_mask, "num_teeth"]
                * merged.loc[valid_feed_mask, "spindle_speed_mean"]
            )
        )
        joined_cache[cache_key] = merged
        return merged

    for sample in samples.itertuples(index=False):
        case_dir = str(sample.case_dir) if hasattr(sample, "case_dir") and pd.notna(sample.case_dir) else None
        operation_id = str(sample.operation_id)
        sample_id = str(sample.sample_id)
        label = str(sample.label)

        try:
            op_df = _load_joined_operation(case_dir, operation_id)
        except Exception as exc:
            skipped += 1
            logger.debug(
                "Skipping pair_casedata sample %s (%s/%s): %s",
                sample_id,
                case_dir,
                operation_id,
                exc,
            )
            continue

        t_end = sample.event_timestamp
        if label in active_positive_labels:
            t_end = t_end - pd.Timedelta(seconds=float(sample.gap_seconds))
        t_start = t_end - pd.Timedelta(seconds=float(sample.window_seconds))

        window = op_df[(op_df["timestamp"] >= t_start) & (op_df["timestamp"] < t_end)].copy()
        trim_seconds = float(sample.trim_seconds_removed)
        if trim_seconds > 0.0:
            trimmed_end = t_end - pd.Timedelta(seconds=trim_seconds)
            window = window[window["timestamp"] < trimmed_end].copy()
        if window.empty:
            skipped += 1
            continue

        window["operation_id"] = sample_id
        window["raw_operation_id"] = operation_id
        window["label"] = label
        window["sample_id"] = sample_id
        window["window_seconds"] = float(sample.window_seconds)
        window["gap_seconds"] = float(sample.gap_seconds)
        window["trim_seconds_removed"] = trim_seconds
        window["event_timestamp"] = sample.event_timestamp
        if case_dir is not None:
            window["case_dir"] = case_dir
        windows.append(window)

    if not windows:
        logger.warning("No raw pair_casedata windows could be reconstructed from %s", feature_path)
        return pd.DataFrame()

    df = pd.concat(windows, ignore_index=True, sort=False)
    logger.info(
        "Reconstructed pair_casedata rows: %d across %d labelled windows (%d skipped) from %s",
        len(df),
        df["operation_id"].nunique() if "operation_id" in df.columns else 0,
        skipped,
        feature_path,
    )
    return df


def _load_site_a_line2(data_dir: Optional[str] = None) -> Any:
    """Load Site_a_line2 features."""
    import pandas as pd

    candidates = [
        Path(data_dir) / "site_a_line2_features.csv" if data_dir else None,
        _FEATURES_DIR / "site_a_line2_features.csv",
        _SITE_A_LINE2_DIR / "site_a_line2_features.csv",
    ]
    for p in candidates:
        if p and p.exists():
            logger.info("Loading Site_a_line2 features from %s", p)
            return pd.read_csv(p)

    # No precomputed feature table found. Dataset-specific loaders are not
    # part of this distribution — point ``data_dir`` at a prepared feature
    # CSV, or add an adapter alongside ``agents/processing/dataset_loader.py``.
    logger.warning(
        "No harmonic feature table found under %s — returning empty frame",
        data_dir or _SITE_A_LINE2_DIR,
    )
    return pd.DataFrame()


def _load_pair_raw(data_dir: Optional[str] = None) -> Any:
    """Load labelled FFT peak parquet files for pair-input training."""
    import pandas as pd

    root = Path(data_dir) if data_dir else _PAIR_RAW_DIR
    parquet_files = []
    if root.is_file() and root.suffix.lower() == ".parquet":
        parquet_files = [root]
    elif root.is_dir():
        parquet_files = sorted(root.rglob("*.parquet"))

    if not parquet_files:
        logger.warning("No pair_raw parquet files found in %s", root)
        return pd.DataFrame()

    allowed_labels = {"normal", "pre_break", "break_event"}
    frames = []
    for path in parquet_files:
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Could not read pair_raw parquet %s: %s", path, exc)
            continue

        if "label" not in frame.columns:
            inferred_label = None
            stem = path.stem.lower()
            if "pre_break" in stem:
                inferred_label = "pre_break"
            elif "normal" in stem:
                inferred_label = "normal"
            elif "break" in stem:
                inferred_label = "break_event"
            if inferred_label is None:
                logger.debug("Skipping unlabeled pair_raw parquet %s", path)
                continue
            frame = frame.copy()
            frame["label"] = inferred_label

        frame = frame[frame["label"].isin(allowed_labels)].copy()
        if frame.empty:
            continue

        frame["source_file"] = str(path)
        if "operation_id" not in frame.columns:
            if {"session", "engagement_idx"}.issubset(frame.columns):
                frame["operation_id"] = (
                    frame["session"].astype(str)
                    + ":"
                    + frame["engagement_idx"].astype(str)
                    + ":"
                    + frame["label"].astype(str)
                )
            else:
                frame["operation_id"] = path.stem + ":" + frame["label"].astype(str)
        frames.append(frame)

    if not frames:
        logger.warning("No usable pair_raw rows found in %s", root)
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded pair_raw parquet rows: %d from %d files", len(df), len(frames))
    return df


def _load_stoppage_1hz(data_dir: Optional[str] = None) -> Any:
    """Load 1 Hz stoppage-series NPZ and flatten it into a training DataFrame."""
    import numpy as np
    import pandas as pd

    candidates = []
    if data_dir:
        supplied = Path(data_dir)
        if supplied.is_file() and supplied.suffix.lower() == ".npz":
            candidates.append(supplied)
        elif supplied.is_dir():
            candidates.extend(
                supplied / name
                for name in ("stoppage_raw_series.npz", "breakage_raw_series.npz")
            )
    candidates.extend([
        _FEATURES_DIR / "stoppage_raw_series.npz",
        _FEATURES_DIR / "breakage_raw_series.npz",
    ])

    npz_path = next((path for path in candidates if path.exists()), None)
    if npz_path is None:
        logger.warning("No stoppage_1hz NPZ found in candidates=%s", [str(c) for c in candidates])
        return pd.DataFrame()

    try:
        with np.load(npz_path, allow_pickle=True) as npz:
            data = np.asarray(npz["data"], dtype=np.float32)
            sample_ids = npz["sample_ids"] if "sample_ids" in npz else np.arange(data.shape[0])
            labels = npz["labels"] if "labels" in npz else np.array(["unknown"] * data.shape[0])
            channel_names = [str(name) for name in npz["channel_names"]]
    except Exception as exc:
        logger.warning("Could not load stoppage_1hz NPZ %s: %s", npz_path, exc)
        return pd.DataFrame()

    if data.ndim != 3:
        logger.warning("Unexpected stoppage_1hz data shape in %s: %s", npz_path, data.shape)
        return pd.DataFrame()

    rows = []
    for sample_idx, sample_id in enumerate(sample_ids):
        label = str(labels[sample_idx]) if sample_idx < len(labels) else "unknown"
        for timestep in range(data.shape[2]):
            row = {
                channel_names[ch_idx]: float(data[sample_idx, ch_idx, timestep])
                for ch_idx in range(min(len(channel_names), data.shape[1]))
                if np.isfinite(data[sample_idx, ch_idx, timestep])
            }
            row["operation_id"] = str(sample_id)
            row["label"] = label
            row["timestep"] = int(timestep)
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("Loaded stoppage_1hz rows: %d from %s", len(df), npz_path)
    return df


def _load_csv(path: str) -> Any:
    """Generic CSV loader."""
    import pandas as pd
    p = Path(path)
    if p.is_file():
        return pd.read_csv(p)
    # If directory, load all CSVs and concat
    if p.is_dir():
        dfs = [pd.read_csv(f) for f in sorted(p.glob("*.csv"))]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return pd.DataFrame()

"""Phase 2 (Test) and Phase 3 (Eval) — run events through the pipeline.

Methodology fixes applied (cf. LOG_EXPERIMENT_METHODOLOGY_FIXES.md):
  - Production-aligned 4-rule scoring with weighted average + multi-rule bonus
  - Data-driven pattern calibration (p95 of normal training set)
  - MULTIPLICATIVE prior boost (not additive) for real score leverage
  - Online model retraining from accumulated feedback
  - Missed-event feedback path to avoid confirmation bias
  - Rolling z-score anomaly deviation rule
  - Counterfactual on ALL eval samples (not just flagged)
  - Threshold adaptation from accumulated feedback
  - Temporal ordering enforced
  - Prediction-flip tracking for meaningful feedback impact measurement
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PATTERN_KEYS, FAULT_SEVERITY, LEAKY_COLUMNS, METADATA_COLUMNS, ExperimentConfig
from .pattern_registry import get_registry as _get_pattern_registry

# Pattern discovery engine — learns new patterns from confirmed events
from backend.agents.patterns.discovery import PatternDiscovery

# KG / SINDIT integration
from backend.agents.storage.store import MemoryStore
from backend.agents.storage.pattern_index import PatternIndex
from backend.agents.core.schemas import Memory, PatternKey as SchemaPatternKey
from backend.agents.core.context import CuttingContext

# Unified scoring — same scorer as the live backend pipeline
from backend.agents.memory.scorer import SignificanceScorer, SignificanceConfig, SignificanceResult
from backend.agents.memory.prior_math import prior_from_counts

# API-mode client for routing events through the live backend
from .api_client import ExperimentAPIClient

logger = logging.getLogger(__name__)


# =====================================================================
# Data structures
# =====================================================================


@dataclass
class SampleResult:
    """Per-sample outcome from the evaluation loop."""

    sample_id: str = ""
    label: str = ""  # ground-truth: "pre_stoppage" | "normal"
    operation_id: str = ""
    tool_number: str = ""

    # Pipeline output
    significance_score: float = 0.0
    action: str = ""  # IGNORE / STORE / ALERT / CRITICAL
    predicted_positive: bool = False  # score >= calibrated threshold
    memory_id: Optional[str] = None

    # Scoring breakdown (for diagnostics)
    raw_model_score: float = 0.0
    pattern_rule_score: float = 0.0
    anomaly_z_score: float = 0.0
    prior_boost: float = 0.0
    prior_multiplier: float = 1.0
    multi_rule_bonus: float = 0.0
    n_rules_triggered: int = 0
    detected_patterns: List[str] = field(default_factory=list)

    # Supervised model output
    supervised_score: float = 0.0  # RF predicted probability
    unsupervised_score: float = 0.0  # 4-rule composite
    combined_score: float = 0.0  # weighted combination of both
    weight_supervised: float = 0.0  # current supervised weight
    weight_unsupervised: float = 0.0  # current unsupervised weight

    # Tool context
    tool_prior: float = 0.5  # tool-specific stop-rate prior
    tool_multiplier: float = 1.0  # tool prior boost factor

    # Feedback (Phase 3 only)
    feedback_given: bool = False
    feedback_action: str = ""  # CONFIRM / DISMISS / ""
    feedback_source: str = ""  # "flagged" | "missed_event" | ""
    model_retrained: bool = False  # was the model retrained after this sample?

    # Unified scorer — score from the SAME SignificanceScorer used in the live pipeline
    unified_score: Optional[float] = None
    unified_action: str = ""  # action from unified scorer
    unified_triggered_rules: int = 0  # number of rules triggered by unified scorer
    score_trace: List[Dict[str, Any]] = field(default_factory=list)

    # API-mode: authoritative score from the backend's production scorer
    api_score: Optional[float] = None
    api_action: str = ""  # action returned by the API orchestrator

    # Counterfactual (eval only): score with initial priors (before any feedback)
    counterfactual_score: Optional[float] = None
    # Did feedback change the binary prediction for this sample?
    prediction_flipped: bool = False

    # Prior snapshot after this sample
    prior_snapshot: Dict[str, float] = field(default_factory=dict)

    # LLM explanation (from API-mode orchestrator)
    explanation: Optional[str] = None
    explanation_source: Optional[str] = None  # "llm" | "fallback" | None
    alert_line: Optional[str] = None
    alert_line_source: Optional[str] = None  # "llm" | "fallback" | None

    # Knowledge Graph integration
    stored_in_memory: bool = False
    co_occurring_pairs: List[Tuple[str, str]] = field(default_factory=list)
    propagated_prior_deltas: Dict[str, float] = field(default_factory=dict)
    sindit_context: Dict[str, Any] = field(default_factory=dict)
    model_breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseResult:
    """Aggregated output of one evaluation phase."""

    phase: str = ""  # "test" | "eval"
    operation: str = ""
    n_samples: int = 0

    # X7: number of samples where the unified scorer threw and a crude fallback
    # score was substituted. Non-zero means the reported metrics are partly
    # built on fallback scores — surfaced in the summary and to_dict().
    n_scorer_fallbacks: int = 0

    sample_results: List[SampleResult] = field(default_factory=list)

    # Score distributions
    scores_positive: List[float] = field(default_factory=list)
    scores_negative: List[float] = field(default_factory=list)

    # Prior history for learning curve
    prior_history: List[Dict[str, float]] = field(default_factory=list)

    # Calibration threshold (from training)
    threshold: float = 0.5
    # Adapted threshold (after feedback in eval phase)
    adapted_threshold: float = 0.5

    # Timing
    duration_s: float = 0.0

    # Prediction flip tracking
    n_predictions_flipped: int = 0

    # Online retraining tracking
    n_model_retrains: int = 0

    # Model weight history (supervised vs unsupervised over time)
    weight_history: List[Dict[str, float]] = field(default_factory=list)

    # Tool prior history (tool priors over time)
    tool_prior_history: List[Dict[str, float]] = field(default_factory=list)

    # Knowledge Graph tracking
    co_occurrence_graph: Dict[str, int] = field(default_factory=dict)
    stored_memories_count: int = 0
    sindit_context_summary: Dict[str, Any] = field(default_factory=dict)
    all_propagated_deltas: List[Dict[str, float]] = field(default_factory=list)

    # Feedback audit trail
    feedback_events: List[Dict[str, Any]] = field(default_factory=list)

    # Pattern discovery tracking
    n_discovered_patterns: int = 0
    n_suppression_patterns: int = 0
    discovered_pattern_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        # Build prior_history: pattern → list of prior values over time
        prior_history: Dict[str, List[float]] = {}
        for snap in self.prior_history:
            for k, v in snap.items():
                prior_history.setdefault(k, []).append(v)

        # Raw score lists (filter NaN for safe JSON serialisation)
        scores_pos = [float(s) for s in self.scores_positive if np.isfinite(float(s))]
        scores_neg = [float(s) for s in self.scores_negative if np.isfinite(float(s))]

        return {
            "phase": self.phase,
            "operation": self.operation,
            "n_samples": self.n_samples,
            "n_scorer_fallbacks": self.n_scorer_fallbacks,
            "threshold": self.threshold,
            "adapted_threshold": self.adapted_threshold,
            "duration_s": self.duration_s,
            "score_stats_positive": _arr_stats(self.scores_positive),
            "score_stats_negative": _arr_stats(self.scores_negative),
            "scores_positive": scores_pos,
            "scores_negative": scores_neg,
            "n_positive": len(self.scores_positive),
            "n_negative": len(self.scores_negative),
            "sample_count": len(self.sample_results),
            "n_predictions_flipped": self.n_predictions_flipped,
            "n_model_retrains": self.n_model_retrains,
            "has_supervised_model": len(self.weight_history) > 0,
            "has_tool_priors": len(self.tool_prior_history) > 0,
            # ── Learning curves & evolution ──
            "weight_history": self.weight_history,
            "tool_prior_history": self.tool_prior_history,
            "prior_history": prior_history,
            # ── Knowledge graph ──
            "co_occurrence_graph": self.co_occurrence_graph,
            "stored_memories_count": self.stored_memories_count,
            "sindit_context_summary": self.sindit_context_summary,
            "n_propagation_events": len(self.all_propagated_deltas),
            "all_propagated_deltas": self.all_propagated_deltas,
            "feedback_events": self.feedback_events,
            "pattern_feedback_summary": _summarize_feedback_events(self.feedback_events),
            # ── Pattern discovery ──
            "n_discovered_patterns": self.n_discovered_patterns,
            "n_suppression_patterns": self.n_suppression_patterns,
            "discovered_pattern_keys": self.discovered_pattern_keys,
            # ── Per-sample results (full scoring breakdown) ──
            "sample_results": [
                {
                    "sample_id": sr.sample_id,
                    "label": sr.label,
                    "operation_id": sr.operation_id,
                    "tool_number": sr.tool_number,
                    "significance_score": round(sr.significance_score, 4),
                    "action": sr.action,
                    "predicted_positive": sr.predicted_positive,
                    "memory_id": sr.memory_id,
                    "raw_model_score": round(sr.raw_model_score, 4),
                    "pattern_rule_score": round(sr.pattern_rule_score, 4),
                    "anomaly_z_score": round(sr.anomaly_z_score, 4),
                    "prior_boost": round(sr.prior_boost, 4),
                    "prior_multiplier": round(sr.prior_multiplier, 4),
                    "multi_rule_bonus": round(sr.multi_rule_bonus, 4),
                    "n_rules_triggered": sr.n_rules_triggered,
                    "detected_patterns": sr.detected_patterns,
                    "supervised_score": round(sr.supervised_score, 4),
                    "unsupervised_score": round(sr.unsupervised_score, 4),
                    "combined_score": round(sr.combined_score, 4),
                    "tool_prior": round(sr.tool_prior, 4),
                    "tool_multiplier": round(sr.tool_multiplier, 4),
                    "unified_score": round(sr.unified_score, 4) if sr.unified_score is not None else None,
                    "unified_action": sr.unified_action,
                    "unified_triggered_rules": sr.unified_triggered_rules,
                    "score_trace": sr.score_trace,
                    "feedback_given": sr.feedback_given,
                    "feedback_action": sr.feedback_action,
                    "feedback_source": sr.feedback_source,
                    "model_retrained": sr.model_retrained,
                    "counterfactual_score": round(sr.counterfactual_score, 4) if sr.counterfactual_score is not None else None,
                    "prediction_flipped": sr.prediction_flipped,
                    "prior_snapshot": {
                        k: round(float(v), 4) for k, v in sr.prior_snapshot.items()
                    },
                    "model_breakdown": sr.model_breakdown,
                    "explanation": sr.explanation,
                    "explanation_source": sr.explanation_source,
                    "alert_line": sr.alert_line,
                    "alert_line_source": sr.alert_line_source,
                    "stored_in_memory": sr.stored_in_memory,
                    "co_occurring_pairs": [list(pair) for pair in sr.co_occurring_pairs],
                    "propagated_prior_deltas": {
                        k: round(float(v), 4) for k, v in sr.propagated_prior_deltas.items()
                    },
                    "sindit_context": sr.sindit_context,
                }
                for sr in self.sample_results
            ],
        }


def _arr_stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    a = np.array(vals)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "median": float(np.median(a)),
    }


def _summarize_feedback_events(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate a per-pattern summary from the feedback audit trail."""
    summary: Dict[str, Dict[str, Any]] = {}
    for event in events:
        action = str(event.get("feedback_action") or "")
        for update in event.get("pattern_updates", []):
            if not isinstance(update, dict):
                continue
            pattern_key = update.get("pattern_key")
            if not isinstance(pattern_key, str) or not pattern_key:
                continue
            bucket = summary.setdefault(pattern_key, {
                "polarity": update.get("polarity"),
                "n_feedback_events": 0,
                "n_confirms": 0,
                "n_dismissals": 0,
                "total_prior_delta": 0.0,
                "mean_prior_delta": 0.0,
                "max_abs_prior_delta": 0.0,
                "last_prior": round(float(update.get("new_prior", 0.0)), 4),
            })
            delta = float(update.get("delta", 0.0) or 0.0)
            bucket["n_feedback_events"] += 1
            if action == "CONFIRM":
                bucket["n_confirms"] += 1
            elif action == "DISMISS":
                bucket["n_dismissals"] += 1
            bucket["total_prior_delta"] = round(float(bucket["total_prior_delta"]) + delta, 4)
            bucket["max_abs_prior_delta"] = round(max(float(bucket["max_abs_prior_delta"]), abs(delta)), 4)
            bucket["last_prior"] = round(float(update.get("new_prior", bucket["last_prior"])), 4)
            if update.get("polarity"):
                bucket["polarity"] = update.get("polarity")

    for bucket in summary.values():
        n_events = max(1, int(bucket["n_feedback_events"]))
        bucket["mean_prior_delta"] = round(float(bucket["total_prior_delta"]) / n_events, 4)
    return summary


# =====================================================================
# Production-aligned scoring engine
# =====================================================================


def _detect_patterns(row: pd.Series, config: ExperimentConfig) -> List[str]:
    """Detect patterns using the extensible PatternRegistry.

    Delegates to the central pattern registry, which runs all enabled
    detectors (built-in + domain + time-series-derived).  Calibrated
    thresholds from config are passed through so registry detectors
    that need them can use them.
    """
    registry = _get_pattern_registry()

    # Build a flat features dict from the pandas row
    features: Dict[str, float] = {}
    for col in row.index:
        try:
            v = row[col]
            if pd.notna(v):
                features[col] = float(v)
        except (ValueError, TypeError):
            pass

    # Build thresholds dict from config (pattern_* attributes)
    thresholds: Dict[str, Any] = {
        attr: getattr(config, attr)
        for attr in dir(config)
        if attr.startswith("pattern_") and not callable(getattr(config, attr))
    }

    # Run all enabled detectors
    patterns = registry.detect_all(features, thresholds)

    return patterns


def _safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    """Safely extract a float from a row, returning default if missing/NaN."""
    try:
        val = row.get(col, default)
        if pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


# =====================================================================
# Vectorized pattern detection — batch version of _detect_patterns
# =====================================================================

def _detect_patterns_batch(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> List[List[str]]:
    """Detect patterns for all rows at once using vectorized comparisons.

    Returns a list of pattern-key lists, one per row — same semantics as
    calling ``_detect_patterns(row, config)`` per row, but 2-5× faster
    because threshold comparisons are vectorized across the full DataFrame.
    """
    registry = _get_pattern_registry()
    n = len(df)

    # Build thresholds dict from config (pattern_* attributes)
    thresholds: Dict[str, Any] = {
        attr: getattr(config, attr)
        for attr in dir(config)
        if attr.startswith("pattern_") and not callable(getattr(config, attr))
    }

    # Pre-compute numeric columns once
    numeric_cols = {}
    for col in df.columns:
        try:
            numeric_cols[col] = pd.to_numeric(df[col], errors="coerce").values
        except Exception:
            pass

    # Run each detector vectorised across all rows
    # detector(features, thresholds) → bool  per row
    pattern_defs = registry.list_patterns()
    # pattern_masks: pattern_key → bool array (N,)
    pattern_masks: Dict[str, np.ndarray] = {}

    for pdef in pattern_defs:
        if not pdef.enabled:
            continue
        # Build per-row features dicts would be slow; instead call detector
        # once per row but catch fast-path: most detectors check 1-2 columns.
        # We'll attempt a vectorized fast-path for column-threshold detectors,
        # falling back to per-row for complex detectors.
        mask = np.zeros(n, dtype=bool)
        try:
            # Try calling detector with a probe to see if it's simple
            # For the common case, just iterate and call (still faster than
            # iterator + all the other per-row work that's now batched)
            for i in range(n):
                features_i = {col: float(vals[i]) for col, vals in numeric_cols.items()
                              if not np.isnan(vals[i])}
                try:
                    mask[i] = pdef.detector(features_i, thresholds)
                except Exception:
                    pass
        except Exception:
            pass
        pattern_masks[pdef.name] = mask

    # Build per-row pattern key lists
    result: List[List[str]] = [[] for _ in range(n)]
    for pk, mask in pattern_masks.items():
        for i in np.where(mask)[0]:
            result[i].append(pk)

    return result


# =====================================================================
# Unified scoring bridge — calls the SAME scorer used in the live pipeline
# =====================================================================

def _create_unified_scorer(config: ExperimentConfig, priors_path: Optional[str] = None) -> SignificanceScorer:
    """Create a SignificanceScorer with weights matching the experiment config."""
    sig_config = SignificanceConfig(
        weight_classical_alert=config.weight_classical_alert,
        weight_pattern_rule=config.weight_pattern_rule,
        weight_protective_pattern=getattr(config, "weight_protective_pattern", 0.20),
        weight_anomaly_deviation=config.weight_anomaly_deviation,
        weight_historical_prior=config.weight_historical_prior if hasattr(config, 'weight_historical_prior') else 0.30,
        store_threshold=config.store_threshold,
        alert_threshold=config.alert_threshold,
        critical_threshold=config.critical_threshold,
        anomaly_z_threshold=config.anomaly_z_threshold,
        enabled_rules=getattr(config, "enabled_rules", None),
    )
    return SignificanceScorer(config=sig_config, priors_path=priors_path)


def _build_cutting_context(
    operation_id: str,
    tool_number: str,
) -> CuttingContext:
    """Construct a CuttingContext for an experiment sample so the scorer is
    NOT context-blind.

    Mirrors the breakage path (``BreakageFeatureExtractor.row_to_event``):
    a constant 5-axis machine type plus a per-tool ``tool_type``/``tool_id``,
    with the operation carried in ``extra`` for traceability and warm-transfer
    / MaaS export. The constant ``machine_type`` is the least-specific key, so
    seeded feedback resolves there (preserving aggregate-prior numerics) while
    the per-tool key activates the hierarchical context path.
    """
    tn = str(tool_number or "").strip()
    if tn.endswith(".0"):
        tn = tn[:-2]
    tool_label = f"T{tn}" if tn and tn.lower() not in ("", "nan", "none") else None
    extra: Dict[str, Any] = {}
    op = str(operation_id or "").strip()
    if op:
        extra["operation_id"] = op
    return CuttingContext(
        machine_type="CNC_5axis",
        tool_id=tool_label,
        tool_type=tool_label,
        extra=extra,
    )


def _score_via_unified_scorer(
    scorer: SignificanceScorer,
    pattern_keys: List[str],
    raw_model_score: float,
    anomaly_z: float,
    model_confidence: float = 1.0,
    pattern_priors: Optional[Dict[str, float]] = None,
    feedback_counts: Optional[Dict[str, Dict[str, float]]] = None,
    context: Optional[CuttingContext] = None,
    extra_signals: Optional[Dict[str, float]] = None,
) -> SignificanceResult:
    """Bridge from the experiment evaluator's data to the live pipeline scorer.

    This ensures experiment results use the SAME scoring logic as the live
    streaming pipeline, eliminating the dual-world scoring divergence.

    ``context`` is the per-sample CuttingContext. Passing it (rather than
    ``None``) is what makes the experiment context-aware: the scorer resolves
    context-conditioned weight profiles and context-scoped priors instead of
    the global-only path. Seeded feedback is mirrored into the context scope
    (see ``seed_feedback_counts``) so the historical-prior rule still sees the
    accumulated evidence.
    """
    # Convert string pattern keys to PatternKey objects
    patterns = [SchemaPatternKey(key=pk) for pk in pattern_keys]

    # P2: seed the scorer with the ACTUAL confirm/dismiss counts so it derives
    # the prior exactly as it would online AND its evidence-damping sees the
    # true observation volume. (The old code inverted a prior *value* to a fixed
    # 20 notional observations, destroying the real count and diverging from
    # production.)
    #
    # MERGE, don't branch: use exact counts where we have them, and fall back to
    # the value->notional approximation for any pattern that has only a *seeded*
    # prior and no counts (discovered-pattern priors, registry defaults).
    # A plain if/elif would drop those seeded priors to neutral 0.5 the moment
    # any feedback existed anywhere in the phase — a silent regression.
    counts_map = feedback_counts or {}
    priors_map = pattern_priors or {}
    seeded: Dict[str, Dict[str, float]] = {}
    for pk in set(counts_map) | set(priors_map):
        counts = counts_map.get(pk)
        if counts is not None:
            c = float(counts.get("confirm", 0.0))
            d = float(counts.get("dismiss", 0.0))
            if c > 0.0 or d > 0.0:
                seeded[pk] = {"confirm": c, "dismiss": d}
                continue
        prior_val = priors_map.get(pk)
        if prior_val is None:
            continue
        # No real counts: approximate from the seeded prior value at a low
        # notional weight so the prior nudges rather than dominates.
        notional_total = 20
        c_i = max(0, int(float(prior_val) * (notional_total + 2) - 1))
        d_i = notional_total - c_i
        seeded[pk] = {"confirm": float(max(0, c_i)), "dismiss": float(max(0, d_i))}
    if seeded:
        scorer.seed_feedback_counts(seeded, context=context)

    # Pass the model score and confidence as external signals
    external_signals = {
        "anomaly_detector_score": raw_model_score,
        "model_confidence": model_confidence,
        "anomaly_z": anomaly_z,
    }
    if extra_signals:
        external_signals.update(extra_signals)

    return scorer.score(
        patterns=patterns,
        metrics=None,
        context=context,
        session_id=None,
        external_signals=external_signals,
    )


# =====================================================================
# Legacy experiment scoring (DEPRECATED – P5)
# Retained for counterfactual comparisons only. The unified scorer
# (_score_via_unified_scorer) is now the primary evaluation path.
# =====================================================================

def _compute_significance_score(
    raw_model_score: float,
    pattern_keys: List[str],
    anomaly_z: float,
    pattern_priors: Dict[str, float],
    config: ExperimentConfig,
    model_confidence: float = 1.0,
) -> Tuple[float, float, float, float, float, int]:
    """Composite scoring with MULTIPLICATIVE prior boost.

    Formula:
        1. Evaluate 3 non-historical rules independently.
        2. base_score = weighted_average(triggered non-historical rules).
        3. multi_rule_bonus = min(0.2, 0.05 * n_triggered).
        4. prior_multiplier = 1.0 + prior_boost_weight * (max_prior - 0.5)
           - With weight=2.0, prior=0.8 -> multiplier=1.6x (60% boost)
           - With weight=2.0, prior=0.3 -> multiplier=0.6x (40% penalty)
        5. final = min(1.0, base_score * prior_multiplier + multi_rule_bonus)

    Returns (score, pattern_rule_score, prior_boost, prior_multiplier,
             multi_rule_bonus, n_rules_triggered).
    """
    triggered_scores: List[float] = []
    triggered_weights: List[float] = []

    # Rule 1: Classical alert — fires if model score > 0.7 (production threshold)
    if raw_model_score > 0.7:
        scaled = raw_model_score * model_confidence
        triggered_scores.append(scaled)
        triggered_weights.append(config.weight_classical_alert)

    # Rule 2: Pattern match — fires if any pattern detected, score from FAULT_SEVERITY
    pattern_rule_score = 0.0
    if pattern_keys:
        sev_values = [FAULT_SEVERITY.get(pk, 0.7) for pk in pattern_keys]
        pattern_rule_score = max(sev_values) if sev_values else 0.0
        triggered_scores.append(pattern_rule_score)
        triggered_weights.append(config.weight_pattern_rule)

    # Rule 3: Anomaly deviation — fires if z-score > anomaly_z_threshold
    if anomaly_z > config.anomaly_z_threshold:
        anom_score = min(1.0, anomaly_z / (config.anomaly_z_threshold * 2.0))
        triggered_scores.append(anom_score)
        triggered_weights.append(config.weight_anomaly_deviation)

    n_rules_triggered = len(triggered_scores)

    # Base score: weighted average of triggered non-historical rules
    if triggered_scores:
        base_score = sum(s * w for s, w in zip(triggered_scores, triggered_weights)) / sum(triggered_weights)
    else:
        base_score = 0.0

    # Multi-rule bonus
    multi_rule_bonus = min(0.2, 0.05 * n_rules_triggered)

    # Rule 4: Historical prior — MULTIPLICATIVE boost on base_score
    # This gives priors real leverage: a high prior (0.8) amplifies the
    # base score by 1.6x, while a low prior (0.3) suppresses it to 0.6x.
    prior_values = [pattern_priors.get(pk, 0.5) for pk in PATTERN_KEYS]
    max_prior = max(prior_values) if prior_values else 0.5
    prior_multiplier = 1.0 + config.prior_boost_weight * (max_prior - 0.5)
    prior_multiplier = max(0.1, prior_multiplier)  # floor: never fully zero out
    prior_boost = base_score * (prior_multiplier - 1.0)  # diagnostic: how much prior added/removed

    final = min(1.0, base_score * prior_multiplier + multi_rule_bonus)
    final = max(0.0, final)

    return final, pattern_rule_score, prior_boost, prior_multiplier, multi_rule_bonus, n_rules_triggered


# =====================================================================
# SINDIT context simulation & co-occurrence propagation
# =====================================================================


def _simulate_sindit_context(row: pd.Series) -> Dict[str, Any]:
    """Simulate SINDIT digital-twin context from CSV sensor columns.

    Supports both stoppage (spindle_actual_mean, feed_actual_mean) and
    breakage/Site_a_line2 (spindle_speed_mean, feed_rate_mean) column naming.
    """
    # Try stoppage column names first, then Site_a_line2 alternatives
    spindle = _safe_float(row, "spindle_actual_mean") or _safe_float(row, "spindle_speed_mean")
    feed = _safe_float(row, "feed_actual_mean") or _safe_float(row, "feed_rate_mean")
    fo_mean = _safe_float(row, "feed_override_mean", 100.0)
    power = _safe_float(row, "power_spindle_mean")
    tool_id = str(row.get("tool_id") or row.get("tool_number") or "")
    context = {
        "spindle_speed": spindle,
        "feed_rate": feed,
        "tool_id": tool_id,
        "feed_override": fo_mean,
        "machine_state": "degraded" if fo_mean < 50 else "normal",
        "power_level": power,
    }

    tool_type = str(row.get("tool_type") or "").strip()
    if tool_type:
        context["tool_type"] = tool_type

    machine_id = str(row.get("machine_id") or row.get("session") or "").strip()
    if machine_id:
        context["machine_id"] = machine_id

    machine_family = str(row.get("machine_family") or "").strip()
    if machine_family:
        context["machine_family"] = machine_family

    tool_material = str(row.get("tool_material") or "").strip()
    if tool_material:
        context["tool_material"] = tool_material

    tool_diameter = _safe_float(row, "tool_diameter")
    if tool_diameter > 0:
        context["tool_diameter"] = tool_diameter

    tool_length = _safe_float(row, "tool_length")
    if tool_length > 0:
        context["tool_length"] = tool_length

    num_teeth = _safe_float(row, "num_teeth")
    if num_teeth > 0:
        context["num_teeth"] = int(round(num_teeth))

    extra: Dict[str, Any] = {}
    operation_id = str(row.get("operation_id") or "").strip()
    if operation_id:
        extra["operation_id"] = operation_id
    session = str(row.get("session") or "").strip()
    if session:
        extra["session"] = session
    sindit_tool_iri = str(row.get("sindit_tool_iri") or "").strip()
    if sindit_tool_iri:
        extra["sindit_tool_iri"] = sindit_tool_iri
    if extra:
        context["extra"] = extra

    return context


def _propagate_prior(
    pattern_key: str,
    delta: float,
    co_occurrence: Dict[Tuple[str, str], int],
    pattern_priors: Dict[str, float],
    decay: float = 0.3,
) -> Dict[str, float]:
    """Propagate a prior update along co-occurrence edges.

    Mirrors Neo4jMemoryStore.propagate_prior_update:
      applied_delta = delta * decay * min(weight / 10, 1.0)
    """
    propagated: Dict[str, float] = {}
    for (a, b), weight in co_occurrence.items():
        if a == pattern_key:
            neighbor = b
        elif b == pattern_key:
            neighbor = a
        else:
            continue
        if neighbor in pattern_priors:
            applied = delta * decay * min(weight / 10.0, 1.0)
            pattern_priors[neighbor] = max(0.01, min(0.99, pattern_priors[neighbor] + applied))
            propagated[neighbor] = applied
    return propagated


def _integrate_promoted_discoveries(
    pattern_priors: Dict[str, float],
    feedback_counts: Dict[str, Dict[str, float]],
    discovered: Optional[List[Any]],
    *,
    feedback_action: str,
) -> List[str]:
    """Insert newly promoted discovered patterns into the runtime prior tables.

    The discovery engine can promote both confirmed-event discoveries and
    dismissed-event suppression patterns. The evaluator needs to mirror those
    promotions into its mutable `pattern_priors` / `feedback_counts` state so
    the patterns become directly learnable during the current run.
    """
    inserted: List[str] = []
    if not discovered:
        return inserted

    if feedback_action not in {"CONFIRM", "DISMISS"}:
        raise ValueError(f"Unsupported feedback_action: {feedback_action}")

    for dp in discovered:
        key = getattr(dp, "key", None)
        if not getattr(dp, "promoted", False):
            continue
        if not isinstance(key, str) or not key or key in pattern_priors:
            continue

        pattern_priors[key] = float(getattr(dp, "prior", 0.5))
        confirmation_count = max(0, int(getattr(dp, "confirmation_count", 0) or 0))
        if feedback_action == "CONFIRM":
            feedback_counts[key] = {"confirm": confirmation_count, "dismiss": 0}
        else:
            feedback_counts[key] = {"confirm": 0, "dismiss": confirmation_count}
        inserted.append(key)

    return inserted


# =====================================================================
# Shared evaluation loop
# =====================================================================


def evaluate_phase(
    df: pd.DataFrame,
    config: ExperimentConfig,
    *,
    phase: str,
    feedback_enabled: bool,
    threshold: float = 0.5,
    initial_priors_path: Optional[Path] = None,
    train_normal_features: Optional[np.ndarray] = None,
    progress_callback: Optional[Any] = None,
) -> PhaseResult:
    """Run one evaluation phase (test or eval) through the full pipeline.

    Parameters
    ----------
    df : DataFrame
        Rows to evaluate (sorted by event_timestamp for temporal ordering).
    config : ExperimentConfig
        Experiment configuration.
    phase : str
        "test" or "eval".
    feedback_enabled : bool
        If True, simulate operator feedback after flagged and some missed events.
    threshold : float
        Decision threshold from Phase-1 calibration.
    initial_priors_path : Path or None
        Path to the priors JSON to start from. If None, uses config.baseline_priors_path.
    train_normal_features : ndarray or None
        Original normal training features for online retraining.
        When feedback accumulates, the model is retrained on this augmented
        with dismissed samples (operator-confirmed normals).
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
    logger.info(
        "Phase %s: evaluating %d samples (feedback=%s, threshold=%.3f)",
        phase, len(df), feedback_enabled, threshold,
    )

    # --- Sort by timestamp for temporal ordering ----------------------
    if "event_timestamp" in df.columns:
        df = df.sort_values("event_timestamp").reset_index(drop=True)
        logger.info("Sorted %d samples by event_timestamp", len(df))

    # --- Load model -----------------------------------------------
    # use_seed_model=False removes the one-class anomaly model entirely: it is
    # not loaded or scored, raw_model_score is forced to 0 (so the
    # classical_alert rule is inert), and online retraining is skipped. Useful
    # when the seed model is near-random and hurts the pipeline (see the config
    # exploration), or to run a patterns-only / AAD-only configuration.
    _seed_enabled = getattr(config, "use_seed_model", True)
    model = SeedModel(config=SeedModelConfig(random_state=config.random_seed))
    if _seed_enabled:
        model.load(config.seed_model_path)
        assert model.is_trained, "Model must be pre-trained before evaluation"

    # AAD combiner: a feedback-trained linear model over the fired patterns +
    # anomaly score. Instantiated only when selected, so it costs nothing
    # otherwise. Scored each sample and updated on each feedback event.
    _aad_enabled = bool(config.enabled_rules and "aad_combiner" in config.enabled_rules)
    aad_model = None
    if _aad_enabled:
        from backend.agents.processing.aad_combiner import AADCombiner
        aad_model = AADCombiner()

    # Compute model confidence from training stats
    n_train = model._training_stats.get("n_samples", 0) if hasattr(model, "_training_stats") else 0
    model_confidence = min(1.0, n_train / 500)

    # --- Load supervised model if available -----------------------
    supervised_clf = None
    supervised_feature_cols: List[str] = []
    if config.use_supervised_model and config.supervised_model_path.exists():
        import joblib
        sup_data = joblib.load(config.supervised_model_path)
        supervised_clf = sup_data["model"]
        supervised_feature_cols = sup_data["feature_cols"]
        logger.info(
            "Loaded supervised %s model (%d features)",
            sup_data.get("model_type", "RF"), len(supervised_feature_cols),
        )

    # --- Load tool priors if available ----------------------------
    tool_priors_data: Dict[str, float] = {}
    tool_feedback_counts: Dict[str, Dict[str, int]] = {}
    if config.use_tool_priors and config.tool_priors_path.exists():
        with open(config.tool_priors_path) as f:
            tp_data = json.load(f)
        tool_priors_data = tp_data.get("tool_priors", {})
        tool_feedback_counts = {
            t: dict(c) for t, c in tp_data.get("feedback_counts", {}).items()
        }
        logger.info("Loaded tool priors for %d tools", len(tool_priors_data))
    initial_tool_priors = dict(tool_priors_data)

    # --- Model weight state (supervised vs unsupervised) ----------
    w_sup = config.weight_supervised
    w_unsup = config.weight_unsupervised
    model_weight_shift = 0.0

    # --- Knowledge Graph integration ---------------------------------
    memory_store = None
    pattern_index = None
    api_client: Optional[ExperimentAPIClient] = None
    co_occurrence: Dict[Tuple[str, str], int] = {}

    if config.api_mode:
        # API mode: route all events through the live backend API.
        # Co-occurrence, prior propagation, SINDIT enrichment, and
        # Neo4j persistence are all handled server-side.
        api_client = ExperimentAPIClient(
            config.api_base_url,
            timeout=getattr(config, "api_request_timeout", 120.0),
        )
        if not api_client.check_health():
            if config.api_mode_strict:
                api_client.close()
                raise RuntimeError(
                    f"Backend API at {config.api_base_url} is not reachable while api_mode_strict is enabled"
                )
            logger.warning(
                "Backend API at %s is not reachable — falling back to local store",
                config.api_base_url,
            )
            api_client.close()
            api_client = None
        else:
            logger.info("API mode: events will be routed through %s", config.api_base_url)

    if config.use_memory_store and api_client is None:
        memory_store = MemoryStore(db_path=":memory:", enable_ann=False, enable_embeddings=False)
        pattern_index = PatternIndex()
        logger.info("MemoryStore + PatternIndex initialized (in-memory)")

    # --- Load priors into a mutable copy --------------------------
    priors_path = initial_priors_path or config.baseline_priors_path
    with open(priors_path) as f:
        priors_data = json.load(f)

    pattern_priors = dict(priors_data.get("pattern_priors", {}))
    feedback_counts = {
        pk: dict(counts)
        for pk, counts in priors_data.get("feedback_counts", {}).items()
    }

    # Load calibrated pattern thresholds if present in priors file
    cal_thresholds = priors_data.get("calibrated_pattern_thresholds", {})
    if cal_thresholds:
        from .trainer import _apply_calibrated_thresholds
        _apply_calibrated_thresholds(config, cal_thresholds)

    # --- Unified scorer (same as live backend pipeline) ---------------
    unified_scorer = _create_unified_scorer(config)
    if cal_thresholds:
        logger.info("Applied calibrated pattern thresholds from priors file")

    # Ensure all tracked keys are present
    for pk in PATTERN_KEYS:
        pattern_priors.setdefault(pk, 0.5)
        feedback_counts.setdefault(pk, {"confirm": 0, "dismiss": 0})

    # Save initial priors for counterfactual analysis
    initial_priors = dict(pattern_priors)
    # A1: a SECOND unified scorer used only for the counterfactual. It is seeded
    # with the frozen initial counts on every call and never receives feedback,
    # so the counterfactual differs from the primary path by feedback ALONE —
    # not by a different scoring formula (the old counterfactual used the
    # deprecated legacy scorer, conflating the two). Kept as a separate instance
    # so re-seeding it cannot perturb the live scorer's state.
    cf_scorer = _create_unified_scorer(config)
    initial_feedback_counts = {pk: dict(counts) for pk, counts in feedback_counts.items()}

    # --- Prepare result -------------------------------------------
    result = PhaseResult(
        phase=phase,
        operation=config.test_op if phase == "test" else config.eval_op,
        n_samples=len(df),
        threshold=threshold,
        adapted_threshold=threshold,
    )

    # --- Pattern Discovery engine ---------------------------------
    # Learns new patterns from confirmed events and suppression patterns
    # from dismissed events. Feeds baseline on every sample.
    _discovery_dir = Path(__file__).resolve().parents[2] / "data"
    discovery = PatternDiscovery(data_dir=_discovery_dir)
    # Feed all training-set features into baseline if possible (cold start mitigation)
    if train_normal_features is not None:
        from backend.agents.processing.classical_models import FEATURE_NAMES
        for fv_row in train_normal_features:
            baseline_dict = {name: float(fv_row[i]) for i, name in enumerate(FEATURE_NAMES) if i < len(fv_row)}
            discovery.update_baseline(baseline_dict)

    # Record initial prior snapshot
    result.prior_history.append(dict(pattern_priors))

    # --- Noise / sparsity for variant experiments -----------------
    rng = random.Random(config.random_seed)
    # Legacy noise_rate kept for backward compat; new confidence-dependent
    # noise (noise_rate_base / noise_rate_ambiguity) takes priority when set.
    legacy_noise_rate = config.noise_rate if feedback_enabled else 0.0
    feedback_every_n = config.feedback_every_n if feedback_enabled else 1

    # --- Feedback realism state -----------------------------------
    import heapq as _heapq
    _feedback_delay_q: List[tuple] = []       # min-heap of (apply_at_idx, payload)
    _feedback_delay = config.feedback_delay_samples if feedback_enabled else 0

    # --- Rolling window for anomaly z-score -----------------------
    recent_scores: List[float] = []
    max_history = 100

    # --- Threshold adaptation state --------------------------------
    current_threshold = threshold
    threshold_shift = 0.0
    secondary_threshold = threshold - config.secondary_threshold_offset

    # --- Online retraining state -----------------------------------
    dismissed_features: List[np.ndarray] = []
    confirmed_features: List[np.ndarray] = []
    total_feedback_events = 0
    retrain_count = 0

    # ==================================================================
    # BATCH PRE-COMPUTATION — avoids per-sample iterrows / model calls
    # ==================================================================
    n_total = len(df)

    # (a) Vectorized 28-feature matrix for SeedModel scoring
    _X_all = batch_features_from_df(df, col_map=_COL_MAP)

    # (b) Batch SeedModel scoring (IF + LOF ensemble in one vectorized call)
    # When the seed model is removed, all anomaly scores are 0 (the classical
    # rule then never fires; patterns + feedback + AAD carry the decision).
    _all_model_scores = model.score_batch(_X_all) if _seed_enabled else np.zeros(len(_X_all))

    # (c) Batch pattern detection
    _all_pattern_keys = _detect_patterns_batch(df, config)

    # (d) Batch supervised model scoring (if available)
    _all_sup_scores: Optional[np.ndarray] = None
    if supervised_clf is not None and supervised_feature_cols:
        try:
            _X_sup = df[supervised_feature_cols].fillna(0).values.astype(np.float64)
            _all_sup_scores = supervised_clf.predict_proba(_X_sup)[:, 1]
        except Exception as e:
            logger.warning("Batch supervised scoring failed, will fall back per-row: %s", e)

    # (e) Pre-extract numeric columns for discovery baseline
    _numeric_cols_for_discovery: Dict[str, np.ndarray] = {}
    for col in df.columns:
        try:
            _numeric_cols_for_discovery[col] = pd.to_numeric(df[col], errors="coerce").values
        except Exception:
            pass

    # (f) Pre-extract metadata arrays for fast indexed access
    _labels = df["label"].values
    _sample_ids = df["sample_id"].values if "sample_id" in df.columns else np.arange(n_total)
    _operation_ids = df["operation_id"].values if "operation_id" in df.columns else np.full(n_total, "")
    _tool_numbers = df["tool_number"].values if "tool_number" in df.columns else np.full(n_total, "")

    logger.info(
        "Batch pre-computation done: %d samples, %d features, model scores range [%.3f, %.3f]",
        n_total, _X_all.shape[1],
        float(np.min(_all_model_scores)), float(np.max(_all_model_scores)),
    )

    # --- Process each sample (using pre-computed arrays) ----------
    feedback_count = 0
    _model_retrained = False  # track when model changes (invalidates pre-computed scores)

    # API batch buffer: accumulate events and flush in chunks to eliminate
    # per-sample HTTP round-trip overhead.
    _api_batch: List[Dict[str, Any]] = []  # event payloads
    _api_batch_srs: List[Any] = []  # corresponding SampleResult objects
    _api_batch_size = config.api_batch_size if api_client is not None else 0

    def _flush_api_batch() -> None:
        """Submit buffered events as a batch and apply API results."""
        nonlocal _api_batch, _api_batch_srs
        if not _api_batch:
            return
        results = api_client.submit_events_batch(_api_batch)  # type: ignore[union-attr]
        for buf_sr, api_result in zip(_api_batch_srs, results):
            if api_result:
                buf_sr.memory_id = api_result["memory_id"] or ""
                buf_sr.stored_in_memory = bool(api_result["memory_id"])
                api_score = float(api_result["significance_score"])
                api_action = str(api_result.get("action", buf_sr.action))
                buf_sr.api_score = api_score
                buf_sr.api_action = api_action
                buf_sr.significance_score = api_score
                buf_sr.predicted_positive = api_score >= current_threshold
                buf_sr.action = api_action.upper()
                buf_sr.model_breakdown = api_result.get("model_breakdown") or {}
                buf_sr.explanation = api_result.get("explanation")
                buf_sr.explanation_source = api_result.get("explanation_source")
                buf_sr.alert_line = api_result.get("alert_line")
                buf_sr.alert_line_source = api_result.get("alert_line_source")
                buf_sr.prior_boost = float(api_result.get("prior_boost") or 0.0)
                buf_sr.pattern_rule_score = float(api_result.get("pattern_rule_score") or 0.0)
                _triggered = api_result.get("triggered_rules") or []
                buf_sr.n_rules_triggered = max(buf_sr.n_rules_triggered, len(_triggered))
                server_pattern_keys = [str(pk) for pk in (api_result.get("pattern_keys") or []) if str(pk)]
                buf_sr.detected_patterns = server_pattern_keys
            elif config.api_mode_strict:
                raise RuntimeError(
                    "Backend API batch event submission returned an unprocessed result: "
                    f"{getattr(api_client, 'last_error', None) or '<no detail captured>'}"
                )
        _api_batch.clear()
        _api_batch_srs.clear()

    for idx in range(n_total):
        # Emit progress at the start of each sample (score snapshot
        # is emitted later, after scoring completes).
        if progress_callback is not None:
            try:
                progress_callback(idx, n_total, None)
            except Exception:
                pass
        sr = SampleResult(
            sample_id=str(_sample_ids[idx]),
            label=str(_labels[idx]),
            operation_id=str(_operation_ids[idx]),
            tool_number=str(_tool_numbers[idx]),
        )

        # Use pre-computed features and model score (re-score if model was retrained)
        fv = _X_all[idx]
        if not _seed_enabled:
            raw_model_score = 0.0
        elif _model_retrained:
            raw_model_score = model.score(fv)
        else:
            raw_model_score = float(_all_model_scores[idx])
        sr.raw_model_score = raw_model_score

        # Use pre-computed pattern detection
        pattern_keys = list(_all_pattern_keys[idx])

        # Feed feature dict to discovery baseline (every sample, not just confirmed)
        all_features_flat: Dict[str, float] = {
            col: float(vals[idx])
            for col, vals in _numeric_cols_for_discovery.items()
            if not np.isnan(vals[idx])
        }
        discovery.update_baseline(all_features_flat)

        # Augment pattern_keys with any discovered patterns that match
        discovered_matches = discovery.match_event(all_features_flat)
        if discovered_matches:
            pattern_keys = pattern_keys + discovered_matches

        sr.detected_patterns = pattern_keys

        # Compute anomaly z-score from rolling baseline
        anomaly_z = 0.0
        if len(recent_scores) >= 10:
            baseline_mean = np.mean(recent_scores)
            baseline_std = np.std(recent_scores)
            if baseline_std > 1e-9:
                anomaly_z = (raw_model_score - baseline_mean) / baseline_std
        sr.anomaly_z_score = anomaly_z

        # Update rolling baseline
        recent_scores.append(raw_model_score)
        if len(recent_scores) > max_history:
            recent_scores = recent_scores[-max_history:]

        # --- Unified scorer (PRIMARY — same as live pipeline) ---------
        # Legacy scorer removed from main path (moved to counterfactual only)
        # Per-sample context (NOT context-blind). Built outside the try so it is
        # guaranteed in scope for the counterfactual block below.
        sample_context = _build_cutting_context(sr.operation_id, sr.tool_number)
        # AAD combiner score (feedback-trained over fired patterns + anomaly).
        aad_signals = None
        if aad_model is not None:
            aad_signals = {"aad_score": aad_model._prob(pattern_keys, raw_model_score)}
        try:
            unified_result = _score_via_unified_scorer(
                unified_scorer, pattern_keys, raw_model_score,
                anomaly_z, model_confidence, pattern_priors, feedback_counts,
                context=sample_context, extra_signals=aad_signals,
            )
            sr.unified_score = unified_result.score
            sr.unified_action = unified_result.action.value if unified_result.action else ""
            sr.unified_triggered_rules = len(unified_result.triggered_rules) if unified_result.triggered_rules else 0
            sr.score_trace = list(unified_result.score_trace or [])
            significance_score = unified_result.score
            sr.n_rules_triggered = sr.unified_triggered_rules
        except Exception as e:
            # X7: this fallback substitutes a crude score (raw_model_score*0.5)
            # for the real scorer. It must NOT be silent — a systematic scorer
            # break would otherwise masquerade as a mediocre-but-plausible run.
            # Count it; the phase summary fails loudly if the rate is high.
            result.n_scorer_fallbacks += 1
            if result.n_scorer_fallbacks == 1:
                logger.warning("Unified scorer failed (using fallback score); first occurrence:", exc_info=True)
            else:
                logger.warning("Unified scorer failed (using fallback score): %s", e)
            significance_score = raw_model_score * 0.5  # fallback
            sr.unified_score = significance_score
            sr.unified_action = ""
            sr.unified_triggered_rules = 0
            sr.score_trace = []

        sr.unsupervised_score = significance_score

        # --- Supervised model scoring (use pre-computed batch) --------
        sup_score = 0.0
        if supervised_clf is not None and supervised_feature_cols:
            if _all_sup_scores is not None:
                sup_score = float(_all_sup_scores[idx])
            else:
                try:
                    _row = df.iloc[idx]
                    safe_vals = [float(_row.get(c, 0.0) or 0.0) for c in supervised_feature_cols]
                    safe_arr = np.array(safe_vals).reshape(1, -1)
                    sup_score = float(supervised_clf.predict_proba(safe_arr)[0, 1])
                except Exception as e:
                    logger.debug("Supervised scoring failed: %s", e)
                    sup_score = 0.5
        sr.supervised_score = sup_score

        # --- Combined score: weighted blend of both models -----------
        if supervised_clf is not None:
            combined = (w_unsup * significance_score + w_sup * sup_score) / (w_unsup + w_sup)
        else:
            combined = significance_score
        sr.weight_supervised = w_sup
        sr.weight_unsupervised = w_unsup

        # --- Tool prior boost ----------------------------------------
        tool_num_str = sr.tool_number
        tool_p = tool_priors_data.get(tool_num_str, config.tool_prior_default)
        sr.tool_prior = tool_p
        tool_multiplier = 1.0 + config.tool_prior_weight * (tool_p - 0.5)
        tool_multiplier = max(0.5, tool_multiplier)  # floor
        sr.tool_multiplier = tool_multiplier

        # Final combined score
        final_score = min(1.0, max(0.0, combined * tool_multiplier))
        sr.combined_score = final_score
        sr.significance_score = final_score  # Use combined as the decision score
        sr.predicted_positive = final_score >= current_threshold

        # Classify action
        if significance_score >= config.critical_threshold:
            sr.action = "CRITICAL"
        elif significance_score >= config.alert_threshold:
            sr.action = "ALERT"
        elif significance_score >= config.store_threshold:
            sr.action = "STORE"
        else:
            sr.action = "IGNORE"

        # --- SINDIT context simulation --------------------------------
        sindit_ctx: Dict[str, Any] = {}
        if config.use_sindit_simulation:
            _row = df.iloc[idx]
            sindit_ctx = _simulate_sindit_context(_row)
            sr.sindit_context = sindit_ctx

        # --- API-mode: POST event to live backend and use its score ---
        # When api_mode is on, the API is the single source of truth.
        # We submit every sample (not just significant ones) so the API's
        # orchestrator scores it through the production pipeline.  The
        # returned significance_score overrides the local scoring above.
        if api_client is not None:
            session_id = f"{phase}_{config.eval_op if phase == 'eval' else config.test_op}"
            cutting_ctx_dict = sindit_ctx if sindit_ctx else None
            fd = {name: float(fv[i]) for i, name in enumerate(FEATURE_NAMES) if i < len(fv)}
            api_event = {
                "session_id": session_id,
                "pattern_keys": pattern_keys,
                "derive_patterns": bool(config.api_use_server_patterns),
                "metadata": {
                    "tool_number": sr.tool_number,
                    "operation_id": sr.operation_id,
                    "raw_model_score": sr.raw_model_score,
                    "supervised_score": sr.supervised_score,
                    "combined_score": sr.combined_score,
                    "experiment_fast_path": config.experiment_fast_path,
                    "await_explanation": not config.experiment_fast_path,
                    "feedback_scope_user_id": config.feedback_user_id,
                    **(
                        {"generate_explanations_override": config.api_generate_explanations}
                        if config.api_generate_explanations is not None
                        else {}
                    ),
                },
                "annotation_text": f"Sample {sr.sample_id}: {sr.action} (score={sr.significance_score:.3f})",
                "cutting_context": cutting_ctx_dict,
                "tags": [sr.action, phase],
                "metrics": fd,
            }
            if _api_batch_size <= 1:
                api_result = api_client.submit_event(
                    session_id=api_event["session_id"],
                    pattern_keys=api_event["pattern_keys"],
                    metadata=api_event["metadata"],
                    significance_score=sr.significance_score,
                    annotation=api_event["annotation_text"],
                    cutting_context=api_event["cutting_context"],
                    label=sr.label,
                    tags=api_event["tags"],
                    metrics=api_event["metrics"],
                    derive_patterns=api_event["derive_patterns"],
                )
                if api_result:
                    sr.memory_id = api_result["memory_id"] or ""
                    sr.stored_in_memory = bool(api_result["memory_id"])
                    api_score = float(api_result["significance_score"])
                    api_action = str(api_result.get("action", sr.action))
                    server_pattern_keys = [str(pk) for pk in (api_result.get("pattern_keys") or []) if str(pk)]
                    if config.api_use_server_patterns:
                        pattern_keys = server_pattern_keys
                        sr.detected_patterns = server_pattern_keys
                    sr.api_score = api_score
                    sr.api_action = api_action
                    sr.significance_score = api_score
                    sr.predicted_positive = api_score >= current_threshold
                    sr.action = api_action.upper()
                    sr.model_breakdown = api_result.get("model_breakdown") or {}
                    sr.explanation = api_result.get("explanation")
                    sr.explanation_source = api_result.get("explanation_source")
                    sr.alert_line = api_result.get("alert_line")
                    sr.alert_line_source = api_result.get("alert_line_source")
                    sr.prior_boost = float(api_result.get("prior_boost") or 0.0)
                    sr.pattern_rule_score = float(api_result.get("pattern_rule_score") or 0.0)
                    _triggered = api_result.get("triggered_rules") or []
                    sr.n_rules_triggered = max(sr.n_rules_triggered, len(_triggered))
                elif config.api_mode_strict:
                    raise RuntimeError(
                        "Backend API event submission failed while api_mode_strict is enabled: "
                        f"{getattr(api_client, 'last_error', None) or '<no detail captured>'}"
                    )
            else:
                _api_batch.append(api_event)
                _api_batch_srs.append(sr)
                if len(_api_batch) >= _api_batch_size:
                    _flush_api_batch()

        # --- Store in MemoryStore (non-API local mode) ----------------
        elif sr.action in ("STORE", "ALERT", "CRITICAL"):
            session_id = f"{phase}_{config.eval_op if phase == 'eval' else config.test_op}"
            cutting_ctx_dict = sindit_ctx if sindit_ctx else None
            if memory_store is not None:
                # Local mode: store in ephemeral MemoryStore
                try:
                    cutting_ctx = CuttingContext(
                        spindle_speed=sindit_ctx.get("spindle_speed"),
                        feed_rate=sindit_ctx.get("feed_rate"),
                        tool_id=sindit_ctx.get("tool_id"),
                        machine_state=sindit_ctx.get("machine_state"),
                    ) if sindit_ctx else None
                    mem = Memory(
                        session_id=session_id,
                        time_range=(float(idx), float(idx) + 1.0),
                        annotation_text=f"Sample {sr.sample_id}: {sr.action} (score={sr.significance_score:.3f})",
                        pattern_keys=[SchemaPatternKey(key=pk) for pk in pattern_keys],
                        label=sr.label,
                        tags=[sr.action, phase],
                        metadata={
                            "tool_number": sr.tool_number,
                            "operation_id": sr.operation_id,
                            "raw_model_score": sr.raw_model_score,
                            "supervised_score": sr.supervised_score,
                            "combined_score": sr.combined_score,
                            **(cutting_ctx.model_dump() if cutting_ctx else {}),
                        },
                    )
                    mem_id = memory_store.store(mem)
                    sr.memory_id = mem_id
                    sr.stored_in_memory = True
                    if pattern_index is not None:
                        pattern_index.add(mem_id, pattern_keys)
                except Exception as e:
                    logger.debug("Memory store error: %s", e)

        # --- Co-occurrence tracking -----------------------------------
        if config.use_co_occurrence and len(pattern_keys) >= 2:
            for i_p, pk_a in enumerate(pattern_keys):
                for pk_b in pattern_keys[i_p + 1:]:
                    pair = tuple(sorted([pk_a, pk_b]))
                    co_occurrence[pair] = co_occurrence.get(pair, 0) + 1
                    sr.co_occurring_pairs.append(pair)

        # --- Score by label for distribution tracking ----------------
        is_positive = sr.label == "pre_stoppage"
        if is_positive:
            result.scores_positive.append(sr.significance_score)
        else:
            result.scores_negative.append(sr.significance_score)

        # --- Counterfactual (A1): same event, priors FROZEN at initial,
        # scored through the SAME unified scorer (cf_scorer). The delta vs the
        # primary path is therefore attributable to feedback alone. ---
        if feedback_enabled:
            try:
                cf_result = _score_via_unified_scorer(
                    cf_scorer, pattern_keys, raw_model_score, anomaly_z,
                    model_confidence,
                    pattern_priors=initial_priors,
                    feedback_counts=initial_feedback_counts,
                    context=sample_context,
                )
                cf_unsup = cf_result.score
            except Exception:
                cf_unsup = raw_model_score * 0.5  # mirror the primary fallback
            # Apply the SAME decision layers as the primary path, but with the
            # INITIAL weights/tool priors. In the faithful pipeline (A2) these
            # are identities, so cf reduces to cf_unsup; in an ablation run they
            # match the primary formula with frozen parameters.
            denom = config.weight_unsupervised + config.weight_supervised
            if supervised_clf is not None and denom > 0:
                cf_combined = (
                    config.weight_unsupervised * cf_unsup
                    + config.weight_supervised * sup_score
                ) / denom
            else:
                cf_combined = cf_unsup
            if config.use_tool_priors:
                initial_tool_p = initial_tool_priors.get(tool_num_str, config.tool_prior_default)
                cf_tool_mult = max(0.5, 1.0 + config.tool_prior_weight * (initial_tool_p - 0.5))
            else:
                cf_tool_mult = 1.0
            cf_score = min(1.0, max(0.0, cf_combined * cf_tool_mult))
            sr.counterfactual_score = cf_score
            cf_predicted = cf_score >= threshold  # original threshold, no adaptation
            if sr.predicted_positive != cf_predicted:
                sr.prediction_flipped = True
                result.n_predictions_flipped += 1

        # --- Feedback ------------------------------------------------
        # Helper: apply a dequeued feedback payload to the pipeline state.
        # This is factored out so both immediate and delayed paths share
        # the same logic.
        def _apply_feedback(fb_payload: Dict[str, Any], applied_idx: int) -> None:
            nonlocal pattern_priors, feedback_counts, co_occurrence
            nonlocal threshold_shift, current_threshold, secondary_threshold
            nonlocal model_weight_shift, w_sup, w_unsup
            nonlocal tool_priors_data, tool_feedback_counts
            nonlocal dismissed_features, confirmed_features
            nonlocal total_feedback_events, retrain_count, model, model_confidence
            nonlocal recent_scores
            nonlocal discovery
            nonlocal _model_retrained

            fb_was_significant = fb_payload["was_significant"]
            fb_pattern_keys = fb_payload["pattern_keys"]
            fb_fv = fb_payload["fv"]
            fb_tool_num = fb_payload["tool_num_str"]
            fb_sr = fb_payload["sr"]
            fb_memory_id = fb_payload.get("memory_id")
            fb_source_idx = int(fb_payload.get("source_idx", applied_idx))

            # AAD combiner: one online gradient step toward the feedback label,
            # using the same features it scored on (fired patterns + anomaly).
            if aad_model is not None:
                aad_model.update(
                    fb_pattern_keys,
                    float(getattr(fb_sr, "raw_model_score", 0.0) or 0.0),
                    bool(fb_was_significant),
                )

            threshold_before = current_threshold
            tool_prior_before = tool_priors_data.get(fb_tool_num, config.tool_prior_default) if fb_tool_num else None
            model_weights_before = {
                "w_sup": round(float(w_sup), 4),
                "w_unsup": round(float(w_unsup), 4),
                "shift": round(float(model_weight_shift), 4),
            }

            fb_sr.feedback_action = "CONFIRM" if fb_was_significant else "DISMISS"

            # Update priors (Beta-Binomial) with optional per-pattern weighting
            update_keys = fb_pattern_keys if fb_pattern_keys else []
            if not update_keys:
                logger.debug("No patterns detected — skipping prior update")

            # Compute per-pattern weights (severity-based or uniform)
            if config.feedback_per_pattern_weighting and len(update_keys) > 1:
                from .config import FAULT_SEVERITY
                sev = {pk: FAULT_SEVERITY.get(pk, 0.5) for pk in update_keys}
                total_sev = sum(sev.values()) or 1.0
                pk_weights = {pk: sev[pk] / total_sev * len(update_keys) for pk in update_keys}
            else:
                pk_weights = {pk: 1.0 for pk in update_keys}

            pattern_updates: List[Dict[str, Any]] = []
            registry = _get_pattern_registry()

            for pk in update_keys:
                if pk not in pattern_priors:
                    pattern_priors[pk] = 0.5
                feedback_counts.setdefault(pk, {"confirm": 0, "dismiss": 0})
                old_prior_val = pattern_priors[pk]
                w = pk_weights.get(pk, 1.0)
                if fb_was_significant:
                    feedback_counts[pk]["confirm"] += w
                else:
                    feedback_counts[pk]["dismiss"] += w
                # P1: derive the prior with the shared estimator (prior_math)
                # so the experiment's prior matches the live scorer exactly for
                # identical counts (was a plain alpha/(alpha+beta) Beta mean).
                pattern_priors[pk], _ = prior_from_counts(
                    feedback_counts[pk]["confirm"], feedback_counts[pk]["dismiss"]
                )
                pdef = registry.get(pk)
                pattern_updates.append({
                    "pattern_key": pk,
                    "polarity": getattr(pdef, "polarity", None),
                    "weight": round(float(w), 4),
                    "old_prior": round(float(old_prior_val), 4),
                    "new_prior": round(float(pattern_priors[pk]), 4),
                    "delta": round(float(pattern_priors[pk] - old_prior_val), 4),
                    "confirm_count": round(float(feedback_counts[pk]["confirm"]), 4),
                    "dismiss_count": round(float(feedback_counts[pk]["dismiss"]), 4),
                })

                # Co-occurrence prior propagation
                if config.use_co_occurrence and co_occurrence:
                    delta_val = pattern_priors[pk] - old_prior_val
                    if abs(delta_val) > 1e-6:
                        propagated = _propagate_prior(
                            pk, delta_val, co_occurrence,
                            pattern_priors, config.co_occurrence_decay,
                        )
                        if propagated:
                            fb_sr.propagated_prior_deltas.update(propagated)
                            result.all_propagated_deltas.append(propagated)

            # Record feedback in MemoryStore or via API
            if fb_memory_id:
                if api_client is not None:
                    try:
                        ok = api_client.submit_feedback(
                            memory_id=fb_memory_id,
                            action=fb_sr.feedback_action,
                            user_id=config.feedback_user_id,
                            pattern_keys=update_keys,
                        )
                        if config.api_mode_strict and not ok:
                            raise RuntimeError(
                                "Backend API feedback submission failed while api_mode_strict is enabled: "
                                f"{getattr(api_client, 'last_error', None) or '<no detail captured>'}"
                            )
                    except Exception as e:
                        if config.api_mode_strict:
                            raise
                        logger.debug("API feedback error: %s", e)
                elif memory_store is not None:
                    try:
                        memory_store.add_feedback_event(
                            memory_id=fb_memory_id,
                            action=fb_sr.feedback_action,
                            user_id=config.feedback_user_id,
                            pattern_keys=update_keys,
                        )
                    except Exception as e:
                        logger.debug("Feedback store error: %s", e)
            elif api_client is not None and fb_was_significant and fb_sr.feedback_source == "missed_event":
                try:
                    missed_result = api_client.submit_missed_event(
                        session_id=f"{phase}_{config.eval_op}",
                        user_id=config.feedback_user_id,
                        pattern_keys=update_keys,
                        raw_metrics=fb_payload.get("row_data", {}),
                        reason="missed_event",
                        derive_patterns=bool(config.api_use_server_patterns),
                    )
                    if config.api_mode_strict and missed_result is None:
                        raise RuntimeError(
                            "Backend API missed-event submission failed while api_mode_strict is enabled: "
                            f"{getattr(api_client, 'last_error', None) or '<no detail captured>'}"
                        )
                except Exception as e:
                    if config.api_mode_strict:
                        raise
                    logger.debug("API missed-event error: %s", e)

            # --- Pattern Discovery: learn from confirmed/dismissed events ---
            try:
                _fb_features = {}
                for col in df.columns:
                    try:
                        v = fb_payload.get("row_data", {}).get(col)
                        if v is not None and pd.notna(v):
                            _fb_features[col] = float(v)
                    except (ValueError, TypeError):
                        pass
                if len(_fb_features) > 5:
                    if fb_was_significant:
                        discovered = discovery.analyse_confirmed_event(
                            features=_fb_features,
                            existing_pattern_keys=fb_pattern_keys,
                            scorer=unified_scorer,
                            memory_id=fb_memory_id,
                            session_id=f"{phase}_{config.eval_op}",
                        )
                        _integrate_promoted_discoveries(
                            pattern_priors,
                            feedback_counts,
                            discovered,
                            feedback_action="CONFIRM",
                        )
                    else:
                        dismissed = discovery.analyse_dismissed_event(
                            features=_fb_features,
                            existing_pattern_keys=fb_pattern_keys,
                            memory_id=fb_memory_id,
                            session_id=f"{phase}_{config.eval_op}",
                        )
                        _integrate_promoted_discoveries(
                            pattern_priors,
                            feedback_counts,
                            dismissed,
                            feedback_action="DISMISS",
                        )
            except Exception as e:
                logger.debug("Discovery feedback processing failed: %s", e)

            # Threshold adaptation
            if fb_was_significant:
                threshold_shift = max(
                    -config.threshold_adaptation_max,
                    threshold_shift - config.threshold_adaptation_rate,
                )
            else:
                threshold_shift = min(
                    config.threshold_adaptation_max,
                    threshold_shift + config.threshold_adaptation_rate,
                )
            current_threshold = threshold + threshold_shift
            secondary_threshold = current_threshold - config.secondary_threshold_offset

            # Model weight shifting
            if supervised_clf is not None:
                if fb_was_significant:
                    model_weight_shift += config.model_weight_shift_per_feedback
                else:
                    model_weight_shift -= config.model_weight_shift_per_feedback
                model_weight_shift = max(
                    -config.model_weight_shift_max,
                    min(config.model_weight_shift_max, model_weight_shift),
                )
                w_sup = config.weight_supervised + model_weight_shift
                w_unsup = config.weight_unsupervised - model_weight_shift
                w_sup = max(0.1, min(0.9, w_sup))
                w_unsup = max(0.1, min(0.9, w_unsup))

            # Tool prior update
            if config.use_tool_priors and fb_tool_num:
                if fb_tool_num not in tool_feedback_counts:
                    tool_feedback_counts[fb_tool_num] = {"confirm": 0, "dismiss": 0}
                if fb_was_significant:
                    tool_feedback_counts[fb_tool_num]["confirm"] += 1
                else:
                    tool_feedback_counts[fb_tool_num]["dismiss"] += 1
                tc = tool_feedback_counts[fb_tool_num]
                base = initial_tool_priors.get(fb_tool_num, config.tool_prior_default)
                n_total = tc["confirm"] + tc["dismiss"]
                data_prior = (tc["confirm"] + 1) / (n_total + 2)
                blend_weight = min(1.0, n_total / 10)
                tool_priors_data[fb_tool_num] = (1 - blend_weight) * base + blend_weight * data_prior
            current_threshold = threshold + threshold_shift
            secondary_threshold = current_threshold - config.secondary_threshold_offset

            # Collect features for online retraining
            if fb_was_significant:
                confirmed_features.append(fb_fv.copy())
            else:
                dismissed_features.append(fb_fv.copy())
            total_feedback_events += 1

            # Online model retraining (skipped when the seed model is removed)
            if (
                _seed_enabled
                and config.online_retrain_enabled
                and feedback_enabled
                and train_normal_features is not None
                and total_feedback_events >= config.online_retrain_min_feedback
                and total_feedback_events % config.online_retrain_interval == 0
            ):
                parts = [train_normal_features]
                if dismissed_features:
                    parts.append(np.array(dismissed_features))
                augmented = np.vstack(parts)
                new_model = SeedModel(config=SeedModelConfig())
                new_model.train(augmented)
                model = new_model
                n_train = model._training_stats.get("n_samples", 0)
                model_confidence = min(1.0, n_train / 500)
                recent_scores.clear()
                retrain_count += 1
                fb_sr.model_retrained = True
                _model_retrained = True  # invalidate pre-computed batch scores
                logger.info(
                    "Online retrain #%d after %d feedback events: "
                    "%d normal + %d dismissed = %d training samples",
                    retrain_count, total_feedback_events,
                    len(train_normal_features), len(dismissed_features),
                    len(augmented),
                )

            # Record prior history
            result.prior_history.append(dict(pattern_priors))
            result.weight_history.append({"w_sup": w_sup, "w_unsup": w_unsup, "shift": model_weight_shift})
            result.tool_prior_history.append(dict(tool_priors_data))
            result.feedback_events.append({
                "source_sample_id": fb_sr.sample_id,
                "source_operation_id": fb_sr.operation_id,
                "source_label": fb_sr.label,
                "feedback_action": fb_sr.feedback_action,
                "feedback_source": fb_sr.feedback_source,
                "was_significant": fb_was_significant,
                "source_index": fb_source_idx,
                "applied_at_index": applied_idx,
                "applied_after_samples": max(0, applied_idx - fb_source_idx),
                "memory_id": fb_memory_id,
                "detected_patterns": list(fb_pattern_keys),
                "pattern_updates": pattern_updates,
                "propagated_prior_deltas": {
                    k: round(float(v), 4) for k, v in fb_sr.propagated_prior_deltas.items()
                },
                "threshold_before": round(float(threshold_before), 4),
                "threshold_after": round(float(current_threshold), 4),
                "tool_prior_before": round(float(tool_prior_before), 4) if tool_prior_before is not None else None,
                "tool_prior_after": round(float(tool_priors_data.get(fb_tool_num, config.tool_prior_default)), 4) if fb_tool_num else None,
                "model_weights_before": model_weights_before,
                "model_weights_after": {
                    "w_sup": round(float(w_sup), 4),
                    "w_unsup": round(float(w_unsup), 4),
                    "shift": round(float(model_weight_shift), 4),
                },
                "model_retrained": fb_sr.model_retrained,
            })

        # --- Drain any delayed feedback whose time has come -----------
        if feedback_enabled and _feedback_delay_q:
            while _feedback_delay_q and _feedback_delay_q[0][0] <= idx:
                _, queued_payload = _heapq.heappop(_feedback_delay_q)
                _apply_feedback(queued_payload, idx)

        if feedback_enabled:
            give_feedback = False
            feedback_source = ""

            # Path A: flagged event (predicted positive)
            # Uses alert-fatigue decay: response_prob decays with total alerts.
            if sr.predicted_positive:
                feedback_count += 1
                import math as _math
                response_prob = config.feedback_response_rate * _math.exp(
                    -config.feedback_fatigue_decay * feedback_count
                )
                # Legacy feedback_every_n gate (backward compat)
                if feedback_every_n > 1:
                    passes_legacy = (feedback_count % feedback_every_n == 0)
                else:
                    passes_legacy = True
                if passes_legacy and rng.random() < response_prob:
                    give_feedback = True
                    feedback_source = "flagged"

            # Path B: missed-event feedback — DECOUPLED from model score.
            # Triggers on actual FN (label=positive AND not predicted positive).
            # This simulates "the machine stopped and the operator noticed
            # the model didn't flag it" — independent of the model's confidence.
            elif (
                is_positive
                and not sr.predicted_positive
                and config.missed_event_feedback_rate > 0
                and rng.random() < config.missed_event_feedback_rate
            ):
                give_feedback = True
                feedback_source = "missed_event"

            # Path C: Negative sampling (disabled by default — no real analogue)
            elif (
                not sr.predicted_positive
                and not is_positive
                and config.negative_sampling_enabled
                and pattern_keys
                and rng.random() < config.negative_sampling_rate
            ):
                give_feedback = True
                feedback_source = "negative_sample"

            if give_feedback:
                sr.feedback_given = True
                sr.feedback_source = feedback_source

                # Confidence-dependent noise: borderline samples get noisier
                # feedback than obvious ones.  Use the decision score
                # (sr.significance_score) which may be the API's score.
                ambiguity = 0.0
                decision_score = sr.significance_score
                if current_threshold > 1e-9:
                    ambiguity = max(0.0, min(1.0,
                        1.0 - abs(decision_score - current_threshold) / current_threshold
                    ))
                noise_prob = config.noise_rate_base + config.noise_rate_ambiguity * ambiguity
                # Fall back to legacy uniform noise_rate if new params are both 0
                if noise_prob < 1e-9 and legacy_noise_rate > 0:
                    noise_prob = legacy_noise_rate

                if noise_prob > 0 and rng.random() < noise_prob:
                    was_significant = not is_positive  # flipped
                else:
                    was_significant = is_positive

                sr.feedback_action = "CONFIRM" if was_significant else "DISMISS"

                # Build payload for immediate or delayed application
                fb_payload = {
                    "was_significant": was_significant,
                    "pattern_keys": pattern_keys,
                    "fv": fv,
                    "tool_num_str": tool_num_str,
                    "sr": sr,
                    "memory_id": sr.memory_id,
                    "source_idx": idx,
                    "row_data": dict(all_features_flat),
                }

                if _feedback_delay > 0:
                    # Queue for later application
                    apply_at = idx + _feedback_delay
                    _heapq.heappush(_feedback_delay_q, (apply_at, fb_payload))
                else:
                    # Apply immediately (oracle mode — default)
                    _apply_feedback(fb_payload, idx)

        sr.prior_snapshot = dict(pattern_priors)

        # Emit score snapshot so the experiment runner can stream it to
        # the UI for real-time charts.
        if progress_callback is not None:
            try:
                progress_callback(idx, n_total, {
                    "idx": idx,
                    "raw_model_score": round(sr.raw_model_score, 4),
                    "supervised_score": round(sr.supervised_score, 4),
                    "combined_score": round(sr.combined_score, 4),
                    "significance_score": round(sr.significance_score, 4),
                    "anomaly_z_score": round(sr.anomaly_z_score, 4),
                    "prior_boost": round(sr.prior_boost, 4),
                    "pattern_rule_score": round(sr.pattern_rule_score, 4),
                    "label": sr.label,
                    "predicted_positive": sr.predicted_positive,
                    "threshold": round(current_threshold, 4),
                })
            except Exception:
                pass

        result.sample_results.append(sr)

    # --- Flush remaining API batch buffer ----------------------------
    if api_client is not None:
        _flush_api_batch()

    # --- Drain any remaining delayed feedback after loop ends ---------
    if feedback_enabled and _feedback_delay_q:
        logger.info("Draining %d delayed feedback events after loop", len(_feedback_delay_q))
        while _feedback_delay_q:
            _, queued_payload = _heapq.heappop(_feedback_delay_q)
            _apply_feedback(queued_payload, n_total)

    result.adapted_threshold = current_threshold
    result.n_model_retrains = retrain_count
    result.duration_s = time.time() - t0

    # X7: surface scorer fallbacks loudly. A high rate means the reported
    # metrics are built substantially on crude fallback scores, not the real
    # scorer — that must not pass silently.
    if result.n_scorer_fallbacks:
        frac = result.n_scorer_fallbacks / max(1, result.n_samples)
        logger.warning(
            "Phase '%s': unified scorer fell back on %d/%d samples (%.1f%%) — "
            "metrics partly reflect fallback scores, not the real scorer.",
            phase, result.n_scorer_fallbacks, result.n_samples, frac * 100.0,
        )

    # --- KG summary ---------------------------------------------------
    if api_client is not None:
        # In API mode, query the backend for the final state
        result.stored_memories_count = sum(
            1 for s in result.sample_results if s.stored_in_memory
        )
        api_co = api_client.get_co_occurrence()
        if api_co:
            result.co_occurrence_graph = api_co
        else:
            result.co_occurrence_graph = {
                f"{a}|{b}": w for (a, b), w in co_occurrence.items()
            }
        result.api_mode_used = True
    elif memory_store is not None:
        result.stored_memories_count = memory_store.count()
        result.co_occurrence_graph = {
            f"{a}|{b}": w for (a, b), w in co_occurrence.items()
        }

    if config.use_sindit_simulation or config.sindit_live:
        states = [
            s.sindit_context.get("machine_state", "unknown")
            for s in result.sample_results if s.sindit_context
        ]
        result.sindit_context_summary = {
            "n_normal": states.count("normal"),
            "n_degraded": states.count("degraded"),
            "total": len(states),
        }

    # Clean up resources
    if api_client is not None:
        try:
            api_client.close()
        except Exception:
            pass
    if memory_store is not None:
        try:
            memory_store.close()
        except Exception:
            pass
    logger.info(
        "KG summary: %d memories stored, %d co-occurrence pairs, %d propagation events",
        result.stored_memories_count, len(result.co_occurrence_graph),
        len(result.all_propagated_deltas),
    )

    n_fb = sum(1 for s in result.sample_results if s.feedback_given)
    n_flagged_fb = sum(1 for s in result.sample_results if s.feedback_source == "flagged")
    n_missed_fb = sum(1 for s in result.sample_results if s.feedback_source == "missed_event")
    n_neg_fb = sum(1 for s in result.sample_results if s.feedback_source == "negative_sample")

    # --- Discovery summary --------------------------------------------
    all_discovered = discovery.get_patterns()
    promoted = {k: v for k, v in all_discovered.items() if v.promoted and not k.startswith("suppressed:")}
    suppressed = {k: v for k, v in all_discovered.items() if v.promoted and k.startswith("suppressed:")}
    result.n_discovered_patterns = len(promoted)
    result.n_suppression_patterns = len(suppressed)
    result.discovered_pattern_keys = list(promoted.keys()) + list(suppressed.keys())

    logger.info(
        "Phase %s complete: %d samples, %d feedback events "
        "(%d flagged, %d missed-event), %d predictions flipped, "
        "%d model retrains, %d discovered patterns, %d suppression patterns, %.1fs",
        phase, len(df), n_fb, n_flagged_fb, n_missed_fb,
        result.n_predictions_flipped, retrain_count,
        result.n_discovered_patterns, result.n_suppression_patterns,
        result.duration_s,
    )

    # --- Persist final priors to shared file for the backend server ----
    # The backend scorer reads from data/pattern_priors.json.  Write the
    # final state so the live /scorer/priors endpoint picks it up.
    if feedback_enabled and n_fb > 0 and api_client is None and config.persist_shared_priors:
        try:
            from datetime import datetime, timezone as _tz
            shared_priors_path = Path(__file__).resolve().parents[2] / "data" / "pattern_priors.json"
            shared_data = {
                "pattern_priors": pattern_priors,
                "feedback_counts": feedback_counts,
                "updated_at": datetime.now(_tz.utc).isoformat(),
                "source": f"experiment_{phase}",
            }
            shared_priors_path.parent.mkdir(parents=True, exist_ok=True)
            with open(shared_priors_path, "w") as _fp:
                json.dump(shared_data, _fp, indent=2)
            logger.info(
                "Saved %d pattern priors to shared file %s",
                len(pattern_priors), shared_priors_path,
            )
        except Exception as e:
            logger.warning("Failed to save shared priors: %s", e)

        # --- Persist discovered patterns so they survive sandbox restore ---
        # Experiment-discovered patterns are saved independently from the
        # snapshot/restore mechanism.  The PatternDiscovery engine writes
        # to data/discovered_patterns.json automatically, but we also
        # register any promoted discoveries with the shared pattern_priors
        # so the scorer picks them up in subsequent experiments.
        try:
            for dk, dp in discovery.get_promoted_patterns().items():
                if dk not in shared_data["pattern_priors"]:
                    shared_data["pattern_priors"][dk] = dp.prior
                    shared_data["feedback_counts"][dk] = {
                        "confirm": dp.confirmation_count, "dismiss": 0,
                    }
            with open(shared_priors_path, "w") as _fp:
                json.dump(shared_data, _fp, indent=2)
            if discovery.get_promoted_patterns():
                logger.info(
                    "Persisted %d discovered pattern priors (survive sandbox restore)",
                    len(discovery.get_promoted_patterns()),
                )
        except Exception as e:
            logger.debug("Failed to persist discovered pattern priors: %s", e)

    return result


# =====================================================================
# Phase 2 & 3 convenience wrappers
# =====================================================================


def run_test_phase(
    test_df: pd.DataFrame,
    config: ExperimentConfig,
    threshold: float = 0.5,
    train_normal_features: Optional[np.ndarray] = None,
    progress_callback: Optional[Any] = None,
) -> PhaseResult:
    """Phase 2: Baseline evaluation with NO feedback."""
    return evaluate_phase(
        test_df,
        config,
        phase="test",
        feedback_enabled=False,
        threshold=threshold,
        train_normal_features=train_normal_features,
        progress_callback=progress_callback,
    )


def run_eval_phase(
    eval_df: pd.DataFrame,
    config: ExperimentConfig,
    threshold: float = 0.5,
    warm_priors_path: Optional[Path] = None,
    train_normal_features: Optional[np.ndarray] = None,
    progress_callback: Optional[Any] = None,
) -> PhaseResult:
    """Phase 3: Evaluation WITH feedback active.

    Parameters
    ----------
    warm_priors_path : Path or None
        If config.eval_variant == "warm", pass the priors from a simulated
        test-with-feedback run here.
    train_normal_features : ndarray or None
        Normal training features for online retraining.
    """
    initial_priors = warm_priors_path if config.eval_variant == "warm" else None
    return evaluate_phase(
        eval_df,
        config,
        phase="eval",
        feedback_enabled=True,
        threshold=threshold,
        initial_priors_path=initial_priors,
        train_normal_features=train_normal_features,
        progress_callback=progress_callback,
    )

"""Model-score enrichment for memory events.

Extracted from ``MemoryEventOrchestrator`` (Phase 2). Running the classical
anomaly detectors and the harmonic scorer over an event, and folding their
output into the event's external signals, is scoring work rather than
coordination.

Both are free functions: their collaborators (the detectors, and the harmonic
row-history buffer the caller owns) are passed in explicitly instead of being
reached for through ``self``. ``row_history`` is mutated in place — it is the
caller's rolling per-session window.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # orchestrator imports this module — keep the arrow one-way
    from .orchestrator import MemoryEvent

logger = logging.getLogger(__name__)


def enrich_with_classical_scores(
    event: MemoryEvent, *, anomaly_detector: Any
) -> Dict[str, Any]:
    """Extract features from event and score with the classical detector.

    [PROTOTYPE_CLASSICAL_RL_V1]
    Builds a feature dict from the event's raw_metrics (preferred, 17 CNC
    features), falling back to WindowMetrics and external_signals.

    Returns:
        Dict of external signals (anomaly_detector_score, model_confidence, etc.)
    """
    feature_dict: Dict[str, float] = {}

    # Prefer raw_metrics (flat CNC feature dict) — this is the sync fix.
    # raw_metrics uses the same feature names as FEATURE_NAMES in
    # classical_models.py, ensuring the model scores the same data
    # that generated the pattern_keys.
    if event.raw_metrics:
        for key, value in event.raw_metrics.items():
            if isinstance(value, (int, float)):
                feature_dict[str(key)] = float(value)
    elif event.metrics:
        # Fallback: generic WindowMetrics (bridge / real-time path)
        # metrics may be a dict or a WindowMetrics dataclass
        import dataclasses as _dc
        m = _dc.asdict(event.metrics) if _dc.is_dataclass(event.metrics) else event.metrics
        for key, value in m.items():
            if isinstance(value, (int, float)):
                feature_dict[str(key)] = float(value)

    # Also pull numeric values from external_signals (but don't overwrite
    # features that came from raw_metrics / metrics).
    if event.external_signals:
        for key, value in event.external_signals.items():
            if isinstance(value, (int, float)) and key not in feature_dict:
                feature_dict[str(key)] = float(value)

    cutting_ctx = (
        event.cutting_context.model_dump() if event.cutting_context else None
    )

    return anomaly_detector.score_window(
        feature_dict,
        cutting_context=cutting_ctx,
    )


def enrich_with_harmonic_score(
    event: MemoryEvent, *, harmonic_scorer: Any, row_history: Dict[str, Any]
) -> Dict[str, Any]:
    """Score the event using the harmonic context-weighted CNN.

    [HARMONIC_CONTEXT_V1]
    Extracts harmonic features + context params from the event's
    raw_metrics (pre_extracted mode) and returns signals suitable
    for merging into external_signals.
    """
    if harmonic_scorer is None or not harmonic_scorer.is_available():
        return {}

    existing = event.external_signals or {}
    if isinstance(existing.get("harmonic_context_score"), (int, float)):
        return {}

    import numpy as np
    cfg = harmonic_scorer.config

    # Build feature dict from event
    feature_dict: Dict[str, float] = {}
    if event.raw_metrics:
        feature_dict.update({
            k: float(v) for k, v in event.raw_metrics.items()
            if isinstance(v, (int, float))
        })
    if event.external_signals:
        for k, v in event.external_signals.items():
            if isinstance(v, (int, float)) and k not in feature_dict:
                feature_dict[k] = float(v)

    if not feature_dict:
        return {}

    try:
        from ..processing.harmonic_features import (
            extract_context_params,
            extract_harmonic_matrix_from_df,
            resolve_spindle_speed_source_column,
            runtime_context_normalize,
            runtime_context_param_stats,
        )
        from ..processing.harmonic_peak_pairs import (
            build_pair_feature_labels,
            discover_peak_pair_columns,
            extract_peak_pairs_from_df,
        )
        import pandas as _pd

        # Extract context params
        runtime_stats = runtime_context_param_stats(cfg)
        normalize_context = runtime_context_normalize(cfg)
        ctx_vec = extract_context_params(
            feature_dict,
            cfg.context_param_keys,
            {k: v.get("source_column", k) for k, v in cfg.context_param_stats.items()}
            if cfg.context_param_stats else cfg.context_param_sources,
            runtime_stats,
            normalize=normalize_context,
        )

        scorer_kind = str(getattr(cfg, "scorer_kind", "context") or "context").strip().lower()

        # Build the current event row, then accumulate a real per-session
        # history instead of relying on zero padding inside the scorer for
        # short windows.
        row_df = _pd.DataFrame([feature_dict])
        maxlen = max(1, int(getattr(cfg, "cnn_window", 1)))
        session_key = f"{scorer_kind}:{event.session_id or '__global__'}"
        history = row_history.get(session_key)
        if history is None or history.maxlen != maxlen:
            history = deque(maxlen=maxlen)
            row_history[session_key] = history

        if scorer_kind == "pair":
            pair_specs = discover_peak_pair_columns(
                list(row_df.columns),
                frequency_patterns=list(getattr(cfg, "pair_frequency_column_patterns", []) or []),
                amplitude_patterns=list(getattr(cfg, "pair_amplitude_column_patterns", []) or []),
                k_peaks=int(getattr(cfg, "k_peaks", 5)),
            )
            if not pair_specs:
                return {}

            pair_labels = list(getattr(cfg, "harmonic_columns", []) or [])
            if not pair_labels:
                pair_labels = build_pair_feature_labels(pair_specs)
                cfg.harmonic_columns = pair_labels

            spindle_speed_col = resolve_spindle_speed_source_column(cfg)
            p_mat = extract_peak_pairs_from_df(
                row_df,
                pair_specs,
                spindle_speed_col=spindle_speed_col,
                k_peaks=int(getattr(cfg, "k_peaks", 5)),
                f_max_rel=float(getattr(cfg, "f_max_rel", 12.0)),
            )
            if p_mat.shape[0] == 0 or p_mat.shape[1] == 0:
                return {}

            for row in np.asarray(p_mat, dtype=np.float32):
                history.append(np.asarray(row, dtype=np.float32))
        else:
            h_mat = extract_harmonic_matrix_from_df(row_df, cfg.harmonic_columns)
            if h_mat.shape[0] == 0 or h_mat.shape[1] == 0:
                return {}

            for row in np.asarray(h_mat, dtype=np.float32):
                history.append(np.asarray(row, dtype=np.float32))

        if len(history) < maxlen:
            return {}

        h_window = np.asarray(list(history), dtype=np.float32)
        result = harmonic_scorer.score(h_window, ctx_vec)
        threshold = result.get(
            "decision_threshold",
            getattr(cfg, "decision_threshold", 0.5),
        )
        numeric_threshold = float(threshold) if isinstance(threshold, (int, float)) else 0.5
        score_value = float(result.get("harmonic_context_score", 0.5))
        return {
            "harmonic_context_score": score_value,
            "harmonic_context_source": result.get("model_source", ""),
            "harmonic_context_threshold": numeric_threshold,
            "harmonic_context_triggered": bool(score_value >= numeric_threshold),
        }
    except Exception as e:
        logger.debug("Harmonic context enrichment error: %s", e)
        return {}

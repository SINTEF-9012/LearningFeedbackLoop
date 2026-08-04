"""Assembly of the evidence bundle handed to the explainer.

Extracted from ``MemoryEventOrchestrator`` (Phase 2). The orchestrator's job is
to coordinate; gathering feature evidence, model scores, per-pattern feedback
stats, retrieved similar memories and cutting context into one
:class:`~backend.agents.llm.explainer.ExplanationContext` is its own concern,
and it only ever needed two collaborators — the scorer and the store — which are
now passed in explicitly rather than reached for through ``self``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from .retriever import MemoryMatch
from .scorer import SignificanceResult

if TYPE_CHECKING:  # orchestrator imports this module — keep the arrow one-way
    from .orchestrator import MemoryEvent

logger = logging.getLogger(__name__)


def build_explanation_context(
    event: MemoryEvent,
    significance: SignificanceResult,
    similar_memories: List[MemoryMatch],
    *,
    scorer: Any,
    store: Any,
) -> "ExplanationContext":
    """Assemble an :class:`ExplanationContext` with all available evidence.

    Pulls data from the scorer, store (feedback & co-occurrence), and
    the raw event to give the LLM everything it needs for a grounded
    explanation.
    """
    from ..llm.explainer import ExplanationContext

    pattern_keys = [p.key for p in (event.patterns or [])]

    # --- Feature evidence: map each pattern to triggering measurements ---
    feature_evidence: Dict[str, List[Dict[str, Any]]] = {}
    raw = event.raw_metrics or {}
    if raw:
        # Try to use the pattern registry for threshold mapping
        try:
            from ..patterns.registry import get_registry
            registry = get_registry()
        except Exception:
            logger.debug("Failed to load pattern registry for explanation context", exc_info=True)
            registry = None

        # Build a threshold lookup from detector defaults so we can
        # populate the feature-evidence thresholds (Issue #9 fix).
        _default_thresholds: Dict[str, float] = {
            "power_spindle_delta_max": 15.0,
            "power_y_delta_max": 10.0,
            "vib_severity_x_delta_max": 0.8,
            "chatter_freq_x_slope": 5.0,
            "chatter_freq_x_slope_abs": 5.0,
            "feed_override_delta_mean": -10.0,
            "feed_override_min": 50.0,
            "corr_spindle_power_vib_x": 0.3,
            "power_spindle_slope": 5.0,
        }
        # Merge scorer-level thresholds if available
        if hasattr(self, "scorer") and hasattr(scorer, "config"):
            sc = scorer.config
            _default_thresholds["chatter_ratio"] = getattr(sc, "chatter_ratio_threshold", 5.0)
            _default_thresholds["anomaly_score"] = getattr(sc, "anomaly_score_threshold", 0.7)

        for pk in pattern_keys:
            ev_list: List[Dict[str, Any]] = []
            # Look up the pattern definition from the registry for column info
            pdef = registry.get(pk) if registry else None
            if pdef and pdef.columns:
                for col in pdef.columns:
                    val = raw.get(col)
                    if val is not None:
                        thresh = _default_thresholds.get(col)
                        direction = "above"
                        # Feed override threshold is negative (below = bad)
                        if thresh is not None and thresh < 0:
                            direction = "below"
                        ev_list.append({
                            "feature": col,
                            "value": float(val),
                            "threshold": thresh,
                            "direction": direction,
                        })
            else:
                # Heuristic: pattern key often references a source_metric
                for pat_obj in (event.patterns or []):
                    if pat_obj.key == pk and getattr(pat_obj, "source_metric", None):
                        src = pat_obj.source_metric
                        # Find matching columns in raw_metrics
                        for col_name, col_val in raw.items():
                            if src in col_name and isinstance(col_val, (int, float)):
                                thresh = _default_thresholds.get(col_name)
                                ev_list.append({
                                    "feature": col_name,
                                    "value": float(col_val),
                                    "threshold": thresh,
                                    "direction": "above",
                                })
            if ev_list:
                feature_evidence[pk] = ev_list

    # --- Classical model signals ---
    classical_model: Dict[str, float] = {}
    sigs = event.external_signals or {}
    for key in ("anomaly_detector_score", "model_confidence",
                 "breakage_prediction", "isolation_forest_score", "lof_score"):
        val = sigs.get(key)
        if val is not None and isinstance(val, (int, float)):
            classical_model[key] = float(val)

    # --- Feedback stats per pattern ---
    feedback_stats: Dict[str, Dict[str, Any]] = {}
    for pk in pattern_keys:
        confirms, dismisses = 0, 0
        prior_val = significance.pattern_priors.get(pk)
        # Query the store for feedback counts
        if hasattr(store, "get_feedback_counts"):
            try:
                confirms, dismisses = store.get_feedback_counts(
                    pattern_key=pk,
                )
            except Exception:
                logger.debug(
                    "Failed to load feedback counts for explanation context pattern %s",
                    pk,
                    exc_info=True,
                )
                pass
        # Fall back to scorer priors if no store feedback
        if prior_val is None:
            prior_val = scorer.get_pattern_prior(pk)
        feedback_stats[pk] = {
            "confirms": confirms,
            "dismisses": dismisses,
            "prior": float(prior_val) if prior_val is not None else None,
        }

    # --- Co-occurrence context ---
    co_occurrence: List[Dict[str, Any]] = []
    if hasattr(store, "get_co_occurring_patterns"):
        # Compute total event count per pattern for normalisation so that
        # raw co-occurrence counts are converted to a strength ratio.
        # strength = weight / max(count_a, count_b) — i.e., what fraction
        # of the more-frequent pattern's appearances include this pair.
        # This avoids high-frequency patterns dominating (Issue #6 fix).
        _pattern_event_counts: Dict[str, int] = {}
        if hasattr(store, "get_feedback_counts"):
            for pk in pattern_keys:
                try:
                    c, d = store.get_feedback_counts(pattern_key=pk)
                    _pattern_event_counts[pk] = max(c + d, 1)
                except Exception:
                    logger.debug(
                        "Failed to load co-occurrence denominator counts for pattern %s",
                        pk,
                        exc_info=True,
                    )
                    _pattern_event_counts[pk] = 1

        seen_pairs: set = set()
        for pk in pattern_keys:
            try:
                edges = store.get_co_occurring_patterns(pk, top_k=5)
                for edge in edges:
                    pair = tuple(sorted([edge["source"], edge["target"]]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        w = edge["weight"]
                        # Normalised strength (0–1)
                        denom = max(
                            _pattern_event_counts.get(edge["source"], 1),
                            _pattern_event_counts.get(edge["target"], 1),
                        )
                        edge["strength"] = round(w / denom, 2) if denom else 0
                        co_occurrence.append(edge)
            except Exception:
                logger.debug(
                    "Failed to load co-occurring patterns for explanation context pattern %s",
                    pk,
                    exc_info=True,
                )
                pass
        # Sort by weight descending, limit to 8
        co_occurrence.sort(key=lambda x: x.get("weight", 0), reverse=True)
        co_occurrence = co_occurrence[:8]

    # --- Raw metrics excerpt (top features by absolute value) ---
    raw_metrics_excerpt: Dict[str, float] = {}
    if raw:
        # Pick features that appear in evidence + top by magnitude
        evidence_cols = set()
        for ev_list in feature_evidence.values():
            for ev in ev_list:
                evidence_cols.add(ev.get("feature", ""))
        # Start with evidence columns
        for col in sorted(evidence_cols):
            if col in raw and isinstance(raw[col], (int, float)):
                raw_metrics_excerpt[col] = float(raw[col])
        # Fill up to 10 with highest-magnitude remaining features
        remaining = [
            (k, abs(float(v)))
            for k, v in raw.items()
            if isinstance(v, (int, float)) and k not in raw_metrics_excerpt
        ]
        remaining.sort(key=lambda x: x[1], reverse=True)
        for k, _ in remaining[:10 - len(raw_metrics_excerpt)]:
            raw_metrics_excerpt[k] = float(raw[k])

    return ExplanationContext(
        pattern_keys=pattern_keys,
        significance=significance,
        feature_evidence=feature_evidence,
        classical_model=classical_model,
        feedback_stats=feedback_stats,
        co_occurrence=co_occurrence,
        similar_memories=similar_memories,
        cutting_context=event.cutting_context,
        raw_metrics_excerpt=raw_metrics_excerpt,
        curated_fallback=(event.metadata or {}).get("curated_explanation"),
    )

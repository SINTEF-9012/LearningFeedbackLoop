from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
router = APIRouter()


class ThresholdRecommendation(BaseModel):
    """A single threshold/config recommendation from LLM or heuristic analysis."""

    parameter: str
    current_value: Optional[float] = None
    recommended_value: float
    reason: str


class ExperimentSummaryRequest(BaseModel):
    """Request for LLM-generated experiment analysis."""

    run_id: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    focus: str = "overall"
    include_threshold_recommendations: bool = True


class ExperimentSummaryResponse(BaseModel):
    summary: str
    recommendations: List[str] = Field(default_factory=list)
    threshold_recommendations: List[ThresholdRecommendation] = Field(default_factory=list)
    source: str = "llm"
    key_metrics: Dict[str, Any] = Field(default_factory=dict)


@router.post("/experiment/summary", response_model=ExperimentSummaryResponse)
async def experiment_llm_summary(request: ExperimentSummaryRequest):
    """Generate an LLM-powered analysis of experiment results."""
    results = request.results or {}

    phases = results.get("phases", [])
    if not phases:
        for key, fallback in [("test_phase", "test"), ("eval_phase", "eval")]:
            phase_data = results.get(key) or results.get(fallback)
            if phase_data and isinstance(phase_data, dict):
                if "phase" not in phase_data:
                    phase_data = {**phase_data, "phase": fallback.replace("_phase", "")}
                phases.append(phase_data)
        if not phases and results.get("folds"):
            for fold in results["folds"]:
                for phase_name in ("test", "eval"):
                    phase_data = fold.get(phase_name)
                    if phase_data and isinstance(phase_data, dict):
                        phases.append({**phase_data, "phase": f"{phase_name} (fold)"})

    key_metrics: Dict[str, Any] = {}
    analysis_parts: List[str] = []

    for phase_data in phases:
        phase_name = phase_data.get("phase", "unknown")
        samples = phase_data.get("sample_results", [])
        n_total = len(samples)
        n_flagged = sum(1 for sample in samples if sample.get("predicted_positive"))
        n_pre_stoppage = sum(1 for sample in samples if sample.get("label") == "pre_stoppage")
        n_correct_flags = sum(
            1
            for sample in samples
            if sample.get("predicted_positive") and sample.get("label") == "pre_stoppage"
        )
        n_false_pos = sum(
            1
            for sample in samples
            if sample.get("predicted_positive") and sample.get("label") != "pre_stoppage"
        )
        n_missed = sum(
            1
            for sample in samples
            if not sample.get("predicted_positive") and sample.get("label") == "pre_stoppage"
        )

        precision = n_correct_flags / max(n_flagged, 1)
        recall = n_correct_flags / max(n_pre_stoppage, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        key_metrics[phase_name] = {
            "total": n_total,
            "flagged": n_flagged,
            "true_positives": n_correct_flags,
            "false_positives": n_false_pos,
            "missed": n_missed,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

        analysis_parts.append(
            f"Phase '{phase_name}': {n_total} samples, {n_flagged} flagged, "
            f"TP={n_correct_flags}, FP={n_false_pos}, Missed={n_missed}, "
            f"Precision={precision:.1%}, Recall={recall:.1%}, F1={f1:.1%}"
        )

    prior_changes = results.get("prior_changes", {})
    if prior_changes:
        analysis_parts.append(f"Prior changes: {json.dumps(prior_changes, indent=2)}")

    comparison = results.get("comparison", {})
    if comparison:
        comp_parts = []
        for key in (
            "delta_f1",
            "pct_f1_improvement",
            "delta_precision",
            "delta_recall",
            "n_feedback_events",
            "test_f1",
            "eval_f1",
        ):
            value = comparison.get(key)
            if value is not None:
                comp_parts.append(f"{key}={value}")
        if comp_parts:
            analysis_parts.append(f"Comparison: {', '.join(comp_parts)}")

    all_patterns: Dict[str, int] = {}
    for phase_data in phases:
        for sample in phase_data.get("sample_results", []):
            for pattern in sample.get("detected_patterns", []):
                all_patterns[pattern] = all_patterns.get(pattern, 0) + 1
    if all_patterns:
        analysis_parts.append(f"Pattern frequencies: {json.dumps(all_patterns, indent=2)}")

    analysis_text = "\n".join(analysis_parts)
    prompt = (
        "You are a senior manufacturing process engineer reviewing an experiment "
        "that tested a tool breakage detection system.\n\n"
        f"Results:\n{analysis_text}\n\n"
        f"Focus area: {request.focus}\n\n"
        "Provide:\n"
        "1. A concise summary of the results (2-3 sentences)\n"
        "2. Key observations about the detection quality\n"
        "3. Specific, actionable recommendations to improve the system\n"
        "4. Which pattern priors should be adjusted and in which direction\n"
    )

    summary = ""
    recommendations: List[str] = []
    source = "fallback"

    try:
        from ..llm.rag import LLMAgent

        agent = LLMAgent()
        if agent.is_available():
            result = await agent.handle_request(prompt)
            if result and result.get("answer"):
                full_text = result["answer"]
                rec_section = False
                for line in full_text.split("\n"):
                    stripped = line.strip()
                    if "recommendation" in stripped.lower() or "should" in stripped.lower():
                        rec_section = True
                    if rec_section and stripped.startswith(("-", "•", "*")):
                        recommendations.append(stripped.lstrip("-•* "))
                summary = full_text
                source = "llm"
    except Exception as exc:
        logger.debug("LLM experiment summary failed: %s", exc)

    if not summary:
        parts = ["## Experiment Results Summary\n"]
        for phase_name, metrics in key_metrics.items():
            parts.append(
                f"**{phase_name}**: F1={metrics['f1']:.1%}, "
                f"Precision={metrics['precision']:.1%}, Recall={metrics['recall']:.1%}"
            )
            if metrics["false_positives"] > 3:
                recommendations.append(
                    f"High false positive rate in {phase_name} ({metrics['false_positives']} FP) - consider raising alert_threshold"
                )
            if metrics["missed"] > 2:
                recommendations.append(
                    f"Missed {metrics['missed']} events in {phase_name} - consider lowering store_threshold or increasing pattern sensitivity"
                )
        summary = "\n".join(parts)

        for pattern, count in sorted(all_patterns.items(), key=lambda item: -item[1]):
            if count >= 3:
                recommendations.append(
                    f"Pattern '{pattern}' fired {count} times - review its prior weight"
                )

    threshold_recs: List[ThresholdRecommendation] = []
    if request.include_threshold_recommendations:
        threshold_recs = _build_threshold_recommendations(
            key_metrics,
            all_patterns,
            results,
            source == "llm",
        )

    return ExperimentSummaryResponse(
        summary=summary,
        recommendations=recommendations,
        threshold_recommendations=threshold_recs,
        source=source,
        key_metrics=key_metrics,
    )


def _build_threshold_recommendations(
    key_metrics: Dict[str, Any],
    all_patterns: Dict[str, int],
    results: Dict[str, Any],
    llm_available: bool,
) -> List[ThresholdRecommendation]:
    """Build structured threshold recommendations."""
    recs: List[ThresholdRecommendation] = []

    if llm_available:
        try:
            recs = _llm_threshold_recommendations(key_metrics, all_patterns, results)
            if recs:
                return recs
        except Exception as exc:
            logger.debug("LLM threshold tuning failed, using heuristic: %s", exc)

    config = results.get("config", {})
    current_store = config.get("store_threshold", 0.3)
    current_alert = config.get("alert_threshold", 0.6)

    for phase_name, metrics in key_metrics.items():
        false_positives = metrics.get("false_positives", 0)
        missed = metrics.get("missed", 0)
        total = metrics.get("total", 1)
        precision = metrics.get("precision", 1.0)
        recall = metrics.get("recall", 1.0)

        if false_positives > 3 and precision < 0.7:
            bump = min(0.1, false_positives / total * 0.5)
            recs.append(
                ThresholdRecommendation(
                    parameter="alert_threshold",
                    current_value=current_alert,
                    recommended_value=round(min(0.9, current_alert + bump), 3),
                    reason=f"{phase_name}: {false_positives} FPs (precision {precision:.0%}) - raise to reduce false alerts",
                )
            )

        if missed > 2 and recall < 0.7:
            drop = min(0.1, missed / total * 0.5)
            recs.append(
                ThresholdRecommendation(
                    parameter="store_threshold",
                    current_value=current_store,
                    recommended_value=round(max(0.05, current_store - drop), 3),
                    reason=f"{phase_name}: {missed} missed events (recall {recall:.0%}) - lower to capture more",
                )
            )

    for pattern, count in sorted(all_patterns.items(), key=lambda item: -item[1]):
        if count >= 5:
            recs.append(
                ThresholdRecommendation(
                    parameter=f"prior:{pattern}",
                    current_value=None,
                    recommended_value=0.7,
                    reason=f"Pattern '{pattern}' fired {count} times - consider boosting its prior",
                )
            )

    return recs


def _llm_threshold_recommendations(
    key_metrics: Dict[str, Any],
    all_patterns: Dict[str, int],
    results: Dict[str, Any],
) -> List[ThresholdRecommendation]:
    """Call LLM with JSON format for structured threshold recommendations."""
    from ..llm.explainer import LLMExplainer

    explainer = LLMExplainer()
    if not explainer.is_available():
        return []

    config = results.get("config", {})
    metrics_summary = json.dumps(key_metrics, indent=2)
    patterns_summary = json.dumps(
        dict(sorted(all_patterns.items(), key=lambda item: -item[1])[:10]),
        indent=2,
    )

    prompt = (
        "You are a manufacturing process engineer tuning a tool breakage detection system.\n\n"
        f"Current config: store_threshold={config.get('store_threshold', 0.3)}, "
        f"alert_threshold={config.get('alert_threshold', 0.6)}, "
        f"prior_boost_weight={config.get('prior_boost_weight', 0.15)}\n\n"
        f"Experiment metrics:\n{metrics_summary}\n\n"
        f"Pattern frequencies:\n{patterns_summary}\n\n"
        "Return a JSON object with this exact schema:\n"
        '{"recommendations": [{"parameter": "<name>", "recommended_value": <float>, "reason": "<why>"}]}\n\n'
        "Valid parameter names: store_threshold, alert_threshold, prior_boost_weight, "
        "weight_classical_alert, weight_harmonic_alert, weight_pattern_rule, weight_anomaly_deviation, weight_historical_prior.\n"
        "Values must be floats between 0.0 and 1.0. Give 2-5 recommendations."
    )

    import httpx as _httpx

    if explainer.config.provider == "groq":
        url = f"{explainer.config.groq_api_url}/chat/completions"
        payload = {
            "model": explainer.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 600,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {explainer.config.groq_api_key}",
            "Content-Type": "application/json",
        }
        connect_timeout = float(os.environ.get("GROQ_CONNECT_TIMEOUT", "5.0"))
        resp = _httpx.post(
            url,
            json=payload,
            headers=headers,
            timeout=_httpx.Timeout(
                connect=connect_timeout,
                read=float(explainer.config.timeout),
                write=connect_timeout,
                pool=connect_timeout,
            ),
        )
        resp.raise_for_status()
        result = resp.json()
        text = ""
        choices = result.get("choices", [])
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                text = str(message.get("content") or "")
    else:
        url = str(explainer.config.ollama_url or "").replace("/api/generate", "/api/chat")
        payload = {
            "model": explainer.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"num_predict": 600},
        }
        connect_timeout = float(os.environ.get("OLLAMA_CONNECT_TIMEOUT", "5.0"))
        resp = _httpx.post(
            url,
            json=payload,
            headers={},
            timeout=_httpx.Timeout(
                connect=connect_timeout,
                read=float(explainer.config.timeout),
                write=connect_timeout,
                pool=connect_timeout,
            ),
        )
        resp.raise_for_status()
        result = resp.json()
        text = ""
        if isinstance(result, dict):
            message = result.get("message")
            if isinstance(message, dict):
                text = str(message.get("content") or "")
            if not text:
                text = str(result.get("response") or "")

    if not text:
        return []

    parsed = json.loads(text)
    raw_recs = parsed.get("recommendations", [])
    if not isinstance(raw_recs, list):
        return []

    recs: List[ThresholdRecommendation] = []
    for item in raw_recs[:6]:
        if not isinstance(item, dict):
            continue
        parameter = str(item.get("parameter", "")).strip()
        recommended_value = item.get("recommended_value")
        reason = str(item.get("reason", "")).strip()
        if parameter and recommended_value is not None and reason:
            try:
                recs.append(
                    ThresholdRecommendation(
                        parameter=parameter,
                        recommended_value=round(float(recommended_value), 4),
                        reason=reason,
                    )
                )
            except (ValueError, TypeError):
                continue
    return recs
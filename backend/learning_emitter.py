"""Helpers for publishing learnings onto the learnings bus."""

from __future__ import annotations

from copy import deepcopy
import inspect
import logging
import os
import time
from typing import Any, Dict, Optional

from .agents.core.batch_context import extract_batch_context
from .events import publish_learning
from .ingestion.schema import LearningEnvelope


logger = logging.getLogger(__name__)

_SCRUBBED_FEEDBACK_KEYS = {
    "comment",
    "comments",
    "message",
    "note",
    "notes",
    "text",
    "user_id",
    "operator_id",
    "confirmed_by",
    "dismissed_by",
}


def _learning_provenance() -> Dict[str, Optional[str]]:
    tenant_id = (
        os.environ.get("KNOWLEDGE_TENANT_ID", "").strip()
        or os.environ.get("TENANT_ID", "").strip()
        or None
    )
    site_id = (
        os.environ.get("LFL_SITE_ID", "").strip()
        or os.environ.get("SITE_ID", "").strip()
        or None
    )
    return {
        "tenant_id": tenant_id,
        "site_id": site_id,
        "pii_scrub_level": "symbolic_only",
    }


def _scrub_feedback_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    scrubbed: Dict[str, Any] = {}
    for key, value in dict(data or {}).items():
        if key in _SCRUBBED_FEEDBACK_KEYS:
            continue
        scrubbed[key] = value
    return scrubbed


async def _resolve_memory(store: Any, memory_id: str) -> Any:
    if store is None:
        return None
    try:
        if hasattr(store, "get") and callable(getattr(store, "get")):
            memory = store.get(memory_id)
        elif hasattr(store, "get_memory") and callable(getattr(store, "get_memory")):
            memory = store.get_memory(memory_id)
            if inspect.isawaitable(memory):
                memory = await memory
        else:
            return None
    except Exception:
        logger.debug("Could not resolve memory %s for learning envelope", memory_id, exc_info=True)
        return None
    return memory


def _memory_field(memory: Any, field_name: str, default: Any = None) -> Any:
    if memory is None:
        return default
    if isinstance(memory, dict):
        return memory.get(field_name, default)
    return getattr(memory, field_name, default)


async def _resolve_session_id(store: Any, memory_id: str) -> Optional[str]:
    memory = await _resolve_memory(store, memory_id)
    session_id = _memory_field(memory, "session_id")
    return str(session_id) if session_id else None


def _memory_metadata(memory: Any) -> Dict[str, Any]:
    metadata = _memory_field(memory, "metadata", {})
    return dict(metadata or {}) if isinstance(metadata, dict) else {}


def _memory_pattern_keys(memory: Any) -> list[str]:
    keys: list[str] = []
    for pattern in _memory_field(memory, "pattern_keys", []) or []:
        key = getattr(pattern, "key", None)
        if key is None and isinstance(pattern, dict):
            key = pattern.get("key")
        if key:
            keys.append(str(key))
    return keys


def _memory_cutting_context(metadata: Dict[str, Any]) -> Dict[str, Any]:
    raw = metadata.get("cutting_context")
    return dict(raw) if isinstance(raw, dict) else {}


def _memory_batch(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    batch = extract_batch_context(metadata)
    if batch is None:
        return None
    return batch.model_dump(mode="json")


def _memory_pattern_priors(metadata: Dict[str, Any]) -> Dict[str, float]:
    candidates = [
        metadata.get("pattern_priors"),
        (metadata.get("significance") or {}).get("pattern_priors") if isinstance(metadata.get("significance"), dict) else None,
        (metadata.get("external_signals") or {}).get("pattern_priors") if isinstance(metadata.get("external_signals"), dict) else None,
    ]
    out: Dict[str, float] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key, value in candidate.items():
            if isinstance(value, (int, float)):
                out[str(key)] = float(value)
    return out


def build_feedback_learning_envelope(
    memory_id: str,
    action: Any,
    data: Dict[str, Any],
    *,
    session_id: Optional[str] = None,
    batch: Optional[Dict[str, Any]] = None,
) -> LearningEnvelope:
    action_str = getattr(action, "value", None) or str(action)
    operator_id = str(
        data.get("user_id")
        or data.get("operator_id")
        or data.get("confirmed_by")
        or data.get("dismissed_by")
        or "operator"
    )
    provenance = _learning_provenance()
    return LearningEnvelope(
        kind="feedback_event",
        ts_unix=time.time(),
        session_id=str(session_id or ""),
        source="feedback_loop",
        payload={
            "memory_id": memory_id,
            "action": action_str,
            "operator_id": operator_id,
            "feedback": _scrub_feedback_payload(data),
        },
        batch=dict(batch) if isinstance(batch, dict) else None,
        tenant_id=provenance["tenant_id"],
        site_id=provenance["site_id"],
        pii_scrub_level=provenance["pii_scrub_level"],
    )


def build_tool_learning_envelope(
    *,
    session_id: str,
    action: str,
    tool_snapshot: Dict[str, Any],
    pattern_keys: Optional[list[str]] = None,
    memory_id: Optional[str] = None,
    significance_score: Optional[float] = None,
    alert_dispatched: Optional[bool] = None,
    pattern_priors: Optional[Dict[str, float]] = None,
    operator_id: Optional[str] = None,
    batch: Optional[Dict[str, Any]] = None,
) -> LearningEnvelope:
    provenance = _learning_provenance()
    payload: Dict[str, Any] = {
        "action": str(action),
        "machine_id": tool_snapshot.get("machine_id"),
        "machine_family": tool_snapshot.get("machine_family"),
        "tool_number": tool_snapshot.get("tool_number"),
        "tool_id": tool_snapshot.get("tool_id"),
        "tool_uri": tool_snapshot.get("tool_uri"),
        "harmonic_ready": bool(tool_snapshot.get("harmonic_ready")),
        "effective_ctx": deepcopy(tool_snapshot.get("effective_ctx") or {}),
        "anomaly_stats": deepcopy(tool_snapshot.get("anomaly_stats") or {}),
        "patterns": list(pattern_keys or []),
    }
    if memory_id:
        payload["memory_id"] = str(memory_id)
    if significance_score is not None:
        payload["significance_score"] = float(significance_score)
    if alert_dispatched is not None:
        payload["alert_dispatched"] = bool(alert_dispatched)
    if pattern_priors:
        payload["pattern_priors"] = dict(pattern_priors)
    if operator_id:
        payload["operator_id"] = str(operator_id)

    return LearningEnvelope(
        kind="tool_event",
        ts_unix=time.time(),
        session_id=str(session_id or ""),
        source="tool_audit",
        payload=payload,
        batch=dict(batch) if isinstance(batch, dict) else None,
        tenant_id=provenance["tenant_id"],
        site_id=provenance["site_id"],
        pii_scrub_level=provenance["pii_scrub_level"],
    )


def _pattern_keys(patterns: Any) -> list[str]:
    keys: list[str] = []
    for pattern in patterns or []:
        key = getattr(pattern, "key", None)
        if key is None and isinstance(pattern, dict):
            key = pattern.get("key")
        keys.append(str(key if key is not None else pattern))
    return keys


def _serialize_significance(significance: Any) -> Dict[str, Any]:
    if significance is None:
        return {}

    serializer = getattr(significance, "to_dict", None)
    if callable(serializer):
        try:
            serialized = serializer()
        except Exception:
            logger.debug("Could not serialize significance via to_dict", exc_info=True)
        else:
            if isinstance(serialized, dict):
                return dict(serialized)

    payload: Dict[str, Any] = {}
    for field_name in (
        "score",
        "is_significant",
        "reasons",
        "triggered_rules",
        "pattern_priors",
        "prior_boost",
        "score_trace",
    ):
        value = getattr(significance, field_name, None)
        if value is not None:
            payload[field_name] = value

    action = getattr(significance, "action", None)
    if action is not None:
        payload["action"] = getattr(action, "value", str(action))

    return payload


def _serialize_time_range(time_range: Any) -> Optional[Dict[str, Any]]:
    if time_range is None:
        return None

    payload: Dict[str, Any] = {}
    for field_name in ("i0", "i1", "t0", "t1", "fs"):
        value = getattr(time_range, field_name, None)
        if value is not None:
            payload[field_name] = value

    return payload or None


def build_scored_learning_envelope(
    *,
    session_id: str,
    memory_id: Optional[str],
    significance: Any,
    patterns: Any,
    external_signals: Optional[Dict[str, Any]] = None,
    model_breakdown: Optional[Dict[str, Any]] = None,
    alert_dispatched: bool = False,
    similar_memory_count: int = 0,
    time_range: Any = None,
    batch: Optional[Dict[str, Any]] = None,
) -> LearningEnvelope:
    provenance = _learning_provenance()
    payload: Dict[str, Any] = {
        "patterns": _pattern_keys(patterns),
        "significance": _serialize_significance(significance),
        "external_signals": dict(external_signals or {}),
        "model_breakdown": dict(model_breakdown or {}),
        "alert_dispatched": bool(alert_dispatched),
        "similar_memory_count": int(similar_memory_count),
    }
    if memory_id:
        payload["memory_id"] = str(memory_id)

    serialized_time_range = _serialize_time_range(time_range)
    if serialized_time_range is not None:
        payload["time_range"] = serialized_time_range

    return LearningEnvelope(
        kind="scored_event",
        ts_unix=time.time(),
        session_id=str(session_id or ""),
        source="memory_orchestrator",
        payload=payload,
        batch=dict(batch) if isinstance(batch, dict) else None,
        tenant_id=provenance["tenant_id"],
        site_id=provenance["site_id"],
        pii_scrub_level=provenance["pii_scrub_level"],
    )


async def publish_scored_learning(
    *,
    session_id: str,
    memory_id: Optional[str],
    significance: Any,
    patterns: Any,
    external_signals: Optional[Dict[str, Any]] = None,
    model_breakdown: Optional[Dict[str, Any]] = None,
    alert_dispatched: bool = False,
    similar_memory_count: int = 0,
    time_range: Any = None,
    batch: Optional[Dict[str, Any]] = None,
) -> None:
    envelope = build_scored_learning_envelope(
        session_id=session_id,
        memory_id=memory_id,
        significance=significance,
        patterns=patterns,
        external_signals=external_signals,
        model_breakdown=model_breakdown,
        alert_dispatched=alert_dispatched,
        similar_memory_count=similar_memory_count,
        time_range=time_range,
        batch=batch,
    )
    await publish_learning(envelope)


async def publish_tool_learning(
    *,
    session_id: str,
    action: str,
    tool_snapshot: Dict[str, Any],
    pattern_keys: Optional[list[str]] = None,
    memory_id: Optional[str] = None,
    significance_score: Optional[float] = None,
    alert_dispatched: Optional[bool] = None,
    pattern_priors: Optional[Dict[str, float]] = None,
    operator_id: Optional[str] = None,
    batch: Optional[Dict[str, Any]] = None,
) -> None:
    envelope = build_tool_learning_envelope(
        session_id=session_id,
        action=action,
        tool_snapshot=tool_snapshot,
        pattern_keys=pattern_keys,
        memory_id=memory_id,
        significance_score=significance_score,
        alert_dispatched=alert_dispatched,
        pattern_priors=pattern_priors,
        operator_id=operator_id,
        batch=batch,
    )
    await publish_learning(envelope)


def build_insight_learning_envelope(
    *,
    session_id: str,
    memory_id: str,
    explanation: Optional[str],
    explanation_source: Optional[str],
    alert_line: Optional[str],
    alert_line_source: Optional[str],
    batch: Optional[Dict[str, Any]] = None,
) -> Optional[LearningEnvelope]:
    if not explanation and not alert_line:
        return None

    provenance = _learning_provenance()
    payload: Dict[str, Any] = {"memory_id": str(memory_id)}
    if explanation:
        payload["explanation"] = explanation
        payload["explanation_source"] = explanation_source
    if alert_line:
        payload["alert_line"] = alert_line
        payload["alert_line_source"] = alert_line_source

    return LearningEnvelope(
        kind="insight_event",
        ts_unix=time.time(),
        session_id=str(session_id or ""),
        source="llm_explainer",
        payload=payload,
        batch=dict(batch) if isinstance(batch, dict) else None,
        tenant_id=provenance["tenant_id"],
        site_id=provenance["site_id"],
        pii_scrub_level=provenance["pii_scrub_level"],
    )


async def publish_insight_learning(
    *,
    session_id: str,
    memory_id: str,
    explanation: Optional[str],
    explanation_source: Optional[str],
    alert_line: Optional[str],
    alert_line_source: Optional[str],
    batch: Optional[Dict[str, Any]] = None,
) -> None:
    envelope = build_insight_learning_envelope(
        session_id=session_id,
        memory_id=memory_id,
        explanation=explanation,
        explanation_source=explanation_source,
        alert_line=alert_line,
        alert_line_source=alert_line_source,
        batch=batch,
    )
    if envelope is None:
        return
    await publish_learning(envelope)


def create_feedback_learning_callback(store: Any):
    async def _callback(memory_id: str, action: Any, data: Dict[str, Any]) -> None:
        memory = await _resolve_memory(store, memory_id)
        session_id = _memory_field(memory, "session_id") or ""
        metadata = _memory_metadata(memory)
        batch = _memory_batch(metadata)
        envelope = build_feedback_learning_envelope(
            memory_id,
            action,
            data,
            session_id=session_id,
            batch=batch,
        )
        await publish_learning(envelope)

        cutting_context = _memory_cutting_context(metadata)
        if not cutting_context:
            return

        operator_id = envelope.payload.get("operator_id")
        pattern_keys = _memory_pattern_keys(memory)
        pattern_priors = _memory_pattern_priors(metadata)
        action_str = envelope.payload.get("action") or str(action)

        try:
            from .agents.sindit.tool_audit import record_tool_feedback

            tool_snapshot = record_tool_feedback(
                str(session_id or ""),
                cutting_context,
                action=action_str,
                memory_id=memory_id,
                pattern_keys=pattern_keys,
                pattern_priors=pattern_priors,
                operator_id=str(operator_id) if operator_id else None,
            )
        except Exception:
            logger.debug("Tool feedback recording skipped for %s", memory_id, exc_info=True)
            return

        if tool_snapshot is None:
            return

        await publish_tool_learning(
            session_id=str(session_id or ""),
            action=str(action_str),
            tool_snapshot=tool_snapshot,
            pattern_keys=pattern_keys,
            memory_id=memory_id,
            pattern_priors=pattern_priors,
            operator_id=str(operator_id) if operator_id else None,
            batch=batch,
        )

    return _callback
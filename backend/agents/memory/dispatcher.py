"""
Alert Dispatcher - Push significant events to connected clients.

# ===========================================================================
# DRAFT/PROTOTYPE - Tag: [PROTOTYPE_LLM_MEMORY_V1]
# This module handles WebSocket broadcasting of significant events.
# Simple implementation for prototyping - production may need Redis pub/sub.
# ===========================================================================

Responsibilities:
1. Manage client subscriptions for alerts
2. Format and dispatch significant events
3. Rate limiting to avoid alert fatigue
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Any, Set
from collections import defaultdict

from backend.agents.llm.alert_doc_linker import propose_alert_doc_links
from backend.agents.llm.docs_backend import get_docs_backend
from backend.agents.usecase import resolve_usecase
from backend.session_logs import append_session_log

from ..core.schemas import Memory, PatternKey, MemoryQueryResult
from ..patterns.signatures import normalize_signature_key
from .scorer import SignificanceResult, SignificanceAction

logger = logging.getLogger(__name__)

SessionPauseHandler = Callable[[str, Dict[str, Any]], Any]


_PATTERN_DISPLAY_NAMES: dict[str, str] = {
    "signature:hf_burst_periodicity_loss": "High-frequency burst with periodicity loss",
    "signature:modulated_tooth_passing_vibration": "Modulated tooth-passing vibration",
    "signature:irregular_tooth_passing": "Irregular tooth-passing pattern",
    "signature:spindle_shift_phase_change": "Spindle-order shift with phase change",
    "spectral:hf_burst": "High-frequency energy burst",
    "spectral:modulated_vibration": "Modulated vibration",
    "spectral:irregular_tooth_passing": "Irregular tooth-passing pattern",
    "spectral:spindle_freq_shift": "Spindle-frequency shift",
    "temporal:periodicity_loss": "Loss of periodicity",
    "temporal:impulsive_burst": "Impulsive burst",
    "temporal:phase_shift": "Phase shift",
    "amp:loud": "High amplitude",
    "amp:increasing": "Rising amplitude",
    "kurtosis:heavy-tails": "Heavy-tailed impulse",
}


def _append_dispatch_log(session_id: str, payload: Dict[str, Any]) -> None:
    try:
        append_session_log(session_id, payload)
    except Exception:
        logger.debug("Session log append failed for %s", session_id, exc_info=True)


def _pattern_display_name(pattern_key: str) -> str:
    canonical = normalize_signature_key(pattern_key)
    label = _PATTERN_DISPLAY_NAMES.get(canonical) or _PATTERN_DISPLAY_NAMES.get(str(pattern_key).strip())
    if label:
        return label

    raw = canonical if canonical.startswith("signature:") else str(pattern_key).strip()
    text = raw.replace("_", " ").replace(":", " ").strip()
    return text[:1].upper() + text[1:] if text else "Observation"


def _extract_signature_payload(patterns: List[PatternKey]) -> Optional[Dict[str, Any]]:
    if not patterns:
        return None

    by_canonical: Dict[str, PatternKey] = {}
    signature_patterns: List[tuple[str, PatternKey]] = []
    for pattern in patterns:
        raw_key = str(getattr(pattern, "key", "") or "").strip()
        if not raw_key:
            continue
        canonical = normalize_signature_key(raw_key)
        by_canonical.setdefault(canonical, pattern)
        if canonical.startswith("signature:"):
            signature_patterns.append((canonical, pattern))

    if not signature_patterns:
        return None

    signature_key, primary = max(
        signature_patterns,
        key=lambda item: float(getattr(item[1], "confidence", 0.0) or 0.0),
    )
    primary_meta = dict(getattr(primary, "additional", {}) or {})
    supporting_patterns = [
        str(key).strip()
        for key in (primary_meta.get("supporting_patterns") or [])
        if str(key).strip()
    ]
    indicators_present = primary_meta.get("indicators_present")
    indicators_required = primary_meta.get("indicators_required")

    indicator_details: List[Dict[str, Any]] = []
    for support_key in supporting_patterns:
        canonical_support = normalize_signature_key(support_key)
        matched = by_canonical.get(canonical_support)
        matched_meta = dict(getattr(matched, "additional", {}) or {}) if matched is not None else {}
        confidence = None
        raw_conf = getattr(matched, "confidence", None) if matched is not None else None
        if isinstance(raw_conf, (int, float)):
            confidence = float(raw_conf)
        elif isinstance(matched_meta.get("confidence"), (int, float)):
            confidence = float(matched_meta["confidence"])

        source_metric = None
        raw_source_metric = getattr(matched, "source_metric", None) if matched is not None else None
        if isinstance(raw_source_metric, str) and raw_source_metric.strip():
            source_metric = raw_source_metric.strip()
        elif isinstance(matched_meta.get("source_metric"), str):
            source_metric = str(matched_meta["source_metric"]).strip() or None

        reason = str(matched_meta.get("reason") or "").strip() or None
        indicator_details.append({
            "key": canonical_support if canonical_support.startswith("signature:") else support_key,
            "label": _pattern_display_name(support_key),
            "confidence": confidence,
            "reason": reason,
            "source_metric": source_metric,
        })

    return {
        "primary_observation_key": signature_key,
        "primary_observation_label": _pattern_display_name(signature_key),
        "indicators_present": int(indicators_present) if isinstance(indicators_present, (int, float)) else None,
        "indicators_required": int(indicators_required) if isinstance(indicators_required, (int, float)) else None,
        "indicator_details": indicator_details,
    }


def _signature_summary(
    signature_payload: Dict[str, Any],
    *,
    similar_count: int = 0,
    similar_history: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, str]:
    label = str(signature_payload.get("primary_observation_label") or "Observation").strip() or "Observation"
    present = signature_payload.get("indicators_present")
    required = signature_payload.get("indicators_required")

    headline = f"Possible {label}"
    if isinstance(present, int) and isinstance(required, int) and required > 0:
        headline += f" — {present}/{required} indicators present"

    effective_similar_count = len(similar_history or []) or similar_count
    tail_parts: List[str] = []
    if effective_similar_count:
        tail_parts.append(f"{effective_similar_count} similar events in memory")
    label_hint = _historical_label_hint(similar_history)
    if label_hint:
        tail_parts.append(f'historical label "{label_hint}"')
    if tail_parts:
        headline += f" ({'; '.join(tail_parts)})"
    return headline, "fallback"


# ── Severity / category derivation ────────────────────────────────────────────

def _derive_severity(
    score: float,
    critical: float = 0.85,
    warning: float = 0.6,
) -> str:
    """Map a significance score to a severity label."""
    if score >= critical:
        return "CRITICAL"
    if score >= warning:
        return "WARNING"
    return "INFO"


_CATEGORY_RULES: list[tuple[str, str]] = [
    ("signature:modulated_tooth_passing_vibration", "Vibration Modulation"),
    ("signature:irregular_tooth_passing", "Tooth-Passing Irregularity"),
    ("signature:spindle_shift_phase_change", "Spindle-Order Shift"),
    ("signature:hf_burst_periodicity_loss", "High-Frequency Burst"),
    ("chatter",  "Vibration Modulation"),
    ("ratio_fx_fy", "Force Ratio Shift"),
    ("anomaly",  "Anomaly"),
    ("spectral", "Frequency"),
    ("freq:",    "Frequency"),
    ("amp:",     "Amplitude"),
    ("external", "External"),
    ("slip",     "Spindle-Order Shift"),
    ("adhesion", "Tooth-Passing Irregularity"),
    ("breakage", "High-Frequency Burst"),
    ("wear",     "Load Drift"),
]


def _derive_category(pattern_keys: List[str]) -> str:
    """Derive a human-readable category from pattern keys (first match)."""
    for pk in pattern_keys:
        lower = pk.lower()
        for substr, label in _CATEGORY_RULES:
            if substr in lower:
                return label
    return ""


_RUNTIME_REASON_LABELS: dict[str, str] = {
    "chatter severity": "Vibration modulation severity",
    "workpiece slip likelihood": "Spindle-order shift likelihood",
    "significant pattern": "Significant observation pattern",
    "critical pattern type": "Critical observation type",
}

_RUNTIME_TERM_REWRITES: list[tuple[str, str]] = [
    ("Tool breakage", "High-frequency burst with periodicity loss"),
    ("tool breakage", "high-frequency burst with periodicity loss"),
    ("Tool break", "High-frequency burst with periodicity loss"),
    ("tool break", "high-frequency burst with periodicity loss"),
    ("Chatter", "Vibration modulation"),
    ("chatter", "vibration modulation"),
    ("Chip adhesion", "Tooth-passing irregularity"),
    ("chip adhesion", "tooth-passing irregularity"),
    ("Workpiece slip", "Spindle-order shift"),
    ("workpiece slip", "spindle-order shift"),
    ("Tool wear", "Baseline load drift"),
    ("tool wear", "baseline load drift"),
]


def _rewrite_runtime_reason(reason: str) -> str:
    raw = str(reason or "").strip()
    if not raw:
        return ""

    normalized = raw.replace("_", " ")
    label, sep, remainder = normalized.partition(":")
    mapped_label = _RUNTIME_REASON_LABELS.get(label.strip().lower())
    if sep and mapped_label:
        normalized = f"{mapped_label}: {remainder.strip()}"

    for old, new in _RUNTIME_TERM_REWRITES:
        normalized = normalized.replace(old, new)
    return normalized.strip()


def _historical_label_hint(similar_history: Optional[List[Dict[str, Any]]]) -> str:
    for entry in similar_history or []:
        label = str((entry or {}).get("label") or "").strip()
        if label:
            return label.replace("_", " ").replace("-", " ").strip()
    return ""


def _reason_suppression_signature(reasons: Optional[List[str]]) -> Optional[str]:
    labels: List[str] = []
    for raw_reason in reasons or []:
        rewritten = _rewrite_runtime_reason(str(raw_reason or ""))
        label, _, _ = rewritten.partition(":")
        normalized = label.strip().lower().replace("_", " ")
        if not normalized:
            continue
        if normalized in {"alert", "store", "ignore", "routine observation"}:
            continue
        labels.append(normalized)

    unique = sorted(set(labels))
    if not unique:
        return None
    return "reason:" + "|".join(unique[:3])


def _fallback_summary(
    pattern_keys: List[str],
    reasons: List[str],
    score: float,
    *,
    similar_count: int = 0,
    similar_history: Optional[List[Dict[str, Any]]] = None,
    default_reason: str = "Routine observation",
) -> tuple[str, str]:
    reason = ""
    for raw_reason in reasons:
        candidate = _rewrite_runtime_reason(str(raw_reason))
        if candidate and candidate.lower() not in {"alert", "store", "ignore"}:
            reason = candidate
            break

    if not reason:
        reason = _derive_category(pattern_keys) or default_reason

    effective_similar_count = len(similar_history or []) or similar_count
    tail_parts: List[str] = []
    if effective_similar_count:
        tail_parts.append(f"{effective_similar_count} similar events in memory")
    label_hint = _historical_label_hint(similar_history)
    if label_hint:
        tail_parts.append(f'historical label "{label_hint}"')
    tail = f" ({'; '.join(tail_parts)})" if tail_parts else ""
    return f"{reason} \u2014 significance {score:.0%}{tail}", "fallback"


def _serialize_time_range(time_range: Any) -> Optional[Dict[str, Any]]:
    """Return a JSON-safe time range dict when the source object carries one."""
    if time_range is None:
        return None
    if hasattr(time_range, "model_dump"):
        try:
            return time_range.model_dump()
        except Exception:
            pass
    if isinstance(time_range, (tuple, list)) and len(time_range) >= 2:
        try:
            return {
                "t0": float(time_range[0]),
                "t1": float(time_range[1]),
            }
        except Exception:
            return None

    out: Dict[str, Any] = {}
    for key in ("i0", "i1", "t0", "t1", "fs"):
        value = getattr(time_range, key, None)
        if value is None:
            continue
        if key in ("i0", "i1"):
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out or None


# ── Demo LLM summaries ────────────────────────────────────────────────────────
# When no real LLM is available, these provide realistic examples of what an
# LLM-generated alert description looks like in the UI.  Keyed by substring
# match against pattern_keys (case-insensitive, first match wins).

_DEMO_LLM_SUMMARIES: list[tuple[str, str]] = [
    ("chatter", (
        "Modulated vibration observed during cutting — the dominant frequency "
        "component at ~480 Hz is elevated relative to the local baseline and "
        "aligns with a tooth-passing modulation pattern at the current spindle "
        "speed. "
        "Recommended action: reduce depth of cut by 15-20% or shift spindle "
        "speed away from the current excitation band. "
        "This observation pattern has repeated in recent similar "
        "similar aluminium roughing operations."
    )),
    ("power_spike", (
        "Sustained spindle load increase observed — spindle power has been at 92% "
        "capacity for over 8 seconds during heavy roughing on 4140 steel. "
        "Vibration severity is 18.7 mm/s (critical threshold: 11.2 mm/s), "
        "and the force ratio Fx/Fy has diverged beyond 5:1, indicating "
        "asymmetric cutting loads or changing radial engagement. "
        "Recommended action: reduce radial depth of cut to 70% of tool diameter "
        "and verify coolant flow to the cutting zone."
    )),
    ("anomaly", (
        "Anomalous deviation detected by the ensemble model (IF + LOF) — the current "
        "window shows a significant departure from the learned baseline in "
        "both the force ratio and spectral energy distribution. The anomaly "
        "score of 0.75 exceeds the alert threshold. "
        "Monitor closely over the next few "
        "cutting passes."
    )),
    ("breakage", (
        "High-frequency burst with periodicity loss observed — force peaks in the Z-axis have "
        "increased 40% over the last 3 windows while spindle current shows "
        "intermittent spikes. This same observation pattern has appeared in "
        "historically similar events involving titanium alloys. "
        "Immediate recommendation: pause at the next safe opportunity and inspect "
        "the cutting edge before continuing."
    )),
    ("wear", (
        "Progressive baseline load drift observed — the cutting force baseline "
        "has drifted upward by 12% compared to the start of this operation, "
        "and the surface finish proxy (high-frequency vibration energy) is "
        "rising. Consider scheduling an inspection or tool change at the next "
        "natural break point to avoid unplanned downtime."
    )),
    ("spectral_peak", (
        "Unusual spectral peak identified at ~380 Hz — this frequency does "
        "not correspond to any expected harmonic of the spindle speed or "
        "tooth-passing frequency. It may reflect structural resonance or "
        "another excitation mode at the current depth "
        "of cut. The peak amplitude is 2.3× the baseline level."
    )),
]


def _demo_llm_summary(
    pattern_keys: List[str],
    score: float,
    similar_count: int = 0,
) -> Optional[tuple[str, str]]:
    """Return (summary, 'llm') if pattern keys match a demo LLM template."""
    joined = " ".join(pattern_keys).lower()
    for substr, template in _DEMO_LLM_SUMMARIES:
        if substr in joined:
            return template, "llm"
    return None


# [PROTOTYPE_LLM_MEMORY_V1] - Alert message schema
@dataclass
class SignificantEventAlert:
    """Alert message pushed to clients."""
    event_id: str  # Usually memory_id
    session_id: str
    timestamp: datetime
    
    # Significance info
    significance_score: float
    action: SignificanceAction
    reasons: List[str]
    
    # Pattern info
    pattern_keys: List[str]
    primary_observation_key: Optional[str] = None
    primary_observation_label: Optional[str] = None
    indicators_present: Optional[int] = None
    indicators_required: Optional[int] = None
    indicator_details: List[Dict[str, Any]] = field(default_factory=list)

    # Transport metadata
    message_type: str = "significant_event"
    time_range: Optional[Dict[str, Any]] = None

    # Derived taxonomy — computed server-side so the UI doesn't have to
    severity: str = ""         # CRITICAL | WARNING | INFO
    category: str = ""         # Chatter | Anomaly | Frequency | …
    persistence_label: Optional[str] = None  # candidate | recurring

    # Prior learning info
    prior_boost: float = 0.0   # Additive boost in additive mode; raw prior in multiplicative mode
    pattern_priors: Dict[str, float] = field(default_factory=dict)  # pattern → prior
    historical_prior: Optional[float] = None
    prior_factor: Optional[float] = None
    prior_mode: Optional[str] = None  # "multiplicative" | "additive" — disambiguates prior_boost

    # Ordered component-contribution trace from the scorer (explainability)
    score_trace: List[Dict[str, Any]] = field(default_factory=list)

    # Recurrence lifecycle for this signature (set by dispatcher just before emit)
    recurrence: Optional[Dict[str, Any]] = None
    
    # Summary (LLM-generated if available)
    summary: Optional[str] = None

    # Where the summary came from ('llm' | 'fallback' | None)
    summary_source: Optional[str] = None

    # Detailed explanation (grounded, multi-sentence) — separate from short summary
    explanation: Optional[str] = None
    explanation_source: Optional[str] = None

    # Documentation citations proposed for the alert at emit time.
    doc_links: List[Dict[str, Any]] = field(default_factory=list)
    
    # Related memories (if retrieved)
    similar_memory_ids: List[str] = field(default_factory=list)
    similar_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context
    cutting_context: Optional[Dict[str, Any]] = None
    metrics_summary: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        significance: Dict[str, Any] = {
            "score": self.significance_score,
            "action": self.action.value,
            "reasons": self.reasons,
            "prior_boost": self.prior_boost,
            "pattern_priors": self.pattern_priors,
        }
        if self.historical_prior is not None:
            significance["historical_prior"] = self.historical_prior
        if self.prior_factor is not None:
            significance["prior_factor"] = self.prior_factor
        if self.prior_mode is not None:
            significance["prior_mode"] = self.prior_mode
        if self.score_trace:
            significance["score_trace"] = self.score_trace

        return {
            "type": self.message_type,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "time_range": self.time_range,
            "severity": self.severity,
            "category": self.category,
            "persistence_label": self.persistence_label,
            "recurrence": self.recurrence,
            "significance": significance,
            "patterns": self.pattern_keys,
            "primary_observation_key": self.primary_observation_key,
            "primary_observation_label": self.primary_observation_label,
            "indicators_present": self.indicators_present,
            "indicators_required": self.indicators_required,
            "indicator_details": self.indicator_details,
            "summary": self.summary,
            "summary_source": self.summary_source,
            "explanation": self.explanation,
            "explanation_source": self.explanation_source,
            "doc_links": self.doc_links,
            "similar_memories": self.similar_memory_ids,
            "similar_history": self.similar_history,
            "context": self.cutting_context,
            "metrics": self.metrics_summary,
        }

    def to_scored_dict(self) -> Dict[str, Any]:
        """Like to_dict but marked as scored_event (for inference panel)."""
        if self.message_type == "scored_event":
            return self.to_dict()
        d = self.to_dict()
        d["type"] = "scored_event"
        return d
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class _ExplanationPayload:
    """Lightweight wrapper so explanation_update messages can be sent
    through the same WebSocket queue that carries ``SignificantEventAlert``
    objects (the WS handler calls ``.to_json()`` on every item)."""

    __slots__ = ("_json", "_dict")

    def __init__(self, json_str: str, payload: Dict[str, Any]):
        self._json = json_str
        self._dict = payload

    def to_json(self) -> str:          # noqa: D401
        return self._json

    def to_dict(self) -> Dict[str, Any]:
        return self._dict


# [PROTOTYPE_LLM_MEMORY_V1] - Rate limiting config
@dataclass
class RateLimitConfig:
    """Rate limiting to prevent alert fatigue."""
    min_interval_seconds: float = 5.0  # Minimum time between alerts per session
    max_alerts_per_minute: int = 10  # Maximum alerts per session per minute
    cooldown_on_dismiss: float = 30.0  # Cooldown after user dismisses (seconds)
    signature_recurrence_seconds: float = 10.0  # Window for candidate -> recurring upgrade
    # Suppress re-emitting an alert for the same signature while it persists,
    # unless the score moves materially or severity escalates. Stops the
    # "4 identical alerts in a row" pattern for sustained events.
    signature_suppress_seconds: float = 30.0
    signature_score_change_threshold: float = 0.10


# [PROTOTYPE_LLM_MEMORY_V1] - Dispatcher
class AlertDispatcher:
    """
    Manages alert subscriptions and dispatches significant events.
    
    [INTEGRATION_POINT] Should be wired to WebSocket endpoints in app.py
    """
    
    def __init__(self, rate_config: Optional[RateLimitConfig] = None):
        self.rate_config = rate_config or RateLimitConfig()
        self._session_pause_handler: Optional[SessionPauseHandler] = None
        
        # Subscriber queues: session_id -> List[Queue]
        self._subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        
        # Global subscribers (all sessions)
        self._global_subscribers: List[asyncio.Queue] = []
        
        # Rate limiting state: session_id -> last alert time
        self._last_alert_time: Dict[str, datetime] = {}
        
        # Alert count tracking: session_id -> list of recent alert times
        self._recent_alerts: Dict[str, List[datetime]] = defaultdict(list)
        
        # Cooldowns from dismissals
        self._cooldowns: Dict[str, datetime] = {}

        # Observation persistence state: session_id -> signature -> last emitted time
        self._recent_signatures: Dict[str, Dict[str, datetime]] = defaultdict(dict)

        # Per-signature dispatch state for suppression of sustained events:
        # session_id -> signature_key -> (last_dispatch_time, last_score, last_severity)
        self._last_signature_dispatch: Dict[str, Dict[str, tuple[datetime, float, str]]] = defaultdict(dict)

        # Per-signature lifecycle for recurrence reporting:
        # session_id -> signature_key -> dict(first_seen, last_seen, occurrences, suppressed)
        self._signature_lifecycle: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

        # Session-level operator mute state for noisy recurring signatures.
        # session_id -> signature_key -> dict(muted_at, source, reason)
        self._muted_signatures: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    def set_session_pause_handler(self, handler: Optional[SessionPauseHandler]) -> None:
        self._session_pause_handler = handler

    def _derive_persistence_label(
        self,
        session_id: str,
        pattern_keys: List[str],
        emitted_at: datetime,
    ) -> Optional[str]:
        window_seconds = max(0.0, float(self.rate_config.signature_recurrence_seconds))
        if window_seconds <= 0:
            return None

        signature_keys = []
        for key in pattern_keys:
            canonical = normalize_signature_key(key)
            if canonical.startswith("signature:"):
                signature_keys.append(canonical)

        if not signature_keys:
            return None

        session_signatures = self._recent_signatures[session_id]
        recurring = False
        cutoff = emitted_at - timedelta(seconds=window_seconds)
        stale = [key for key, ts in session_signatures.items() if ts < cutoff]
        for key in stale:
            session_signatures.pop(key, None)

        for key in signature_keys:
            last_seen = session_signatures.get(key)
            if last_seen and (emitted_at - last_seen).total_seconds() <= window_seconds:
                recurring = True
            session_signatures[key] = emitted_at

        return "recurring" if recurring else "candidate"
    
    def subscribe(
        self, 
        session_id: Optional[str] = None,
        maxsize: int = 64
    ) -> asyncio.Queue:
        """
        Subscribe to alerts.
        
        Args:
            session_id: If provided, only receive alerts for this session.
                       If None, receive all alerts.
            maxsize: Queue size limit
        
        Returns:
            Queue that will receive SignificantEventAlert objects
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        
        if session_id:
            self._subscribers[session_id].append(queue)
            logger.debug(f"Client subscribed to alerts for session {session_id}")
        else:
            self._global_subscribers.append(queue)
            logger.debug("Client subscribed to global alerts")
        
        return queue
    
    def unsubscribe(self, queue: asyncio.Queue, session_id: Optional[str] = None):
        """Remove a subscriber."""
        if session_id and session_id in self._subscribers:
            try:
                self._subscribers[session_id].remove(queue)
            except ValueError:
                pass
        else:
            try:
                self._global_subscribers.remove(queue)
            except ValueError:
                pass

    async def _propose_doc_links(
        self,
        *,
        pattern_keys: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        machine_uri: Optional[str] = None,
        cutting_context: Optional[Dict[str, Any]] = None,
        channel_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not pattern_keys:
            return []

        merged_metadata = dict(metadata or {})
        context_dict = dict(cutting_context or {}) if isinstance(cutting_context, dict) else {}
        if context_dict and "cutting_context" not in merged_metadata:
            merged_metadata["cutting_context"] = context_dict

        machine_hint = (
            context_dict.get("machine_id")
            or merged_metadata.get("machine_id")
            or machine_uri
            or merged_metadata.get("machine_uri")
            or merged_metadata.get("machine_iri")
            or merged_metadata.get("sindit_asset_iri")
            or context_dict.get("machine_type")
        )
        usecase = resolve_usecase(
            metadata=merged_metadata,
            machine_uri=machine_uri or merged_metadata.get("machine_uri") or merged_metadata.get("machine_iri"),
            machine=machine_hint,
            fallback_generic=False,
        )
        if usecase is None and not machine_hint:
            return []

        try:
            payload = await propose_alert_doc_links(
                get_docs_backend(),
                pattern_keys=pattern_keys,
                usecase=usecase,
                machine=machine_hint,
                cutting_context=context_dict,
                channel_names=list(channel_names or []),
            )
        except Exception:
            logger.debug("Alert doc-link lookup failed for patterns=%s", pattern_keys, exc_info=True)
            return []

        return list(payload.get("doc_links") or [])

    async def propose_doc_links_for_memory(
        self,
        *,
        memory: Memory,
        cutting_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        pattern_keys = [
            str(pattern.key).strip()
            for pattern in (getattr(memory, "pattern_keys", None) or [])
            if str(getattr(pattern, "key", "")).strip()
        ]
        return await self._propose_doc_links(
            pattern_keys=pattern_keys,
            metadata=getattr(memory, "metadata", None),
            machine_uri=getattr(memory, "machine_uri", None),
            cutting_context=cutting_context,
            channel_names=list(getattr(memory, "channels", []) or []),
        )
    
    async def dispatch(
        self,
        memory: Memory,
        significance: SignificanceResult,
        similar_memories: Optional[List[MemoryQueryResult]] = None,
        similar_history: Optional[List[Dict[str, Any]]] = None,
        summary: Optional[str] = None,
        summary_source: Optional[str] = None,
        explanation: Optional[str] = None,
        explanation_source: Optional[str] = None,
        cutting_context: Optional[Dict[str, Any]] = None,
        metrics_summary: Optional[Dict[str, Any]] = None,
        persistence_label: Optional[str] = None,
        doc_links: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Dispatch a significant event alert.
        
        Args:
            memory: The memory record for this event
            significance: Significance scoring result
            similar_memories: Related historical memories
            summary: LLM-generated summary
            cutting_context: Cutting conditions
            metrics_summary: Brief metrics summary
        
        Returns:
            True if alert was dispatched, False if rate limited
        """
        session_id = memory.session_id
        
        pattern_keys_list = [p.key for p in memory.pattern_keys]
        suppression_signature = self._suppression_signature(
            pattern_keys_list,
            list(getattr(significance, "reasons", []) or []),
        )

        muted_snapshot = self.get_muted_signature(session_id, pattern_keys_list)
        if muted_snapshot is not None:
            logger.debug(
                "Alert suppressed (signature muted) for session %s sig=%s",
                session_id,
                muted_snapshot.get("signature"),
            )
            _append_dispatch_log(
                session_id,
                {
                    "phase": "alert_suppressed",
                    "session_id": session_id,
                    "event_id": memory.id,
                    "reason": "signature_muted",
                    "signature": muted_snapshot.get("signature"),
                    "score": significance.score,
                    "action": significance.action.value,
                    "patterns": pattern_keys_list,
                    "reasons": list(significance.reasons or []),
                    "mute": muted_snapshot,
                },
            )
            return False

        # Check rate limiting
        if not self._check_rate_limit(session_id):
            logger.debug(f"Alert rate limited for session {session_id}")
            _append_dispatch_log(
                session_id,
                {
                    "phase": "alert_suppressed",
                    "session_id": session_id,
                    "event_id": memory.id,
                    "reason": "rate_limited",
                    "score": significance.score,
                    "action": significance.action.value,
                    "patterns": pattern_keys_list,
                    "reasons": list(significance.reasons or []),
                },
            )
            return False
        
        # Check cooldown
        if self._is_in_cooldown(session_id):
            logger.debug(f"Alert in cooldown for session {session_id}")
            _append_dispatch_log(
                session_id,
                {
                    "phase": "alert_suppressed",
                    "session_id": session_id,
                    "event_id": memory.id,
                    "reason": "cooldown",
                    "score": significance.score,
                    "action": significance.action.value,
                    "patterns": pattern_keys_list,
                    "reasons": list(significance.reasons or []),
                },
            )
            return False

        # Suppress repeat emissions for a sustained signature unless the score
        # has moved materially or the severity has escalated. The inference
        # panel still receives scored_event updates via broadcast_scored_event,
        # so the operator can see the underlying activity \u2014 we just stop
        # showering them with identical alert cards for the same persistent event.
        if suppression_signature is not None:
            suppress_window = float(self.rate_config.signature_suppress_seconds or 0.0)
            if suppress_window > 0:
                now = datetime.now(timezone.utc)
                state = self._last_signature_dispatch[session_id].get(suppression_signature)
                if state is not None:
                    last_time, last_score, last_severity = state
                    elapsed = (now - last_time).total_seconds()
                    new_severity = _derive_severity(float(significance.score or 0.0))
                    severity_escalated = self._severity_rank(new_severity) > self._severity_rank(last_severity)
                    score_change = abs(float(significance.score or 0.0) - float(last_score))
                    if (
                        elapsed < suppress_window
                        and not severity_escalated
                        and score_change < float(self.rate_config.signature_score_change_threshold or 0.0)
                    ):
                        logger.debug(
                            "Alert suppressed (signature unchanged) for session %s sig=%s",
                            session_id, suppression_signature,
                        )
                        suppressed_snapshot = self._touch_signature_lifecycle(
                            session_id, suppression_signature, suppressed=True,
                        )
                        _append_dispatch_log(
                            session_id,
                            {
                                "phase": "alert_suppressed",
                                "session_id": session_id,
                                "event_id": memory.id,
                                "reason": "signature_unchanged",
                                "signature": suppression_signature,
                                "score": significance.score,
                                "last_score": last_score,
                                "score_change": score_change,
                                "elapsed_seconds": elapsed,
                                "action": significance.action.value,
                                "severity": new_severity,
                                "patterns": pattern_keys_list,
                                "recurrence": suppressed_snapshot,
                            },
                        )
                        return False

        # Build alert
        # Demo-friendly: ensure a human-readable summary even when the LLM explainer
        # is disabled/unavailable, and avoid showing raw pattern keys in the summary.
        signature_payload = _extract_signature_payload(list(memory.pattern_keys or []))
        if not summary:
            pkeys = [p.key for p in memory.pattern_keys]
            score = float(getattr(significance, "score", 0.0) or 0.0)
            sim_n = len(similar_memories or [])

            # Try demo LLM summaries first (pattern-matched templates)
            demo = _demo_llm_summary(pkeys, score, sim_n)
            if demo:
                summary, summary_source = demo
            elif (
                signature_payload
                and isinstance(signature_payload.get("indicators_present"), int)
                and isinstance(signature_payload.get("indicators_required"), int)
            ):
                summary, summary_source = _signature_summary(
                    signature_payload,
                    similar_count=sim_n,
                    similar_history=list(similar_history or []),
                )
            else:
                summary, summary_source = _fallback_summary(
                    pkeys,
                    list(getattr(significance, "reasons", []) or []),
                    score,
                    similar_count=sim_n,
                    similar_history=list(similar_history or []),
                    default_reason="Significant observation",
                )
        else:
            summary_source = summary_source or "llm"

        pkeys = [p.key for p in memory.pattern_keys]
        emitted_at = datetime.now(timezone.utc)
        if persistence_label is None:
            persistence_label = self._derive_persistence_label(session_id, pkeys, emitted_at)

        recurrence_snapshot = self._touch_signature_lifecycle(
            session_id, suppression_signature, suppressed=False,
        )
        if doc_links is None:
            doc_links = await self.propose_doc_links_for_memory(
                memory=memory,
                cutting_context=cutting_context,
            )

        alert = SignificantEventAlert(
            event_id=memory.id,
            session_id=session_id,
            timestamp=emitted_at,
            message_type="significant_event",
            time_range=_serialize_time_range(getattr(memory, "time_range", None)),
            significance_score=significance.score,
            action=significance.action,
            reasons=significance.reasons,
            pattern_keys=pkeys,
            primary_observation_key=signature_payload.get("primary_observation_key") if signature_payload else None,
            primary_observation_label=signature_payload.get("primary_observation_label") if signature_payload else None,
            indicators_present=signature_payload.get("indicators_present") if signature_payload else None,
            indicators_required=signature_payload.get("indicators_required") if signature_payload else None,
            indicator_details=list(signature_payload.get("indicator_details") or []) if signature_payload else [],
            severity=_derive_severity(significance.score),
            category=_derive_category(pkeys),
            persistence_label=persistence_label,
            prior_boost=getattr(significance, "prior_boost", 0.0),
            pattern_priors=dict(getattr(significance, "pattern_priors", {}) or {}),
            historical_prior=getattr(significance, "historical_prior", None),
            prior_factor=getattr(significance, "prior_factor", None),
            prior_mode=getattr(significance, "prior_mode", None),
            score_trace=list(getattr(significance, "score_trace", []) or []),
            recurrence=recurrence_snapshot,
            summary=summary,
            summary_source=summary_source,
            explanation=explanation,
            explanation_source=explanation_source,
            doc_links=doc_links,
            similar_memory_ids=[m.memory.id for m in (similar_memories or [])],
            similar_history=list(similar_history or []),
            cutting_context=cutting_context,
            metrics_summary=metrics_summary,
        )
        
        # Send to subscribers
        await self._broadcast(alert, session_id)

        _append_dispatch_log(
            session_id,
            {
                "phase": "significant_event",
                "session_id": session_id,
                "event_id": alert.event_id,
                "time_range": alert.time_range,
                "score": alert.significance_score,
                "action": alert.action.value,
                "severity": alert.severity,
                "category": alert.category,
                "persistence_label": alert.persistence_label,
                "patterns": list(alert.pattern_keys),
                "reasons": list(alert.reasons),
                "summary": alert.summary,
                "summary_source": alert.summary_source,
                "explanation": alert.explanation,
                "explanation_source": alert.explanation_source,
                "doc_links": list(alert.doc_links),
                "similar_memory_ids": list(alert.similar_memory_ids),
                "similar_history": list(alert.similar_history),
                "metrics": alert.metrics_summary,
                "context": alert.cutting_context,
            },
        )
        
        # Update rate limiting state
        self._record_alert(session_id)
        self._record_signature_dispatch(
            session_id,
            pattern_keys_list,
            float(significance.score or 0.0),
            alert.severity,
            reasons=list(getattr(significance, "reasons", []) or []),
        )

        if self._session_pause_handler is not None:
            try:
                self._session_pause_handler(
                    session_id,
                    {
                        "event_id": memory.id,
                        "severity": alert.severity,
                        "score": float(significance.score or 0.0),
                        "action": significance.action.value,
                    },
                )
            except Exception:
                logger.exception("Alert pause handler failed for session %s", session_id)

        logger.info(f"Dispatched alert for event {memory.id} (score={significance.score:.2f})")
        return True
    
    def set_cooldown(self, session_id: str):
        """
        Set cooldown after user dismisses an alert.
        
        [INTEGRATION_POINT] Should be called when user dismisses alert.
        """
        cooldown_until = datetime.now(timezone.utc) + timedelta(
            seconds=self.rate_config.cooldown_on_dismiss
        )
        self._cooldowns[session_id] = cooldown_until
        logger.debug(f"Set cooldown for session {session_id} until {cooldown_until}")
    
    def clear_cooldown(self, session_id: str):
        """Clear cooldown for a session."""
        self._cooldowns.pop(session_id, None)

    def reset_session_gating(self, session_id: str) -> None:
        """Clear both rate-limit and dismiss-cooldown gating for a session.

        Used by the Demo Director so each scripted event reliably dispatches an
        alert — otherwise the 5 s min-interval and the 30 s post-dismiss cooldown
        silently suppress the next fired event (nothing to open / confirm).
        """
        self._last_alert_time.pop(session_id, None)
        self._cooldowns.pop(session_id, None)

    async def broadcast_scored_event(
        self,
        significance: SignificanceResult,
        memory: Optional[Memory] = None,
        event: Optional[Any] = None,
        cutting_context: Optional[Dict[str, Any]] = None,
        metrics_summary: Optional[Dict[str, Any]] = None,
        persistence_label: Optional[str] = None,
    ) -> None:
        """
        Broadcast a scored event to all subscribers without rate limiting.

        Used by the inference panel to show ALL event results (including
        sub-threshold ones that would not trigger an alert).
        """
        if memory is not None:
            event_id = memory.id
            session_id = memory.session_id
            pattern_objects = list(getattr(memory, "pattern_keys", None) or [])
            if not pattern_objects and event is not None:
                pattern_objects = list(getattr(event, "patterns", None) or [])
            pkeys = [p.key for p in pattern_objects]
        elif event is not None:
            event_id = f"scored:{event.session_id}:{int(getattr(event.time_range, 'i1', 0))}"
            session_id = event.session_id
            pkeys = [p.key for p in (event.patterns or [])]
            pattern_objects = list(event.patterns or [])
        else:
            raise ValueError("broadcast_scored_event requires either memory or event")

        # Try demo LLM summaries first, then humanised fallback
        score = float(getattr(significance, "score", 0.0) or 0.0)
        signature_payload = _extract_signature_payload(pattern_objects)
        demo = _demo_llm_summary(pkeys, score)
        if demo:
            summary, summary_source = demo
        elif (
            signature_payload
            and isinstance(signature_payload.get("indicators_present"), int)
            and isinstance(signature_payload.get("indicators_required"), int)
        ):
            summary, summary_source = _signature_summary(signature_payload)
        else:
            summary, summary_source = _fallback_summary(
                pkeys,
                list(getattr(significance, "reasons", []) or []),
                score,
            )

        emitted_at = datetime.now(timezone.utc)
        if persistence_label is None:
            persistence_label = self._derive_persistence_label(session_id, pkeys, emitted_at)

        alert = SignificantEventAlert(
            event_id=event_id,
            session_id=session_id,
            timestamp=emitted_at,
            message_type="scored_event",
            time_range=_serialize_time_range(
                getattr(memory, "time_range", None) if memory is not None else getattr(event, "time_range", None)
            ),
            significance_score=significance.score,
            action=significance.action,
            reasons=significance.reasons,
            pattern_keys=pkeys,
            primary_observation_key=signature_payload.get("primary_observation_key") if signature_payload else None,
            primary_observation_label=signature_payload.get("primary_observation_label") if signature_payload else None,
            indicators_present=signature_payload.get("indicators_present") if signature_payload else None,
            indicators_required=signature_payload.get("indicators_required") if signature_payload else None,
            indicator_details=list(signature_payload.get("indicator_details") or []) if signature_payload else [],
            severity=_derive_severity(significance.score),
            category=_derive_category(pkeys),
            persistence_label=persistence_label,
            prior_boost=getattr(significance, "prior_boost", 0.0),
            pattern_priors=dict(getattr(significance, "pattern_priors", {}) or {}),
            historical_prior=getattr(significance, "historical_prior", None),
            prior_factor=getattr(significance, "prior_factor", None),
            prior_mode=getattr(significance, "prior_mode", None),
            score_trace=list(getattr(significance, "score_trace", []) or []),
            summary=summary,
            summary_source=summary_source,
            cutting_context=cutting_context,
            metrics_summary=metrics_summary,
        )
        # Broadcast with scored_event type (no rate limit, no cooldown check)
        for queue in list(self._subscribers.get(session_id, [])):
            try:
                queue.put_nowait(alert)
            except asyncio.QueueFull:
                pass
        for queue in list(self._global_subscribers):
            try:
                queue.put_nowait(alert)
            except asyncio.QueueFull:
                pass
        _append_dispatch_log(
            session_id,
            {
                "phase": "scored_event",
                "session_id": session_id,
                "event_id": alert.event_id,
                "time_range": alert.time_range,
                "score": alert.significance_score,
                "action": alert.action.value,
                "severity": alert.severity,
                "category": alert.category,
                "patterns": list(alert.pattern_keys),
                "reasons": list(alert.reasons),
                "summary": alert.summary,
                "summary_source": alert.summary_source,
                "metrics": alert.metrics_summary,
                "context": alert.cutting_context,
            },
        )
        logger.debug("Broadcast scored event %s (score=%.2f)", event_id, significance.score)

    async def broadcast_explanation_update(
        self,
        *,
        memory_id: str,
        session_id: str,
        explanation: Optional[str] = None,
        explanation_source: Optional[str] = None,
        alert_line: Optional[str] = None,
        alert_line_source: Optional[str] = None,
        guardrail_outcome: Optional[Dict[str, Any]] = None,
        recommendation: Optional[str] = None,
    ) -> None:
        """Broadcast an LLM explanation that completed in the background.

        Sent as a lightweight ``explanation_update`` message so the
        frontend can patch the corresponding alert/scored-event card
        without a full page refresh.

        ``guardrail_outcome`` (when present) carries the Tier-1 output-rail
        audit trail ({"action", "reasons", "checks"}) for the UI/audit log.
        """
        payload = {
            "type": "explanation_update",
            "event_id": memory_id,
            "session_id": session_id,
            "explanation": explanation,
            "explanation_source": explanation_source,
            "alert_line": alert_line,
            "alert_line_source": alert_line_source,
            "summary": alert_line,          # alias for UI compatibility
            "summary_source": alert_line_source,
            "recommendation": recommendation,   # immediate breakage-avoidance action (two-tier model)
            "guardrail_outcome": guardrail_outcome,
        }
        msg_json = json.dumps(payload)
        # Push to session-specific subscribers
        for queue in list(self._subscribers.get(session_id, [])):
            try:
                queue.put_nowait(_ExplanationPayload(msg_json, payload))
            except asyncio.QueueFull:
                pass
        # Push to global subscribers
        for queue in list(self._global_subscribers):
            try:
                queue.put_nowait(_ExplanationPayload(msg_json, payload))
            except asyncio.QueueFull:
                pass
        _append_dispatch_log(
            session_id,
            {
                "phase": "explanation_update",
                **payload,
            },
        )
        logger.debug("Broadcast explanation_update for %s", memory_id)

    async def _broadcast(self, alert: SignificantEventAlert, session_id: str):
        """Send alert to all relevant subscribers."""
        # Session-specific subscribers
        for queue in list(self._subscribers.get(session_id, [])):
            try:
                queue.put_nowait(alert)
            except asyncio.QueueFull:
                logger.warning(f"Alert queue full for session {session_id}")
        
        # Global subscribers
        for queue in list(self._global_subscribers):
            try:
                queue.put_nowait(alert)
            except asyncio.QueueFull:
                logger.warning("Global alert queue full")
    
    def _check_rate_limit(self, session_id: str) -> bool:
        """Check if we can send an alert (rate limiting)."""
        now = datetime.now(timezone.utc)
        
        # Check minimum interval
        last_time = self._last_alert_time.get(session_id)
        if last_time:
            elapsed = (now - last_time).total_seconds()
            if elapsed < self.rate_config.min_interval_seconds:
                return False
        
        # Check max per minute
        recent = self._recent_alerts.get(session_id, [])
        one_minute_ago = now - timedelta(minutes=1)
        recent = [t for t in recent if t > one_minute_ago]
        self._recent_alerts[session_id] = recent
        
        if len(recent) >= self.rate_config.max_alerts_per_minute:
            return False
        
        return True
    
    def _is_in_cooldown(self, session_id: str) -> bool:
        """Check if session is in cooldown."""
        cooldown_until = self._cooldowns.get(session_id)
        if cooldown_until and datetime.now(timezone.utc) < cooldown_until:
            return True
        return False
    
    def _record_alert(self, session_id: str):
        """Record that an alert was sent."""
        now = datetime.now(timezone.utc)
        self._last_alert_time[session_id] = now
        self._recent_alerts[session_id].append(now)

    def _record_signature_dispatch(
        self,
        session_id: str,
        pattern_keys: List[str],
        score: float,
        severity: str,
        reasons: Optional[List[str]] = None,
    ) -> None:
        """Track per-signature dispatch state for sustained-event suppression."""
        signature = self._suppression_signature(pattern_keys, reasons)
        if signature is None:
            return
        self._last_signature_dispatch[session_id][signature] = (
            datetime.now(timezone.utc), float(score or 0.0), severity or "",
        )

    @staticmethod
    def _suppression_signature(
        pattern_keys: List[str],
        reasons: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Return a stable key for suppression. Prefers signature: patterns,\n        falls back to a canonicalised join of all pattern keys.\n        """
        canonical = sorted({normalize_signature_key(k) for k in pattern_keys if k})
        signature_only = [k for k in canonical if k.startswith("signature:")]
        chosen = signature_only or canonical
        if chosen:
            return "|".join(chosen)
        return _reason_suppression_signature(reasons)

    @staticmethod
    def _severity_rank(severity: str) -> int:
        return {"INFO": 0, "WARNING": 1, "CRITICAL": 2}.get((severity or "").upper(), 0)

    def get_signature_lifecycle(
        self, session_id: str, pattern_keys: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Read-only snapshot of the per-signature lifecycle state.

        Callers (e.g. the LLM explainer pipeline) use this to attach
        \"\u00d7N occurrences\" context to a generated description without
        mutating dispatcher state.
        """
        signature = self._suppression_signature(pattern_keys)
        if signature is None:
            return None
        state = self._signature_lifecycle.get(session_id, {}).get(signature)
        if state is None:
            return None
        first_seen = state.get("first_seen")
        last_seen = state.get("last_seen")
        return {
            "signature": signature,
            "first_seen": first_seen.isoformat() if isinstance(first_seen, datetime) else None,
            "last_seen": last_seen.isoformat() if isinstance(last_seen, datetime) else None,
            "occurrences": int(state.get("occurrences", 0)),
            "suppressed_since_last_emit": int(state.get("suppressed_since_last_emit", 0)),
        }

    def get_muted_signature(
        self,
        session_id: str,
        pattern_keys: Optional[List[str]] = None,
        *,
        signature: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the mute record for a signature in a session, if any."""
        resolved_signature = signature or self._suppression_signature(pattern_keys or [])
        if resolved_signature is None:
            return None
        state = self._muted_signatures.get(session_id, {}).get(resolved_signature)
        if state is None:
            return None
        muted_at = state.get("muted_at")
        return {
            "signature": resolved_signature,
            "muted_at": muted_at.isoformat() if isinstance(muted_at, datetime) else None,
            "source": state.get("source"),
            "reason": state.get("reason"),
        }

    def set_signature_muted(
        self,
        session_id: str,
        *,
        pattern_keys: Optional[List[str]] = None,
        signature: Optional[str] = None,
        source: str = "operator",
        reason: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Mute a recurring signature for the remainder of the session."""
        resolved_signature = signature or self._suppression_signature(pattern_keys or [])
        if resolved_signature is None:
            return None
        state = {
            "muted_at": datetime.now(timezone.utc),
            "source": source,
            "reason": reason,
        }
        self._muted_signatures[session_id][resolved_signature] = state
        return self.get_muted_signature(session_id, signature=resolved_signature)

    def clear_signature_muted(
        self,
        session_id: str,
        *,
        pattern_keys: Optional[List[str]] = None,
        signature: Optional[str] = None,
    ) -> bool:
        """Remove a mute for a signature in a session."""
        resolved_signature = signature or self._suppression_signature(pattern_keys or [])
        if resolved_signature is None:
            return False
        removed = self._muted_signatures.get(session_id, {}).pop(resolved_signature, None)
        return removed is not None

    def _touch_signature_lifecycle(
        self, session_id: str, signature: Optional[str], *, suppressed: bool
    ) -> Optional[Dict[str, Any]]:
        """Update the per-signature lifecycle record and return a snapshot for emission.

        Returns None when there is no signature to track. The snapshot is intended
        to be attached to the outgoing alert so the UI can render \"\u00d7N occurrences
        since first_seen\" and \"M suppressed since last emit\".
        """
        if signature is None:
            return None
        now = datetime.now(timezone.utc)
        bucket = self._signature_lifecycle[session_id]
        state = bucket.get(signature)
        # Expire stale state (no activity for 5 \u00d7 suppress window or 5 min, whichever larger)
        suppress_window = max(float(self.rate_config.signature_suppress_seconds or 0.0), 60.0)
        if state is not None:
            last_seen = state.get("last_seen")
            if isinstance(last_seen, datetime) and (now - last_seen).total_seconds() > suppress_window * 5:
                state = None
        if state is None:
            state = {
                "first_seen": now,
                "last_seen": now,
                "occurrences": 0,
                "suppressed_since_last_emit": 0,
            }
            bucket[signature] = state
        state["last_seen"] = now
        if suppressed:
            state["suppressed_since_last_emit"] = int(state.get("suppressed_since_last_emit", 0)) + 1
        else:
            state["occurrences"] = int(state.get("occurrences", 0)) + 1
            state["suppressed_since_last_emit"] = 0
        first_seen_iso = state["first_seen"].isoformat() if isinstance(state["first_seen"], datetime) else None
        # Stable episode identity (plan 1.4): a signature + its first_seen names
        # one episode; first_seen resets when the state expires, so a later
        # recurrence of the same signature is a distinct episode. The operator UI
        # echoes this back on feedback so learning updates dedupe per episode.
        episode_id = f"{signature}::{first_seen_iso}" if first_seen_iso else signature
        return {
            "signature": signature,
            "episode_id": episode_id,
            "first_seen": first_seen_iso,
            "last_seen": state["last_seen"].isoformat() if isinstance(state["last_seen"], datetime) else None,
            "occurrences": int(state.get("occurrences", 0)),
            "suppressed_since_last_emit": int(state.get("suppressed_since_last_emit", 0)),
        }


# [PROTOTYPE_LLM_MEMORY_V1] - Global dispatcher instance
_dispatcher: Optional[AlertDispatcher] = None


def get_dispatcher() -> AlertDispatcher:
    """Get or create the global AlertDispatcher instance."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = AlertDispatcher()
    return _dispatcher


async def dispatch_significant_event(
    memory: Memory,
    significance: SignificanceResult,
    **kwargs
) -> bool:
    """Convenience function to dispatch via global dispatcher."""
    dispatcher = get_dispatcher()
    return await dispatcher.dispatch(memory, significance, **kwargs)

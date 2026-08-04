"""
Agent C — build a `model_breakdown` dict from event external signals.

Each memory-event response carries a normalized per-model snapshot so UI
components (and the LLM) can attribute the score to a specific detector.

Shape::

    {
        "classical": {
            "anomaly_score": float | None,
            "model_confidence": float | None,
            "isolation_forest": float | None,
            "lof": float | None,
            "ensemble": float | None,
            "breakage_probability": float | None,
        },
        "harmonic": {
            "score": float | None,
            "source": str | None,
        },
        "stoppage": {
            "probability": float | None,
            "eta_s": float | None,
            "label": str | None,
        },
        "online": {
            "probability": float | None,
            "running": bool | None,
        },
        "available": [names],      # sub-model namespaces with any real data
    }

Every field is optional and defaults to ``None``. Consumers must tolerate
missing keys.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Key maps: canonical output key ← list of accepted external-signal aliases
# ---------------------------------------------------------------------------

_CLASSICAL_KEYS: Dict[str, tuple[str, ...]] = {
    "anomaly_score": ("anomaly_detector_score", "anomaly_score"),
    "model_confidence": ("model_confidence",),
    "isolation_forest": ("isolation_forest_score", "iforest_score"),
    "lof": ("lof_score",),
    "ensemble": ("ensemble_score",),
    "breakage_probability": ("breakage_prediction", "breakage_probability"),
}

_HARMONIC_KEYS: Dict[str, tuple[str, ...]] = {
    "score": ("harmonic_context_score",),
    "source": ("harmonic_context_source",),
}

_STOPPAGE_KEYS: Dict[str, tuple[str, ...]] = {
    "probability": ("stoppage_probability", "stoppage_prob"),
    "eta_s": ("stoppage_eta_s", "eta_s", "time_to_stop_s"),
    "label": ("stoppage_label",),
}

_ONLINE_KEYS: Dict[str, tuple[str, ...]] = {
    "probability": ("online_probability", "online_prob", "online_score"),
    "running": ("online_running",),
}


def _pick_number(signals: Dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    for k in keys:
        v = signals.get(k) if signals else None
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _pick_string(signals: Dict[str, Any], keys: tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = signals.get(k) if signals else None
        if isinstance(v, str) and v:
            return v
    return None


def _pick_bool(signals: Dict[str, Any], keys: tuple[str, ...]) -> Optional[bool]:
    for k in keys:
        v = signals.get(k) if signals else None
        if isinstance(v, bool):
            return v
    return None


def _build_section(
    signals: Dict[str, Any],
    number_keys: Dict[str, tuple[str, ...]],
    string_keys: Optional[Dict[str, tuple[str, ...]]] = None,
    bool_keys: Optional[Dict[str, tuple[str, ...]]] = None,
) -> Dict[str, Any]:
    section: Dict[str, Any] = {}
    for out_key, aliases in number_keys.items():
        section[out_key] = _pick_number(signals, aliases)
    if string_keys:
        for out_key, aliases in string_keys.items():
            section[out_key] = _pick_string(signals, aliases)
    if bool_keys:
        for out_key, aliases in bool_keys.items():
            section[out_key] = _pick_bool(signals, aliases)
    return section


def build_model_breakdown(
    external_signals: Optional[Dict[str, Any]],
    *,
    feature_schema_version: Optional[int] = None,
    feature_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the canonical per-model attribution dict for a memory event."""
    signals: Dict[str, Any] = dict(external_signals or {})

    classical = _build_section(signals, _CLASSICAL_KEYS)
    harmonic = _build_section(
        signals,
        {"score": _HARMONIC_KEYS["score"]},
        string_keys={"source": _HARMONIC_KEYS["source"]},
    )
    stoppage = _build_section(
        signals,
        {"probability": _STOPPAGE_KEYS["probability"], "eta_s": _STOPPAGE_KEYS["eta_s"]},
        string_keys={"label": _STOPPAGE_KEYS["label"]},
    )
    online = _build_section(
        signals,
        {"probability": _ONLINE_KEYS["probability"]},
        bool_keys={"running": _ONLINE_KEYS["running"]},
    )

    available: List[str] = []
    for name, block in (
        ("classical", classical),
        ("harmonic", harmonic),
        ("stoppage", stoppage),
        ("online", online),
    ):
        if any(v is not None for v in block.values()):
            available.append(name)

    breakdown: Dict[str, Any] = {
        "classical": classical,
        "harmonic": harmonic,
        "stoppage": stoppage,
        "online": online,
        "available": available,
    }
    if feature_schema_version is not None:
        breakdown["feature_schema_version"] = int(feature_schema_version)
    if feature_count is not None:
        breakdown["feature_count"] = int(feature_count)
    return breakdown

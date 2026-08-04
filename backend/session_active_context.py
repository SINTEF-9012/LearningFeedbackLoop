from __future__ import annotations

from typing import Any, Dict, Mapping

from backend.agents.sindit.runtime_context import resolve_runtime_metadata


_REQUIRED_FIELD_LABELS = {
    "tool_diameter": "tool diameter",
    "num_teeth": "number of teeth",
    "spindle_speed": "spindle speed",
    "feed_rate": "feed rate",
}


def build_active_session_context(session: Mapping[str, Any]) -> Dict[str, Any] | None:
    data = session.get("data")
    metadata = session.get("metadata")
    if not isinstance(data, Mapping) or not data:
        return None
    if not isinstance(metadata, Mapping):
        metadata = {}

    sample_index = _active_sample_index(session, data)
    signals = _sample_signals(data, sample_index)
    if not signals and not metadata:
        return None

    resolved = resolve_runtime_metadata(
        metadata,
        {
            "source": metadata.get("source"),
            "metadata": dict(metadata),
            "signals": signals,
        },
    )

    casedata = dict(resolved.get("casedata") or {})
    cutting_context = dict(casedata.get("cutting_context") or {})
    extra = dict(cutting_context.get("extra") or {})
    source_config = dict(session.get("source_config") or {})

    machine_id = _text(resolved.get("machine_id")) or _text(casedata.get("case_dir")) or _text(source_config.get("case_dir"))
    operation_id = _text(casedata.get("operation_id")) or _text(source_config.get("operation_id"))
    dataset_id = _text(resolved.get("dataset_id")) or _text(casedata.get("dataset_id"))
    machine_family = _text(resolved.get("machine_family")) or _text(extra.get("machine_family"))
    tool_number = _canonical_int(extra.get("tool_number")) or _canonical_int(signals.get("Tool_Number")) or _canonical_int(signals.get("tool_number"))
    tool_id = _text(cutting_context.get("tool_id")) or (f"T{tool_number}" if tool_number is not None else None)
    sindit_tool_iri = _text(extra.get("sindit_tool_iri"))

    missing_fields: list[str] = []
    if tool_number is None and tool_id is None:
        missing_fields.append("tool number")
    for field, label in _REQUIRED_FIELD_LABELS.items():
        if cutting_context.get(field) is None:
            missing_fields.append(label)

    ready = len(missing_fields) == 0
    hover_detail = (
        "Enough tool data available for harmonics."
        if ready
        else f"Missing: {', '.join(missing_fields)}."
    )

    tool_geometry_bits: list[str] = []
    if cutting_context.get("tool_diameter") is not None:
        tool_geometry_bits.append(f"d {_format_number(cutting_context.get('tool_diameter'))} mm")
    if cutting_context.get("num_teeth") is not None:
        tool_geometry_bits.append(f"z {_format_int(cutting_context.get('num_teeth'))}")
    tool_type = _text(cutting_context.get("tool_type"))
    if tool_type:
        tool_geometry_bits.append(tool_type)

    if not any((machine_id, operation_id, tool_id, tool_number, dataset_id)):
        return None

    return {
        "dataset_id": dataset_id,
        "machine_id": machine_id,
        "operation_id": operation_id,
        "machine_family": machine_family,
        "tool_number": tool_number,
        "tool_id": tool_id,
        "tool_label": tool_id or (f"T{tool_number}" if tool_number is not None else "Unknown tool"),
        "tool_geometry": " · ".join(tool_geometry_bits),
        "tool_ready": ready,
        "missing_fields": missing_fields,
        "hover_detail": hover_detail,
        "spindle_speed": _coerce_float(cutting_context.get("spindle_speed")),
        "feed_rate": _coerce_float(cutting_context.get("feed_rate")),
        "sindit_tool_iri": sindit_tool_iri,
        "part_label": operation_id or machine_id or dataset_id,
        "part_detail": machine_id if machine_id and operation_id else dataset_id or machine_family,
        "sample_index": sample_index,
    }


def _active_sample_index(session: Mapping[str, Any], data: Mapping[str, Any]) -> int:
    lengths = [len(series) for series in data.values() if isinstance(series, list) and series]
    if not lengths:
        return 0
    total_samples = min(lengths)
    if total_samples <= 0:
        return 0
    position = int(session.get("position", 0) or 0)
    return max(0, min(total_samples - 1, position - 1 if position > 0 else 0))


def _sample_signals(data: Mapping[str, Any], sample_index: int) -> Dict[str, Any]:
    signals: Dict[str, Any] = {}
    for channel, series in data.items():
        if not isinstance(series, list) or sample_index >= len(series):
            continue
        signals[str(channel)] = series[sample_index]
    return signals


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonical_int(value: Any) -> int | None:
    text = _text(value)
    if text is None:
        return None
    if text.upper().startswith("T"):
        text = text[1:]
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return "?"
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return f"{numeric:.2f}".rstrip("0").rstrip(".")


def _format_int(value: Any) -> str:
    numeric = _canonical_int(value)
    return str(numeric) if numeric is not None else "?"
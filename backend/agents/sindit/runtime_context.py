from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Mapping

from backend.agents.processing.tool_lookup import (
    FAMILY_MACHINE_A1,
    FAMILY_BUILDER_B12,
    FAMILY_PRESS_C_20_0482_010,
    resolve_machine_family,
    resolve_tool_context,
)

from .asset_catalog import build_machine_asset


_MACHINE_URI_KEYS = ("machine_uri", "machine_iri", "sindit_asset_iri", "asset_iri")
_MACHINE_ID_KEYS = ("machine_id", "machine", "asset_id", "machine_name", "case_dir")
_MACHINE_FAMILY_KEYS = ("machine_family", "family")
_DATASET_ID_KEYS = ("dataset_id", "source_dataset_id")
_OPERATION_ID_KEYS = ("operation_id", "operation", "of_id", "part_id", "part")
_SPINDLE_SPEED_KEYS = (
    "Spindle_Speed_Actual",
    "Spindle_Speed_Commanded",
    "spindle_speed",
    "spindle",
    "n",
)
_FEED_RATE_KEYS = (
    "Feed_Rate_Actual",
    "Feed_Rate_Commanded",
    "feed_rate",
    "feed",
    "vf",
)
_TOOL_NUMBER_KEYS = (
    "Tool_Number",
    "tool_number",
    "tool",
    "tool_id",
    "Cnc_Tool_Number_RT",
    "Cnc_Tool_Number",
    "CNC_Tool_Number",
)
# String tool *identity* (e.g. Site_a_line2's SG_Active_tool_name) — the stable id used
# to resolve tools when magazine slot numbering is unreliable.
_TOOL_ID_KEYS = (
    "SG_Active_tool_name",
    "active_tool_name",
    "tool_name",
)
_TEETH_KEYS = (
    "num_teeth",
    "NumberOfTeeth",
    "CNC_parameters_teeth_num",
    "teeth",
    "z",
)
_LIVE_METADATA_NESTS = ("metadata", "signals", "frame")


def resolve_runtime_metadata(
    metadata: Mapping[str, Any] | None,
    payload: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved = deepcopy(dict(metadata or {}))
    payload_dict = dict(payload or {})

    source = _extract_source(resolved, payload_dict)
    if source is not None:
        resolved["source"] = source

    explicit_machine_uri = _extract_machine_uri(resolved, payload_dict)
    machine_id = _extract_machine_id(resolved, payload_dict)
    machine_uri = explicit_machine_uri or _machine_uri_for_id(machine_id)
    if machine_id is not None:
        resolved["machine_id"] = machine_id
    if machine_uri is not None:
        resolved["machine_uri"] = machine_uri
        resolved["machine_iri"] = machine_uri
        resolved["sindit_asset_iri"] = machine_uri

    machine_family = _extract_machine_family(resolved, payload_dict, machine_id, source)
    if machine_family is not None:
        resolved["machine_family"] = machine_family

    dataset_id = _extract_dataset_id(resolved, payload_dict, machine_id, machine_family, source)
    if dataset_id is not None:
        resolved["dataset_id"] = dataset_id
        resolved["source_dataset_id"] = dataset_id

    casedata = deepcopy(dict(resolved.get("casedata") or {}))
    if dataset_id is not None:
        casedata["dataset_id"] = dataset_id
    if machine_id is not None:
        casedata.setdefault("case_dir", machine_id)

    operation_id = _extract_operation_id(resolved, payload_dict)
    if operation_id is not None:
        casedata["operation_id"] = operation_id

    cutting_context = deepcopy(dict(casedata.get("cutting_context") or {}))
    extra = deepcopy(dict(cutting_context.get("extra") or {}))

    if machine_id is not None:
        cutting_context["machine_id"] = machine_id
    if machine_family is not None:
        extra["machine_family"] = machine_family
    if machine_uri is not None:
        extra["sindit_asset_iri"] = machine_uri

    spindle_speed = _extract_numeric(payload_dict, resolved, _SPINDLE_SPEED_KEYS)
    if spindle_speed is not None:
        cutting_context["spindle_speed"] = spindle_speed

    feed_rate = _extract_numeric(payload_dict, resolved, _FEED_RATE_KEYS)
    if feed_rate is not None:
        cutting_context["feed_rate"] = feed_rate

    raw_teeth = _extract_integer(payload_dict, resolved, _TEETH_KEYS)
    tool_number = _extract_tool_number(resolved, payload_dict)
    tool_id = _extract_tool_id(resolved, payload_dict)
    if (tool_number is not None or tool_id is not None) and machine_family is not None:
        tool_context = resolve_tool_context(
            machine_family,
            tool_number,
            dataset_id=dataset_id,
            machine_id=machine_id,
            raw_teeth=raw_teeth,
            tool_id=tool_id,
        )
        for source_key, target_key in (
            ("tool_id", "tool_id"),
            ("tool_type", "tool_type"),
            ("tool_diameter", "tool_diameter"),
            ("tool_length", "tool_length"),
            ("tool_material", "tool_material"),
            ("num_teeth", "num_teeth"),
        ):
            value = tool_context.get(source_key)
            if value is not None:
                cutting_context[target_key] = value
        if tool_context.get("tool_number") is not None:
            extra["tool_number"] = int(tool_context["tool_number"])
        if tool_context.get("sindit_tool_iri") is not None:
            extra["sindit_tool_iri"] = tool_context["sindit_tool_iri"]
    elif raw_teeth is not None:
        cutting_context["num_teeth"] = raw_teeth

    if extra:
        cutting_context["extra"] = extra
    if cutting_context:
        casedata["cutting_context"] = cutting_context
    if casedata:
        resolved["casedata"] = casedata

    return resolved


def resolve_payload_machine_asset(
    payload: Mapping[str, Any] | None,
    *,
    default_asset_uri: str | None = None,
    default_label: str | None = None,
) -> Dict[str, str]:
    resolved = resolve_runtime_metadata({}, payload)
    asset_uri = (
        _text(resolved.get("sindit_asset_iri"))
        or _text(resolved.get("machine_uri"))
        or _text(resolved.get("machine_iri"))
        or default_asset_uri
        or "urn:lfl:asset:cnc-machine-1"
    )
    machine_id = _text(resolved.get("machine_id")) or default_label or asset_uri
    return {
        "asset_uri": asset_uri,
        "label": machine_id,
        "machine_id": machine_id,
    }


def _extract_source(metadata: Mapping[str, Any], payload: Mapping[str, Any]) -> str | None:
    payload_meta = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else None
    for candidate in (
        payload.get("source"),
        payload_meta.get("source") if payload_meta else None,
        metadata.get("source"),
    ):
        text = _text(candidate)
        if text is not None:
            return text
    return None


def _extract_machine_uri(metadata: Mapping[str, Any], payload: Mapping[str, Any]) -> str | None:
    for container in _machine_uri_sources(payload, metadata):
        for key in _MACHINE_URI_KEYS:
            text = _text(container.get(key))
            if text is not None:
                return text
    return None


# Keys whose *values* may embed a machine name (e.g. a source tag like
# "site_b_machine_b1_of00001") even when no explicit machine_id is present.
_MACHINE_HINT_KEYS = (
    "source",
    "dataset_id",
    "source_dataset_id",
    "dataset",
    "operation_id",
    "case_dir",
    "machine_name",
)


def _alnum(text: Any) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _machine_hint_tokens() -> tuple[tuple[str, str], ...]:
    """Build ``(token, machine_id)`` pairs from the machine-family registry,
    ordered most-specific (longest token) first, so a model code like
    ``machine_b1`` wins over the ambiguous maker token ``site_b``.
    """
    try:
        from backend.agents.processing.tool_lookup import load_machine_family_registry

        registry = load_machine_family_registry()
    except Exception:
        return ()
    seen: set[tuple[str, str]] = set()
    entries: list[tuple[str, str]] = []
    for machine_ids in registry.values():
        for mid in machine_ids:
            for part in [mid, *mid.split(" - ")]:
                tok = _alnum(part)
                if len(tok) >= 3 and (tok, mid) not in seen:
                    seen.add((tok, mid))
                    entries.append((tok, mid))
    entries.sort(key=lambda x: len(x[0]), reverse=True)
    return tuple(entries)


def _machine_id_from_hint(*texts: str | None) -> str | None:
    """Resolve a machine_id from free-text hints (source/dataset/operation) by
    matching known machine-name tokens. Returns ``None`` when nothing matches."""
    hay = _alnum(" ".join(t for t in texts if t))
    if not hay:
        return None
    for tok, mid in _machine_hint_tokens():
        if tok in hay:
            return mid
    return None


def _extract_machine_id(
    metadata: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str | None:
    for container in _machine_id_sources(payload, metadata):
        for key in _MACHINE_ID_KEYS:
            text = _text(container.get(key))
            if text is not None:
                return text
    # Fallback: derive the machine from a naming hint embedded in the session's
    # source / dataset / operation tags, so a session tagged only with e.g.
    # "site_b_machine_b1_of00001" still resolves to its SINDIT asset.
    hints: list[str] = []
    for container in _machine_id_sources(payload, metadata):
        for key in _MACHINE_HINT_KEYS:
            text = _text(container.get(key))
            if text is not None:
                hints.append(text)
    return _machine_id_from_hint(*hints)


def _extract_machine_family(
    metadata: Mapping[str, Any],
    payload: Mapping[str, Any],
    machine_id: str | None,
    source: str | None,
) -> str | None:
    for container in _ordered_sources(payload, metadata):
        for key in _MACHINE_FAMILY_KEYS:
            text = _text(container.get(key))
            if text is not None:
                return text.strip().lower()

        cutting_context = container.get("cutting_context")
        if isinstance(cutting_context, Mapping):
            extra = cutting_context.get("extra")
            if isinstance(extra, Mapping):
                text = _text(extra.get("machine_family"))
                if text is not None:
                    return text.strip().lower()

    if machine_id is not None:
        resolved = _text(resolve_machine_family(machine_id))
        if resolved is not None:
            return resolved.strip().lower()

    source_text = (source or "").strip().lower()
    if "site_a_line2" in source_text:
        return FAMILY_MACHINE_A1
    if "site_c" in source_text:
        return FAMILY_PRESS_C_20_0482_010
    if source_text in {"simulated_casedata", "casedata", "simulated_file"}:
        return FAMILY_BUILDER_B12
    return None


def _extract_dataset_id(
    metadata: Mapping[str, Any],
    payload: Mapping[str, Any],
    machine_id: str | None,
    machine_family: str | None,
    source: str | None,
) -> str | None:
    for container in _ordered_sources(payload, metadata):
        for key in _DATASET_ID_KEYS:
            text = _text(container.get(key))
            if text is not None:
                return text.strip().lower()

    machine_text = (machine_id or "").strip().lower()
    source_text = (source or "").strip().lower()
    family_text = (machine_family or "").strip().lower()

    if machine_text == "olddata":
        return "site_b_olddata"
    if "site_a_line2" in source_text:
        return "site_a_line2"
    if machine_text.startswith("site_c") or family_text == FAMILY_PRESS_C_20_0482_010:
        return "site_c_casedata"
    if source_text in {"simulated_casedata", "casedata", "simulated_file"}:
        if family_text == FAMILY_MACHINE_A1 or "site_a" in machine_text:
            return "site_a_casedata"
        return "site_b_casedata"
    if family_text == FAMILY_MACHINE_A1:
        return "site_a_line2"
    if family_text == FAMILY_BUILDER_B12:
        return "site_b_casedata"
    return None


def _extract_operation_id(metadata: Mapping[str, Any], payload: Mapping[str, Any]) -> str | None:
    for container in _ordered_sources(payload, metadata):
        for key in _OPERATION_ID_KEYS:
            operation_id = _canonical_operation_id(container.get(key))
            if operation_id is not None:
                return operation_id
    return None


def _extract_numeric(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    aliases: tuple[str, ...],
) -> float | None:
    for container in _live_value_sources(payload, metadata):
        for alias in aliases:
            value = _coerce_float(container.get(alias))
            if value is not None:
                return value
    return None


def _extract_integer(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    aliases: tuple[str, ...],
) -> int | None:
    for container in _live_value_sources(payload, metadata):
        for alias in aliases:
            value = _canonical_int(container.get(alias))
            if value is not None:
                return value
    return None


def _extract_tool_number(metadata: Mapping[str, Any], payload: Mapping[str, Any]) -> int | None:
    for container in _live_value_sources(payload, metadata):
        for alias in _TOOL_NUMBER_KEYS:
            value = _canonical_int(container.get(alias))
            if value is not None:
                return value
    return None


def _extract_tool_id(metadata: Mapping[str, Any], payload: Mapping[str, Any]) -> str | None:
    for container in _live_value_sources(payload, metadata):
        for alias in _TOOL_ID_KEYS:
            value = _text(container.get(alias))
            if value is not None:
                return value
    return None


def _machine_uri_sources(payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    payload_meta = payload.get("metadata")
    if isinstance(payload_meta, Mapping):
        sources.append(payload_meta)
    sources.append(payload)
    sources.extend(_metadata_nested_sources(metadata))
    return sources


def _machine_id_sources(payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    payload_meta = payload.get("metadata")
    if isinstance(payload_meta, Mapping):
        sources.extend(_metadata_nested_sources(payload_meta))
        sources.append(payload_meta)
    sources.append(payload)
    sources.extend(_metadata_nested_sources(metadata))
    sources.append(metadata)
    return sources


def _ordered_sources(payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    payload_meta = payload.get("metadata")
    if isinstance(payload_meta, Mapping):
        sources.extend(_metadata_nested_sources(payload_meta))
        sources.append(payload_meta)
    sources.extend(_metadata_nested_sources(metadata))
    sources.append(metadata)
    sources.append(payload)
    return sources


def _live_value_sources(payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    for key in _LIVE_METADATA_NESTS:
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            sources.append(candidate)
            sources.extend(_metadata_nested_sources(candidate))
    sources.extend(_metadata_nested_sources(metadata))
    sources.append(metadata)
    return sources


def _metadata_nested_sources(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    casedata = metadata.get("casedata")
    if isinstance(casedata, Mapping):
        sources.append(casedata)
        cutting_context = casedata.get("cutting_context")
        if isinstance(cutting_context, Mapping):
            sources.append(cutting_context)
            extra = cutting_context.get("extra")
            if isinstance(extra, Mapping):
                sources.append(extra)
    machining = metadata.get("machining")
    if isinstance(machining, Mapping):
        sources.append(machining)
    mqtt = metadata.get("mqtt")
    if isinstance(mqtt, Mapping):
        sources.append(mqtt)
    return sources


def _canonical_operation_id(value: Any) -> str | None:
    text = _text(value)
    if text is None:
        return None
    upper = text.upper().strip()
    if upper.startswith("OF"):
        return upper
    numeric = _canonical_int(text)
    if numeric is not None and numeric > 0:
        return f"OF{numeric}"
    return upper or None


def _machine_uri_for_id(machine_id: str | None) -> str | None:
    text = _text(machine_id)
    if text is None:
        return None
    if text.startswith("urn:lfl:asset:"):
        return text
    return str(build_machine_asset(text, label=text)["uri"])


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
    if isinstance(value, bool):
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "resolve_payload_machine_asset",
    "resolve_runtime_metadata",
]
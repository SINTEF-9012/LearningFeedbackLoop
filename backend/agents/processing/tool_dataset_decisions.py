"""Dataset-scoped tool decision persistence for runtime enrichment.

Confirmed dataset-tool decisions are persisted as concrete tool-context
snapshots so runtime resolution does not depend on the audit/UI layer or on
transient graph/runtime evidence remaining available later.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_TOOL_DECISIONS_PATH = REPO_ROOT / "data" / "tools" / "dataset_tool_decisions.json"
TOOL_SELECTION_MODES = frozenset({"default", "master", "reference", "runtime", "sindit", "manual"})
TOOL_DECISION_STATUSES = frozenset({"pending", "confirmed", "rejected"})
_CONTEXT_FIELDS = (
    "tool_id",
    "tool_type",
    "tool_diameter",
    "num_teeth",
    "tool_length",
    "tool_material",
)


def load_tool_dataset_decisions(path: Path | str | None = None) -> dict[tuple[str, str, int], Dict[str, Any]]:
    target = Path(path) if path is not None else DATASET_TOOL_DECISIONS_PATH
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to load dataset tool decisions from %s", target, exc_info=True)
        return {}

    raw_items = payload.get("decisions") if isinstance(payload, dict) else []
    out: dict[tuple[str, str, int], Dict[str, Any]] = {}
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        dataset_id = _text(item.get("dataset_id"))
        machine_family = _text(item.get("machine_family"))
        tool_number = _canonical_tool_number(item.get("tool_number"))
        if dataset_id is None or machine_family is None or tool_number is None:
            continue
        selection_mode = _normalize_selection_mode(item.get("selection_mode"))
        status = _normalize_status(item.get("status"))
        reference_tool_number = _canonical_tool_number(item.get("reference_tool_number"))
        out[(dataset_id, machine_family, tool_number)] = {
            "dataset_id": dataset_id,
            "machine_family": machine_family,
            "tool_number": tool_number,
            "selection_mode": selection_mode,
            "status": status,
            "reference_tool_number": reference_tool_number,
            "updated_at": _text(item.get("updated_at")),
            "updated_by": _text(item.get("updated_by")),
            "notes": _text(item.get("notes")),
            "resolved_context": _clean_resolved_context(item.get("resolved_context")),
            "resolved_sources": _clean_resolved_sources(item.get("resolved_sources")),
        }
    return out


def save_tool_dataset_decision(
    *,
    dataset_id: str,
    machine_family: str,
    tool_number: int,
    status: str,
    selection_mode: str = "default",
    reference_tool_number: int | None = None,
    updated_by: str | None = None,
    notes: str | None = None,
    resolved_context: Mapping[str, Any] | None = None,
    resolved_sources: Mapping[str, Any] | None = None,
    path: Path | str | None = None,
) -> Dict[str, Any]:
    target = Path(path) if path is not None else DATASET_TOOL_DECISIONS_PATH
    normalized_dataset_id = _text(dataset_id)
    normalized_family = _text(machine_family)
    normalized_tool_number = _canonical_tool_number(tool_number)
    if normalized_dataset_id is None or normalized_family is None or normalized_tool_number is None:
        raise ValueError("dataset_id, machine_family, and tool_number are required")

    normalized_selection = _normalize_selection_mode(selection_mode)
    normalized_status = _normalize_status(status)
    normalized_reference_tool_number = _canonical_tool_number(reference_tool_number)
    cleaned_context = _clean_resolved_context(resolved_context)
    cleaned_sources = _clean_resolved_sources(resolved_sources)
    if normalized_status != "confirmed":
        cleaned_context = {}
        cleaned_sources = {}

    decision = {
        "dataset_id": normalized_dataset_id,
        "machine_family": normalized_family,
        "tool_number": normalized_tool_number,
        "selection_mode": normalized_selection,
        "status": normalized_status,
        "reference_tool_number": normalized_reference_tool_number,
        "updated_at": _now_iso(),
        "updated_by": _text(updated_by),
        "notes": _text(notes),
        "resolved_context": cleaned_context,
        "resolved_sources": cleaned_sources,
    }

    current = load_tool_dataset_decisions(target)
    current[(normalized_dataset_id, normalized_family, normalized_tool_number)] = decision
    payload = {
        "version": 2,
        "updated_at": decision["updated_at"],
        "decisions": sorted(
            current.values(),
            key=lambda item: (item["dataset_id"], item["machine_family"], int(item["tool_number"])),
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return dict(decision)


def resolve_confirmed_tool_context(
    dataset_id: str | None,
    machine_family: str | None,
    tool_number: int | float | str | None,
    *,
    path: Path | str | None = None,
) -> dict[str, Any] | None:
    normalized_dataset_id = _text(dataset_id)
    normalized_family = _text(machine_family)
    normalized_tool_number = _canonical_tool_number(tool_number)
    if normalized_dataset_id is None or normalized_family is None or normalized_tool_number is None:
        return None
    decision = load_tool_dataset_decisions(path).get((normalized_dataset_id, normalized_family, normalized_tool_number))
    if not decision or decision.get("status") != "confirmed":
        return None
    context = dict(decision.get("resolved_context") or {})
    return context or None


def _normalize_selection_mode(value: Any) -> str:
    text = (_text(value) or "default").lower()
    if text not in TOOL_SELECTION_MODES:
        raise ValueError(f"Unsupported selection_mode: {value}")
    return text


def _normalize_status(value: Any) -> str:
    text = (_text(value) or "pending").lower()
    if text not in TOOL_DECISION_STATUSES:
        raise ValueError(f"Unsupported status: {value}")
    return text


def _clean_resolved_context(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    out: dict[str, Any] = {}
    tool_id = _text(payload.get("tool_id"))
    if tool_id is not None:
        out["tool_id"] = tool_id
    tool_type = _text(payload.get("tool_type"))
    if tool_type is not None:
        out["tool_type"] = tool_type
    tool_diameter = _parse_float(payload.get("tool_diameter"))
    if tool_diameter is not None:
        out["tool_diameter"] = tool_diameter
    num_teeth = _parse_int(payload.get("num_teeth"))
    if num_teeth is not None and num_teeth > 0:
        out["num_teeth"] = num_teeth
    tool_length = _parse_float(payload.get("tool_length"))
    if tool_length is not None:
        out["tool_length"] = tool_length
    tool_material = _text(payload.get("tool_material"))
    if tool_material is not None:
        out["tool_material"] = tool_material
    return out


def _clean_resolved_sources(payload: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        return {}
    out: dict[str, str] = {}
    for field in _CONTEXT_FIELDS:
        value = _text(payload.get(field))
        if value is not None:
            out[field] = value
    return out


def _canonical_tool_number(value: Any) -> int | None:
    text = _text(value)
    if text is None:
        return None
    if text.lower().startswith("t"):
        text = text[1:]
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "DATASET_TOOL_DECISIONS_PATH",
    "TOOL_DECISION_STATUSES",
    "TOOL_SELECTION_MODES",
    "load_tool_dataset_decisions",
    "resolve_confirmed_tool_context",
    "save_tool_dataset_decision",
]
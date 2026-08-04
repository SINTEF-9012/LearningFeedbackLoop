"""Tool-audit aggregation for SINDIT, tool master, and runtime observations."""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from backend.agents.processing.tool_lookup import (
    ToolSpec,
    load_machine_family_registry,
    load_tool_master,
    resolve_machine_family,
)
from backend.agents.processing.tool_dataset_decisions import (
    load_tool_dataset_decisions,
    save_tool_dataset_decision,
)
from backend.agents.processing.tool_lookup_coverage import collect_coverage_items
from backend.agents.processing.tool_reference_catalog import (
    load_site_b_critical_references,
    load_use_case_operation_index,
)

from .asset_catalog import build_machine_asset

logger = logging.getLogger(__name__)

_RUNTIME_OBSERVATIONS: dict[tuple[str, str, int], Dict[str, Any]] = {}
_RUNTIME_LOCK = threading.Lock()
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "data" / "tools"
_TOOL_URI_RE = re.compile(r"^urn:lfl:tool:(.+)-t(\d+)$", re.IGNORECASE)
_PROFILE_LABELS: dict[str, str] = {
    "default": "Default recommendation",
    "master": "Workbook master",
    "reference": "Secondary reference",
    "runtime": "Observed runtime context",
    "sindit": "SINDIT graph",
    "manual": "Manual override",
}
_DATASET_SORT_ORDER = {
    "site_b_casedata": 0,
    "site_b_olddata": 1,
    "site_c_casedata": 2,
    "site_a_casedata": 3,
    "site_a_line2": 4,
}
_DATASET_CONTEXT: dict[str, dict[str, Any]] = {
    "site_b_casedata": {
        "label": "Site_b casedata",
        "shared_workpiece": True,
        "workpiece_note": "MACHINE_B1 / CASE_B1 and MACHINE_B2 / CASE_B2 are two machines cutting the same workpiece.",
    },
    "site_b_olddata": {
        "label": "Site_b olddata",
        "shared_workpiece": False,
        "workpiece_note": None,
    },
    "site_c_casedata": {
        "label": "SITE_C casedata",
        "shared_workpiece": False,
        "workpiece_note": None,
    },
    "site_a_casedata": {
        "label": "Site_a casedata",
        "shared_workpiece": False,
        "workpiece_note": None,
    },
    "site_a_line2": {
        "label": "Site_a_line2",
        "shared_workpiece": False,
        "workpiece_note": None,
    },
}
_DATA_REVIEW_FLAGS = {
    "missing_master_spec",
    "missing_tool_diameter",
    "missing_num_teeth",
    "missing_tool_type",
    "diameter_mismatch_mm",
    "teeth_mismatch",
    "tool_type_mismatch",
    "tool_length_mismatch_mm",
    "duplicate_tool_assets",
}
_ANOMALY_COUNT_FIELDS = (
    "scored_count",
    "significant_count",
    "alerted_count",
    "confirmed_count",
    "dismissed_count",
)


def clear_runtime_observations() -> None:
    with _RUNTIME_LOCK:
        _RUNTIME_OBSERVATIONS.clear()


def _empty_anomaly_stats() -> Dict[str, Any]:
    return {
        "scored_count": 0,
        "significant_count": 0,
        "alerted_count": 0,
        "confirmed_count": 0,
        "dismissed_count": 0,
        "last_score": None,
        "last_memory_id": None,
        "last_patterns": [],
        "last_triggered_rules": [],
        "last_pattern_priors": {},
        "pattern_counts": {},
        "last_event_at": None,
        "last_feedback_at": None,
        "last_feedback_action": None,
        "last_feedback_patterns": [],
        "last_operator_id": None,
    }


def _resolve_tool_identity(cutting_context: Any) -> tuple[Dict[str, Any], Dict[str, Any]] | None:
    ctx = _ctx_to_dict(cutting_context)
    extra = dict(ctx.get("extra") or {})
    machine_id = _text(ctx.get("machine_id"))
    tool_id = _text(ctx.get("tool_id"))
    tool_uri = _text(extra.get("sindit_tool_iri"))
    tool_number = _canonical_tool_number(extra.get("tool_number"))
    family = _text(extra.get("machine_family"))

    if tool_number is None and tool_uri:
        parsed = _parse_tool_uri(tool_uri)
        if parsed is not None:
            family = family or parsed[0]
            tool_number = parsed[1]

    if tool_number is None and tool_id:
        tool_number = _canonical_tool_number(tool_id)

    if family is None and machine_id is not None:
        family = resolve_machine_family(machine_id)

    if family is None or tool_number is None:
        return None

    return (
        {
            "machine_id": machine_id,
            "tool_id": tool_id,
            "tool_uri": tool_uri or _expected_tool_uri(family, tool_number),
            "tool_number": tool_number,
            "machine_family": family,
        },
        ctx,
    )


def _effective_ctx_from_ctx(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "tool_diameter": ctx.get("tool_diameter"),
        "num_teeth": ctx.get("num_teeth"),
        "tool_type": ctx.get("tool_type"),
        "tool_length": ctx.get("tool_length"),
        "tool_material": ctx.get("tool_material"),
        "spindle_speed": ctx.get("spindle_speed"),
        "feed_rate": ctx.get("feed_rate"),
    }


def _ensure_runtime_observation(
    *,
    session_id: str,
    identity: Mapping[str, Any],
    now_iso: str,
    effective_ctx: Mapping[str, Any],
) -> Dict[str, Any]:
    key = (str(session_id), str(identity["machine_family"]), int(identity["tool_number"]))
    existing = _RUNTIME_OBSERVATIONS.get(key)
    if existing is None:
        existing = {
            "session_id": str(session_id),
            "machine_id": identity.get("machine_id"),
            "machine_family": identity.get("machine_family"),
            "tool_number": identity.get("tool_number"),
            "tool_id": identity.get("tool_id"),
            "tool_uri": identity.get("tool_uri"),
            "seen_count": 0,
            "first_seen_at": now_iso,
            "last_seen_at": now_iso,
            "effective_ctx": dict(effective_ctx or {}),
            "anomaly_stats": _empty_anomaly_stats(),
        }
        _RUNTIME_OBSERVATIONS[key] = existing
        return existing

    if existing.get("first_seen_at") is None:
        existing["first_seen_at"] = now_iso
    if identity.get("machine_id") is not None:
        existing["machine_id"] = identity.get("machine_id")
    if identity.get("tool_id") is not None:
        existing["tool_id"] = identity.get("tool_id")
    if identity.get("tool_uri") is not None:
        existing["tool_uri"] = identity.get("tool_uri")
    if not isinstance(existing.get("anomaly_stats"), dict):
        existing["anomaly_stats"] = _empty_anomaly_stats()
    return existing


def _merge_effective_ctx(target: Dict[str, Any], effective_ctx: Mapping[str, Any]) -> None:
    for field, value in effective_ctx.items():
        if value is not None and _should_update_effective_ctx(field, target.get("effective_ctx", {}).get(field), value):
            target.setdefault("effective_ctx", {})[field] = value


def _clean_pattern_list(values: Sequence[Any] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        text = _text(value)
        if text is not None:
            out.append(text)
    return out


def _clean_pattern_priors(values: Mapping[str, Any] | None) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for key, value in (values or {}).items():
        text_key = _text(key)
        numeric = _as_float(value)
        if text_key is None or numeric is None:
            continue
        cleaned[text_key] = round(numeric, 6)
    return cleaned


def _harmonic_ready_from_effective_ctx(effective_ctx: Mapping[str, Any]) -> bool:
    return all(
        effective_ctx.get(field) is not None
        for field in ("tool_diameter", "num_teeth", "spindle_speed", "feed_rate")
    )


def _resolve_effective_harmonic_ctx(
    runtime_layer: Mapping[str, Any] | None,
    master_layer: Mapping[str, Any] | None,
    sindit_layer: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    runtime_effective = dict((runtime_layer or {}).get("effective_ctx") or {})
    return {
        "tool_diameter": _first_non_none(
            runtime_effective.get("tool_diameter"),
            master_layer and master_layer.get("diameter_mm"),
            sindit_layer and sindit_layer.get("tool_diameter"),
        ),
        "num_teeth": _first_non_none(
            runtime_effective.get("num_teeth"),
            master_layer and master_layer.get("teeth"),
            sindit_layer and sindit_layer.get("num_teeth"),
        ),
        "spindle_speed": runtime_effective.get("spindle_speed"),
        "feed_rate": runtime_effective.get("feed_rate"),
    }


def _runtime_tool_snapshot(observation: Mapping[str, Any]) -> Dict[str, Any]:
    effective_ctx = dict(observation.get("effective_ctx") or {})
    return {
        "session_id": observation.get("session_id"),
        "machine_id": observation.get("machine_id"),
        "machine_family": observation.get("machine_family"),
        "tool_number": observation.get("tool_number"),
        "tool_id": observation.get("tool_id"),
        "tool_uri": observation.get("tool_uri"),
        "effective_ctx": deepcopy(effective_ctx),
        "harmonic_ready": _harmonic_ready_from_effective_ctx(effective_ctx),
        "anomaly_stats": deepcopy(observation.get("anomaly_stats") or _empty_anomaly_stats()),
    }


def record_tool_observation(session_id: str, cutting_context: Any) -> None:
    resolved = _resolve_tool_identity(cutting_context)
    if resolved is None:
        return
    identity, ctx = resolved
    effective_ctx = _effective_ctx_from_ctx(ctx)
    now_iso = _now_iso()

    with _RUNTIME_LOCK:
        existing = _ensure_runtime_observation(
            session_id=session_id,
            identity=identity,
            now_iso=now_iso,
            effective_ctx=effective_ctx,
        )
        existing["seen_count"] = int(existing.get("seen_count", 0)) + 1
        existing["last_seen_at"] = now_iso
        _merge_effective_ctx(existing, effective_ctx)


def record_tool_anomaly(
    session_id: str,
    cutting_context: Any,
    *,
    memory_id: str | None = None,
    significance_score: Any = None,
    significant: bool | None = None,
    alert_dispatched: bool = False,
    pattern_keys: Sequence[Any] | None = None,
    triggered_rules: Sequence[Any] | None = None,
    pattern_priors: Mapping[str, Any] | None = None,
) -> Dict[str, Any] | None:
    resolved = _resolve_tool_identity(cutting_context)
    if resolved is None:
        return None
    identity, ctx = resolved
    effective_ctx = _effective_ctx_from_ctx(ctx)
    now_iso = _now_iso()
    score = _as_float(significance_score)
    patterns = _clean_pattern_list(pattern_keys)
    rules = _clean_pattern_list(triggered_rules)
    priors = _clean_pattern_priors(pattern_priors)

    with _RUNTIME_LOCK:
        existing = _ensure_runtime_observation(
            session_id=session_id,
            identity=identity,
            now_iso=now_iso,
            effective_ctx=effective_ctx,
        )
        existing["last_seen_at"] = now_iso
        _merge_effective_ctx(existing, effective_ctx)

        stats = existing.setdefault("anomaly_stats", _empty_anomaly_stats())
        stats["scored_count"] = int(stats.get("scored_count", 0)) + 1
        if significant is True:
            stats["significant_count"] = int(stats.get("significant_count", 0)) + 1
        if alert_dispatched:
            stats["alerted_count"] = int(stats.get("alerted_count", 0)) + 1
        if score is not None:
            stats["last_score"] = round(score, 6)
        if memory_id:
            stats["last_memory_id"] = str(memory_id)
        if patterns:
            stats["last_patterns"] = patterns
        if rules:
            stats["last_triggered_rules"] = rules
        if priors:
            stats["last_pattern_priors"] = priors
        counts = stats.setdefault("pattern_counts", {})
        for pattern_key in patterns:
            counts[pattern_key] = int(counts.get(pattern_key, 0)) + 1
        stats["last_event_at"] = now_iso
        return _runtime_tool_snapshot(existing)


def record_tool_feedback(
    session_id: str,
    cutting_context: Any,
    *,
    action: Any,
    memory_id: str | None = None,
    pattern_keys: Sequence[Any] | None = None,
    pattern_priors: Mapping[str, Any] | None = None,
    operator_id: str | None = None,
) -> Dict[str, Any] | None:
    resolved = _resolve_tool_identity(cutting_context)
    if resolved is None:
        return None
    identity, ctx = resolved
    effective_ctx = _effective_ctx_from_ctx(ctx)
    now_iso = _now_iso()
    action_text = _text(action) or "feedback"
    patterns = _clean_pattern_list(pattern_keys)
    priors = _clean_pattern_priors(pattern_priors)

    with _RUNTIME_LOCK:
        existing = _ensure_runtime_observation(
            session_id=session_id,
            identity=identity,
            now_iso=now_iso,
            effective_ctx=effective_ctx,
        )
        existing["last_seen_at"] = now_iso
        _merge_effective_ctx(existing, effective_ctx)

        stats = existing.setdefault("anomaly_stats", _empty_anomaly_stats())
        if action_text == "confirm":
            stats["confirmed_count"] = int(stats.get("confirmed_count", 0)) + 1
        elif action_text == "dismiss":
            stats["dismissed_count"] = int(stats.get("dismissed_count", 0)) + 1
        if memory_id:
            stats["last_memory_id"] = str(memory_id)
        if patterns:
            stats["last_feedback_patterns"] = patterns
        if priors:
            stats["last_pattern_priors"] = priors
        if operator_id:
            stats["last_operator_id"] = str(operator_id)
        stats["last_feedback_action"] = action_text
        stats["last_feedback_at"] = now_iso
        return _runtime_tool_snapshot(existing)


def _should_update_effective_ctx(field: str, existing_value: Any, new_value: Any) -> bool:
    if field not in {"spindle_speed", "feed_rate"}:
        return True

    existing_numeric = _as_float(existing_value)
    new_numeric = _as_float(new_value)
    if new_numeric is None:
        return True
    if existing_numeric is None:
        return True
    if existing_numeric > 0 and new_numeric <= 0:
        return False
    return True


def _is_newer_iso(candidate: Any, current: Any) -> bool:
    candidate_text = _text(candidate)
    current_text = _text(current)
    if candidate_text is None:
        return False
    if current_text is None:
        return True
    return candidate_text > current_text


def build_tool_audit_rows(
    *,
    master: Mapping[tuple[str, int], ToolSpec] | None = None,
    graph: Mapping[tuple[str, int], Sequence[Dict[str, Any]]] | None = None,
    runtime: Mapping[tuple[str, int], Dict[str, Any]] | None = None,
    family_registry: Mapping[str, Sequence[str]] | None = None,
    reference_index: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    process_plan_index: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    sindit_available: bool,
    now: datetime | None = None,
) -> list[Dict[str, Any]]:
    tool_master = master or load_tool_master()
    graph_map = graph or {}
    runtime_map = runtime or {}
    registry = family_registry or load_machine_family_registry()
    reference_map = reference_index or load_site_b_critical_references()
    process_map = process_plan_index or load_use_case_operation_index()
    registry_keys = set(registry.keys())
    current_time = now or datetime.now(timezone.utc)

    all_keys = set(tool_master.keys()) | set(graph_map.keys()) | set(runtime_map.keys())
    rows: list[Dict[str, Any]] = []

    for family, tool_number in sorted(all_keys):
        master_spec = tool_master.get((family, tool_number))
        graph_candidates = list(graph_map.get((family, tool_number), []))
        graph_primary = graph_candidates[0] if graph_candidates else None
        runtime_obs = runtime_map.get((family, tool_number))
        reference_layer = deepcopy(reference_map.get((family, tool_number))) if (family, tool_number) in reference_map else None
        process_layer = deepcopy(process_map.get((family, tool_number))) if (family, tool_number) in process_map else None

        master_layer = _master_layer(master_spec, registry.get(family, [])) if master_spec else None
        sindit_layer = _graph_layer(graph_primary, len(graph_candidates)) if graph_primary else None
        runtime_layer = _runtime_layer(runtime_obs) if runtime_obs else None

        flags: list[str] = []
        if master_spec is None:
            flags.append("missing_master_spec")
        if sindit_available and not graph_candidates:
            flags.append("missing_sindit_asset")
        if sindit_available and len(graph_candidates) > 1:
            flags.append("duplicate_tool_assets")

        effective = (runtime_layer or {}).get("effective_ctx") or {}
        effective_d = _first_non_none(
            effective.get("tool_diameter"),
            master_layer and master_layer.get("diameter_mm"),
            sindit_layer and sindit_layer.get("tool_diameter"),
        )
        effective_z = _first_non_none(
            effective.get("num_teeth"),
            master_layer and master_layer.get("teeth"),
            sindit_layer and sindit_layer.get("num_teeth"),
        )
        effective_type = _first_non_none(
            effective.get("tool_type"),
            master_layer and master_layer.get("tool_type"),
            sindit_layer and sindit_layer.get("tool_type"),
        )

        if effective_d is None:
            flags.append("missing_tool_diameter")
        if effective_z is None:
            flags.append("missing_num_teeth")
        if effective_type is None:
            flags.append("missing_tool_type")

        if master_layer and sindit_layer:
            if _numeric_mismatch(master_layer.get("diameter_mm"), sindit_layer.get("tool_diameter")):
                flags.append("diameter_mismatch_mm")
            if _int_mismatch(master_layer.get("teeth"), sindit_layer.get("num_teeth")):
                flags.append("teeth_mismatch")
            if _string_mismatch(master_layer.get("tool_type"), sindit_layer.get("tool_type")):
                flags.append("tool_type_mismatch")
            if _numeric_mismatch(master_layer.get("tool_length_mm"), sindit_layer.get("tool_length")):
                flags.append("tool_length_mismatch_mm")

        if runtime_layer and family not in registry_keys:
            flags.append("family_resolution_miss")

        if sindit_available and sindit_layer and _is_stale_import(
            sindit_layer.get("last_imported_at"),
            sindit_layer.get("source_workbook") or (master_layer and master_layer.get("source_workbook")),
            current_time,
        ):
            flags.append("stale_import")

        effective_harmonic_ctx = _resolve_effective_harmonic_ctx(
            runtime_layer,
            master_layer,
            sindit_layer,
        )
        harmonic_ready = _harmonic_ready_from_effective_ctx(effective_harmonic_ctx)
        if not harmonic_ready:
            flags.append("harmonic_not_ready")

        rows.append(
            {
                "machine_family": family,
                "tool_number": tool_number,
                "tool_uri": _first_non_none(
                    runtime_layer and runtime_layer.get("tool_uri"),
                    sindit_layer and sindit_layer.get("asset_uri"),
                    _expected_tool_uri(family, tool_number),
                ),
                "master": master_layer,
                "sindit": sindit_layer,
                "runtime": runtime_layer,
                "reference": reference_layer,
                "process_plan": process_layer,
                "flags": sorted(set(flags)),
                "harmonic_ready": harmonic_ready,
            }
        )

    return rows


def build_tool_audit_summary(rows: Iterable[Mapping[str, Any]], *, sindit_available: bool) -> Dict[str, Any]:
    items = list(rows)
    return {
        "sindit_available": sindit_available,
        "tools_seen": len(items),
        "discrepancies": sum(1 for row in items if row.get("flags")),
        "harmonic_ready": sum(1 for row in items if row.get("harmonic_ready") is True),
        "missing_diameter": sum(1 for row in items if "missing_tool_diameter" in row.get("flags", [])),
        "missing_teeth": sum(1 for row in items if "missing_num_teeth" in row.get("flags", [])),
        "missing_sindit_asset": sum(1 for row in items if "missing_sindit_asset" in row.get("flags", [])),
        "family_resolution_miss": sum(1 for row in items if "family_resolution_miss" in row.get("flags", [])),
    }


def filter_tool_audit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    session_id: str | None = None,
    machine_id: str | None = None,
    family: str | None = None,
    tool_number: int | None = None,
    only_discrepancies: bool = False,
) -> list[Dict[str, Any]]:
    expected_machine_uri = None
    if machine_id:
        expected_machine_uri = build_machine_asset(machine_id, label=machine_id)["uri"]

    out: list[Dict[str, Any]] = []
    for row in rows:
        runtime = row.get("runtime") or {}
        master = row.get("master") or {}
        sindit = row.get("sindit") or {}
        if family and row.get("machine_family") != family:
            continue
        if tool_number is not None and row.get("tool_number") != tool_number:
            continue
        if session_id and session_id not in (runtime.get("session_ids") or []):
            continue
        if machine_id:
            runtime_machine_ids = runtime.get("machine_ids") or []
            master_machine_ids = master.get("machine_ids") or []
            sindit_machine_uris = sindit.get("machine_uris") or []
            if (
                machine_id not in runtime_machine_ids
                and machine_id not in master_machine_ids
                and expected_machine_uri not in sindit_machine_uris
            ):
                continue
        if only_discrepancies and not row.get("flags"):
            continue
        out.append(dict(row))
    return out


async def collect_tool_audit_payload(
    *,
    client: Any | None = None,
    session_id: str | None = None,
    machine_id: str | None = None,
    family: str | None = None,
    tool_number: int | None = None,
    only_discrepancies: bool = False,
) -> Dict[str, Any]:
    rows = build_tool_audit_rows(
        master=load_tool_master(),
        graph=await _load_graph_snapshot(client) if client is not None else {},
        runtime=_runtime_snapshot(),
        family_registry=load_machine_family_registry(),
        sindit_available=client is not None,
    )
    filtered = filter_tool_audit_rows(
        rows,
        session_id=session_id,
        machine_id=machine_id,
        family=family,
        tool_number=tool_number,
        only_discrepancies=only_discrepancies,
    )
    return {
        "sindit_available": client is not None,
        "summary": build_tool_audit_summary(filtered, sindit_available=client is not None),
        "items": filtered,
        "total": len(filtered),
    }


def build_tool_dataset_overview(
    *,
    coverage_items: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    decisions: Mapping[tuple[str, str, int], Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    decision_map = decisions or {}
    audit_index = {
        (str(row.get("machine_family") or "").strip().lower(), int(row.get("tool_number") or 0)): deepcopy(dict(row))
        for row in audit_rows
        if row.get("machine_family") and row.get("tool_number") is not None
    }

    grouped: dict[tuple[str, str, int], Dict[str, Any]] = {}
    part_summaries_by_dataset: dict[str, dict[tuple[str | None, str | None], Dict[str, Any]]] = {}
    for item in coverage_items:
        dataset_meta = _dataset_meta_from_coverage_item(item)
        family = _text(item.get("machine_family"))
        machine_id = _text(item.get("machine_id"))
        operation_id = _text(item.get("operation_id"))
        if family is None:
            continue

        dataset_parts = part_summaries_by_dataset.setdefault(dataset_meta["dataset_id"], {})
        part_key = (machine_id, operation_id)
        existing_part = dataset_parts.get(part_key)
        if existing_part is None:
            existing_part = {
                "machine_id": machine_id,
                "operation_id": operation_id,
                "label": " / ".join(part for part in (machine_id, operation_id) if part) or dataset_meta["label"],
                "valid_rows": 0,
                "resolved_dz_rows": 0,
                "harmonic_ready_rows": 0,
                "observed_tools": set(),
                "resolved_dz_tools": set(),
                "harmonic_ready_tools": set(),
            }
            dataset_parts[part_key] = existing_part
        existing_part["valid_rows"] += int(item.get("valid_tool_rows") or 0)
        existing_part["resolved_dz_rows"] += int(item.get("resolved_dz_rows") or 0)
        existing_part["harmonic_ready_rows"] += int(item.get("harmonic_ready_rows") or 0)
        existing_part["observed_tools"].update(item.get("unique_tools") or [])
        existing_part["resolved_dz_tools"].update(item.get("resolved_dz_tools") or [])
        existing_part["harmonic_ready_tools"].update(item.get("harmonic_ready_tools") or [])

        for raw_tool_number in item.get("unique_tools") or []:
            tool_number = _canonical_tool_number(raw_tool_number)
            if tool_number is None:
                continue
            key = (dataset_meta["dataset_id"], family, tool_number)
            current = grouped.get(key)
            if current is None:
                current = {
                    "dataset_id": dataset_meta["dataset_id"],
                    "dataset_label": dataset_meta["label"],
                    "machine_family": family,
                    "tool_number": tool_number,
                    "machine_ids": set(),
                    "operation_ids": set(),
                    "coverage": {
                        "observed": True,
                        "master": False,
                        "diameter": False,
                        "teeth": False,
                        "diameter_and_teeth": False,
                        "harmonic_ready": False,
                    },
                }
                grouped[key] = current
            if machine_id is not None:
                current["machine_ids"].add(machine_id)
            if operation_id is not None:
                current["operation_ids"].add(operation_id)

            coverage = current["coverage"]
            coverage["master"] = coverage["master"] or tool_number in set(item.get("resolved_master_tools") or [])
            coverage["diameter"] = coverage["diameter"] or tool_number in set(item.get("resolved_d_tools") or [])
            coverage["teeth"] = coverage["teeth"] or tool_number in set(item.get("resolved_z_tools") or [])
            coverage["diameter_and_teeth"] = coverage["diameter_and_teeth"] or tool_number in set(item.get("resolved_dz_tools") or [])
            coverage["harmonic_ready"] = coverage["harmonic_ready"] or tool_number in set(item.get("harmonic_ready_tools") or [])

    datasets: list[Dict[str, Any]] = []
    by_dataset: dict[str, Dict[str, Any]] = {}
    for (dataset_id, family, tool_number), item in grouped.items():
        audit = deepcopy(audit_index.get((family, tool_number)))
        if audit is None:
            audit = {
                "machine_family": family,
                "tool_number": tool_number,
                "flags": ["missing_audit_row"],
                "harmonic_ready": False,
            }
        decision = deepcopy(decision_map.get((dataset_id, family, tool_number))) if (dataset_id, family, tool_number) in decision_map else None
        profiles = _build_tool_profiles(audit, decision)
        available_profiles = [mode for mode, profile in profiles.items() if profile.get("available")]
        if "default" not in available_profiles:
            available_profiles.insert(0, "default")
        recommended_profile = "default"
        selected_profile = _selected_profile(decision, available_profiles)
        review_flags = [flag for flag in audit.get("flags", []) if flag in _DATA_REVIEW_FLAGS]
        review_flags = _effective_review_flags(
            review_flags,
            profiles,
            profile_mode=selected_profile if decision and decision.get("status") == "confirmed" else "default",
        )
        certainty, reasons = _tool_dataset_certainty(
            audit,
            profiles,
            review_flags,
            profile_mode=selected_profile if decision and decision.get("status") == "confirmed" else "default",
        )

        row = {
            "dataset_id": dataset_id,
            "dataset_label": item["dataset_label"],
            "machine_family": family,
            "tool_number": tool_number,
            "machine_ids": sorted(item["machine_ids"]),
            "operation_ids": sorted(item["operation_ids"]),
            "operation_count": len(item["operation_ids"]),
            "coverage": dict(item["coverage"]),
            "profiles": profiles,
            "available_profiles": available_profiles,
            "recommended_profile": recommended_profile,
            "selected_profile": selected_profile,
            "decision": decision,
            "decision_status": (decision or {}).get("status") or "pending",
            "certainty": certainty,
            "certainty_reasons": reasons,
            "review_flags": review_flags,
            "evidence_sources": _tool_evidence_sources(audit),
            "audit": audit,
        }

        dataset_bucket = by_dataset.get(dataset_id)
        if dataset_bucket is None:
            dataset_bucket = {
                "dataset_id": dataset_id,
                "label": item["dataset_label"],
                "shared_workpiece": False,
                "workpiece_note": None,
                "machine_ids": set(),
                "machine_families": set(),
                "operation_ids": set(),
                "tools": [],
            }
            by_dataset[dataset_id] = dataset_bucket
        dataset_meta = _dataset_context_meta(dataset_id)
        dataset_bucket["shared_workpiece"] = bool(dataset_meta.get("shared_workpiece"))
        dataset_bucket["workpiece_note"] = dataset_meta.get("workpiece_note")
        dataset_bucket["machine_ids"].update(row["machine_ids"])
        dataset_bucket["machine_families"].add(family)
        dataset_bucket["operation_ids"].update(row["operation_ids"])
        dataset_bucket["tools"].append(row)

    for dataset_id, dataset in sorted(by_dataset.items(), key=lambda item: (_DATASET_SORT_ORDER.get(item[0], 99), item[1]["label"])):
        tools = sorted(dataset["tools"], key=lambda row: (row["machine_family"], row["tool_number"]))
        part_summaries = []
        dataset_parts = part_summaries_by_dataset.get(dataset_id) or {}
        for part in sorted(dataset_parts.values(), key=lambda payload: ((_text(payload.get("machine_id")) or ""), (_text(payload.get("operation_id")) or ""))):
            valid_rows = int(part.get("valid_rows") or 0)
            resolved_dz_rows = int(part.get("resolved_dz_rows") or 0)
            harmonic_ready_rows = int(part.get("harmonic_ready_rows") or 0)
            observed_tools = sorted(int(value) for value in set(part.get("observed_tools") or []))
            resolved_dz_tools = sorted(int(value) for value in set(part.get("resolved_dz_tools") or []))
            harmonic_ready_tools = sorted(int(value) for value in set(part.get("harmonic_ready_tools") or []))
            part_summaries.append({
                "machine_id": part.get("machine_id"),
                "operation_id": part.get("operation_id"),
                "label": part.get("label"),
                "valid_rows": valid_rows,
                "resolved_dz_rows": resolved_dz_rows,
                "harmonic_ready_rows": harmonic_ready_rows,
                "observed_tools": observed_tools,
                "resolved_dz_tools": resolved_dz_tools,
                "harmonic_ready_tools": harmonic_ready_tools,
                "resolved_dz_row_pct": _percent_ratio(resolved_dz_rows, valid_rows),
                "harmonic_ready_row_pct": _percent_ratio(harmonic_ready_rows, valid_rows),
                "resolved_dz_tool_pct": _percent_ratio(len(resolved_dz_tools), len(observed_tools)),
                "harmonic_ready_tool_pct": _percent_ratio(len(harmonic_ready_tools), len(observed_tools)),
            })

        observed_tools = sorted({int(row["tool_number"]) for row in tools})
        resolved_dz_tools = sorted(int(row["tool_number"]) for row in tools if row["coverage"].get("diameter_and_teeth") is True)
        harmonic_ready_tools = sorted(int(row["tool_number"]) for row in tools if row["coverage"].get("harmonic_ready") is True)
        valid_rows = sum(int(part.get("valid_rows") or 0) for part in part_summaries)
        resolved_dz_rows = sum(int(part.get("resolved_dz_rows") or 0) for part in part_summaries)
        harmonic_ready_rows = sum(int(part.get("harmonic_ready_rows") or 0) for part in part_summaries)
        harmonic_summary = {
            "observed_tools": len(observed_tools),
            "resolved_dz_tools": len(resolved_dz_tools),
            "harmonic_ready_tools": len(harmonic_ready_tools),
            "resolved_dz_tool_pct": _percent_ratio(len(resolved_dz_tools), len(observed_tools)),
            "harmonic_ready_tool_pct": _percent_ratio(len(harmonic_ready_tools), len(observed_tools)),
            "valid_rows": valid_rows,
            "resolved_dz_rows": resolved_dz_rows,
            "harmonic_ready_rows": harmonic_ready_rows,
            "resolved_dz_row_pct": _percent_ratio(resolved_dz_rows, valid_rows),
            "harmonic_ready_row_pct": _percent_ratio(harmonic_ready_rows, valid_rows),
            "ready_parts": sum(1 for part in part_summaries if int(part.get("harmonic_ready_rows") or 0) > 0),
            "total_parts": len(part_summaries),
        }
        summary = {
            "tool_count": len(tools),
            "certain_count": sum(1 for row in tools if row["certainty"] == "certain"),
            "defaulted_count": sum(1 for row in tools if row["certainty"] == "defaulted"),
            "needs_review_count": sum(1 for row in tools if row["certainty"] == "needs_review"),
            "confirmed_count": sum(1 for row in tools if row["decision_status"] == "confirmed"),
            "rejected_count": sum(1 for row in tools if row["decision_status"] == "rejected"),
            "pending_count": sum(1 for row in tools if row["decision_status"] == "pending"),
            "master_backed_count": sum(1 for row in tools if row["coverage"].get("master") is True),
        }
        datasets.append({
            "dataset_id": dataset_id,
            "label": dataset["label"],
            "machine_ids": sorted(dataset["machine_ids"]),
            "machine_families": sorted(dataset["machine_families"]),
            "shared_workpiece": bool(dataset.get("shared_workpiece")),
            "workpiece_note": dataset.get("workpiece_note"),
            "operation_count": len(dataset["operation_ids"]),
            "harmonic_summary": harmonic_summary,
            "part_summaries": part_summaries,
            "summary": summary,
            "tools": tools,
        })

    return {
        "datasets": datasets,
        "total_datasets": len(datasets),
        "total_tools": sum(len(dataset["tools"]) for dataset in datasets),
    }


async def collect_tool_dataset_overview_payload(
    *,
    client: Any | None = None,
    dataset_id: str | None = None,
) -> Dict[str, Any]:
    rows = build_tool_audit_rows(
        master=load_tool_master(),
        graph=await _load_graph_snapshot(client) if client is not None else {},
        runtime=_runtime_snapshot(),
        family_registry=load_machine_family_registry(),
        sindit_available=client is not None,
    )
    payload = build_tool_dataset_overview(
        coverage_items=collect_coverage_items(_REPO_ROOT),
        audit_rows=rows,
        decisions=load_tool_dataset_decisions(),
    )
    if dataset_id:
        normalized_dataset_id = _text(dataset_id)
        payload["datasets"] = [dataset for dataset in payload["datasets"] if dataset["dataset_id"] == normalized_dataset_id]
        payload["total_datasets"] = len(payload["datasets"])
        payload["total_tools"] = sum(len(dataset["tools"]) for dataset in payload["datasets"])
    payload["sindit_available"] = client is not None
    return payload


async def _load_graph_snapshot(client: Any) -> dict[tuple[str, int], list[Dict[str, Any]]]:
    assets = await client.get_assets()
    connections = await client.get_connections()
    machine_uris_by_tool: dict[str, list[str]] = {}
    for rel in connections:
        source = _text(rel.get("sourceUri") or rel.get("relationshipSource") or rel.get("source"))
        target = _text(rel.get("targetUri") or rel.get("relationshipTarget") or rel.get("target"))
        rel_type = _text(rel.get("relationshipType") or rel.get("type"))
        if source and target and rel_type == "HAS_TOOL":
            machine_uris_by_tool.setdefault(target, []).append(source)

    grouped: dict[tuple[str, int], list[Dict[str, Any]]] = {}
    for asset in assets:
        asset_uri = _text(asset.get("uri") or asset.get("nodeUri"))
        if asset_uri is None or not asset_uri.startswith("urn:lfl:tool:"):
            continue
        parsed = _parse_tool_uri(asset_uri)
        if parsed is None:
            continue
        properties = await client.get_properties(asset_uri)
        property_map = {_property_name(prop): _property_value(prop) for prop in properties if _property_name(prop)}
        metadata = asset.get("metadata") or {}
        candidate = {
            "asset_uri": asset_uri,
            "label": asset.get("label"),
            "tool_diameter": _as_float(_first_non_none(property_map.get("ToolDiameter"), metadata.get("diameterMm"))),
            "num_teeth": _as_int(_first_non_none(property_map.get("NumberOfTeeth"), metadata.get("teeth"))),
            "tool_type": _text(_first_non_none(property_map.get("ToolType"), metadata.get("geometry"))),
            "tool_length": _as_float(property_map.get("ToolLength")),
            "tool_material": _text(_first_non_none(property_map.get("ToolMaterial"), metadata.get("material"))),
            "last_imported_at": _text(property_map.get("LastImportedAt")),
            "source_workbook": _text(property_map.get("SourceWorkbook")),
            "properties": property_map,
            "machine_uris": sorted(set(machine_uris_by_tool.get(asset_uri, []))),
        }
        grouped.setdefault(parsed, []).append(candidate)
    return grouped


def _runtime_snapshot() -> dict[tuple[str, int], Dict[str, Any]]:
    with _RUNTIME_LOCK:
        snapshots = [dict(value) for value in _RUNTIME_OBSERVATIONS.values()]

    grouped: dict[tuple[str, int], Dict[str, Any]] = {}
    for obs in snapshots:
        key = (obs["machine_family"], int(obs["tool_number"]))
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = {
                "session_ids": [obs["session_id"]],
                "machine_ids": [obs["machine_id"]] if obs.get("machine_id") else [],
                "tool_id": obs.get("tool_id"),
                "tool_uri": obs.get("tool_uri"),
                "seen_count": int(obs.get("seen_count", 0)),
                "first_seen_at": obs.get("first_seen_at"),
                "last_seen_at": obs.get("last_seen_at"),
                "effective_ctx": dict(obs.get("effective_ctx") or {}),
                "anomaly_stats": deepcopy(obs.get("anomaly_stats") or _empty_anomaly_stats()),
            }
            continue

        if obs["session_id"] not in existing["session_ids"]:
            existing["session_ids"].append(obs["session_id"])
        machine_id = obs.get("machine_id")
        if machine_id and machine_id not in existing["machine_ids"]:
            existing["machine_ids"].append(machine_id)
        existing["seen_count"] += int(obs.get("seen_count", 0))
        existing["first_seen_at"] = min(filter(None, [existing.get("first_seen_at"), obs.get("first_seen_at")]), default=None)
        existing["last_seen_at"] = max(filter(None, [existing.get("last_seen_at"), obs.get("last_seen_at")]), default=None)
        if obs.get("tool_id"):
            existing["tool_id"] = obs["tool_id"]
        if obs.get("tool_uri"):
            existing["tool_uri"] = obs["tool_uri"]
        for field, value in (obs.get("effective_ctx") or {}).items():
            if value is not None and _should_update_effective_ctx(
                field,
                existing.get("effective_ctx", {}).get(field),
                value,
            ):
                existing["effective_ctx"][field] = value

        existing_stats = existing.setdefault("anomaly_stats", _empty_anomaly_stats())
        obs_stats = obs.get("anomaly_stats") or {}
        for field in _ANOMALY_COUNT_FIELDS:
            existing_stats[field] = int(existing_stats.get(field, 0)) + int(obs_stats.get(field, 0) or 0)

        pattern_counts = existing_stats.setdefault("pattern_counts", {})
        for pattern_key, count in (obs_stats.get("pattern_counts") or {}).items():
            clean_key = _text(pattern_key)
            if clean_key is None:
                continue
            pattern_counts[clean_key] = int(pattern_counts.get(clean_key, 0)) + int(count or 0)

        if _is_newer_iso(obs_stats.get("last_event_at"), existing_stats.get("last_event_at")):
            for field in (
                "last_event_at",
                "last_score",
                "last_memory_id",
                "last_patterns",
                "last_triggered_rules",
                "last_pattern_priors",
            ):
                existing_stats[field] = deepcopy(obs_stats.get(field))

        if _is_newer_iso(obs_stats.get("last_feedback_at"), existing_stats.get("last_feedback_at")):
            for field in (
                "last_feedback_at",
                "last_feedback_action",
                "last_feedback_patterns",
                "last_operator_id",
                "last_memory_id",
                "last_pattern_priors",
            ):
                existing_stats[field] = deepcopy(obs_stats.get(field))

    for payload in grouped.values():
        payload["session_ids"].sort()
        payload["machine_ids"].sort()
    return grouped


def _master_layer(spec: ToolSpec, machine_ids: Sequence[str]) -> Dict[str, Any]:
    return {
        "tool_id": spec.tool_id,
        "description": spec.description,
        "tool_type": spec.tool_type,
        "diameter_mm": spec.diameter_mm,
        "teeth": spec.teeth,
        "tool_length_mm": spec.tool_length_mm,
        "tool_material": spec.tool_substrate,
        "source_workbook": spec.source,
        "machine_ids": list(machine_ids),
    }


def _graph_layer(primary: Dict[str, Any], asset_count: int) -> Dict[str, Any]:
    return {
        **primary,
        "asset_count": asset_count,
    }


def _runtime_layer(obs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "session_ids": list(obs.get("session_ids") or []),
        "machine_ids": list(obs.get("machine_ids") or []),
        "tool_id": obs.get("tool_id"),
        "tool_uri": obs.get("tool_uri"),
        "seen_count": obs.get("seen_count"),
        "first_seen_at": obs.get("first_seen_at"),
        "last_seen_at": obs.get("last_seen_at"),
        "effective_ctx": dict(obs.get("effective_ctx") or {}),
        "anomaly_stats": deepcopy(obs.get("anomaly_stats") or _empty_anomaly_stats()),
    }


def _ctx_to_dict(cutting_context: Any) -> Dict[str, Any]:
    if cutting_context is None:
        return {}
    model_dump = getattr(cutting_context, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump())
    if isinstance(cutting_context, dict):
        return dict(cutting_context)
    return dict(getattr(cutting_context, "__dict__", {}) or {})


def _property_name(prop: Mapping[str, Any]) -> str | None:
    for key in ("propertyName", "label", "name"):
        text = _text(prop.get(key))
        if text is not None:
            return text
    return None


def _property_value(prop: Mapping[str, Any]) -> Any:
    return _first_non_none(prop.get("propertyValue"), prop.get("value"), prop.get("staticValue"))


def _parse_tool_uri(uri: str | None) -> tuple[str, int] | None:
    text = _text(uri)
    if text is None:
        return None
    match = _TOOL_URI_RE.match(text)
    if not match:
        return None
    return match.group(1).lower(), int(match.group(2))


def _expected_tool_uri(family: str, tool_number: int) -> str:
    return f"urn:lfl:tool:{family}-t{tool_number}"


def _canonical_tool_number(value: Any) -> int | None:
    text = _text(value)
    if text is None:
        return None
    if text.lower().startswith("t"):
        text = text[1:]
    try:
        return int(float(text))
    except ValueError:
        return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _numeric_mismatch(left: Any, right: Any, tolerance: float = 1e-3) -> bool:
    left_value = _as_float(left)
    right_value = _as_float(right)
    if left_value is None or right_value is None:
        return False
    return abs(left_value - right_value) > tolerance


def _int_mismatch(left: Any, right: Any) -> bool:
    left_value = _as_int(left)
    right_value = _as_int(right)
    if left_value is None or right_value is None:
        return False
    return left_value != right_value


def _string_mismatch(left: Any, right: Any) -> bool:
    left_value = _text(left)
    right_value = _text(right)
    if left_value is None or right_value is None:
        return False
    return left_value.strip().lower() != right_value.strip().lower()


def _is_stale_import(last_imported_at: str | None, source_workbook: str | None, now: datetime) -> bool:
    imported_at = _parse_datetime(last_imported_at)
    if imported_at is None:
        return True
    freshness_floor = now - timedelta(hours=24)
    source_mtime = _source_mtime(source_workbook)
    if source_mtime is not None and source_mtime > freshness_floor:
        freshness_floor = source_mtime
    return imported_at < freshness_floor


def _source_mtime(source_workbook: str | None) -> datetime | None:
    source = _text(source_workbook)
    if source is None:
        return None
    latest: datetime | None = None
    for part in [chunk.strip() for chunk in source.split("+") if chunk.strip()]:
        path = (_SOURCE_ROOT / part).resolve()
        if not path.is_file():
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        latest = mtime if latest is None or mtime > latest else latest
    return latest


def _parse_datetime(value: str | None) -> datetime | None:
    text = _text(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_decision_status(value: Any) -> str:
    text = (_text(value) or "pending").lower()
    if text not in {"pending", "confirmed", "rejected"}:
        return "pending"
    return text


def _dataset_context_meta(dataset_id: str) -> Dict[str, Any]:
    meta = _DATASET_CONTEXT.get(dataset_id) or {}
    return {
        "dataset_id": dataset_id,
        "label": meta.get("label") or dataset_id.replace("_", " "),
        "shared_workpiece": bool(meta.get("shared_workpiece")),
        "workpiece_note": _text(meta.get("workpiece_note")),
    }


def _dataset_meta_from_coverage_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    raw_dataset = (_text(item.get("dataset")) or "").lower()
    machine_id = (_text(item.get("machine_id")) or "").lower()
    if raw_dataset == "site_a_line2":
        return _dataset_context_meta("site_a_line2")
    if raw_dataset in {"site_a", "site_a_casedata"} or machine_id.startswith("site_a"):
        return _dataset_context_meta("site_a_casedata")
    if raw_dataset == "site_c":
        return _dataset_context_meta("site_c_casedata")
    if machine_id == "olddata":
        return _dataset_context_meta("site_b_olddata")
    return _dataset_context_meta("site_b_casedata")


def _profile_from_decision_snapshot(
    decision: Mapping[str, Any],
    *,
    label: str,
) -> Dict[str, Any]:
    resolved_context = dict(decision.get("resolved_context") or {})
    resolved_sources = dict(decision.get("resolved_sources") or {})
    return {
        "label": label,
        "available": any(
            resolved_context.get(field) is not None
            for field in ("tool_id", "tool_diameter", "num_teeth", "tool_type", "tool_length", "tool_material")
        ),
        "tool_id": _profile_value(resolved_context.get("tool_id"), _text(resolved_sources.get("tool_id"))),
        "diameter_mm": _profile_value(resolved_context.get("tool_diameter"), _text(resolved_sources.get("tool_diameter"))),
        "teeth": _profile_value(resolved_context.get("num_teeth"), _text(resolved_sources.get("num_teeth"))),
        "tool_type": _profile_value(resolved_context.get("tool_type"), _text(resolved_sources.get("tool_type"))),
        "tool_length_mm": _profile_value(resolved_context.get("tool_length"), _text(resolved_sources.get("tool_length"))),
        "tool_material": _profile_value(resolved_context.get("tool_material"), _text(resolved_sources.get("tool_material"))),
        "description": _profile_value(None, None),
        "notes": _text(decision.get("notes")),
        "reference_tool_number": _canonical_tool_number(decision.get("reference_tool_number")),
    }


def _merge_profile(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(dict(base))
    for field in (
        "tool_id",
        "diameter_mm",
        "teeth",
        "tool_type",
        "tool_length_mm",
        "tool_material",
        "description",
    ):
        payload = dict(overlay.get(field) or {})
        if payload.get("value") is not None or payload.get("source") is not None:
            merged[field] = payload
    if overlay.get("label") is not None:
        merged["label"] = overlay.get("label")
    merged["available"] = bool(merged.get("available")) or bool(overlay.get("available"))
    if overlay.get("notes") is not None:
        merged["notes"] = overlay.get("notes")
    if overlay.get("reference_tool_number") is not None:
        merged["reference_tool_number"] = overlay.get("reference_tool_number")
    return merged


def _apply_confirmed_decision_profile(
    profiles: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    out = {key: deepcopy(value) for key, value in profiles.items()}
    if not decision or decision.get("status") != "confirmed":
        return out
    selection_mode = _text(decision.get("selection_mode")) or "default"
    label = _PROFILE_LABELS.get(selection_mode, _PROFILE_LABELS["default"])
    overlay = _profile_from_decision_snapshot(decision, label=label)
    out[selection_mode] = _merge_profile(out.get(selection_mode) or {"label": label, "available": False}, overlay)
    return out


def _build_tool_profiles(row: Mapping[str, Any], decision: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    master = row.get("master") or {}
    reference = row.get("reference") or {}
    runtime = row.get("runtime") or {}
    effective_ctx = runtime.get("effective_ctx") or {}
    sindit = row.get("sindit") or {}
    dimensions = reference.get("dimensions") or {}

    profiles = {
        "master": {
            "label": _PROFILE_LABELS["master"],
            "available": bool(master),
            "tool_id": _profile_value(master.get("tool_id"), "master"),
            "diameter_mm": _profile_value(master.get("diameter_mm"), "master"),
            "teeth": _profile_value(master.get("teeth"), "master"),
            "tool_type": _profile_value(master.get("tool_type"), "master"),
            "tool_length_mm": _profile_value(master.get("tool_length_mm"), "master"),
            "tool_material": _profile_value(master.get("tool_material"), "master"),
            "description": _profile_value(master.get("description"), "master"),
        },
        "reference": {
            "label": _PROFILE_LABELS["reference"],
            "available": bool(reference),
            "tool_id": _profile_value(None, None),
            "diameter_mm": _profile_value(
                _first_non_none(dimensions.get("head_diameter_mm"), dimensions.get("arbour_diameter_mm")),
                "reference",
            ),
            "teeth": _profile_value(None, None),
            "tool_type": _profile_value(None, None),
            "tool_length_mm": _profile_value(dimensions.get("overall_length_mm"), "reference"),
            "tool_material": _profile_value(None, None),
            "description": _profile_value(reference.get("description"), "reference"),
        },
        "runtime": {
            "label": _PROFILE_LABELS["runtime"],
            "available": bool(runtime) and any(
                effective_ctx.get(field) is not None for field in ("tool_diameter", "num_teeth", "tool_type")
            ),
            "tool_id": _profile_value(runtime.get("tool_id"), "runtime"),
            "diameter_mm": _profile_value(effective_ctx.get("tool_diameter"), "runtime"),
            "teeth": _profile_value(effective_ctx.get("num_teeth"), "runtime"),
            "tool_type": _profile_value(effective_ctx.get("tool_type"), "runtime"),
            "tool_length_mm": _profile_value(effective_ctx.get("tool_length"), "runtime"),
            "tool_material": _profile_value(effective_ctx.get("tool_material"), "runtime"),
            "description": _profile_value(runtime.get("tool_id"), "runtime"),
        },
        "sindit": {
            "label": _PROFILE_LABELS["sindit"],
            "available": bool(sindit),
            "tool_id": _profile_value(None, None),
            "diameter_mm": _profile_value(sindit.get("tool_diameter"), "sindit"),
            "teeth": _profile_value(sindit.get("num_teeth"), "sindit"),
            "tool_type": _profile_value(sindit.get("tool_type"), "sindit"),
            "tool_length_mm": _profile_value(sindit.get("tool_length"), "sindit"),
            "tool_material": _profile_value(sindit.get("tool_material"), "sindit"),
            "description": _profile_value(_first_non_none(sindit.get("label"), sindit.get("asset_uri")), "sindit"),
        },
    }
    profiles["default"] = {
        "label": _PROFILE_LABELS["default"],
        "available": True,
        "tool_id": _first_profile_value(
            profiles["master"]["tool_id"],
            profiles["runtime"]["tool_id"],
        ),
        "diameter_mm": _first_profile_value(
            profiles["master"]["diameter_mm"],
            profiles["reference"]["diameter_mm"],
            profiles["runtime"]["diameter_mm"],
            profiles["sindit"]["diameter_mm"],
        ),
        "teeth": _first_profile_value(
            profiles["master"]["teeth"],
            profiles["runtime"]["teeth"],
            profiles["sindit"]["teeth"],
        ),
        "tool_type": _first_profile_value(
            profiles["master"]["tool_type"],
            profiles["runtime"]["tool_type"],
            profiles["sindit"]["tool_type"],
        ),
        "tool_length_mm": _first_profile_value(
            profiles["master"]["tool_length_mm"],
            profiles["reference"]["tool_length_mm"],
            profiles["runtime"]["tool_length_mm"],
            profiles["sindit"]["tool_length_mm"],
        ),
        "tool_material": _first_profile_value(
            profiles["master"]["tool_material"],
            profiles["runtime"]["tool_material"],
            profiles["sindit"]["tool_material"],
        ),
        "description": _first_profile_value(
            profiles["master"]["description"],
            profiles["reference"]["description"],
            profiles["sindit"]["description"],
            profiles["runtime"]["description"],
        ),
    }
    profiles["manual"] = {
        **deepcopy(profiles["default"]),
        "label": _PROFILE_LABELS["manual"],
        "available": True,
        "notes": None,
        "reference_tool_number": None,
    }
    return _apply_confirmed_decision_profile(profiles, decision)


def build_tool_dataset_decision_snapshot(
    row: Mapping[str, Any],
    selection_mode: str,
    *,
    reference_row: Mapping[str, Any] | None = None,
    manual_num_teeth: int | None = None,
) -> Dict[str, Any]:
    normalized_selection = _text(selection_mode) or "default"
    if normalized_selection not in _PROFILE_LABELS:
        normalized_selection = "default"

    base_row = reference_row if normalized_selection == "manual" and reference_row is not None else row
    profiles = base_row.get("profiles") if isinstance(base_row.get("profiles"), Mapping) else _build_tool_profiles(base_row)
    profile_mode = "default" if normalized_selection == "manual" else normalized_selection
    profile = profiles.get(profile_mode) or profiles.get("default") or {}
    resolved_context: dict[str, Any] = {}
    resolved_sources: dict[str, str] = {}
    for context_field, profile_field in (
        ("tool_id", "tool_id"),
        ("tool_type", "tool_type"),
        ("tool_diameter", "diameter_mm"),
        ("num_teeth", "teeth"),
        ("tool_length", "tool_length_mm"),
        ("tool_material", "tool_material"),
    ):
        payload = profile.get(profile_field) or {}
        if payload.get("value") is not None:
            resolved_context[context_field] = payload.get("value")
        source = _text(payload.get("source"))
        if source is not None:
            resolved_sources[context_field] = source

    if normalized_selection == "manual" and manual_num_teeth is not None:
        if int(manual_num_teeth) <= 0:
            raise ValueError("manual_num_teeth must be positive")
        resolved_context["num_teeth"] = int(manual_num_teeth)
        resolved_sources["num_teeth"] = "manual"

    return {
        "selection_mode": normalized_selection,
        "resolved_context": resolved_context,
        "resolved_sources": resolved_sources,
    }


def _tool_dataset_certainty(
    row: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, Any]],
    review_flags: Sequence[str],
    *,
    profile_mode: str = "default",
) -> tuple[str, list[str]]:
    active_profile = profiles.get(profile_mode) or profiles.get("default") or {}
    diameter = active_profile.get("diameter_mm") or {}
    teeth = active_profile.get("teeth") or {}
    tool_type = active_profile.get("tool_type") or {}

    reasons: list[str] = []
    if review_flags:
        reasons.extend(f"Review flag: {flag}" for flag in review_flags)
    if diameter.get("value") is None:
        reasons.append("Diameter still unresolved.")
    if teeth.get("value") is None:
        reasons.append("Tooth count still unresolved.")
    if diameter.get("value") is None or teeth.get("value") is None or review_flags:
        return "needs_review", reasons or ["Core tool geometry is incomplete."]

    fallback_fields = []
    for label, field in (("diameter", diameter), ("teeth", teeth), ("type", tool_type)):
        source = field.get("source")
        if source is not None and source != "master":
            fallback_fields.append(f"{label} from {source}")
    if fallback_fields:
        reasons.extend(f"Defaulted {item}." for item in fallback_fields)
        return "defaulted", reasons

    return "certain", ["Workbook master covers the core tool parameters."]


def _effective_review_flags(
    review_flags: Sequence[str],
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    profile_mode: str,
) -> list[str]:
    active_profile = profiles.get(profile_mode) or profiles.get("default") or {}
    out = list(review_flags)
    if (active_profile.get("diameter_mm") or {}).get("value") is not None:
        out = [flag for flag in out if flag != "missing_tool_diameter"]
    if (active_profile.get("teeth") or {}).get("value") is not None:
        out = [flag for flag in out if flag != "missing_num_teeth"]
    if (active_profile.get("tool_type") or {}).get("value") is not None:
        out = [flag for flag in out if flag != "missing_tool_type"]
    return out


def _tool_evidence_sources(row: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    if row.get("master"):
        out.append("master")
    if row.get("reference"):
        out.append("reference")
    if row.get("process_plan"):
        out.append("process_plan")
    if row.get("runtime"):
        out.append("runtime")
    if row.get("sindit"):
        out.append("sindit")
    return out


def _selected_profile(decision: Mapping[str, Any] | None, available_profiles: Sequence[str]) -> str:
    preferred = _text((decision or {}).get("selection_mode")) or "default"
    if preferred in available_profiles:
        return preferred
    return "default"


def _percent_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((float(numerator) / float(denominator)) * 100.0, 2)


def _profile_value(value: Any, source: str | None) -> Dict[str, Any]:
    return {
        "value": value,
        "source": source,
    }


def _first_profile_value(*values: Mapping[str, Any]) -> Dict[str, Any]:
    for value in values:
        if value.get("value") is not None:
            return dict(value)
    return _profile_value(None, None)


__all__ = [
    "build_tool_dataset_decision_snapshot",
    "build_tool_dataset_overview",
    "build_tool_audit_rows",
    "build_tool_audit_summary",
    "clear_runtime_observations",
    "collect_tool_dataset_overview_payload",
    "collect_tool_audit_payload",
    "filter_tool_audit_rows",
    "load_tool_dataset_decisions",
    "record_tool_anomaly",
    "record_tool_feedback",
    "record_tool_observation",
    "save_tool_dataset_decision",
]
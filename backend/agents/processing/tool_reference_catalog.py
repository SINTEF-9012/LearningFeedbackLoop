from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from .tool_lookup import FAMILY_MACHINE_A1, FAMILY_BUILDER_B12, FAMILY_PRESS_C_20_0482_010


TOOLS_DIR = Path(__file__).resolve().parents[3] / "data" / "tools"
SITE_B_CRITICAL_REFERENCE_PATH = TOOLS_DIR / "site_b" / "critical_tool_reference.json"
USE_CASE_OPERATION_SEQUENCE_PATH = TOOLS_DIR / "use_case_operation_sequences.json"

_USE_CASE_FAMILY_MAP = {
    1: FAMILY_BUILDER_B12,
    2: FAMILY_PRESS_C_20_0482_010,
    3: FAMILY_MACHINE_A1,
}


def load_site_b_critical_references(*, refresh: bool = False) -> dict[tuple[str, int], dict[str, Any]]:
    if refresh:
        _load_site_b_critical_references_cached.cache_clear()
    return deepcopy(_load_site_b_critical_references_cached())


def load_use_case_operation_index(*, refresh: bool = False) -> dict[tuple[str, int], dict[str, Any]]:
    if refresh:
        _load_use_case_operation_index_cached.cache_clear()
    return deepcopy(_load_use_case_operation_index_cached())


@lru_cache(maxsize=1)
def _load_site_b_critical_references_cached() -> dict[tuple[str, int], dict[str, Any]]:
    if not SITE_B_CRITICAL_REFERENCE_PATH.is_file():
        return {}
    raw = json.loads(SITE_B_CRITICAL_REFERENCE_PATH.read_text(encoding="utf-8"))
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for item in raw:
        family = str(item.get("machine_family") or FAMILY_BUILDER_B12).strip().lower()
        try:
            tool_number = int(item.get("tool_number"))
        except (TypeError, ValueError):
            continue
        out[(family, tool_number)] = dict(item)
    return out


@lru_cache(maxsize=1)
def _load_use_case_operation_index_cached() -> dict[tuple[str, int], dict[str, Any]]:
    if not USE_CASE_OPERATION_SEQUENCE_PATH.is_file():
        return {}
    raw = json.loads(USE_CASE_OPERATION_SEQUENCE_PATH.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for item in raw:
        try:
            use_case_id = int(item.get("use_case_id"))
        except (TypeError, ValueError):
            continue
        family = _USE_CASE_FAMILY_MAP.get(use_case_id)
        if family is None:
            continue
        for tool_number in item.get("tool_numbers") or []:
            try:
                normalized = int(tool_number)
            except (TypeError, ValueError):
                continue
            key = (family, normalized)
            payload = grouped.setdefault(
                key,
                {
                    "use_case_ids": [],
                    "use_case_titles": [],
                    "operation_ids": [],
                    "setups": [],
                    "entries": [],
                },
            )
            if use_case_id not in payload["use_case_ids"]:
                payload["use_case_ids"].append(use_case_id)
            title = str(item.get("use_case_title") or "").strip()
            if title and title not in payload["use_case_titles"]:
                payload["use_case_titles"].append(title)
            operation_id = str(item.get("operation_id") or "").strip()
            if operation_id and operation_id not in payload["operation_ids"]:
                payload["operation_ids"].append(operation_id)
            setup = str(item.get("setup") or "").strip()
            if setup and setup not in payload["setups"]:
                payload["setups"].append(setup)
            payload["entries"].append(
                {
                    "use_case_id": use_case_id,
                    "use_case_title": title or None,
                    "setup": setup or None,
                    "operation_id": operation_id or None,
                    "head": item.get("head"),
                    "op_type": item.get("op_type"),
                    "description": item.get("description"),
                    "tool_raw": item.get("tool_raw"),
                    "slide_number": item.get("slide_number"),
                    "slide_row_index": item.get("slide_row_index"),
                }
            )

    for payload in grouped.values():
        payload["use_case_ids"].sort()
        payload["use_case_titles"].sort()
        payload["operation_ids"].sort()
        payload["setups"].sort()
        payload["entries"].sort(key=lambda item: (item.get("use_case_id") or 0, item.get("slide_number") or 0, item.get("slide_row_index") or 0))
    return grouped


__all__ = [
    "load_site_b_critical_references",
    "load_use_case_operation_index",
]

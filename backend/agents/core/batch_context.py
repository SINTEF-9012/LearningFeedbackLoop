"""Shared batch context helpers for event and learning flows."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from pydantic import BaseModel


class BatchContext(BaseModel):
    """Minimal batch identity carried across events, memories, and learnings."""

    batch_id: str
    unit_index: Optional[int] = None
    unit_count: Optional[int] = None
    recipe_id: Optional[str] = None


_BATCH_FIELD_ALIASES = {
    "batch_id": ("batch_id", "batchId"),
    "unit_index": ("unit_index", "unitIndex"),
    "unit_count": ("unit_count", "unitCount"),
    "recipe_id": ("recipe_id", "recipeId"),
}


def _coerce_non_negative_int(value: Any, *, minimum: int = 0) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not numeric.is_integer():
            return None
        coerced = int(numeric)
    if coerced < minimum:
        return None
    return coerced


def _extract_candidate(mapping: Mapping[str, Any]) -> dict[str, Any]:
    candidate: dict[str, Any] = {}
    for field_name, aliases in _BATCH_FIELD_ALIASES.items():
        for alias in aliases:
            value = mapping.get(alias)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            candidate[field_name] = value
            break
    return candidate


def extract_batch_context(*sources: Any) -> Optional[BatchContext]:
    """Best-effort batch extraction from payload, metadata, or session context."""

    collected: dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        nested = source.get("batch")
        for candidate in (nested, source):
            if not isinstance(candidate, Mapping):
                continue
            for key, value in _extract_candidate(candidate).items():
                collected.setdefault(key, value)

    batch_id = str(collected.get("batch_id") or "").strip()
    if not batch_id:
        return None

    recipe_id_raw = collected.get("recipe_id")
    recipe_id = str(recipe_id_raw).strip() if isinstance(recipe_id_raw, str) else None
    if recipe_id == "":
        recipe_id = None

    return BatchContext(
        batch_id=batch_id,
        unit_index=_coerce_non_negative_int(collected.get("unit_index"), minimum=0),
        unit_count=_coerce_non_negative_int(collected.get("unit_count"), minimum=1),
        recipe_id=recipe_id,
    )
"""Helpers for resolving machine/document usecases.

The documentation graph and the memory/feedback graph share one Neo4j
database, but retrieval should stay partitioned by usecase. This module keeps
the normalization logic in one place so the docs backend, chat routes, and any
future graph queries use the same mapping.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

USECASE_GENERIC = "GENERIC"
USECASE_SITE_A = "SITE_A"
USECASE_SITE_B = "SITE_B"
USECASE_SITE_C = "SITE_C"

_USECASE_TOKENS = {
    USECASE_SITE_A: {
        "site_a",
        "site_a_casedata",
        "site_a_line2",
        "machine_a1",
        "a1001",
    },
    USECASE_SITE_B: {
        "site_b",
        "site_b_casedata",
        "site_b_olddata",
        "olddata",
        "builder_b1",
        "builder_b2",
        "machine_b1",
        "machine_b2",
        "b1001",
        "b1002",
    },
    USECASE_SITE_C: {
        "site_c",
        "site_c_casedata",
        "machine_c1",
        "press_c",
        "c1001",
    },
    USECASE_GENERIC: {
        "generic",
        "shared",
        "common",
        "cnc_operations_guide",
    },
}

_NESTED_KEYS = (
    "metadata",
    "cutting_context",
    "casedata",
    "extra",
)


def normalize_usecase(value: Any) -> Optional[str]:
    """Normalize a free-form identifier to a stable usecase code."""
    if value is None:
        return None
    text = _normalize_token(value)
    if not text:
        return None
    for usecase, aliases in _USECASE_TOKENS.items():
        if text == usecase.lower() or text in aliases:
            return usecase
    for usecase, aliases in _USECASE_TOKENS.items():
        if any(alias in text for alias in aliases):
            return usecase
    return None


def resolve_usecase(
    *,
    usecase: Any = None,
    dataset_id: Any = None,
    machine_id: Any = None,
    machine_uri: Any = None,
    machine: Any = None,
    source: Any = None,
    metadata: Optional[Mapping[str, Any]] = None,
    fallback_generic: bool = False,
) -> Optional[str]:
    """Resolve a usecase from explicit fields and nested metadata."""
    for candidate in _iter_candidates(
        usecase=usecase,
        dataset_id=dataset_id,
        machine_id=machine_id,
        machine_uri=machine_uri,
        machine=machine,
        source=source,
        metadata=metadata,
    ):
        resolved = normalize_usecase(candidate)
        if resolved is not None:
            return resolved
    return USECASE_GENERIC if fallback_generic else None


def usecase_aliases(usecase: Optional[str]) -> list[str]:
    """Return lowercase aliases that should stay inside one usecase."""
    resolved = normalize_usecase(usecase)
    if resolved is None:
        return []
    return sorted(_USECASE_TOKENS.get(resolved, set()))


def _iter_candidates(
    *,
    usecase: Any,
    dataset_id: Any,
    machine_id: Any,
    machine_uri: Any,
    machine: Any,
    source: Any,
    metadata: Optional[Mapping[str, Any]],
) -> Iterable[Any]:
    yield usecase
    yield dataset_id
    yield machine_id
    yield machine_uri
    yield machine
    yield source

    if not isinstance(metadata, Mapping):
        return

    direct_keys = (
        "usecase",
        "dataset_id",
        "source_dataset_id",
        "machine_id",
        "machine",
        "machine_uri",
        "machine_iri",
        "sindit_asset_iri",
        "asset_iri",
        "machine_type",
        "source",
        "file_name",
        "file",
    )
    for key in direct_keys:
        if key in metadata:
            yield metadata.get(key)

    for nested_key in _NESTED_KEYS:
        nested = metadata.get(nested_key)
        if isinstance(nested, Mapping):
            yield from _iter_candidates(
                usecase=nested.get("usecase"),
                dataset_id=nested.get("dataset_id") or nested.get("source_dataset_id"),
                machine_id=nested.get("machine_id") or nested.get("machine"),
                machine_uri=nested.get("machine_uri") or nested.get("machine_iri") or nested.get("asset_iri"),
                machine=nested.get("machine") or nested.get("machine_type"),
                source=nested.get("source"),
                metadata=None,
            )


def _normalize_token(value: Any) -> str:
    text = str(value).strip().lower()
    if not text:
        return ""
    cleaned_chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", ":", "/", "."}:
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    return " ".join("".join(cleaned_chars).split())

"""Unified entity schema — Agent K data-density contract (2026-04-24).

Plan §11: every backend entity (Memory, Feedback, Pattern, Model,
Session, Experiment, SINDIT asset) exposes a normalized JSON shape so
a generic ``<EntityView schema=... />`` component can render any of
them without code changes. Adding a new field anywhere upstream
surfaces in the UI automatically via the detail drawer.

The contract is four buckets:

- ``fields`` — free-form scalar/string metadata the user might want to
  see on a detail card. Flat key/value.
- ``tags`` — short, filterable string labels (pattern names, labels,
  regimes). Flat list of strings.
- ``metrics`` — numeric measurements the UI can chart or show as
  badges. Flat key → float.
- ``relationships`` — pointers to other entities (``{kind, id,
  role}``). Enables the unified knowledge-graph view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


ENTITY_KINDS = frozenset(
    {
        "memory",
        "feedback",
        "pattern",
        "model",
        "session",
        "experiment",
        "sindit_asset",
        "knowledge_pack",
    }
)


@dataclass
class EntityRelationship:
    kind: str
    id: str
    role: str

    def to_dict(self) -> Dict[str, str]:
        return {"kind": self.kind, "id": self.id, "role": self.role}


@dataclass
class EntitySchema:
    """Normalised view of a backend entity.

    All four buckets default to empty so adapters only fill in what
    they have. ``to_dict`` is what the API returns.
    """

    kind: str
    id: str
    label: Optional[str] = None
    fields: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    relationships: List[EntityRelationship] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in ENTITY_KINDS:
            raise ValueError(
                f"Unknown entity kind={self.kind!r}; allowed={sorted(ENTITY_KINDS)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["relationships"] = [r.to_dict() if isinstance(r, EntityRelationship) else r for r in self.relationships]
        return d

    def add_relationship(self, kind: str, id: str, role: str) -> "EntitySchema":
        self.relationships.append(EntityRelationship(kind=kind, id=id, role=role))
        return self


# ── Adapters ────────────────────────────────────────────────────────────
#
# Each adapter takes a source dict (the raw entity as stored in Neo4j /
# on disk / in the orchestrator) and returns an :class:`EntitySchema`.
# Adapters are pure: no I/O, no side effects. Missing keys degrade to
# empty buckets rather than raising.


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metrics(source: Mapping[str, Any], candidates: Iterable[str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key in candidates:
        v = _coerce_float(source.get(key))
        if v is not None:
            metrics[key] = v
    return metrics


def memory_to_schema(memory: Mapping[str, Any]) -> EntitySchema:
    mem_id = str(memory.get("id") or memory.get("memory_id") or "unknown")
    session_id = memory.get("session_id")
    patterns = list(memory.get("patterns") or [])
    tags = list(memory.get("tags") or [])
    schema = EntitySchema(
        kind="memory",
        id=mem_id,
        label=memory.get("label") or (patterns[0] if patterns else None),
        fields={
            "created_at": memory.get("created_at"),
            "annotation": memory.get("annotation"),
            "regime": memory.get("regime"),
        },
        tags=sorted(set(tags) | set(patterns)),
        metrics=_extract_metrics(
            memory,
            ["significance_score", "confidence", "severity", "novelty"],
        ),
    )
    if session_id:
        schema.add_relationship("session", str(session_id), role="source")
    for pattern_key in patterns:
        schema.add_relationship("pattern", str(pattern_key), role="observed")
    # Drop None field values for cleanliness.
    schema.fields = {k: v for k, v in schema.fields.items() if v is not None}
    return schema


def feedback_to_schema(feedback: Mapping[str, Any]) -> EntitySchema:
    fb_id = str(feedback.get("id") or feedback.get("feedback_id") or "unknown")
    action = feedback.get("action")
    comment = feedback.get("comment")
    schema = EntitySchema(
        kind="feedback",
        id=fb_id,
        label=str(action) if action else "feedback",
        fields={
            "action": action,
            "comment": comment,
            "created_at": feedback.get("created_at"),
            "operator_id": feedback.get("operator_id"),
        },
        tags=[str(action)] if action else [],
        metrics=_extract_metrics(feedback, ["confidence", "latency_ms"]),
    )
    mem_id = feedback.get("memory_id")
    if mem_id:
        schema.add_relationship("memory", str(mem_id), role="target")
    schema.fields = {k: v for k, v in schema.fields.items() if v is not None}
    return schema


def pattern_to_schema(pattern: Mapping[str, Any]) -> EntitySchema:
    key = str(pattern.get("key") or pattern.get("id") or "unknown")
    prior = _coerce_float(pattern.get("prior"))
    schema = EntitySchema(
        kind="pattern",
        id=key,
        label=pattern.get("label") or key,
        fields={
            "description": pattern.get("description"),
            "first_seen": pattern.get("first_seen"),
            "last_seen": pattern.get("last_seen"),
        },
        tags=list(pattern.get("tags") or []),
        metrics=_extract_metrics(
            pattern,
            ["prior", "support", "precision", "recall", "f1", "occurrences"],
        ),
    )
    if prior is not None and "prior" not in schema.metrics:
        schema.metrics["prior"] = prior
    schema.fields = {k: v for k, v in schema.fields.items() if v is not None}
    return schema


def model_to_schema(model: Mapping[str, Any]) -> EntitySchema:
    mid = str(model.get("id") or model.get("model_id") or model.get("name") or "unknown")
    schema = EntitySchema(
        kind="model",
        id=mid,
        label=model.get("name") or mid,
        fields={
            "trained_at": model.get("trained_at"),
            "dataset": model.get("dataset"),
            "version": model.get("version"),
            "type": model.get("type"),
        },
        tags=list(model.get("tags") or []),
        metrics=_extract_metrics(
            model,
            ["accuracy", "f1", "precision", "recall", "n_samples", "loss"],
        ),
    )
    schema.fields = {k: v for k, v in schema.fields.items() if v is not None}
    return schema


def session_to_schema(session: Mapping[str, Any]) -> EntitySchema:
    sid = str(session.get("id") or session.get("session_id") or "unknown")
    schema = EntitySchema(
        kind="session",
        id=sid,
        label=session.get("name") or sid,
        fields={
            "started_at": session.get("started_at"),
            "ended_at": session.get("ended_at"),
            "dataset": session.get("dataset"),
            "operator": session.get("operator"),
        },
        tags=list(session.get("tags") or []),
        metrics=_extract_metrics(
            session,
            ["duration_s", "n_alerts", "n_memories", "n_samples"],
        ),
    )
    schema.fields = {k: v for k, v in schema.fields.items() if v is not None}
    return schema


def experiment_to_schema(experiment: Mapping[str, Any]) -> EntitySchema:
    eid = str(experiment.get("id") or experiment.get("experiment_id") or "unknown")
    schema = EntitySchema(
        kind="experiment",
        id=eid,
        label=experiment.get("name") or eid,
        fields={
            "status": experiment.get("status"),
            "started_at": experiment.get("started_at"),
            "ended_at": experiment.get("ended_at"),
            "active": experiment.get("active"),
        },
        tags=list(experiment.get("tags") or []),
        metrics=_extract_metrics(
            experiment,
            ["duration_s", "n_phases", "n_operations"],
        ),
    )
    schema.fields = {k: v for k, v in schema.fields.items() if v is not None}
    return schema


def sindit_asset_to_schema(asset: Mapping[str, Any]) -> EntitySchema:
    uri = str(asset.get("uri") or asset.get("id") or "unknown")
    kind_tag = asset.get("lflAssetKind") or asset.get("assetKind")
    metadata = asset.get("metadata") or {}
    schema = EntitySchema(
        kind="sindit_asset",
        id=uri,
        label=asset.get("label") or uri,
        fields={
            "description": asset.get("assetDescription"),
            "assetType": asset.get("assetType"),
            "lflAssetKind": kind_tag,
            **{k: v for k, v in metadata.items() if not isinstance(v, (dict, list))},
        },
        tags=[kind_tag] if kind_tag else [],
        metrics=_extract_metrics(metadata, list(metadata.keys())),
    )
    schema.fields = {k: v for k, v in schema.fields.items() if v is not None}
    return schema


def knowledge_pack_to_schema(pack: Mapping[str, Any]) -> EntitySchema:
    site = str(pack.get("site") or "unknown")
    summary = pack.get("summary") if isinstance(pack.get("summary"), Mapping) else {}
    ctx = pack.get("context") or {}
    schema = EntitySchema(
        kind="knowledge_pack",
        id=site,
        label=f"pack/{site}",
        fields={
            "version": pack.get("version"),
            "built_at": pack.get("built_at"),
            **{f"context.{k}": v for k, v in ctx.items() if v},
        },
        tags=[v for v in ctx.values() if isinstance(v, str) and v],
        metrics={k: float(v) for k, v in summary.items() if isinstance(v, (int, float))},
    )
    schema.fields = {k: v for k, v in schema.fields.items() if v is not None}
    return schema


ADAPTERS = {
    "memory": memory_to_schema,
    "feedback": feedback_to_schema,
    "pattern": pattern_to_schema,
    "model": model_to_schema,
    "session": session_to_schema,
    "experiment": experiment_to_schema,
    "sindit_asset": sindit_asset_to_schema,
    "knowledge_pack": knowledge_pack_to_schema,
}


def to_entity_schema(kind: str, source: Mapping[str, Any]) -> EntitySchema:
    """Dispatch to the correct adapter."""
    if kind not in ADAPTERS:
        raise ValueError(f"Unknown entity kind={kind!r}; allowed={sorted(ADAPTERS)}")
    return ADAPTERS[kind](source)

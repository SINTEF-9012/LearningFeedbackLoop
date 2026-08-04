"""Tests for backend.agents.schema (data-density contract) — Agent K."""

from __future__ import annotations

import pytest

from backend.agents.schema import (
    EntitySchema,
    feedback_to_schema,
    knowledge_pack_to_schema,
    memory_to_schema,
    model_to_schema,
    pattern_to_schema,
    session_to_schema,
    sindit_asset_to_schema,
    to_entity_schema,
    experiment_to_schema,
)


# ── EntitySchema core ──────────────────────────────────────────────────


def test_entity_schema_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        EntitySchema(kind="nope", id="x")


def test_entity_schema_to_dict_has_four_buckets() -> None:
    s = EntitySchema(kind="memory", id="m1")
    d = s.to_dict()
    assert set(d.keys()) >= {"kind", "id", "fields", "tags", "metrics", "relationships"}
    assert d["fields"] == {}
    assert d["tags"] == []
    assert d["metrics"] == {}
    assert d["relationships"] == []


def test_add_relationship_chains_and_serialises() -> None:
    s = EntitySchema(kind="memory", id="m1").add_relationship("pattern", "chatter", "observed")
    d = s.to_dict()
    assert d["relationships"] == [{"kind": "pattern", "id": "chatter", "role": "observed"}]


# ── Adapters ────────────────────────────────────────────────────────────


def test_memory_adapter_extracts_tags_and_relationships() -> None:
    mem = {
        "id": "mem-1",
        "session_id": "sess-7",
        "patterns": ["chatter:high_ratio", "TOOL_BREAKAGE"],
        "tags": ["roughing"],
        "significance_score": 0.82,
        "annotation": "operator noticed vibration",
        "regime": "roughing",
        "created_at": "2026-04-24T10:00:00",
    }
    schema = memory_to_schema(mem)
    assert schema.kind == "memory"
    assert schema.id == "mem-1"
    # Tags union of patterns + tags, deduped.
    assert "chatter:high_ratio" in schema.tags
    assert "roughing" in schema.tags
    # Relationships include session + each pattern.
    rel_kinds = {r.kind for r in schema.relationships}
    assert rel_kinds == {"session", "pattern"}
    assert schema.metrics["significance_score"] == pytest.approx(0.82)
    # None values stripped.
    assert "regime" in schema.fields and schema.fields["regime"] == "roughing"


def test_memory_adapter_handles_missing_fields() -> None:
    schema = memory_to_schema({"id": "m"})
    assert schema.id == "m"
    assert schema.tags == []
    assert schema.relationships == []


def test_feedback_adapter_links_memory() -> None:
    fb = {
        "id": "fb-1",
        "memory_id": "mem-1",
        "action": "CONFIRM",
        "comment": "yes, real breakage",
        "operator_id": "op-42",
    }
    schema = feedback_to_schema(fb)
    assert schema.kind == "feedback"
    assert schema.fields["operator_id"] == "op-42"
    assert schema.relationships[0].kind == "memory"
    assert schema.relationships[0].id == "mem-1"
    assert "CONFIRM" in schema.tags


def test_pattern_adapter_promotes_prior() -> None:
    schema = pattern_to_schema({"key": "chatter", "prior": 0.73, "support": 42})
    assert schema.metrics["prior"] == pytest.approx(0.73)
    assert schema.metrics["support"] == pytest.approx(42.0)


def test_model_adapter_keeps_numeric_metrics() -> None:
    schema = model_to_schema(
        {"name": "seed_cnn", "accuracy": 0.9, "f1": 0.85, "trained_at": "2026-04-01"}
    )
    assert schema.metrics["accuracy"] == pytest.approx(0.9)
    assert schema.fields["trained_at"] == "2026-04-01"


def test_session_adapter_basic() -> None:
    schema = session_to_schema(
        {"id": "sess-1", "started_at": "t0", "ended_at": "t1", "duration_s": 123.4}
    )
    assert schema.kind == "session"
    assert schema.metrics["duration_s"] == pytest.approx(123.4)


def test_experiment_adapter_includes_active() -> None:
    schema = experiment_to_schema(
        {"id": "expt-1", "active": True, "status": "running", "n_phases": 3}
    )
    assert schema.fields["active"] is True
    assert schema.metrics["n_phases"] == pytest.approx(3.0)


def test_sindit_asset_adapter_flattens_metadata() -> None:
    asset = {
        "uri": "urn:lfl:asset:cnc-1",
        "label": "CNC-1",
        "assetType": "urn:samm:sindit.sintef.no:1.0.0#AbstractAsset",
        "lflAssetKind": "Machine",
        "assetDescription": "Primary CNC",
        "metadata": {"max_rpm": 15000, "site": "lab-a", "nested": {"x": 1}},
    }
    schema = sindit_asset_to_schema(asset)
    assert schema.kind == "sindit_asset"
    assert "Machine" in schema.tags
    assert schema.metrics["max_rpm"] == pytest.approx(15000.0)
    # Nested dicts excluded from flat fields.
    assert "nested" not in schema.fields
    assert schema.fields["site"] == "lab-a"


def test_knowledge_pack_adapter() -> None:
    pack = {
        "site": "CNC-1",
        "version": "1.0.0",
        "built_at": "2026-04-24T00:00:00",
        "context": {"machine_type": "cnc", "tool_type": "endmill"},
        "summary": {"priors": 10, "discovered_patterns": 3},
    }
    schema = knowledge_pack_to_schema(pack)
    assert schema.kind == "knowledge_pack"
    assert schema.id == "CNC-1"
    assert schema.metrics["priors"] == pytest.approx(10.0)
    assert schema.fields["context.machine_type"] == "cnc"


# ── Dispatcher ────────────────────────────────────────────────────────


def test_to_entity_schema_dispatches() -> None:
    schema = to_entity_schema("memory", {"id": "m"})
    assert schema.kind == "memory"


def test_to_entity_schema_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        to_entity_schema("galaxy", {"id": "x"})


def test_coerce_ignores_bool_as_metric() -> None:
    # Booleans look float-ish but shouldn't pollute metrics.
    schema = pattern_to_schema({"key": "p", "prior": True})
    assert "prior" not in schema.metrics

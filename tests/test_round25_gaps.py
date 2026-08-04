"""Tests for Round 25 gap fixes:

1. ``SENSOR_PROPERTY_CATALOG`` / ``get_sensor_property_meta`` in
   ``backend.agents.sindit.asset_catalog`` — verifies the hardcoded
   ``_SENSOR_FIELDS`` map in ``bridge.py`` is now sourced from the
   canonical catalog (plan point 6, Agent F).

2. ``Memory.machine_uri`` persistence — verifies the orchestrator's
   ``_create_memory`` populates the cross-reference to SINDIT
   (plan point 2, Agent B).
"""
from __future__ import annotations

from pathlib import Path

from backend.agents.sindit.asset_catalog import (
    SENSOR_PROPERTY_CATALOG,
    get_sensor_property_meta,
)


def _bare_orchestrator():
    from backend.agents.memory.orchestrator import MemoryEventOrchestrator

    orchestrator = object.__new__(MemoryEventOrchestrator)
    orchestrator._convert_metrics = lambda metrics: None
    return orchestrator


def test_sensor_catalog_has_core_fields():
    for field in ("spindle_speed", "feed_rate", "vibration_x", "temperature"):
        assert field in SENSOR_PROPERTY_CATALOG
        entry = SENSOR_PROPERTY_CATALOG[field]
        assert "label" in entry and "unit" in entry


def test_get_sensor_property_meta_known():
    meta = get_sensor_property_meta("spindle_speed")
    assert meta == {"label": "Spindle Speed", "unit": "rpm"}


def test_get_sensor_property_meta_returns_copy():
    # Mutating the result must not affect the catalog.
    meta = get_sensor_property_meta("spindle_speed")
    meta["unit"] = "XXX"
    assert SENSOR_PROPERTY_CATALOG["spindle_speed"]["unit"] == "rpm"


def test_get_sensor_property_meta_unknown():
    meta = get_sensor_property_meta("some_novel_field")
    assert meta["label"] == "Some Novel Field"
    assert meta["unit"] == ""


def test_bridge_sensor_fields_alias_points_to_catalog():
    # Backward-compat alias: ``_SENSOR_FIELDS`` imported by bridge.py
    # must be the same object as ``SENSOR_PROPERTY_CATALOG``.
    from backend.agents.sindit import live_data_bridge as bridge_mod

    assert bridge_mod._SENSOR_FIELDS is SENSOR_PROPERTY_CATALOG


def test_memory_schema_has_machine_uri_field():
    from backend.agents.core.schemas import Memory

    # Field should be declared with Optional[str] default None.
    fields = Memory.model_fields
    assert "machine_uri" in fields
    mem = Memory(session_id="s1", time_range=(0.0, 1.0))
    assert mem.machine_uri is None


def test_orchestrator_create_memory_derives_machine_uri_from_context():
    from backend.agents.memory.orchestrator import (
        MemoryEvent,
    )
    from backend.agents.memory.scorer import SignificanceResult, SignificanceAction
    from backend.agents.core.context import CuttingContext

    orch = _bare_orchestrator()

    ctx = CuttingContext(machine_id="CNC 7")
    event = MemoryEvent(
        session_id="sess-1",
        time_range=(0.0, 1.0),
        cutting_context=ctx,
    )
    sig = SignificanceResult(
        is_significant=True,
        score=0.9,
        action=SignificanceAction.STORE,
        reasons=["test"],
        triggered_rules=[],
    )

    mem = orch._create_memory(event, sig)
    assert mem.machine_uri == "urn:lfl:asset:cnc-7"


def test_orchestrator_create_memory_respects_explicit_uri():
    from backend.agents.memory.orchestrator import (
        MemoryEvent,
    )
    from backend.agents.memory.scorer import SignificanceResult, SignificanceAction

    orch = _bare_orchestrator()

    event = MemoryEvent(
        session_id="sess-2",
        time_range=(0.0, 1.0),
        metadata={"machine_uri": "urn:lfl:asset:pump-17"},
    )
    sig = SignificanceResult(
        is_significant=True,
        score=0.5,
        action=SignificanceAction.STORE,
        reasons=["ok"],
        triggered_rules=[],
    )

    mem = orch._create_memory(event, sig)
    assert mem.machine_uri == "urn:lfl:asset:pump-17"


def test_orchestrator_create_memory_defaults_machine_uri():
    from backend.agents.memory.orchestrator import (
        MemoryEvent,
    )
    from backend.agents.memory.scorer import SignificanceResult, SignificanceAction

    orch = _bare_orchestrator()

    event = MemoryEvent(session_id="sess-3", time_range=(0.0, 1.0))
    sig = SignificanceResult(
        is_significant=True,
        score=0.3,
        action=SignificanceAction.STORE,
        reasons=[],
        triggered_rules=[],
    )

    mem = orch._create_memory(event, sig)
    # Falls back to the default single-machine URN.
    assert mem.machine_uri == "urn:lfl:asset:cnc-machine-1"


def test_orchestrator_create_memory_preserves_curated_event_metadata():
    from backend.agents.memory.orchestrator import (
        MemoryEvent,
    )
    from backend.agents.memory.scorer import SignificanceResult, SignificanceAction
    from backend.agents.core.context import CuttingContext

    orch = _bare_orchestrator()

    event = MemoryEvent(
        session_id="sess-meta",
        time_range=(0.0, 1.0),
        cutting_context=CuttingContext(
            machine_id="MACHINE_A1",
            tool_id="tool-7",
            tool_type="end_mill",
            extra={"sindit_tool_iri": "urn:lfl:tool:machine_a1-t7"},
        ),
        metadata={
            "source": "SITE_A",
            "machine_family": "machine_a1",
            "dataset_id": "site_a_casedata",
            "source_dataset_id": "site_a_line2",
            "machine_uri": "urn:lfl:asset:machine_a1",
            "machine_iri": "urn:test:machine-iri",
            "sindit_asset_iri": "urn:test:asset-iri",
            "casedata": {
                "operation_id": "OF00011",
                "tool_id": "tool-7",
                "root": Path("/tmp/casedata"),
                "case_dir": "Site_a - MACHINE_A1",
                "dataset_id": "site_a_casedata",
            },
        },
    )
    sig = SignificanceResult(
        is_significant=True,
        score=0.7,
        action=SignificanceAction.STORE,
        reasons=["preserve metadata"],
        triggered_rules=[],
    )

    mem = orch._create_memory(event, sig)

    assert mem.metadata["source"] == "SITE_A"
    assert mem.metadata["machine_family"] == "machine_a1"
    assert mem.metadata["dataset_id"] == "site_a_casedata"
    assert mem.metadata["source_dataset_id"] == "site_a_line2"
    assert mem.metadata["machine_iri"] == "urn:test:machine-iri"
    assert mem.metadata["sindit_asset_iri"] == "urn:test:asset-iri"
    assert mem.metadata["casedata"] == {
        "operation_id": "OF00011",
        "tool_id": "tool-7",
        "root": "/tmp/casedata",
        "case_dir": "Site_a - MACHINE_A1",
        "dataset_id": "site_a_casedata",
    }

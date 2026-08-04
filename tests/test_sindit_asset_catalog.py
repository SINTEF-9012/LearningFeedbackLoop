"""Tests for the SINDIT asset-catalog builders (Agent F, 2026-04-24)."""

from __future__ import annotations

import pytest

from backend.agents.sindit.asset_catalog import (
    LFL_ASSET_KIND_KEY,
    LFL_RELATION_TYPES,
    SAMM_ASSET_TYPE,
    SinditCatalog,
    build_controller_program_asset,
    build_default_cnc_kit,
    build_fixture_asset,
    build_machine_asset,
    build_model_metadata_asset,
    build_property,
    build_relationship,
    build_spindle_asset,
    build_tool_asset,
    build_workpiece_asset,
    sync_catalog,
)


# ── URN shape + slug ──────────────────────────────────────────────────


def test_machine_asset_slug_and_fields():
    a = build_machine_asset("CNC Machine 1", make="DMG MORI", model="DMU 50", max_rpm=15000)
    assert a["uri"] == "urn:lfl:asset:cnc-machine-1"
    assert a["assetType"] == SAMM_ASSET_TYPE
    assert a[LFL_ASSET_KIND_KEY] == "Machine"
    assert a["label"] == "CNC Machine 1"
    assert a["metadata"]["make"] == "DMG MORI"
    assert a["metadata"]["maxRpm"] == 15000.0


def test_slug_handles_empty_and_weird_input():
    a = build_tool_asset("   ")
    assert a["uri"].endswith("unknown")
    b = build_tool_asset("Tool/42")
    assert b["uri"] == "urn:lfl:tool:tool-42"


# ── Per-asset builders ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "builder, kw, kind, uri_prefix",
    [
        (build_tool_asset, dict(tool_id="T1", diameter_mm=10, teeth=4), "Tool", "urn:lfl:tool:t1"),
        (build_workpiece_asset, dict(workpiece_id="WP-A", material="Al 7075"), "Workpiece", "urn:lfl:workpiece:wp-a"),
        (build_fixture_asset, dict(fixture_id="FX1", fixture_type="vise"), "Fixture", "urn:lfl:fixture:fx1"),
        (build_spindle_asset, dict(spindle_id="SP1", max_rpm=12000, power_kw=15), "Spindle", "urn:lfl:spindle:sp1"),
        (build_controller_program_asset, dict(program_id="PG_1", controller_version="MELD-V5"), "ControllerProgram", "urn:lfl:program:pg_1"),
    ],
)
def test_asset_builders_shape(builder, kw, kind, uri_prefix):
    a = builder(**kw)
    assert a["uri"] == uri_prefix
    assert a["assetType"] == SAMM_ASSET_TYPE
    assert a[LFL_ASSET_KIND_KEY] == kind
    assert isinstance(a.get("metadata"), dict)


def test_model_metadata_asset_has_trained_at_default():
    a = build_model_metadata_asset("seed_model", n_samples=120, current_f1=0.81)
    assert a["uri"] == "urn:lfl:model:seed_model"
    assert a[LFL_ASSET_KIND_KEY] == "ModelMetadata"
    assert "trainedAt" in a["metadata"]
    assert a["metadata"]["nSamples"] == 120
    assert a["metadata"]["currentF1"] == 0.81


# ── Property + relationship builders ──────────────────────────────────


def test_build_property_uri_stable_from_parent():
    p1 = build_property("urn:lfl:asset:cnc-1", "spindle_speed", value=3200, unit="rpm")
    p2 = build_property("urn:lfl:asset:cnc-1", "spindle_speed", value=0, unit="rpm")
    assert p1["uri"] == p2["uri"]
    assert p1["propertyName"] == "spindle_speed"
    assert p1["propertyUnit"] == "rpm"
    assert p1["assetUri"] == "urn:lfl:asset:cnc-1"
    assert p1["propertyValue"] == "3200"


def test_build_relationship_validates_type():
    rel = build_relationship("urn:lfl:asset:cnc-1", "urn:lfl:tool:t1", "HAS_TOOL")
    assert rel["relationshipType"] == "HAS_TOOL"
    assert rel["sourceUri"].endswith("cnc-1")

    with pytest.raises(ValueError):
        build_relationship("a", "b", "SOMETHING_ELSE")


def test_all_allowed_relationship_types_work():
    for t in LFL_RELATION_TYPES:
        build_relationship("a", "b", t)


# ── Catalog assembly ──────────────────────────────────────────────────


def test_default_cnc_kit_composition():
    kit = build_default_cnc_kit(
        "CNC-1",
        tool_id="T1",
        workpiece_id="WP1",
        fixture_id="FX1",
        spindle_id="SP1",
        program_id="PG1",
    )
    s = kit.summary()
    assert s == {"assets": 6, "properties": 0, "relationships": 5}
    rel_types = {r["relationshipType"] for r in kit.relationships}
    assert rel_types == {"HAS_TOOL", "HAS_WORKPIECE", "HAS_FIXTURE", "HAS_SPINDLE", "RUNS_PROGRAM"}
    # All relationship sources point at the machine IRI.
    machine_uri = next(a["uri"] for a in kit.assets if a[LFL_ASSET_KIND_KEY] == "Machine")
    assert all(r["sourceUri"] == machine_uri for r in kit.relationships)


def test_catalog_extend_and_uris():
    a = SinditCatalog()
    a.add_asset(build_tool_asset("T1"))
    b = SinditCatalog()
    b.add_asset(build_workpiece_asset("WP1"))
    b.add_property(build_property("urn:lfl:tool:t1", "diameter", value=10, unit="mm"))
    a.extend(b)
    assert a.summary() == {"assets": 2, "properties": 1, "relationships": 0}
    assert "urn:lfl:tool:t1" in a.uris()
    assert "urn:lfl:workpiece:wp1" in a.uris()


# ── sync_catalog (async, with fake client) ────────────────────────────


class _FakeClient:
    def __init__(self, fail_asset: bool = False, fail_property: bool = False):
        self.asset_calls: list = []
        self.property_calls: list = []
        self.relationship_calls: list = []
        self.fail_asset = fail_asset
        self.fail_property = fail_property

    async def post_asset(self, payload):
        self.asset_calls.append(payload)
        return None if self.fail_asset else {"ok": True}

    async def post_property(self, payload):
        self.property_calls.append(payload)
        return None if self.fail_property else {"ok": True}

    async def post_relationship(self, payload):
        self.relationship_calls.append(payload)
        return {"ok": True}


@pytest.mark.asyncio
async def test_sync_catalog_calls_client_per_kind():
    kit = build_default_cnc_kit("CNC-1", tool_id="T1", workpiece_id="WP1", fixture_id="FX1")
    kit.add_property(build_property(kit.assets[0]["uri"], "spindle_speed", value=3000, unit="rpm"))
    client = _FakeClient()
    result = await sync_catalog(client, kit)
    assert result["assets_ok"] == 4
    assert result["properties_ok"] == 1
    assert result["relationships_ok"] == 3
    assert result["assets_fail"] == 0
    assert len(client.asset_calls) == 4
    assert len(client.relationship_calls) == 3


@pytest.mark.asyncio
async def test_sync_catalog_records_failures():
    kit = SinditCatalog()
    kit.add_asset(build_tool_asset("T1"))
    kit.add_property(build_property("urn:lfl:tool:t1", "p", value=1))
    client = _FakeClient(fail_asset=True, fail_property=True)
    result = await sync_catalog(client, kit)
    assert result["assets_fail"] == 1 and result["assets_ok"] == 0
    assert result["properties_fail"] == 1 and result["properties_ok"] == 0

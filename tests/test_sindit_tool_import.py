from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.processing.tool_lookup import (
    FAMILY_MACHINE_A1,
    FAMILY_BUILDER_B12,
    MACHINE_FAMILIES_PATH,
    ToolSpec,
    load_machine_family_registry,
    resolve_machine_family,
)
from backend.agents.sindit.import_tool_master import build_tool_master_catalog


def test_machine_family_registry_loads_default_yaml():
    registry = load_machine_family_registry(MACHINE_FAMILIES_PATH, refresh=True)
    assert registry[FAMILY_BUILDER_B12] == [
        "Site_b - MACHINE_B1 - CASE_B1",
        "Site_b - MACHINE_B2 - CASE_B2",
        "olddata",
    ]
    assert registry[FAMILY_MACHINE_A1] == ["Site_a - MACHINE_A1 - CASE_A1"]


def test_resolve_machine_family_uses_registry_and_slug_fallback():
    assert resolve_machine_family("Site_b - MACHINE_B1 - CASE_B1") == FAMILY_BUILDER_B12
    assert resolve_machine_family("Site_a - MACHINE_A1 - CASE_A1") == FAMILY_MACHINE_A1
    assert resolve_machine_family("Unknown Machine / Alpha") == "unknown-machine---alpha"


def test_build_tool_master_catalog_uses_labeled_properties_and_machine_assets():
    master = {
        (FAMILY_BUILDER_B12, 6): ToolSpec(
            machine_family=FAMILY_BUILDER_B12,
            tool_number=6,
            tool_id="T06",
            description="FINISH BORE 65MM DIA",
            tool_type="bore",
            diameter_mm=65.0,
            teeth=1,
            tool_length_mm=128.125,
            tool_substrate="carbide",
            source="site_b/Builder_b1 2 Tooling Database.xlsx",
        )
    }
    family_map = {
        FAMILY_BUILDER_B12: [
            "Site_b - MACHINE_B1 - CASE_B1",
            "Site_b - MACHINE_B2 - CASE_B2",
            "olddata",
        ]
    }

    catalog = build_tool_master_catalog(
        master=master,
        family_to_machine_ids=family_map,
        imported_at="2026-05-13T00:00:00+00:00",
    )

    assert len(catalog.assets) == 5
    assert len(catalog.relationships) == 5

    tool_rel_sources = {
        rel["sourceUri"]
        for rel in catalog.relationships
        if rel["relationshipType"] == "HAS_TOOL"
    }
    assert "urn:lfl:asset:olddata" in tool_rel_sources
    assert "urn:lfl:asset:site_b---machine_b1---case_b1" in tool_rel_sources
    assert "urn:lfl:asset:site_b---machine_b2---case_b2" in tool_rel_sources

    workpiece_assets = [asset for asset in catalog.assets if asset["uri"] == "urn:lfl:workpiece:site_b-casedata-shared-workpiece"]
    assert len(workpiece_assets) == 1

    workpiece_rels = [rel for rel in catalog.relationships if rel["relationshipType"] == "HAS_WORKPIECE"]
    assert {rel["sourceUri"] for rel in workpiece_rels} == {
        "urn:lfl:asset:site_b---machine_b1---case_b1",
        "urn:lfl:asset:site_b---machine_b2---case_b2",
    }
    assert {rel["targetUri"] for rel in workpiece_rels} == {"urn:lfl:workpiece:site_b-casedata-shared-workpiece"}

    properties_by_name = {(prop["assetUri"], prop["propertyName"]): prop for prop in catalog.properties}
    assert properties_by_name[("urn:lfl:tool:builder_b12-t6", "ToolDiameter")]["label"] == "ToolDiameter"
    assert properties_by_name[("urn:lfl:tool:builder_b12-t6", "ToolLength")]["label"] == "ToolLength"
    assert properties_by_name[("urn:lfl:tool:builder_b12-t6", "ToolMaterial")]["label"] == "ToolMaterial"
    assert properties_by_name[("urn:lfl:tool:builder_b12-t6", "LastImportedAt")]["propertyValue"] == "2026-05-13T00:00:00+00:00"
    assert properties_by_name[("urn:lfl:tool:builder_b12-t6", "SourceWorkbook")]["propertyValue"] == "site_b/Builder_b1 2 Tooling Database.xlsx"
    assert properties_by_name[("urn:lfl:workpiece:site_b-casedata-shared-workpiece", "DatasetId")]["propertyValue"] == "site_b_casedata"
    assert properties_by_name[("urn:lfl:workpiece:site_b-casedata-shared-workpiece", "SharedAcrossMachines")]["propertyValue"] == "True"
    assert not any(prop_name == "Material" for (_asset_uri, prop_name) in properties_by_name)
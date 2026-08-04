from __future__ import annotations

import pytest

from backend.agents.processing import tool_dataset_decisions as tool_dataset_decisions_module
from backend.agents.processing.tool_lookup import (
    FAMILY_MACHINE_A1,
    FAMILY_BUILDER_B12,
    FAMILY_PRESS_C_20_0482_010,
    load_tool_master,
    lookup,
    resolve_tool_context,
)


@pytest.fixture(scope="module")
def master():
    return load_tool_master(refresh=True)


def test_machine_a1_encoded_geometry_is_parsed(master):
    spec = master[(FAMILY_MACHINE_A1, 2)]
    assert spec.tool_id == "PART0003"
    assert spec.tool_type == "mill"
    assert spec.diameter_mm == pytest.approx(125.0)
    assert spec.tool_length_mm == pytest.approx(113.0)
    assert spec.teeth is None


def test_machine_a1_ardatza_is_classified_as_tap(master):
    spec = master[(FAMILY_MACHINE_A1, 38)]
    assert spec.description == "ARDATZA M14X2 ZUZENA"
    assert spec.tool_type == "tap"


def test_press_c_ditto_description_keeps_previous_tool_family(master):
    spec = master[(FAMILY_PRESS_C_20_0482_010, 2473)]
    assert spec.description == "Plan-Messerkopf Ingersoll"
    assert spec.tool_type == "mill"


def test_builder_b1_reviewed_merge_is_conservative(master):
    probe = master[(FAMILY_BUILDER_B12, 1)]
    assert probe.tool_type == "probe"
    assert probe.teeth is None

    empty_primary = master[(FAMILY_BUILDER_B12, 5)]
    assert empty_primary.description == "ROUGH BORE. Ø 65"
    assert empty_primary.tool_type == "bore"
    assert empty_primary.diameter_mm == pytest.approx(65.0)
    assert empty_primary.teeth == 2
    assert "Site_b_Tool List Reviewed v2.xlsx" in empty_primary.source

    bore = master[(FAMILY_BUILDER_B12, 6)]
    assert bore.tool_type == "bore"
    assert bore.diameter_mm == pytest.approx(65.0)
    assert bore.tool_length_mm == pytest.approx(128.125)
    assert bore.teeth == 1
    assert "Site_b_Tool List Reviewed v2.xlsx" in bore.source

    bore_fallback = master[(FAMILY_BUILDER_B12, 18)]
    assert bore_fallback.description == "FINISH BORE 100MM DIA"
    assert bore_fallback.tool_type == "bore"
    assert bore_fallback.diameter_mm == pytest.approx(100.0)
    assert bore_fallback.teeth == 1


def test_press_c_split_header_is_parsed(master):
    spec = master[(FAMILY_PRESS_C_20_0482_010, 2432)]
    assert spec.description == "HSS-Kegelsenker 90°"
    assert spec.tool_type == "countersink"
    assert spec.diameter_mm == pytest.approx(20.5)


def test_lookup_returns_copy_and_none_for_missing(master):
    spec = lookup(FAMILY_MACHINE_A1, "2")
    assert spec is not None
    assert spec.tool_id == "PART0003"
    assert lookup(FAMILY_MACHINE_A1, 9999) is None


def test_resolve_tool_context_uses_raw_teeth_fallback(master):
    context = resolve_tool_context(
        FAMILY_MACHINE_A1,
        2,
        machine_id="2026_03_13",
        raw_teeth=4,
    )

    assert context["machine_family"] == FAMILY_MACHINE_A1
    assert context["machine_id"] == "2026_03_13"
    assert context["tool_number"] == 2
    assert context["tool_id"] == "PART0003"
    assert context["tool_type"] == "mill"
    assert context["tool_diameter"] == pytest.approx(125.0)
    assert context["tool_length"] == pytest.approx(113.0)
    assert context["num_teeth"] == 4
    assert context["sindit_tool_iri"] == "urn:lfl:tool:machine_a1-t2"


def test_resolve_tool_context_applies_confirmed_dataset_decision(tmp_path, monkeypatch, master):
    target = tmp_path / "dataset_tool_decisions.json"
    monkeypatch.setattr(tool_dataset_decisions_module, "DATASET_TOOL_DECISIONS_PATH", target)
    tool_dataset_decisions_module.save_tool_dataset_decision(
        dataset_id="site_a_line2",
        machine_family=FAMILY_MACHINE_A1,
        tool_number=2,
        status="confirmed",
        selection_mode="default",
        resolved_context={
            "tool_id": "CONFIRMED-T2",
            "tool_type": "boring_head",
            "tool_diameter": 126.5,
            "tool_length": 115.0,
            "num_teeth": 7,
            "tool_material": "carbide",
        },
        resolved_sources={
            "tool_id": "runtime",
            "tool_type": "reference",
            "tool_diameter": "reference",
            "tool_length": "reference",
            "num_teeth": "runtime",
            "tool_material": "master",
        },
    )

    context = resolve_tool_context(
        FAMILY_MACHINE_A1,
        2,
        dataset_id="site_a_line2",
        machine_id="2026_03_13",
        raw_teeth=4,
    )

    assert context["tool_id"] == "CONFIRMED-T2"
    assert context["tool_type"] == "boring_head"
    assert context["tool_diameter"] == pytest.approx(126.5)
    assert context["tool_length"] == pytest.approx(115.0)
    assert context["num_teeth"] == 7
    assert context["tool_material"] == "carbide"
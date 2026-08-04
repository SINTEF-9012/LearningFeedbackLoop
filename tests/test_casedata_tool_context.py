from __future__ import annotations

import pytest

from backend.agents.core.context import extract_context_from_metadata
from backend.agents.processing.dataset_loader import WindowData


def test_windowdata_cutting_context_uses_tool_lookup_for_known_case_family():
    window = WindowData(
        operation_id="OF00006",
        case_dir="Site_b - MACHINE_B1 - CASE_B1",
        t_start="2026-01-01T00:00:00Z",
        t_end="2026-01-01T00:00:30Z",
        duration_s=30.0,
        n_samples=30,
        features={
            "tool_number": 6.0,
            "spindle_speed_mean": 1800.0,
            "feed_rate_mean": 900.0,
            "temp_head_mean": 23.1,
        },
    )

    ctx = window._derive_cutting_context()

    assert ctx["machine_id"] == "Site_b - MACHINE_B1 - CASE_B1"
    assert ctx["tool_id"] == "T6"
    assert ctx["tool_type"] == "bore"
    assert ctx["tool_diameter"] == pytest.approx(65.0)
    assert ctx["num_teeth"] == 1
    assert ctx["tool_length"] == pytest.approx(128.125)
    assert ctx["extra"]["machine_family"] == "builder_b12"
    assert ctx["extra"]["sindit_tool_iri"] == "urn:lfl:tool:builder_b12-t6"
    assert ctx["extra"]["temperature_head"] == pytest.approx(23.1)


def test_extract_context_from_metadata_maps_new_tool_fields():
    ctx = extract_context_from_metadata(
        {
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
            "casedata": {
                "root": "/tmp/casedata",
                "cutting_context": {
                    "tool_type": "bore",
                    "tool_id": "T6",
                    "tool_length": 128.125,
                    "tool_material": "carbide",
                    "extra": {"sindit_tool_iri": "urn:lfl:tool:builder_b12-t6"},
                },
            },
        }
    )

    assert ctx.tool_type == "bore"
    assert ctx.tool_id == "T6"
    assert ctx.tool_length == pytest.approx(128.125)
    assert ctx.tool_material == "carbide"
    assert ctx.extra["sindit_tool_iri"] == "urn:lfl:tool:builder_b12-t6"
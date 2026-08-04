from __future__ import annotations

import pandas as pd

from backend.agents.experiment.evaluator import _simulate_sindit_context


def test_simulate_sindit_context_passes_through_tool_master_fields() -> None:
    row = pd.Series(
        {
            "spindle_speed_mean": 1800.0,
            "feed_rate_mean": 900.0,
            "power_spindle_mean": 12.5,
            "tool_number": 2,
            "tool_id": "PART0003",
            "tool_type": "mill",
            "tool_diameter": 125.0,
            "num_teeth": 4,
            "tool_length": 113.0,
            "machine_id": "2026_03_13",
            "machine_family": "machine_a1",
            "sindit_tool_iri": "urn:lfl:tool:machine_a1-t2",
            "session": "2026_03_13",
            "operation_id": "OF00011",
        }
    )

    context = _simulate_sindit_context(row)

    assert context["spindle_speed"] == 1800.0
    assert context["feed_rate"] == 900.0
    assert context["tool_id"] == "PART0003"
    assert context["tool_type"] == "mill"
    assert context["tool_diameter"] == 125.0
    assert context["num_teeth"] == 4
    assert context["tool_length"] == 113.0
    assert context["machine_id"] == "2026_03_13"
    assert context["machine_family"] == "machine_a1"
    assert context["extra"]["session"] == "2026_03_13"
    assert context["extra"]["operation_id"] == "OF00011"
    assert context["extra"]["sindit_tool_iri"] == "urn:lfl:tool:machine_a1-t2"
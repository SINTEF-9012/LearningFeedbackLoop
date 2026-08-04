from __future__ import annotations

import pytest

from backend.agents.processing.breakage_detector import BreakageFeatureExtractor


def test_row_to_feature_dict_reads_direct_site_a_line2_feature_names() -> None:
    row = {
        "power_spindle_mean": "10.5",
        "spindle_speed_mean": "1800",
        "feed_rate_mean": "900",
        "temp_head_mean": "22.5",
        "chatter_ratio": "0.25",
    }

    features = BreakageFeatureExtractor.row_to_feature_dict(row)

    assert features["power_spindle_mean"] == pytest.approx(10.5)
    assert features["spindle_speed_mean"] == pytest.approx(1800.0)
    assert features["feed_rate_mean"] == pytest.approx(900.0)
    assert features["temp_head_mean"] == pytest.approx(22.5)
    assert features["chatter_ratio"] == pytest.approx(0.25)


def test_row_to_event_uses_rich_site_a_line2_tool_context() -> None:
    extractor = BreakageFeatureExtractor("/tmp/unused.csv")
    row = {
        "sample_id": "s1",
        "label": "pre_break",
        "operation_id": "OF00011",
        "tool_number": "2",
        "session": "2026_03_13",
        "machine_id": "2026_03_13",
        "machine_family": "machine_a1",
        "tool_id": "PART0003",
        "tool_type": "mill",
        "tool_diameter": "125.0",
        "num_teeth": "4",
        "tool_length": "113.0",
        "tool_material": "carbide",
        "sindit_tool_iri": "urn:lfl:tool:machine_a1-t2",
        "spindle_speed_mean": "1800",
        "feed_rate_mean": "900",
        "power_spindle_mean": "10.0",
    }

    event, meta = extractor.row_to_event(row, session_id="exp-1")

    assert meta.tool_number == "2"
    assert event.cutting_context is not None
    assert event.cutting_context.machine_id == "2026_03_13"
    assert event.cutting_context.tool_id == "PART0003"
    assert event.cutting_context.tool_type == "mill"
    assert event.cutting_context.tool_diameter == pytest.approx(125.0)
    assert event.cutting_context.num_teeth == 4
    assert event.cutting_context.tool_length == pytest.approx(113.0)
    assert event.cutting_context.tool_material == "carbide"
    assert event.cutting_context.spindle_speed == pytest.approx(1800.0)
    assert event.cutting_context.feed_rate == pytest.approx(900.0)
    assert event.cutting_context.extra["machine_family"] == "machine_a1"
    assert event.cutting_context.extra["operation_id"] == "OF00011"
    assert event.cutting_context.extra["sindit_tool_iri"] == "urn:lfl:tool:machine_a1-t2"
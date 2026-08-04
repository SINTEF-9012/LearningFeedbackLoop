"""Tests for the DPP CO2/cost impact-weighting hook in the feedback handler."""

import json
import types

from backend.agents.maas import DPPRegistry
from backend.agents.memory.feedback import MemoryFeedbackHandler


def _dpp_registry(tmp_path, pcf_a3="900.0"):
    d = tmp_path / "dpp"
    d.mkdir()
    (d / "DPP_x.json").write_text(json.dumps({"DPP": [{
        "company": "site_a", "event_id": "PART-HI",
        "carbonFootprint": {"ProductCarbonFootprints": [
            {"PCF in kg CO2eq": pcf_a3, "ProcessName": "SITE_A total A3"},
        ]},
    }]}))
    return DPPRegistry.from_dir(d)


def _memory(event_id=None):
    md = {"event_id": event_id} if event_id else {}
    return types.SimpleNamespace(metadata=md, pattern_keys=[])


def test_multiplier_is_one_when_disabled(tmp_path):
    h = MemoryFeedbackHandler(dpp_registry=_dpp_registry(tmp_path))
    h._impact_weighting = False
    assert h._impact_multiplier(_memory("PART-HI")) == 1.0


def test_multiplier_scales_with_pcf_when_enabled(tmp_path):
    # ref PCF 900, part A3 = 2700 -> 3.0x (also the cap)
    h = MemoryFeedbackHandler(dpp_registry=_dpp_registry(tmp_path, pcf_a3="2700.0"))
    h._impact_weighting = True
    h._impact_ref_pcf_kg = 900.0
    h._impact_max = 3.0
    assert h._impact_multiplier(_memory("PART-HI")) == 3.0


def test_multiplier_capped(tmp_path):
    h = MemoryFeedbackHandler(dpp_registry=_dpp_registry(tmp_path, pcf_a3="18000.0"))
    h._impact_weighting = True
    h._impact_ref_pcf_kg = 900.0
    h._impact_max = 3.0
    assert h._impact_multiplier(_memory("PART-HI")) == 3.0  # 20x clamped to 3.0


def test_multiplier_floored_at_one_for_cheap_parts(tmp_path):
    h = MemoryFeedbackHandler(dpp_registry=_dpp_registry(tmp_path, pcf_a3="100.0"))
    h._impact_weighting = True
    h._impact_ref_pcf_kg = 900.0
    assert h._impact_multiplier(_memory("PART-HI")) == 1.0  # never down-weight


def test_multiplier_one_when_no_dpp_match_or_no_event(tmp_path):
    h = MemoryFeedbackHandler(dpp_registry=_dpp_registry(tmp_path))
    h._impact_weighting = True
    assert h._impact_multiplier(_memory(None)) == 1.0          # no event id
    assert h._impact_multiplier(_memory("PART-HI")) >= 1.0     # match falls back fine


def test_handler_default_is_off():
    # constructed without the env flag -> impact weighting disabled, behaviour unchanged
    h = MemoryFeedbackHandler()
    assert h._impact_weighting is False
    assert h._impact_multiplier(_memory("anything")) == 1.0

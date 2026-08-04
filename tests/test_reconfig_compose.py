"""Tests for composing a reconfiguration proposal from confirmed tool-condition feedback."""

from backend.agents.memory.reconfig import (
    ProcessReconfiguration,
    compose_tool_condition_reconfiguration,
)


def test_broken_proposes_replace_and_requires_confirmation():
    p = compose_tool_condition_reconfiguration(
        context={"machine_type": "gantry_mill", "tool_type": "face_mill"},
        severity="broken", confirmed=3, dismissed=1, tool_id="PART0001", tool_number=1,
    )
    assert isinstance(p, ProcessReconfiguration)
    assert p.requires_operator_confirmation is True
    assert p.applied is False
    assert len(p.tool_actions) == 1
    assert p.tool_actions[0].action == "replace"
    assert p.tool_actions[0].reason_code == "confirmed_tool_broken"
    assert p.tool_actions[0].tool_id == "PART0001"


def test_chipped_proposes_inspect():
    p = compose_tool_condition_reconfiguration(
        context={"tool_type": "face_mill"}, severity="chipped", confirmed=1,
    )
    assert p.tool_actions[0].action == "inspect"
    assert p.tool_actions[0].reason_code == "confirmed_tool_chipped"


def test_parameter_delta_is_bounded_and_low_confidence():
    p = compose_tool_condition_reconfiguration(
        context={"tool_type": "face_mill"}, severity="broken", confirmed=3, dismissed=1,
        feed_reduction_pct=50.0,  # request a large cut...
    )
    assert len(p.parameter_deltas) == 1
    d = p.parameter_deltas[0]
    assert d.parameter == "feed_rate" and d.direction == "decrease"
    assert d.magnitude_pct <= 15.0          # ...but it is capped
    assert d.confidence < 0.3               # volume-shrunk: 4 events is thin evidence
    assert "precautionary" in d.rationale
    assert p.risk == "medium"


def test_no_parameter_delta_when_suppressed():
    p = compose_tool_condition_reconfiguration(
        context={"tool_type": "face_mill"}, severity="broken", confirmed=2,
        suggest_parameter_delta=False,
    )
    assert p.parameter_deltas == []
    assert p.risk == "low"                  # tool action only -> lower risk


def test_more_confirmations_raise_tool_action_confidence():
    low = compose_tool_condition_reconfiguration(
        context={"tool_type": "x"}, severity="broken", confirmed=1)
    high = compose_tool_condition_reconfiguration(
        context={"tool_type": "x"}, severity="broken", confirmed=3)
    assert high.tool_actions[0].confidence > low.tool_actions[0].confidence


def test_co2_impact_appears_in_rationale_and_notes():
    p = compose_tool_condition_reconfiguration(
        context={"tool_type": "face_mill"}, severity="broken", confirmed=2,
        impact_co2_kg_per_catch=921.0,
    )
    assert any("921" in n for n in p.notes)
    assert "921" in p.parameter_deltas[0].rationale


def test_context_is_normalized():
    p = compose_tool_condition_reconfiguration(
        context={"machine_type": " gantry_mill ", "tool_type": "face_mill",
                 "material": "casting_steel", "regime": "roughing"},
        severity="broken", confirmed=1,
    )
    assert p.context["machine_type"] == "gantry_mill"   # trimmed
    assert set(p.context) == {"machine_type", "tool_type", "material", "regime"}

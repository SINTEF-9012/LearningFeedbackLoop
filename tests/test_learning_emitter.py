from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.ingestion.schema import LearningEnvelope
from backend.agents.sindit.tool_audit import clear_runtime_observations, _runtime_snapshot
from backend.learning_emitter import (
    create_feedback_learning_callback,
    publish_insight_learning,
    publish_scored_learning,
)


class FakeMemory:
    session_id = "session-1"
    metadata = {
        "batch": {
            "batch_id": "batch-42",
            "unit_index": 1,
            "unit_count": 5,
            "recipe_id": "recipe-z",
        },
        "cutting_context": {
            "machine_id": "Site_b - MACHINE_B1 - CASE_B1",
            "tool_id": "T55",
            "tool_diameter": 20.0,
            "num_teeth": 4,
            "spindle_speed": 1800.0,
            "feed_rate": 900.0,
            "extra": {
                "machine_family": "builder_b12",
                "tool_number": 55,
                "sindit_tool_iri": "urn:lfl:tool:builder_b12-t55",
            },
        },
        "significance": {
            "pattern_priors": {
                "signature:modulated_tooth_passing_vibration": 0.71,
            }
        },
    }
    pattern_keys = [SimpleNamespace(key="signature:modulated_tooth_passing_vibration")]


class FakeStore:
    def get(self, memory_id: str):
        assert memory_id == "mem-1"
        return FakeMemory()


@pytest.mark.asyncio
async def test_feedback_learning_callback_publishes_learning_envelope(monkeypatch):
    clear_runtime_observations()
    published: list[LearningEnvelope] = []

    async def fake_publish(payload):
        published.append(payload)

    monkeypatch.setattr("backend.learning_emitter.publish_learning", fake_publish)

    callback = create_feedback_learning_callback(FakeStore())
    await callback("mem-1", "confirm", {"user_id": "operator-7", "reason": "verified"})

    assert len(published) == 2

    feedback_envelope = next(item for item in published if item.kind == "feedback_event")
    tool_envelope = next(item for item in published if item.kind == "tool_event")

    assert isinstance(feedback_envelope, LearningEnvelope)
    assert feedback_envelope.session_id == "session-1"
    assert feedback_envelope.payload["memory_id"] == "mem-1"
    assert feedback_envelope.payload["action"] == "confirm"
    assert feedback_envelope.payload["operator_id"] == "operator-7"
    assert feedback_envelope.pii_scrub_level == "symbolic_only"
    assert feedback_envelope.payload["feedback"] == {"reason": "verified"}
    assert feedback_envelope.batch == {
        "batch_id": "batch-42",
        "unit_index": 1,
        "unit_count": 5,
        "recipe_id": "recipe-z",
    }

    assert isinstance(tool_envelope, LearningEnvelope)
    assert tool_envelope.session_id == "session-1"
    assert tool_envelope.payload["action"] == "confirm"
    assert tool_envelope.payload["tool_number"] == 55
    assert tool_envelope.payload["harmonic_ready"] is True
    assert tool_envelope.payload["anomaly_stats"]["confirmed_count"] == 1
    assert tool_envelope.pii_scrub_level == "symbolic_only"
    assert tool_envelope.batch == feedback_envelope.batch

    runtime = _runtime_snapshot()
    assert runtime[("builder_b12", 55)]["anomaly_stats"]["confirmed_count"] == 1


@pytest.mark.asyncio
async def test_publish_scored_learning_publishes_scored_envelope(monkeypatch):
    published: list[LearningEnvelope] = []

    async def fake_publish(payload):
        published.append(payload)

    monkeypatch.setattr("backend.learning_emitter.publish_learning", fake_publish)

    significance = SimpleNamespace(
        score=0.91,
        action=SimpleNamespace(value="alert"),
        is_significant=True,
        reasons=["High anomaly score"],
        triggered_rules=["ANOMALY_HIGH"],
        pattern_priors={"CHATTER": 0.7},
        prior_boost=0.2,
        score_trace=[],
        to_dict=lambda: {
            "score": 0.91,
            "action": "alert",
            "is_significant": True,
            "reasons": ["High anomaly score"],
            "triggered_rules": ["ANOMALY_HIGH"],
            "pattern_priors": {"CHATTER": 0.7},
            "prior_boost": 0.2,
            "score_trace": [],
        },
    )

    await publish_scored_learning(
        session_id="session-2",
        memory_id="mem-2",
        significance=significance,
        patterns=[SimpleNamespace(key="CHATTER")],
        external_signals={"anomaly_detector_score": 0.91},
        model_breakdown={"ensemble": {"score": 0.91}},
        alert_dispatched=True,
        similar_memory_count=2,
        time_range=SimpleNamespace(i0=0, i1=64, t0=0.0, t1=0.64, fs=100.0),
        batch={"batch_id": "batch-score", "unit_index": 3, "unit_count": 10},
    )

    assert len(published) == 1
    envelope = published[0]
    assert isinstance(envelope, LearningEnvelope)
    assert envelope.kind == "scored_event"
    assert envelope.source == "memory_orchestrator"
    assert envelope.session_id == "session-2"
    assert envelope.payload["memory_id"] == "mem-2"
    assert envelope.payload["patterns"] == ["CHATTER"]
    assert envelope.payload["significance"]["action"] == "alert"
    assert envelope.payload["alert_dispatched"] is True
    assert envelope.payload["similar_memory_count"] == 2
    assert envelope.payload["time_range"]["i1"] == 64
    assert envelope.pii_scrub_level == "symbolic_only"
    assert envelope.batch == {
        "batch_id": "batch-score",
        "unit_index": 3,
        "unit_count": 10,
    }


@pytest.mark.asyncio
async def test_publish_insight_learning_publishes_insight_envelope(monkeypatch):
    published: list[LearningEnvelope] = []

    async def fake_publish(payload):
        published.append(payload)

    monkeypatch.setattr("backend.learning_emitter.publish_learning", fake_publish)

    await publish_insight_learning(
        session_id="session-3",
        memory_id="mem-3",
        explanation="Rising vibration aligns with prior wear progression.",
        explanation_source="llm",
        alert_line="Tool wear trend increasing",
        alert_line_source="history",
    )

    assert len(published) == 1
    envelope = published[0]
    assert isinstance(envelope, LearningEnvelope)
    assert envelope.kind == "insight_event"
    assert envelope.source == "llm_explainer"
    assert envelope.session_id == "session-3"
    assert envelope.payload["memory_id"] == "mem-3"
    assert envelope.payload["explanation_source"] == "llm"
    assert envelope.payload["alert_line_source"] == "history"
    assert envelope.pii_scrub_level == "symbolic_only"


@pytest.mark.asyncio
async def test_feedback_learning_envelope_uses_env_provenance_and_scrubs_comment(monkeypatch):
    published: list[LearningEnvelope] = []

    async def fake_publish(payload):
        published.append(payload)

    monkeypatch.setattr("backend.learning_emitter.publish_learning", fake_publish)
    monkeypatch.setenv("KNOWLEDGE_TENANT_ID", "tenant-a")
    monkeypatch.setenv("LFL_SITE_ID", "site-7")

    callback = create_feedback_learning_callback(FakeStore())
    await callback(
        "mem-1",
        "dismiss",
        {
            "user_id": "operator-7",
            "reason": "false alarm",
            "comment": "customer name should stay local",
        },
    )

    feedback_envelope = next(item for item in published if item.kind == "feedback_event")
    assert feedback_envelope.tenant_id == "tenant-a"
    assert feedback_envelope.site_id == "site-7"
    assert feedback_envelope.payload["feedback"] == {"reason": "false alarm"}

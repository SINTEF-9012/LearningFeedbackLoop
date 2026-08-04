import pytest

from backend.agents.core.schemas import Memory, PatternKey, PatternType
from backend.agents.memory.feedback import FeedbackAction, MemoryFeedbackHandler, MemoryFeedbackRequest
from backend.agents.memory.scorer import SignificanceAction, SignificanceScorer
from backend.agents.storage.store import MemoryStore


@pytest.mark.asyncio
async def test_severity_correction_persists_without_dismiss_penalty(tmp_path):
    store = MemoryStore(
        db_path=str(tmp_path / "memories.db"),
        enable_ann=False,
        enable_embeddings=False,
    )
    store.create(
        Memory(
            id="m-severity-correction",
            session_id="s-severity-correction",
            time_range=(0.0, 1.0),
            annotation_text="severity correction",
            pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:test")],
            metadata={
                "significance_score": 0.95,
                "significance_action": "critical",
            },
        )
    )

    scorer = SignificanceScorer(
        priors_path=str(tmp_path / "priors.json"),
        feedback_store=store,
    )
    handler = MemoryFeedbackHandler(memory_store=store, significance_scorer=scorer)

    response = await handler.process_feedback(
        "m-severity-correction",
        MemoryFeedbackRequest(
            action=FeedbackAction.DISMISS,
            user_id="tester",
            reason="wrong severity",
            severity_target="warning",
        ),
    )

    assert response.success is True
    assert response.updated_fields == ["severity_corrected"]
    assert store.get_feedback_counts(pattern_key="CUSTOM:test") == pytest.approx((0.0, 0.0))

    events = store.list_feedback_events("m-severity-correction")
    assert len(events) == 1
    assert events[0]["action"] == "severity_correction"
    assert events[0]["data"]["severity_target"] == "warning"
    assert scorer._local_feedback_counts.get("CUSTOM:test") is None
    assert scorer._severity_calibration["CUSTOM:test"]["weight_sum"] == pytest.approx(1.0)


def test_severity_correction_biases_future_scores_for_same_pattern():
    scorer = SignificanceScorer()
    pattern = PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:test", confidence=1.0)

    baseline = scorer.score([pattern], external_signals={"anomaly_detector_score": 1.0})
    assert baseline.action == SignificanceAction.CRITICAL

    for _ in range(3):
        scorer.record_severity_correction(
            "CUSTOM:test",
            target_severity="warning",
            current_score=baseline.score,
            current_severity=baseline.action.value,
            weight=1.0,
        )

    adjusted = scorer.score([pattern], external_signals={"anomaly_detector_score": 1.0})
    assert adjusted.action == SignificanceAction.ALERT
    assert adjusted.score < baseline.score
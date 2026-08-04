import json

import pytest

from backend.agents.core.context import CuttingContext, OperatingRegime
from backend.agents.core.schemas import Memory, PatternKey, PatternType
from backend.agents.memory.feedback import FeedbackAction, MemoryFeedbackHandler, MemoryFeedbackRequest
from backend.agents.memory.scorer import FEEDBACK_WEIGHTS, SignificanceScorer
from backend.agents.storage.store import MemoryStore


def test_fractional_feedback_counts_roundtrip_in_priors_cache(tmp_path):
    priors_path = tmp_path / "pattern_priors.json"

    scorer = SignificanceScorer(priors_path=str(priors_path))
    scorer.update_pattern_prior("CUSTOM:test", was_significant=False, weight=0.25)

    reloaded = SignificanceScorer(priors_path=str(priors_path))
    assert reloaded._local_feedback_counts["CUSTOM:test"]["dismiss"] == pytest.approx(0.25)


def test_context_scoped_feedback_counts_roundtrip_in_priors_cache(tmp_path):
    priors_path = tmp_path / "pattern_priors.json"
    context = CuttingContext(
        machine_type="cnc",
        tool_type="endmill",
        workpiece_material="al",
        operating_regime=OperatingRegime.ROUGHING,
    )

    scorer = SignificanceScorer(priors_path=str(priors_path))
    scorer.update_pattern_prior("CUSTOM:test", was_significant=True, context=context)

    raw = json.loads(priors_path.read_text())
    assert (
        "machine_type=cnc|tool_type=endmill|workpiece_material=al|operating_regime=roughing"
        in raw["pattern_priors_by_context"]
    )

    reloaded = SignificanceScorer(priors_path=str(priors_path))
    assert reloaded.get_pattern_prior("CUSTOM:test", context=context) > 0.5
    assert reloaded.get_pattern_prior(
        "CUSTOM:test",
        context=CuttingContext(
            machine_type="cnc",
            tool_type="drill",
            workpiece_material="al",
            operating_regime=OperatingRegime.ROUGHING,
        ),
    ) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_dismiss_feedback_reason_uses_weighted_store_counts(tmp_path):
    store = MemoryStore(
        db_path=str(tmp_path / "memories.db"),
        enable_ann=False,
        enable_embeddings=False,
    )
    store.create(
        Memory(
            id="m-weighted-dismiss",
            session_id="s-weighted-dismiss",
            time_range=(0.0, 1.0),
            annotation_text="weighted dismiss",
            pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:test")],
            metadata={},
        )
    )

    scorer = SignificanceScorer(
        priors_path=str(tmp_path / "priors.json"),
        feedback_store=store,
    )
    handler = MemoryFeedbackHandler(memory_store=store, significance_scorer=scorer)

    response = await handler.process_feedback(
        "m-weighted-dismiss",
        MemoryFeedbackRequest(
            action=FeedbackAction.DISMISS,
            user_id="tester",
            reason="false alarm",
        ),
    )

    assert response.success is True
    assert store.get_feedback_counts(pattern_key="CUSTOM:test") == pytest.approx(
        (0.0, FEEDBACK_WEIGHTS["dismiss_with_reason"])
    )
    assert scorer._local_feedback_counts["CUSTOM:test"]["dismiss"] == pytest.approx(
        FEEDBACK_WEIGHTS["dismiss_with_reason"]
    )


@pytest.mark.asyncio
async def test_plain_dismiss_uses_lower_weight_than_reasoned_dismiss(tmp_path):
    store = MemoryStore(
        db_path=str(tmp_path / "memories.db"),
        enable_ann=False,
        enable_embeddings=False,
    )
    store.create(
        Memory(
            id="m-plain-dismiss",
            session_id="s-plain-dismiss",
            time_range=(0.0, 1.0),
            annotation_text="plain dismiss",
            pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:test")],
            metadata={},
        )
    )

    scorer = SignificanceScorer(
        priors_path=str(tmp_path / "priors.json"),
        feedback_store=store,
    )
    handler = MemoryFeedbackHandler(memory_store=store, significance_scorer=scorer)

    response = await handler.process_feedback(
        "m-plain-dismiss",
        MemoryFeedbackRequest(
            action=FeedbackAction.DISMISS,
            user_id="tester",
        ),
    )

    assert response.success is True
    assert store.get_feedback_counts(pattern_key="CUSTOM:test") == pytest.approx(
        (0.0, FEEDBACK_WEIGHTS["dismiss_oneclick"])
    )


@pytest.mark.asyncio
async def test_feedback_handler_updates_context_scoped_prior_without_leaking(tmp_path):
    store = MemoryStore(
        db_path=str(tmp_path / "memories.db"),
        enable_ann=False,
        enable_embeddings=False,
    )
    context = CuttingContext(
        machine_type="cnc",
        tool_type="endmill",
        workpiece_material="al",
        operating_regime=OperatingRegime.ROUGHING,
    )
    store.create(
        Memory(
            id="m-context-confirm",
            session_id="s-context-confirm",
            time_range=(0.0, 1.0),
            annotation_text="context confirm",
            pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:test")],
            metadata={"cutting_context": context.model_dump(mode="json")},
        )
    )

    scorer = SignificanceScorer(priors_path=str(tmp_path / "priors.json"))
    handler = MemoryFeedbackHandler(memory_store=store, significance_scorer=scorer)

    response = await handler.process_feedback(
        "m-context-confirm",
        MemoryFeedbackRequest(
            action=FeedbackAction.CONFIRM,
            user_id="tester",
        ),
    )

    assert response.success is True
    assert scorer.get_pattern_prior("CUSTOM:test", context=context) > 0.5
    assert scorer.get_pattern_prior(
        "CUSTOM:test",
        context=CuttingContext(
            machine_type="cnc",
            tool_type="drill",
            workpiece_material="al",
            operating_regime=OperatingRegime.ROUGHING,
        ),
    ) == pytest.approx(0.5)
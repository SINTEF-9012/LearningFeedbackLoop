import pytest

from backend.agents.core.context import CuttingContext, OperatingRegime
from backend.agents.core.schemas import Memory, PatternKey, PatternType
from backend.agents.memory.cycle_tracker import CycleEnded, CycleTracker
from backend.agents.memory.outcome_correlator import attach_passive_outcome
from backend.agents.memory.scorer import FEEDBACK_WEIGHTS, SignificanceScorer
from backend.agents.storage.store import MemoryStore


def test_cycle_tracker_emits_cycle_end_on_part_transition():
    tracker = CycleTracker()

    assert tracker.observe(
        "session-1",
        {"casedata": {"part_id": "part-1", "operation_id": "op-1"}},
        10.0,
    ) is None
    assert tracker.observe(
        "session-1",
        {"casedata": {"part_id": "part-1", "operation_id": "op-1"}},
        12.5,
    ) is None

    ended = tracker.observe(
        "session-1",
        {"casedata": {"part_id": "part-2", "operation_id": "op-1"}},
        15.0,
    )

    assert ended is not None
    assert ended.session_id == "session-1"
    assert ended.part_id == "part-1"
    assert ended.operation_id == "op-1"
    assert ended.started_at == pytest.approx(10.0)
    assert ended.ended_at == pytest.approx(15.0)


def test_cycle_tracker_flushes_open_cycle_at_last_seen_timestamp():
    tracker = CycleTracker()

    assert tracker.observe(
        "session-flush",
        {"casedata": {"part_id": "part-9", "operation_id": "op-3"}},
        3.0,
    ) is None
    assert tracker.observe(
        "session-flush",
        {"casedata": {"part_id": "part-9", "operation_id": "op-3"}},
        6.5,
    ) is None

    ended = tracker.flush_session("session-flush")

    assert ended is not None
    assert ended.session_id == "session-flush"
    assert ended.part_id == "part-9"
    assert ended.operation_id == "op-3"
    assert ended.started_at == pytest.approx(3.0)
    assert ended.ended_at == pytest.approx(6.5)
    assert tracker.flush_session("session-flush") is None


def test_attach_passive_outcome_marks_only_unreviewed_memories(tmp_path):
    store = MemoryStore(
        db_path=str(tmp_path / "memories.db"),
        enable_ann=False,
        enable_embeddings=False,
    )
    scorer = SignificanceScorer(
        priors_path=str(tmp_path / "priors.json"),
        feedback_store=store,
    )

    store.create(
        Memory(
            id="m-passive-1",
            session_id="session-passive",
            time_range=(0.25, 0.75),
            annotation_text="unreviewed memory",
            pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:passive")],
            metadata={},
        )
    )
    store.create(
        Memory(
            id="m-reviewed",
            session_id="session-passive",
            time_range=(0.8, 1.2),
            annotation_text="reviewed memory",
            pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:reviewed")],
            metadata={},
        )
    )
    store.create(
        Memory(
            id="m-outside",
            session_id="session-passive",
            time_range=(5.0, 6.0),
            annotation_text="outside cycle",
            pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:outside")],
            metadata={},
        )
    )

    store.add_feedback_event(
        memory_id="m-reviewed",
        action="confirm",
        user_id="tester",
        pattern_keys=["CUSTOM:reviewed"],
        data={"reason": "operator confirmed"},
        weight=1.0,
    )

    cycle = CycleEnded(
        session_id="session-passive",
        part_id="part-1",
        operation_id="op-1",
        started_at=0.0,
        ended_at=2.0,
    )

    affected = attach_passive_outcome(
        cycle=cycle,
        memories=store.list(session_id="session-passive"),
        store=store,
        scorer=scorer,
    )

    assert cycle.part_id == "part-1"
    assert affected == 1
    assert store.get_feedback_counts(pattern_key="CUSTOM:passive") == pytest.approx(
        (0.0, FEEDBACK_WEIGHTS["passive_cycle_completed_without_intervention"])
    )
    assert scorer._local_feedback_counts["CUSTOM:passive"]["dismiss"] == pytest.approx(
        FEEDBACK_WEIGHTS["passive_cycle_completed_without_intervention"]
    )

    passive_events = store.list_feedback_events("m-passive-1")
    assert len(passive_events) == 1
    assert passive_events[0]["action"] == "dismiss"
    assert passive_events[0]["data"]["source"] == "passive_cycle_completed_without_intervention"
    assert passive_events[0]["data"]["emitted_by"] == "passive_cycle_tracker"

    reviewed_events = store.list_feedback_events("m-reviewed")
    assert len(reviewed_events) == 1
    assert reviewed_events[0]["action"] == "confirm"

    outside_events = store.list_feedback_events("m-outside")
    assert outside_events == []


def test_attach_passive_outcome_continues_when_feedback_store_is_unavailable(tmp_path):
    class FailingStore:
        def list_feedback_events(self, memory_id, limit=200):
            return []

        def add_feedback_event(self, **kwargs):
            raise RuntimeError("neo4j unavailable")

        def get_feedback_counts(self, *, pattern_key, context_key=None, user_id=None):
            raise RuntimeError("neo4j unavailable")

    store = FailingStore()
    scorer = SignificanceScorer(
        priors_path=str(tmp_path / "priors.json"),
        feedback_store=store,
    )

    cycle = CycleEnded(
        session_id="session-passive",
        part_id="part-1",
        operation_id="op-1",
        started_at=0.0,
        ended_at=2.0,
    )
    memory = Memory(
        id="m-passive-1",
        session_id="session-passive",
        time_range=(0.25, 0.75),
        annotation_text="unreviewed memory",
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:passive")],
        metadata={},
    )

    affected = attach_passive_outcome(
        cycle=cycle,
        memories=[memory],
        store=store,
        scorer=scorer,
    )

    assert affected == 1
    assert scorer._local_feedback_counts["CUSTOM:passive"]["dismiss"] == pytest.approx(
        FEEDBACK_WEIGHTS["passive_cycle_completed_without_intervention"]
    )
    assert scorer.get_pattern_prior("CUSTOM:passive") < 0.5


def test_attach_passive_outcome_updates_context_scoped_prior(tmp_path):
    store = MemoryStore(
        db_path=str(tmp_path / "memories.db"),
        enable_ann=False,
        enable_embeddings=False,
    )
    scorer = SignificanceScorer(priors_path=str(tmp_path / "priors.json"))
    context = CuttingContext(
        machine_type="cnc",
        tool_type="endmill",
        workpiece_material="al",
        operating_regime=OperatingRegime.ROUGHING,
    )

    store.create(
        Memory(
            id="m-passive-context",
            session_id="session-passive-context",
            time_range=(0.25, 0.75),
            annotation_text="context passive",
            pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:passive")],
            metadata={"cutting_context": context.model_dump(mode="json")},
        )
    )

    affected = attach_passive_outcome(
        cycle=CycleEnded(
            session_id="session-passive-context",
            part_id="part-ctx",
            operation_id="op-ctx",
            started_at=0.0,
            ended_at=2.0,
        ),
        memories=store.list(session_id="session-passive-context"),
        store=store,
        scorer=scorer,
    )

    assert affected == 1
    assert scorer.get_pattern_prior("CUSTOM:passive", context=context) < 0.5
    assert scorer.get_pattern_prior(
        "CUSTOM:passive",
        context=CuttingContext(
            machine_type="cnc",
            tool_type="drill",
            workpiece_material="al",
            operating_regime=OperatingRegime.ROUGHING,
        ),
    ) == pytest.approx(0.5)
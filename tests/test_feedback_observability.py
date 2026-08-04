from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from backend.agents.core.schemas import Memory, PatternKey, PatternType
from backend.agents.memory import graph_routes as graph_mod
from backend.agents.memory import router as router_mod
from backend.agents.memory.feedback import MemoryFeedbackHandler
from backend.agents.memory.scorer import FEEDBACK_WEIGHTS, SignificanceScorer
from backend.agents.storage.store import MemoryStore


def test_feedback_stats_include_passive_outcomes_and_effective_weight(tmp_path):
    store = MemoryStore(
        db_path=str(tmp_path / "memories.db"),
        enable_ann=False,
        enable_embeddings=False,
    )
    store.create(
        Memory(
            id="m-observability",
            session_id="s-observability",
            time_range=(0.0, 1.0),
            annotation_text="observability",
            pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:obs")],
            metadata={},
        )
    )

    store.add_feedback_event(
        memory_id="m-observability",
        action="confirm",
        user_id="tester",
        pattern_keys=["CUSTOM:obs"],
        data={"source": "confirm_explicit", "weight": FEEDBACK_WEIGHTS["confirm_explicit"]},
        weight=FEEDBACK_WEIGHTS["confirm_explicit"],
    )
    store.add_feedback_event(
        memory_id="m-observability",
        action="dismiss",
        user_id="system:cycle_tracker",
        pattern_keys=["CUSTOM:obs"],
        data={
            "source": "passive_cycle_completed_without_intervention",
            "weight": FEEDBACK_WEIGHTS["passive_cycle_completed_without_intervention"],
        },
        weight=FEEDBACK_WEIGHTS["passive_cycle_completed_without_intervention"],
    )
    store.add_feedback_event(
        memory_id="m-observability",
        action="severity_correction",
        user_id="tester",
        pattern_keys=["CUSTOM:obs"],
        data={"source": "severity_correction", "severity_target": "warning", "weight": 1.0},
        weight=1.0,
    )

    handler = MemoryFeedbackHandler(memory_store=store)
    stats = handler.get_feedback_stats("m-observability")

    assert stats["total_feedback"] == 3
    assert stats["confirms"] == 1
    assert stats["dismisses"] == 0
    assert stats["severity_corrections"] == 1
    assert stats["passive_outcomes"] == 1
    assert stats["passive_outcome_weight_total"] == pytest.approx(
        FEEDBACK_WEIGHTS["passive_cycle_completed_without_intervention"]
    )
    assert stats["effective_weight_total"] == pytest.approx(2.25)


@pytest.mark.asyncio
async def test_feedback_routes_surface_passive_and_severity_diagnostics(tmp_path, monkeypatch):
    scorer = SignificanceScorer(priors_path=str(tmp_path / "priors.json"))
    pattern_key = "CUSTOM:observability"

    scorer.update_pattern_prior(
        pattern_key,
        was_significant=True,
        weight=FEEDBACK_WEIGHTS["confirm_explicit"],
        source="confirm_explicit",
    )
    scorer.update_pattern_prior(
        pattern_key,
        was_significant=False,
        weight=FEEDBACK_WEIGHTS["passive_cycle_completed_without_intervention"],
        source="passive_cycle_completed_without_intervention",
    )
    scorer.record_severity_correction(
        pattern_key,
        target_severity="warning",
        current_score=0.95,
        current_severity="critical",
        weight=1.0,
    )

    orchestrator = SimpleNamespace(
        scorer=scorer,
        store=SimpleNamespace(list_memories=lambda limit=200, offset=0: []),
    )
    # The feedback graph moved to graph_routes; the scorer-prior handlers below
    # are still in router. Patch both namespaces.
    monkeypatch.setattr(router_mod, "get_orchestrator", lambda: orchestrator)
    monkeypatch.setattr(graph_mod, "get_orchestrator", lambda: orchestrator)

    graph_payload = await graph_mod.get_feedback_graph()
    node = next(item for item in graph_payload["nodes"] if item["id"] == pattern_key)

    assert node["effective_weight_total"] == pytest.approx(2.25)
    assert node["passive_outcome_count"] == 1
    assert node["severity_correction_count"] == 1
    assert node["severity_calibration"]["targets"]["warning"] == pytest.approx(1.0)

    priors_payload = await router_mod.get_scorer_priors(limit=10)
    prior_item = next(item for item in priors_payload["priors"] if item["pattern"] == pattern_key)

    assert prior_item["effective_weight_total"] == pytest.approx(2.25)
    assert prior_item["passive_outcome_count"] == 1
    assert prior_item["severity_calibration"]["targets"]["warning"] == pytest.approx(1.0)

    prior_detail = await router_mod.get_scorer_prior(pattern_key=pattern_key)
    assert prior_detail["effective_weight_total"] == pytest.approx(2.25)
    assert prior_detail["passive_outcome_count"] == 1


def test_update_pattern_prior_syncs_backing_store(tmp_path):
    calls = []

    class SyncStore:
        def sync_pattern_prior(self, pattern_key: str, prior: float) -> None:
            calls.append((pattern_key, prior))

    scorer = SignificanceScorer(
        priors_path=str(tmp_path / "priors.json"),
        feedback_store=SyncStore(),
    )

    scorer.update_pattern_prior(
        "CUSTOM:observability",
        was_significant=True,
        weight=FEEDBACK_WEIGHTS["confirm_explicit"],
        source="confirm_explicit",
    )

    assert calls, "Expected scorer to mirror the recomputed prior into the backing store"
    pattern_key, prior = calls[-1]
    assert pattern_key == "CUSTOM:observability"
    assert prior == pytest.approx(scorer.get_pattern_prior("CUSTOM:observability"))


@pytest.mark.asyncio
async def test_update_pattern_prior_syncs_backing_store_from_async_context(tmp_path):
    calls = []

    class SyncStore:
        def sync_pattern_prior(self, pattern_key: str, prior: float) -> None:
            calls.append((pattern_key, prior))

    scorer = SignificanceScorer(
        priors_path=str(tmp_path / "priors.json"),
        feedback_store=SyncStore(),
    )

    scorer.update_pattern_prior(
        "CUSTOM:observability",
        was_significant=True,
        weight=FEEDBACK_WEIGHTS["confirm_explicit"],
        source="confirm_explicit",
    )

    for _ in range(20):
        if calls:
            break
        await asyncio.sleep(0.01)

    assert calls, "Expected async-context scorer updates to schedule backing-store sync"
    pattern_key, prior = calls[-1]
    assert pattern_key == "CUSTOM:observability"
    assert prior == pytest.approx(scorer.get_pattern_prior("CUSTOM:observability"))


def test_update_pattern_prior_falls_back_to_local_counts_when_store_query_fails(tmp_path):
    class FailingStore:
        def get_feedback_counts(self, *, pattern_key, context_key=None, user_id=None):
            raise RuntimeError("store unavailable")

    scorer = SignificanceScorer(
        priors_path=str(tmp_path / "priors.json"),
        feedback_store=FailingStore(),
    )

    scorer.update_pattern_prior(
        "CUSTOM:observability",
        was_significant=True,
        weight=FEEDBACK_WEIGHTS["confirm_explicit"],
        source="confirm_explicit",
    )

    assert scorer._local_feedback_counts["CUSTOM:observability"]["confirm"] == pytest.approx(
        FEEDBACK_WEIGHTS["confirm_explicit"]
    )
    assert scorer.get_pattern_prior("CUSTOM:observability") > 0.5
from types import SimpleNamespace

import pytest

from backend.agents.memory.feedback import FeedbackAction, MemoryFeedbackRequest, MemoryFeedbackResponse
from backend.agents.memory.memory_feedback_routes import (
    DocLinkFeedbackRequest,
    DocLinkFeedbackValue,
    add_memory_feedback,
    get_memory_feedback,
    set_doc_link_feedback,
)


@pytest.mark.asyncio
async def test_get_memory_feedback_prefers_store_events(monkeypatch):
    class DummyStore:
        def list_feedback_events(self, memory_id, limit):
            return [{"source": "store", "memory_id": memory_id, "limit": limit}]

    class DummyHandler:
        def get_feedback_history(self, memory_id):
            raise AssertionError("store-backed events should be used first")

        def get_feedback_stats(self, memory_id):
            return {"n_events": 1}

    orchestrator = SimpleNamespace(
        get_memory=lambda memory_id: SimpleNamespace(id=memory_id),
        store=DummyStore(),
        feedback_handler=DummyHandler(),
    )
    monkeypatch.setattr(
        "backend.agents.memory.memory_feedback_routes.get_orchestrator",
        lambda: orchestrator,
    )

    response = await get_memory_feedback("mem-1", limit=5)

    assert response.memory_id == "mem-1"
    assert response.events == [{"source": "store", "memory_id": "mem-1", "limit": 5}]
    assert response.stats == {"n_events": 1}


@pytest.mark.asyncio
async def test_add_memory_feedback_sets_cooldown_for_dismiss(monkeypatch):
    captured = {}

    class DummyHandler:
        async def process_feedback(self, memory_id, request):
            captured["memory_id"] = memory_id
            captured["action"] = request.action
            return MemoryFeedbackResponse(
                success=True,
                feedback_id="fb-1",
                memory_id=memory_id,
                action=request.action,
                message="ok",
            )

    class DummyDispatcher:
        def set_cooldown(self, session_id):
            captured["cooldown_session_id"] = session_id

    orchestrator = SimpleNamespace(
        get_memory=lambda memory_id: SimpleNamespace(id=memory_id, session_id="session-1"),
        feedback_handler=DummyHandler(),
        store=SimpleNamespace(),
    )
    monkeypatch.setattr(
        "backend.agents.memory.memory_feedback_routes.get_orchestrator",
        lambda: orchestrator,
    )
    monkeypatch.setattr(
        "backend.agents.memory.memory_feedback_routes.get_dispatcher",
        lambda: DummyDispatcher(),
    )

    response = await add_memory_feedback(
        "mem-1",
        MemoryFeedbackRequest(action=FeedbackAction.DISMISS, user_id="operator"),
    )

    assert response.feedback_id == "fb-1"
    assert captured["action"] == FeedbackAction.DISMISS
    assert captured["cooldown_session_id"] == "session-1"


@pytest.mark.asyncio
async def test_set_doc_link_feedback_updates_store_and_logs_feedback(monkeypatch):
    captured = {}

    class DummyStore:
        def set_doc_link_feedback(self, *, memory_id, doc_id, feedback, user_id, reason=None):
            captured["feedback_update"] = {
                "memory_id": memory_id,
                "doc_id": doc_id,
                "feedback": feedback,
                "user_id": user_id,
                "reason": reason,
            }
            return {
                "id": doc_id,
                "citation": "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1",
                "query_used": "regenerative chatter harmonic vibration tooth passing",
                "pattern_key": "fault:chatter",
                "doc_feedback": feedback,
                "helpful_count": 1,
                "not_helpful_count": 0,
                "feedback_score": 1.0,
                "evidence_entities": [],
            }

        def add_feedback_event(self, **kwargs):
            captured["feedback_event"] = kwargs
            return "fb-doc-1"

    orchestrator = SimpleNamespace(
        get_memory=lambda memory_id: SimpleNamespace(id=memory_id, session_id="session-1"),
        feedback_handler=SimpleNamespace(),
        store=DummyStore(),
    )
    monkeypatch.setattr(
        "backend.agents.memory.memory_feedback_routes.get_orchestrator",
        lambda: orchestrator,
    )

    response = await set_doc_link_feedback(
        "mem-1",
        "doc-1",
        DocLinkFeedbackRequest(
            feedback=DocLinkFeedbackValue.HELPFUL,
            user_id="operator",
            reason="Most actionable citation",
        ),
    )

    assert response.success is True
    assert response.doc_id == "doc-1"
    assert response.feedback == DocLinkFeedbackValue.HELPFUL
    assert response.doc_link["feedback_score"] == 1.0
    assert captured["feedback_update"]["feedback"] == "helpful"
    assert captured["feedback_event"]["action"] == "doc_link_helpful"
    assert captured["feedback_event"]["pattern_keys"] == ["fault:chatter"]
    assert captured["feedback_event"]["data"]["doc_id"] == "doc-1"
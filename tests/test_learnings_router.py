from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.events import publish_learning
from backend.ingestion.schema import LearningEnvelope
from backend.routers.learnings import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_learnings_ws_receives_global_learning_event() -> None:
    client = _client()
    envelope = LearningEnvelope(
        kind="tool_event",
        ts_unix=123.0,
        session_id="session-1",
        source="tool_audit",
        payload={"tool_number": 55, "action": "confirm"},
        batch={"batch_id": "batch-1", "unit_index": 0, "unit_count": 3},
    )

    with client.websocket_connect("/learnings/ws") as ws:
        asyncio.run(publish_learning(envelope))
        payload = ws.receive_json()

    assert payload["kind"] == "tool_event"
    assert payload["session_id"] == "session-1"
    assert payload["payload"]["tool_number"] == 55
    assert payload["payload"]["action"] == "confirm"
    assert payload["batch"] == {"batch_id": "batch-1", "unit_index": 0, "unit_count": 3}


def test_learnings_ws_receives_session_scoped_event() -> None:
    client = _client()
    envelope = LearningEnvelope(
        kind="feedback_event",
        ts_unix=456.0,
        session_id="session-2",
        source="feedback_loop",
        payload={"memory_id": "mem-7", "action": "dismiss"},
    )

    with client.websocket_connect("/learnings/ws/session-2") as ws:
        asyncio.run(publish_learning(envelope))
        payload = ws.receive_json()

    assert payload["kind"] == "feedback_event"
    assert payload["session_id"] == "session-2"
    assert payload["payload"]["memory_id"] == "mem-7"
    assert payload["payload"]["action"] == "dismiss"
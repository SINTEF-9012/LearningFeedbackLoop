"""Tests for backend.routers.feedback — Agent M wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.agents.memory.feedback_async import build_default_pipeline
from backend.routers.feedback import router


@pytest.fixture
def app_with_pipeline(tmp_path: Path) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.feedback_pipeline = build_default_pipeline(
        outbox_path=tmp_path / "outbox.jsonl"
    )
    return app


@pytest.fixture
def client(app_with_pipeline: FastAPI) -> TestClient:
    return TestClient(app_with_pipeline)


# ── HTTP endpoints ────────────────────────────────────────────────────


def test_operators_empty_when_no_feedback(client: TestClient) -> None:
    resp = client.get("/agent/memory/feedback/operators")
    assert resp.status_code == 200
    assert resp.json() == {"operators": []}


def test_operators_summary_after_callback(
    client: TestClient, app_with_pipeline: FastAPI
) -> None:
    import asyncio

    pipeline = app_with_pipeline.state.feedback_pipeline
    asyncio.run(pipeline.callback("m1", "confirm", {"user_id": "op-1"}))
    asyncio.run(pipeline.callback("m2", "confirm", {"user_id": "op-1"}))
    asyncio.run(pipeline.callback("m3", "dismiss", {"user_id": "op-2"}))

    resp = client.get("/agent/memory/feedback/operators")
    assert resp.status_code == 200
    body = resp.json()
    by_id = {op["operator_id"]: op for op in body["operators"]}
    assert by_id["op-1"]["total"] == 2
    assert by_id["op-1"]["actions"]["confirm"] == 2
    assert by_id["op-2"]["total"] == 1


def test_operator_events_most_recent_first(
    client: TestClient, app_with_pipeline: FastAPI
) -> None:
    import asyncio

    pipeline = app_with_pipeline.state.feedback_pipeline
    asyncio.run(pipeline.callback("m1", "confirm", {"user_id": "op-1"}))
    asyncio.run(pipeline.callback("m2", "dismiss", {"user_id": "op-1"}))

    resp = client.get("/agent/memory/feedback/operators/op-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["events"][0]["memory_id"] == "m2"
    assert body["events"][1]["memory_id"] == "m1"


def test_operator_events_unknown_operator_empty(client: TestClient) -> None:
    resp = client.get("/agent/memory/feedback/operators/nobody")
    assert resp.status_code == 200
    assert resp.json() == {"operator_id": "nobody", "count": 0, "events": []}


def test_operator_events_limit(client: TestClient, app_with_pipeline: FastAPI) -> None:
    import asyncio

    pipeline = app_with_pipeline.state.feedback_pipeline
    for i in range(5):
        asyncio.run(pipeline.callback(f"m{i}", "confirm", {"user_id": "op"}))
    resp = client.get("/agent/memory/feedback/operators/op?limit=2")
    assert resp.status_code == 200
    assert resp.json()["count"] == 5
    assert len(resp.json()["events"]) == 2


def test_outbox_status_pending_count(
    client: TestClient, app_with_pipeline: FastAPI
) -> None:
    import asyncio

    pipeline = app_with_pipeline.state.feedback_pipeline
    asyncio.run(pipeline.callback("m1", "confirm", {"user_id": "op"}))
    asyncio.run(pipeline.callback("m2", "confirm", {"user_id": "op"}))

    resp = client.get("/agent/memory/feedback/outbox")
    assert resp.status_code == 200
    assert resp.json() == {"pending": 2, "head": []}


def test_outbox_status_with_head(client: TestClient, app_with_pipeline: FastAPI) -> None:
    import asyncio

    pipeline = app_with_pipeline.state.feedback_pipeline
    asyncio.run(pipeline.callback("m1", "confirm", {"user_id": "op"}))
    asyncio.run(pipeline.callback("m2", "dismiss", {"user_id": "op"}))

    resp = client.get("/agent/memory/feedback/outbox?head=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["pending"] == 2
    assert [e["memory_id"] for e in body["head"]] == ["m1", "m2"]


def test_endpoints_503_without_pipeline() -> None:
    bare = FastAPI()
    bare.include_router(router)
    c = TestClient(bare)
    assert c.get("/agent/memory/feedback/operators").status_code == 503
    assert c.get("/agent/memory/feedback/operators/x").status_code == 503
    assert c.get("/agent/memory/feedback/outbox").status_code == 503


# ── WebSocket ─────────────────────────────────────────────────────────


def test_websocket_receives_broadcast(
    client: TestClient, app_with_pipeline: FastAPI
) -> None:
    import asyncio

    pipeline = app_with_pipeline.state.feedback_pipeline
    with client.websocket_connect("/agent/memory/feedback/ws") as ws:
        # Fire a feedback event; broadcaster fans out to the ws subscriber.
        asyncio.run(pipeline.callback("m1", "confirm", {"user_id": "op-1"}))
        payload = ws.receive_json()
        assert payload["memory_id"] == "m1"
        assert payload["operator_id"] == "op-1"
        assert payload["action"] == "confirm"


def test_websocket_closed_without_pipeline() -> None:
    from starlette.websockets import WebSocketDisconnect as StarletteWSDisconnect

    bare = FastAPI()
    bare.include_router(router)
    c = TestClient(bare)
    with pytest.raises(StarletteWSDisconnect):
        with c.websocket_connect("/agent/memory/feedback/ws"):
            pass

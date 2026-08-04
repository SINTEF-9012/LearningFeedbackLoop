from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.app as app_module
from backend.routers.graph_outbox import router


def test_graph_outbox_status_503_without_store(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)
    monkeypatch.setattr("backend.routers.graph_outbox.get_store", lambda: None)

    with TestClient(app) as client:
        resp = client.get("/agent/memory/graph-outbox")

    assert resp.status_code == 503
    with TestClient(app) as client:
        replay = client.post("/agent/memory/graph-outbox/replay")

    assert replay.status_code == 503


def test_graph_outbox_status_reports_disabled_for_non_neo4j_store(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)

    class _FakeStore:
        pass

    monkeypatch.setattr("backend.routers.graph_outbox.get_store", lambda: _FakeStore())

    with TestClient(app) as client:
        resp = client.get("/agent/memory/graph-outbox")

    assert resp.status_code == 200
    assert resp.json() == {
        "backend": "_fakestore",
        "enabled": False,
        "pending": 0,
        "head": [],
    }

    with TestClient(app) as client:
        replay = client.post("/agent/memory/graph-outbox/replay")

    assert replay.status_code == 200
    assert replay.json() == {
        "backend": "_fakestore",
        "enabled": False,
        "processed": 0,
        "pending_before": 0,
        "pending_after": 0,
    }


def test_graph_outbox_status_returns_pending_head(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)

    class _FakeNeo4jStore:
        def graph_outbox_enabled(self) -> bool:
            return True

        def graph_outbox_pending_count(self) -> int:
            return 2

        def list_pending_graph_writes(self, limit: int = 0):
            rows = [
                {
                    "kind": "trace",
                    "created_at": "2026-05-20T21:00:00+00:00",
                    "sequence": 1,
                    "payload": {"trace_type": "score"},
                },
                {
                    "kind": "doc_links",
                    "created_at": "2026-05-20T21:01:00+00:00",
                    "sequence": 2,
                    "payload": {"memory_id": "mem-1"},
                },
            ]
            return rows[:limit] if limit > 0 else rows

    monkeypatch.setattr("backend.routers.graph_outbox.get_store", lambda: _FakeNeo4jStore())

    with TestClient(app) as client:
        resp = client.get("/agent/memory/graph-outbox?head=5")

    assert resp.status_code == 200
    assert resp.json() == {
        "backend": "_fakeneo4jstore",
        "enabled": True,
        "pending": 2,
        "head": [
            {
                "kind": "trace",
                "created_at": "2026-05-20T21:00:00+00:00",
                "sequence": 1,
                "payload": {"trace_type": "score"},
            },
            {
                "kind": "doc_links",
                "created_at": "2026-05-20T21:01:00+00:00",
                "sequence": 2,
                "payload": {"memory_id": "mem-1"},
            },
        ],
    }


def test_graph_outbox_replay_reports_processed_counts(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)

    class _FakeNeo4jStore:
        def __init__(self) -> None:
            self.pending = 3

        def graph_outbox_enabled(self) -> bool:
            return True

        def graph_outbox_pending_count(self) -> int:
            return self.pending

        def replay_pending_graph_writes(self, limit: int = 0) -> int:
            processed = 2 if limit == 2 else self.pending
            self.pending = max(0, self.pending - processed)
            return processed

    store = _FakeNeo4jStore()
    monkeypatch.setattr("backend.routers.graph_outbox.get_store", lambda: store)

    with TestClient(app) as client:
        resp = client.post("/agent/memory/graph-outbox/replay?limit=2")

    assert resp.status_code == 200
    assert resp.json() == {
        "backend": "_fakeneo4jstore",
        "enabled": True,
        "processed": 2,
        "pending_before": 3,
        "pending_after": 1,
    }


def test_graph_outbox_route_precedes_memory_id_catchall() -> None:
    graph_outbox_index = None
    memory_detail_index = None

    for index, route in enumerate(app_module.app.router.routes):
        path = getattr(route, "path", None)
        if path == "/agent/memory/graph-outbox" and graph_outbox_index is None:
            graph_outbox_index = index
        if path == "/agent/memory/{memory_id}" and memory_detail_index is None:
            memory_detail_index = index

    assert graph_outbox_index is not None
    assert memory_detail_index is not None
    assert graph_outbox_index < memory_detail_index
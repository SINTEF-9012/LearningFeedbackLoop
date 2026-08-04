from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import sindit as sindit_router


def _session_payload() -> dict:
    return {
        "position": 1,
        "data": {"Tool_Number": [7]},
        "metadata": {
            "source": "simulated_casedata",
            "casedata": {
                "case_dir": "Case-1",
                "operation_id": "OP-1",
                "dataset_id": "dataset-1",
                "cutting_context": {
                    "tool_id": "T7",
                    "tool_diameter": 10.0,
                    "num_teeth": 4,
                    "spindle_speed": 1200.0,
                    "feed_rate": 150.0,
                    "extra": {"tool_number": 7},
                },
            },
        },
        "source_config": {
            "case_dir": "Case-1",
            "operation_id": "OP-1",
        },
    }


class _MutableFakeClient:
    def __init__(self, initial_nodes: dict[str, dict] | None = None) -> None:
        self.nodes = dict(initial_nodes or {})
        self.posted_assets: list[dict] = []
        self.updated_nodes: list[tuple[str, dict, bool]] = []

    async def get_node(self, node_uri: str, depth: int = 0):
        return self.nodes.get(node_uri)

    async def post_asset(self, payload: dict):
        self.posted_assets.append(payload)
        self.nodes[payload["uri"]] = {
            "uri": payload["uri"],
            "label": payload.get("label"),
            "assetDescription": payload.get("assetDescription"),
        }
        return {"ok": True}

    async def update_node(self, node_uri: str, fields: dict, *, overwrite: bool = True):
        self.updated_nodes.append((node_uri, fields, overwrite))
        current = {} if overwrite else dict(self.nodes.get(node_uri) or {})
        current.update(fields)
        current["uri"] = node_uri
        self.nodes[node_uri] = current
        return {"ok": True}

    async def close(self) -> None:
        return None


def test_runtime_reconciliation_reports_aligned_runtime_identity(monkeypatch) -> None:
    app = FastAPI()
    app.state.sessions = {"session-1": _session_payload()}
    app.include_router(sindit_router.router)

    class _FakeStore:
        def get_runtime_identity_snapshot(self, *, operation_node_id=None, dataset_id=None):
            return {
                "operation_node": {
                    "id": operation_node_id,
                    "operation_id": "OP-1",
                    "dataset_id": dataset_id,
                },
                "dataset_node": {
                    "id": dataset_id,
                    "source_dataset_id": "site_a_line2",
                },
            }

    class _FakeClient:
        async def get_node(self, node_uri: str, depth: int = 0):
            assert node_uri == "urn:lfl:operation:op-1"
            return {
                "uri": node_uri,
                "metadata": {"active": True, "lastSeenAt": "2026-05-20T22:00:00+00:00"},
            }

        async def close(self) -> None:
            return None

    async def fake_client_factory():
        return _FakeClient(), True, None

    monkeypatch.setattr("backend.routers.sindit.get_store", lambda: _FakeStore())
    monkeypatch.setattr(sindit_router, "_maybe_authenticated_sindit_client", fake_client_factory)

    with TestClient(app) as client:
        resp = client.get("/sindit/reconciliation/runtime")

    assert resp.status_code == 200
    body = resp.json()
    assert body["neo4j_available"] is True
    assert body["sindit_available"] is True
    assert body["sessions_with_issues"] == 0
    row = body["sessions"][0]
    assert row["expected"]["operation_uri"] == "urn:lfl:operation:op-1"
    assert row["neo4j"]["operation_present"] is True
    assert row["neo4j"]["dataset_present"] is True
    assert row["sindit"]["operation_present"] is True
    assert row["sindit"]["operation_active"] is True
    assert row["issues"] == []


def test_runtime_reconciliation_reports_missing_backends_as_issues(monkeypatch) -> None:
    app = FastAPI()
    app.state.sessions = {"session-1": _session_payload()}
    app.include_router(sindit_router.router)

    async def fake_client_factory():
        return None, False, "authentication failed"

    monkeypatch.setattr("backend.routers.sindit.get_store", lambda: None)
    monkeypatch.setattr(sindit_router, "_maybe_authenticated_sindit_client", fake_client_factory)

    with TestClient(app) as client:
        resp = client.get("/sindit/reconciliation/runtime")

    assert resp.status_code == 200
    body = resp.json()
    assert body["neo4j_available"] is False
    assert body["sindit_available"] is False
    assert body["sindit_detail"] == "authentication failed"
    row = body["sessions"][0]
    assert row["issues"] == ["neo4j_unavailable", "sindit_unavailable"]


def test_runtime_operation_ensure_creates_missing_runtime_node(monkeypatch) -> None:
    app = FastAPI()
    app.state.sessions = {"session-1": _session_payload()}
    app.include_router(sindit_router.router)

    class _FakeStore:
        def get_runtime_identity_snapshot(self, *, operation_node_id=None, dataset_id=None):
            return {
                "operation_node": {
                    "id": operation_node_id,
                    "operation_id": "OP-1",
                    "dataset_id": dataset_id,
                },
                "dataset_node": {"id": dataset_id},
            }

    client_impl = _MutableFakeClient()

    async def fake_client_factory():
        return client_impl, True, None

    monkeypatch.setattr("backend.routers.sindit.get_store", lambda: _FakeStore())
    monkeypatch.setattr(sindit_router, "_maybe_authenticated_sindit_client", fake_client_factory)

    with TestClient(app) as client:
        resp = client.post("/sindit/reconciliation/runtime/operation/ensure")

    assert resp.status_code == 200
    body = resp.json()
    assert body["repaired"] == 1
    row = body["sessions"][0]
    assert row["action"] == "created"
    assert row["errors"] == []
    assert row["reconciliation"]["issues"] == []
    assert client_impl.posted_assets[0]["uri"] == "urn:lfl:operation:op-1"
    update_uri, update_fields, overwrite = client_impl.updated_nodes[0]
    assert update_uri == "urn:lfl:operation:op-1"
    assert overwrite is False
    assert update_fields["active"] is True


def test_runtime_operation_ensure_reactivates_inactive_runtime_node(monkeypatch) -> None:
    app = FastAPI()
    app.state.sessions = {"session-1": _session_payload()}
    app.include_router(sindit_router.router)

    class _FakeStore:
        def get_runtime_identity_snapshot(self, *, operation_node_id=None, dataset_id=None):
            return {
                "operation_node": {
                    "id": operation_node_id,
                    "operation_id": "OP-1",
                    "dataset_id": dataset_id,
                },
                "dataset_node": {"id": dataset_id},
            }

    client_impl = _MutableFakeClient(
        {
            "urn:lfl:operation:op-1": {
                "uri": "urn:lfl:operation:op-1",
                "label": "Operation OP-1",
                "active": False,
            }
        }
    )

    async def fake_client_factory():
        return client_impl, True, None

    monkeypatch.setattr("backend.routers.sindit.get_store", lambda: _FakeStore())
    monkeypatch.setattr(sindit_router, "_maybe_authenticated_sindit_client", fake_client_factory)

    with TestClient(app) as client:
        resp = client.post("/sindit/reconciliation/runtime/operation/ensure")

    assert resp.status_code == 200
    body = resp.json()
    assert body["repaired"] == 1
    row = body["sessions"][0]
    assert row["action"] == "reactivated"
    assert row["errors"] == []
    assert row["reconciliation"]["issues"] == []
    assert client_impl.posted_assets == []
    update_uri, update_fields, overwrite = client_impl.updated_nodes[0]
    assert update_uri == "urn:lfl:operation:op-1"
    assert overwrite is False
    assert update_fields["active"] is True
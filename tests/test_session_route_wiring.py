from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from backend.agents.memory.cycle_tracker import CycleEnded
from backend.app import app
from backend.routers import sessions as sessions_router


def test_session_http_routes_are_registered_once_from_sessions_router():
    expected_routes = {
        ("/sessions", "GET"),
        ("/sessions", "POST"),
        ("/sessions/start-demo", "POST"),
        ("/sessions/{session_id}", "GET"),
        ("/sessions/{session_id}", "DELETE"),
        ("/sessions/{session_id}/upload", "POST"),
        ("/sessions/{session_id}/start", "POST"),
        ("/sessions/{session_id}/pause", "POST"),
        ("/sessions/{session_id}/resume", "POST"),
        ("/sessions/{session_id}/replay", "POST"),
        ("/sessions/{session_id}/download", "GET"),
        ("/sessions/{session_id}/metadata", "GET"),
        ("/sessions/{session_id}/source", "GET"),
        ("/sessions/{session_id}/config", "PATCH"),
        ("/sessions/{session_id}/playback", "POST"),
    }

    for path, method in expected_routes:
        routes = [
            route
            for route in app.routes
            if isinstance(route, APIRoute)
            and route.path == path
            and method in route.methods
        ]

        assert len(routes) == 1, (path, method, [route.endpoint.__module__ for route in routes])
        assert routes[0].endpoint.__module__ == "backend.routers.sessions"


def test_analysis_routes_are_registered_once_from_analysis_router():
    expected_routes = {
        ("/sessions/{session_id}/analyze", "GET"),
        ("/sessions/{session_id}/analyze", "POST"),
    }

    for path, method in expected_routes:
        routes = [
            route
            for route in app.routes
            if isinstance(route, APIRoute)
            and route.path == path
            and method in route.methods
        ]

        assert len(routes) == 1, (path, method, [route.endpoint.__module__ for route in routes])
        assert routes[0].endpoint.__module__ == "backend.routers.analysis"


def test_sessions_list_route_returns_status_summaries():
    app.state.sessions.clear()
    app.state.sessions["session-1"] = {
        "session_id": "session-1",
        "config": {"channels": ["A"]},
        "data": {"A": [1.0, 2.0, 3.0]},
        "metadata": {
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
            "casedata": {
                "case_dir": "Site_b - MACHINE_B1 - CASE_B1",
                "operation_id": "OF00001",
            },
        },
        "running": False,
        "paused": False,
        "position": 2,
        "source_name": "simulated_casedata",
        "source_config": {
            "case_dir": "Site_b - MACHINE_B1 - CASE_B1",
            "operation_id": "OF00001",
        },
        "subscribers": [],
        "task": None,
    }

    try:
        route = next(
            route
            for route in app.routes
            if isinstance(route, APIRoute)
            and route.path == "/sessions"
            and "GET" in route.methods
        )
        payload = route.endpoint(request=type("Req", (), {"app": app})())
    finally:
        app.state.sessions.clear()

    assert payload["sessions"] == ["session-1"]
    assert payload["session_summaries"][0]["session_id"] == "session-1"
    assert payload["session_summaries"][0]["status"] == "stopped"
    assert payload["session_summaries"][0]["status_label"] == "Stopped"
    assert payload["session_summaries"][0]["position"] == 2
    assert payload["session_summaries"][0]["total_samples"] == 3
    assert payload["session_summaries"][0]["source"] == "simulated_casedata"
    assert payload["session_summaries"][0]["source_label"] == "Site_b - MACHINE_B1 - CASE_B1 / OF00001"


def test_delete_session_flushes_active_cycle_on_shutdown(monkeypatch):
    client = TestClient(app)
    app.state.sessions.clear()
    app.state.sessions["session-flush"] = {
        "session_id": "session-flush",
        "subscribers": [],
        "fft_subscribers": [],
        "inference_subscribers": [],
        "task": None,
        "fft_task": None,
        "inference_task": None,
        "startup_task": None,
    }

    flushed_cycle = CycleEnded(
        session_id="session-flush",
        part_id="part-2",
        operation_id="op-8",
        started_at=10.0,
        ended_at=14.0,
    )
    attach = AsyncMock(return_value=2)

    class FakeTracker:
        def flush_session(self, session_id: str):
            assert session_id == "session-flush"
            return flushed_cycle

    monkeypatch.setattr(sessions_router, "get_cycle_tracker", lambda: FakeTracker())
    monkeypatch.setattr(sessions_router, "is_memory_initialized", lambda: True)
    monkeypatch.setattr(
        sessions_router,
        "get_orchestrator",
        lambda: SimpleNamespace(attach_passive_cycle_outcome=attach),
    )

    try:
        response = client.delete("/sessions/session-flush")
    finally:
        app.state.sessions.clear()

    assert response.status_code == 200
    assert response.json()["deleted"] == "session-flush"
    assert response.json()["flushed_cycle"] is True
    assert response.json()["passive_feedback_count"] == 2
    attach.assert_awaited_once_with(flushed_cycle)
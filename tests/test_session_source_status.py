from fastapi.testclient import TestClient

from backend.app import app


def test_session_source_status_endpoint_returns_source_metadata():
    client = TestClient(app)
    app.state.sessions.clear()
    app.state.sessions["source-session"] = {
        "session_id": "source-session",
        "config": {},
        "data": {},
        "metadata": {},
        "running": True,
        "paused": False,
        "position": 12,
        "source_name": "simulated_file",
        "source_status": {
            "kind": "simulated_file",
            "connected": True,
            "last_frame_ts": 123.45,
            "lag_ms": 6.0,
            "dropped": 0,
        },
        "subscribers": [],
        "task": None,
    }

    try:
        response = client.get("/sessions/source-session/source")
    finally:
        app.state.sessions.clear()

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "source-session",
        "kind": "simulated_file",
        "running": True,
        "paused": False,
        "position": 12,
        "connected": True,
        "last_frame_ts": 123.45,
        "lag_ms": 6.0,
        "dropped": 0,
    }
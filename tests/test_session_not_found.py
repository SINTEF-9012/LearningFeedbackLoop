import pytest
from fastapi.testclient import TestClient

from backend.app import app


@pytest.fixture(autouse=True)
def _clear_sessions():
    # Ensure clean state between tests
    app.state.sessions.clear()
    yield
    app.state.sessions.clear()


def test_unknown_session_endpoints_return_404():
    client = TestClient(app)
    sid = "does-not-exist"

    cases = [
        ("GET", f"/sessions/{sid}", None),
        ("GET", f"/sessions/{sid}/metadata", None),
        ("GET", f"/sessions/{sid}/download", None),
        ("POST", f"/sessions/{sid}/start", None),
        ("POST", f"/sessions/{sid}/pause", None),
        ("POST", f"/sessions/{sid}/resume", None),
        # Query-param endpoints
        ("GET", f"/sessions/{sid}/analyze", {"channel": "A", "start": 0, "end": 1}),
        ("POST", f"/sessions/{sid}/analyze", {"channel": "A", "start": 0, "end": 1}),
    ]

    for method, url, params in cases:
        resp = client.request(method, url, params=params)
        assert resp.status_code == 404, (method, url, resp.status_code, resp.text)
        detail = (resp.json() or {}).get("detail")
        assert isinstance(detail, str)
        assert "session" in detail.lower() and "not found" in detail.lower()

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app import app


@pytest.fixture(autouse=True)
def _clear_sessions():
    app.state.sessions.clear()
    yield
    app.state.sessions.clear()


@pytest.mark.parametrize(
    "path",
    [
        "/streams/does-not-exist",
        "/agent/memory/alerts/does-not-exist",
        "/sessions/does-not-exist/inference",
    ],
)
def test_missing_session_websockets_close_with_4404(path: str):
    client = TestClient(app)

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(path) as ws:
            ws.receive_text()

    assert exc_info.value.code == 4404
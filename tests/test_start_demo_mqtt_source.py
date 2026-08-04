from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from backend.app import app
from backend.ingestion.mqtt_source import MqttStreamSource
from backend.mqtt_transport import MqttMessage
from backend.routers import sessions as sessions_router


class FakeMqttClient:
    def __init__(self, messages: list[MqttMessage]):
        self._messages = list(messages)
        self._handler = None

    def set_message_handler(self, handler) -> None:
        self._handler = handler

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def subscribe(self, topic: str, qos: int = 0) -> None:
        for message in self._messages:
            if self._handler is not None:
                self._handler(message)

    async def publish(self, topic: str, payload: bytes, *, qos: int = 0, retain: bool = False) -> None:
        raise AssertionError("publish should not be called")


def test_start_demo_can_launch_mqtt_source(monkeypatch):
    fake_client = FakeMqttClient(
        [
            MqttMessage(
                topic="machine/live",
                payload=json.dumps(
                    {
                        "timestamp": "2026-05-07T12:00:00Z",
                        "Power_Spindle": 10.0,
                        "Feed_Rate_Actual": 120.0,
                    }
                ).encode("utf-8"),
            )
        ]
    )
    monkeypatch.setattr(MqttStreamSource, "_make_client", lambda self, session_id: fake_client)
    monkeypatch.setattr(sessions_router, "ensure_mqtt_transport_available", lambda: None)

    client = TestClient(app)
    app.state.sessions.clear()

    try:
        response = client.post(
            "/sessions/start-demo",
            json={
                "source": "mqtt",
                "topic": "machine/live",
                "broker_host": "broker.local",
                "broker_port": 1883,
                "sample_frequency": 1.0,
                "username": "mqtt-user",
                "password": "secret",
                "speed": 1.0,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["source"] == "mqtt"
        assert payload["n_events"] == 0

        session_id = payload["session_id"]
        time.sleep(0.1)
        session = app.state.sessions[session_id]
        assert session["source_name"] == "mqtt"
        assert session["source_config"]["topic"] == "machine/live"
        assert session["config"]["speed"] == 1.0
        assert session["metadata"]["sample_frequency"] == 1.0
        assert session["metadata"]["mqtt"]["broker_host"] == "broker.local"

        source_resp = client.get(f"/sessions/{session_id}/source")
        assert source_resp.status_code == 200
        assert source_resp.json()["kind"] == "mqtt"
        assert source_resp.json()["topic"] == "machine/live"
        assert source_resp.json()["broker_host"] == "broker.local"
        assert source_resp.json()["broker_port"] == 1883
        assert source_resp.json()["username"] == "mqtt-user"
        assert source_resp.json()["password_configured"] is True
    finally:
        for session_id in list(app.state.sessions.keys()):
            client.delete(f"/sessions/{session_id}")
        app.state.sessions.clear()


def test_start_demo_rejects_mqtt_when_transport_dependency_is_unavailable(monkeypatch):
    def fake_ensure_transport() -> None:
        raise RuntimeError("paho-mqtt is required for MQTT transport")

    monkeypatch.setattr(sessions_router, "ensure_mqtt_transport_available", fake_ensure_transport)

    client = TestClient(app)
    app.state.sessions.clear()

    try:
        response = client.post(
            "/sessions/start-demo",
            json={
                "source": "mqtt",
                "topic": "machine/live",
                "broker_host": "broker.local",
                "broker_port": 1883,
            },
        )
        assert response.status_code == 400
        assert "paho-mqtt is required for MQTT transport" in response.json()["detail"]
        assert app.state.sessions == {}
    finally:
        app.state.sessions.clear()
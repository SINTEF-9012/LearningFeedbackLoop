from __future__ import annotations

import asyncio
import json

import pytest

from backend.ingestion.mqtt_source import MqttStreamSource
from backend.ingestion.registry import create_source, registered_sources
from backend.ingestion.schema import FrameEnvelope
from backend.mqtt_transport import MqttMessage
from backend.session_active_context import build_active_session_context


class FakeMqttClient:
    def __init__(self, messages: list[MqttMessage]):
        self._messages = list(messages)
        self._handler = None
        self.connected = False
        self.disconnected = False
        self.subscriptions: list[tuple[str, int]] = []

    def set_message_handler(self, handler) -> None:
        self._handler = handler

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True

    async def subscribe(self, topic: str, qos: int = 0) -> None:
        self.subscriptions.append((topic, qos))
        for message in self._messages:
            if self._handler is not None:
                self._handler(message)

    async def publish(self, topic: str, payload: bytes, *, qos: int = 0, retain: bool = False) -> None:
        raise AssertionError("publish should not be called by the MQTT source")


@pytest.mark.asyncio
async def test_mqtt_source_emits_frames_updates_session_buffers_and_publishes_envelopes(monkeypatch):
    published: list[tuple[str, FrameEnvelope]] = []

    async def fake_publish(session_id: str, payload: FrameEnvelope) -> None:
        published.append((session_id, payload))

    monkeypatch.setattr("backend.ingestion.mqtt_source.publish_feature", fake_publish)

    queue: asyncio.Queue = asyncio.Queue()
    sessions = {
        "session-mqtt": {
            "session_id": "session-mqtt",
            "config": {"speed": 1.0},
            "data": {},
            "metadata": {},
            "running": True,
            "paused": False,
            "position": 0,
            "subscribers": [queue],
            "task": None,
        }
    }

    messages = [
        MqttMessage(
            topic="machine/live",
            payload=json.dumps(
                {
                    "timestamp": "2026-05-07T12:00:00Z",
                    "source": "site_a_line2",
                    "machine_id": "site_a-01",
                    "tool_id": "T12",
                    "operation_id": "OF00013",
                    "dataset_id": "site_a_line2",
                    "Power_Spindle": 10.0,
                    "Feed_Rate_Actual": 120.0,
                }
            ).encode("utf-8"),
        ),
        MqttMessage(
            topic="machine/live",
            payload=json.dumps(
                {
                    "ts_unix": 1_778_152_801.0,
                    "signals": {
                        "Power_Spindle": 11.0,
                        "Feed_Rate_Actual": 121.0,
                    },
                }
            ).encode("utf-8"),
        ),
    ]

    fake_client = FakeMqttClient(messages)
    source = MqttStreamSource(
        sessions,
        topic="machine/live",
        broker_host="broker.local",
        broker_port=1883,
        username="mqtt-user",
        password="secret",
        client_factory=lambda *args, **kwargs: fake_client,
    )

    task = asyncio.create_task(source.run("session-mqtt"))
    first = await asyncio.wait_for(queue.get(), timeout=1.0)
    second = await asyncio.wait_for(queue.get(), timeout=1.0)
    sessions["session-mqtt"]["running"] = False
    await asyncio.wait_for(task, timeout=1.0)
    eos = await asyncio.wait_for(queue.get(), timeout=1.0)

    assert first["Power_Spindle"] == 10.0
    assert first["ts_unix"] == 1_778_155_200.0
    assert second["Feed_Rate_Actual"] == 121.0
    assert eos["eos"] is True
    assert sessions["session-mqtt"]["position"] == 2
    assert sessions["session-mqtt"]["data"]["Power_Spindle"] == [10.0, 11.0]
    assert sessions["session-mqtt"]["metadata"]["sample_frequency"] == 1.0
    assert len(published) == 2
    assert all(isinstance(item[1], FrameEnvelope) for item in published)
    assert published[0][1].signals["Power_Spindle"] == 10.0
    assert published[0][1].frame is not None
    assert published[0][1].frame["ts_unix"] == 1_778_155_200.0
    assert published[0][1].metadata["machine_id"] == "site_a-01"
    assert published[0][1].metadata["tool_id"] == "T12"
    assert sessions["session-mqtt"]["metadata"]["machine_id"] == "site_a-01"
    assert sessions["session-mqtt"]["metadata"]["tool_id"] == "T12"
    assert sessions["session-mqtt"]["metadata"]["source"] == "site_a_line2"
    active_context = build_active_session_context(sessions["session-mqtt"])
    assert active_context is not None
    assert active_context["machine_id"] == "site_a-01"
    assert active_context["tool_number"] == 12
    assert active_context["tool_id"] == "T12"
    status = source.status("session-mqtt")
    assert status["topic"] == "machine/live"
    assert status["broker_host"] == "broker.local"
    assert status["broker_port"] == 1883
    assert status["username"] == "mqtt-user"
    assert status["password_configured"] is True
    assert fake_client.connected is True
    assert fake_client.disconnected is True


def test_source_registry_exposes_mqtt_source():
    assert "mqtt" in registered_sources()

    source = create_source(
        "mqtt",
        {},
        topic="machine/live",
        broker_host="broker.local",
        broker_port=1883,
        client_factory=lambda *args, **kwargs: FakeMqttClient([]),
    )

    assert isinstance(source, MqttStreamSource)


@pytest.mark.asyncio
async def test_mqtt_source_coerces_numeric_strings_from_frame_payload(monkeypatch):
    async def fake_publish(_session_id: str, _payload: FrameEnvelope) -> None:
        return None

    monkeypatch.setattr("backend.ingestion.mqtt_source.publish_feature", fake_publish)

    queue: asyncio.Queue = asyncio.Queue()
    sessions = {
        "session-mqtt": {
            "session_id": "session-mqtt",
            "config": {"speed": 1.0},
            "data": {},
            "metadata": {},
            "running": True,
            "paused": False,
            "position": 0,
            "subscribers": [queue],
            "task": None,
        }
    }

    messages = [
        MqttMessage(
            topic="machine/live",
            payload=json.dumps(
                {
                    "timestamp": "2026-05-07T12:00:02Z",
                    "frame": {
                        "Power_Spindle": "12.5",
                        "Feed_Rate_Actual": "140.0",
                        "status": "OK",
                    },
                }
            ).encode("utf-8"),
        ),
    ]

    fake_client = FakeMqttClient(messages)
    source = MqttStreamSource(
        sessions,
        topic="machine/live",
        broker_host="broker.local",
        broker_port=1883,
        client_factory=lambda *args, **kwargs: fake_client,
    )

    task = asyncio.create_task(source.run("session-mqtt"))
    frame = await asyncio.wait_for(queue.get(), timeout=1.0)
    sessions["session-mqtt"]["running"] = False
    await asyncio.wait_for(task, timeout=1.0)

    assert frame["Power_Spindle"] == 12.5
    assert frame["Feed_Rate_Actual"] == 140.0
    assert sessions["session-mqtt"]["data"]["Power_Spindle"] == [12.5]
    assert sessions["session-mqtt"]["data"]["Feed_Rate_Actual"] == [140.0]
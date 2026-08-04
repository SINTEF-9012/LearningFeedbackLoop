from __future__ import annotations

import asyncio
import json

import pytest

from backend.events import publish_learning
from backend.ingestion.schema import LearningEnvelope
from backend.mqtt_learning_publisher import MqttLearningPublisher


class FakeMqttClient:
    def __init__(self):
        self.connected = False
        self.disconnected = False
        self.published: list[tuple[str, bytes, int, bool]] = []

    def set_message_handler(self, handler) -> None:
        return None

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True

    async def subscribe(self, topic: str, qos: int = 0) -> None:
        raise AssertionError("subscribe should not be called by the learning publisher")

    async def publish(self, topic: str, payload: bytes, *, qos: int = 0, retain: bool = False) -> None:
        self.published.append((topic, payload, qos, retain))


@pytest.mark.asyncio
async def test_mqtt_learning_publisher_publishes_learning_bus_messages():
    fake_client = FakeMqttClient()
    publisher = MqttLearningPublisher(
        topic="machine/learnings",
        broker_host="broker.local",
        broker_port=1883,
        client_factory=lambda *args, **kwargs: fake_client,
    )

    publisher.start()
    deadline = asyncio.get_running_loop().time() + 1.0
    while not fake_client.connected and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)

    await publish_learning(
        LearningEnvelope(
            kind="feedback_event",
            ts_unix=1_778_152_800.0,
            session_id="session-1",
            source="feedback_loop",
            payload={"memory_id": "mem-1", "action": "confirm"},
        )
    )

    deadline = asyncio.get_running_loop().time() + 1.0
    while not fake_client.published and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)

    status = publisher.status()

    await publisher.stop()

    assert fake_client.connected is True
    assert fake_client.disconnected is True
    assert len(fake_client.published) == 1
    topic, payload, qos, retain = fake_client.published[0]
    assert topic == "machine/learnings"
    assert qos == 0
    assert retain is False
    body = json.loads(payload.decode("utf-8"))
    assert body["kind"] == "feedback_event"
    assert body["session_id"] == "session-1"
    assert body["payload"]["memory_id"] == "mem-1"
    assert status["state"] == "connected"
    assert status["published_count"] == 1
    assert status["last_published_at"] is not None
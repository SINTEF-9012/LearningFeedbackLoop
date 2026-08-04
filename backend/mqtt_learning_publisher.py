"""Background MQTT publisher for learning events."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from .events import bus, subscribe_learnings
from .ingestion.schema import LearningEnvelope, envelope_to_dict
from .mqtt_transport import AsyncMqttClient, create_mqtt_client


logger = logging.getLogger(__name__)


def _serialize_learning(payload: Any) -> dict[str, Any]:
    if isinstance(payload, LearningEnvelope):
        return envelope_to_dict(payload)
    if hasattr(payload, "__dataclass_fields__"):
        return envelope_to_dict(payload)
    if isinstance(payload, dict):
        return dict(payload)
    return {"payload": str(payload)}


class MqttLearningPublisher:
    def __init__(
        self,
        *,
        topic: str,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        qos: int = 0,
        retain: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        connect_timeout_s: float = 5.0,
        client_factory: Any = create_mqtt_client,
    ):
        if not topic:
            raise ValueError("MQTT learnings topic is required")
        self._topic = topic
        self._broker_host = broker_host
        self._broker_port = int(broker_port)
        self._qos = int(qos)
        self._retain = bool(retain)
        self._username = username
        self._password = password
        self._connect_timeout_s = float(connect_timeout_s)
        self._client_factory = client_factory
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._queue: Optional[asyncio.Queue] = None
        self._client: Optional[AsyncMqttClient] = None
        self._state = "idle"
        self._last_error: Optional[str] = None
        self._connected_at: Optional[float] = None
        self._last_published_at: Optional[float] = None
        self._published_count = 0

    def status(self) -> dict[str, Any]:
        task_active = self._task is not None and not self._task.done()
        return {
            "topic": self._topic,
            "broker_host": self._broker_host,
            "broker_port": self._broker_port,
            "qos": self._qos,
            "retain": self._retain,
            "state": self._state,
            "task_active": task_active,
            "connected_at": self._connected_at,
            "last_published_at": self._last_published_at,
            "published_count": self._published_count,
            "last_error": self._last_error,
        }

    def start(self) -> asyncio.Task:
        if self._task is not None and not self._task.done():
            return self._task
        self._running = True
        self._state = "starting"
        self._last_error = None
        self._task = asyncio.create_task(self.run())
        return self._task

    async def stop(self) -> None:
        self._running = False
        task = self._task
        if task is None:
            return
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def run(self) -> None:
        queue = subscribe_learnings()
        self._queue = queue
        self._state = "connecting"
        client = self._client_factory(
            self._broker_host,
            self._broker_port,
            client_id="lfl-learnings",
            username=self._username,
            password=self._password,
            connect_timeout_s=self._connect_timeout_s,
        )
        self._client = client

        try:
            await client.connect()
            self._state = "connected"
            self._connected_at = time.time()
            self._last_error = None
            while self._running:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                message = json.dumps(_serialize_learning(payload), default=str).encode("utf-8")
                await client.publish(
                    self._topic,
                    message,
                    qos=self._qos,
                    retain=self._retain,
                )
                self._published_count += 1
                self._last_published_at = time.time()
        except asyncio.CancelledError:
            self._state = "stopping"
            raise
        except Exception as exc:
            self._last_error = str(exc)
            self._state = "error"
            logger.exception("MQTT learning publisher failed")
        finally:
            bus.unsubscribe("learnings", queue)
            if self._client is not None:
                try:
                    await self._client.disconnect()
                except Exception:
                    logger.debug("MQTT learning publisher disconnect failed", exc_info=True)
            self._client = None
            self._queue = None
            self._running = False
            self._task = None
            if self._state != "error":
                self._state = "stopped"
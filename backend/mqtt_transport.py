"""Async-friendly MQTT transport helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Protocol

try:
    import paho.mqtt.client as paho_mqtt
except Exception:  # pragma: no cover - optional dependency
    paho_mqtt = None


def ensure_mqtt_transport_available() -> None:
    if paho_mqtt is None:
        raise RuntimeError("paho-mqtt is required for MQTT transport")


MessageHandler = Callable[["MqttMessage"], Awaitable[None] | None]


@dataclass
class MqttMessage:
    topic: str
    payload: bytes
    qos: int = 0


class AsyncMqttClient(Protocol):
    def set_message_handler(self, handler: MessageHandler) -> None:
        ...

    async def connect(self) -> None:
        ...

    async def disconnect(self) -> None:
        ...

    async def subscribe(self, topic: str, qos: int = 0) -> None:
        ...

    async def publish(
        self,
        topic: str,
        payload: bytes,
        *,
        qos: int = 0,
        retain: bool = False,
    ) -> None:
        ...


class PahoAsyncMqttClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        client_id: str = "",
        username: Optional[str] = None,
        password: Optional[str] = None,
        keepalive: int = 60,
        connect_timeout_s: float = 5.0,
    ):
        ensure_mqtt_transport_available()
        self._host = host
        self._port = port
        self._client_id = client_id
        self._username = username
        self._password = password
        self._keepalive = keepalive
        self._connect_timeout_s = connect_timeout_s
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connect_waiter: Optional[asyncio.Future[None]] = None
        self._handler: Optional[MessageHandler] = None
        self._client = paho_mqtt.Client(client_id=self._client_id or "")
        if username:
            self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def set_message_handler(self, handler: MessageHandler) -> None:
        self._handler = handler

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._connect_waiter = self._loop.create_future()
        self._client.connect(self._host, self._port, self._keepalive)
        self._client.loop_start()
        await asyncio.wait_for(self._connect_waiter, timeout=self._connect_timeout_s)

    async def disconnect(self) -> None:
        try:
            self._client.disconnect()
        finally:
            self._client.loop_stop()

    async def subscribe(self, topic: str, qos: int = 0) -> None:
        result, _mid = self._client.subscribe(topic, qos=qos)
        if result != 0:
            raise RuntimeError(f"MQTT subscribe failed for {topic!r}: rc={result}")

    async def publish(
        self,
        topic: str,
        payload: bytes,
        *,
        qos: int = 0,
        retain: bool = False,
    ) -> None:
        info = self._client.publish(topic, payload, qos=qos, retain=retain)
        if info.rc != 0:
            raise RuntimeError(f"MQTT publish failed for {topic!r}: rc={info.rc}")

    def _on_connect(self, _client, _userdata, _flags, reason_code, _properties=None) -> None:
        if self._loop is None or self._connect_waiter is None or self._connect_waiter.done():
            return
        if int(reason_code) == 0:
            self._loop.call_soon_threadsafe(self._connect_waiter.set_result, None)
            return
        err = RuntimeError(f"MQTT connection failed: rc={int(reason_code)}")
        self._loop.call_soon_threadsafe(self._connect_waiter.set_exception, err)

    def _on_message(self, _client, _userdata, msg) -> None:
        if self._loop is None or self._handler is None:
            return
        message = MqttMessage(topic=msg.topic, payload=bytes(msg.payload), qos=int(getattr(msg, "qos", 0)))

        def _deliver() -> None:
            result = self._handler(message)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)

        self._loop.call_soon_threadsafe(_deliver)


def create_mqtt_client(
    host: str,
    port: int,
    *,
    client_id: str = "",
    username: Optional[str] = None,
    password: Optional[str] = None,
    keepalive: int = 60,
    connect_timeout_s: float = 5.0,
) -> AsyncMqttClient:
    return PahoAsyncMqttClient(
        host,
        port,
        client_id=client_id,
        username=username,
        password=password,
        keepalive=keepalive,
        connect_timeout_s=connect_timeout_s,
    )
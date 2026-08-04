"""Registry-backed live MQTT source."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.events import publish_feature
from backend.ingestion.schema import FrameEnvelope
from backend.mqtt_transport import AsyncMqttClient, MqttMessage, create_mqtt_client


logger = logging.getLogger(__name__)

_RESERVED_KEYS = {
    "kind",
    "session_id",
    "timestamp",
    "ts",
    "ts_unix",
    "position",
    "fs",
    "signals",
    "metadata",
    "source",
    "frame",
}

_TOP_LEVEL_METADATA_KEYS = {
    "asset_id",
    "asset_iri",
    "case_dir",
    "dataset_id",
    "machine",
    "machine_family",
    "machine_id",
    "machine_iri",
    "machine_name",
    "machine_uri",
    "of_id",
    "operation",
    "operation_id",
    "part",
    "part_id",
    "source",
    "source_dataset_id",
    "sindit_asset_iri",
    "tool",
    "tool_id",
    "tool_number",
    "Tool_Number",
    "Cnc_Tool_Number",
    "Cnc_Tool_Number_RT",
    "CNC_Tool_Number",
}

_TOP_LEVEL_METADATA_NEST_KEYS = (
    "casedata",
    "machining",
)


def _parse_ts(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        raw = float(value)
        if raw > 1_000_000_000_000:
            return raw / 1000.0
        return raw
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except ValueError:
            if stripped.endswith("Z"):
                stripped = stripped[:-1] + "+00:00"
            return datetime.fromisoformat(stripped).timestamp()
    return default


def _extract_signals(payload: Dict[str, Any]) -> Dict[str, float]:
    def _coerce_numeric(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        if isinstance(value, dict):
            for key in ("value", "v", "signal", "reading"):
                if key in value:
                    return _coerce_numeric(value.get(key))
        return None

    candidate_sources = []
    for key in ("signals", "frame", "data", "values", "measurements"):
        raw = payload.get(key)
        if isinstance(raw, dict):
            candidate_sources.append(raw.items())
    used_payload_only = not candidate_sources
    if used_payload_only:
        candidate_sources.append(payload.items())

    signals: Dict[str, float] = {}
    for source in candidate_sources:
        for key, value in source:
            if key in _RESERVED_KEYS:
                continue
            numeric = _coerce_numeric(value)
            if numeric is not None:
                signals[key] = numeric
        if signals:
            break

    if not signals and not used_payload_only:
        for key, value in payload.items():
            if key in _RESERVED_KEYS:
                continue
            numeric = _coerce_numeric(value)
            if numeric is not None:
                signals[key] = numeric
    return signals


def _merge_metadata(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_metadata(merged[key], value)
        else:
            merged[key] = deepcopy(value) if isinstance(value, (dict, list)) else value
    return merged


def _top_level_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}

    for key in _TOP_LEVEL_METADATA_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            metadata[key] = value

    for key in _TOP_LEVEL_METADATA_NEST_KEYS:
        value = payload.get(key)
        if isinstance(value, dict) and value:
            metadata[key] = deepcopy(value)

    return metadata


class MqttStreamSource:
    name = "mqtt"

    def __init__(
        self,
        sessions: Dict[str, Dict[str, Any]],
        *,
        topic: str,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        qos: int = 0,
        sample_frequency: float = 1.0,
        username: Optional[str] = None,
        password: Optional[str] = None,
        connect_timeout_s: float = 5.0,
        client_factory: Any = create_mqtt_client,
        queue_maxsize: int = 256,
    ):
        if not topic:
            raise ValueError("MQTT topic is required")
        self._sessions = sessions
        self._topic = topic
        self._broker_host = broker_host
        self._broker_port = int(broker_port)
        self._qos = int(qos)
        self._sample_frequency = float(sample_frequency)
        self._username = username
        self._password = password
        self._connect_timeout_s = float(connect_timeout_s)
        self._client_factory = client_factory
        self._queue_maxsize = max(1, int(queue_maxsize))
        self._clients: Dict[str, AsyncMqttClient] = {}

    def _status_block(self, session_id: str) -> Dict[str, Any]:
        session = self._sessions[session_id]
        status = session.setdefault(
            "source_status",
            {
                "kind": self.name,
                "connected": False,
                "last_frame_ts": None,
                "lag_ms": 0.0,
                "dropped": 0,
                "topic": self._topic,
                "broker_host": self._broker_host,
                "broker_port": self._broker_port,
                "username": self._username,
                "password_configured": bool(self._password),
            },
        )
        status["kind"] = self.name
        status["topic"] = self._topic
        status["broker_host"] = self._broker_host
        status["broker_port"] = self._broker_port
        status["username"] = self._username
        status["password_configured"] = bool(self._password)
        return status

    def start(self, session_id: str, *, startup_delay: float = 0.0) -> asyncio.Task:
        session = self._sessions[session_id]
        session["source_name"] = self.name
        session["_stream_source"] = self

        async def _runner() -> None:
            if startup_delay > 0:
                await asyncio.sleep(startup_delay)
            await self.run(session_id)

        task = asyncio.create_task(_runner())
        session["task"] = task
        return task

    async def stop(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session["running"] = False
        client = self._clients.get(session_id)
        if client is not None:
            try:
                await client.disconnect()
            except Exception:
                logger.debug("MQTT disconnect failed for session %s", session_id, exc_info=True)
        task = session.get("task")
        if isinstance(task, asyncio.Task) and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def status(self, session_id: str) -> Dict[str, Any]:
        return dict(self._status_block(session_id))

    def _make_client(self, session_id: str) -> AsyncMqttClient:
        return self._client_factory(
            self._broker_host,
            self._broker_port,
            client_id=f"lfl-{session_id}",
            username=self._username,
            password=self._password,
            connect_timeout_s=self._connect_timeout_s,
        )

    def _normalize_message(
        self,
        session_id: str,
        message: MqttMessage,
        position: int,
    ) -> tuple[FrameEnvelope, Dict[str, Any]]:
        payload = json.loads(message.payload.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("MQTT payload must decode to a JSON object")

        ts_unix = _parse_ts(
            payload.get("ts_unix") or payload.get("timestamp") or payload.get("ts"),
            time.time(),
        )
        fs = float(payload.get("fs") or payload.get("sample_frequency") or self._sample_frequency)
        signals = _extract_signals(payload)
        metadata = _top_level_metadata(payload)
        if isinstance(payload.get("metadata"), dict):
            metadata = _merge_metadata(metadata, payload["metadata"])
        metadata.setdefault(
            "mqtt",
            {
                "topic": message.topic,
                "broker_host": self._broker_host,
                "broker_port": self._broker_port,
                "qos": message.qos,
            },
        )

        frame = dict(payload.get("frame")) if isinstance(payload.get("frame"), dict) else None
        if frame is None:
            t_value = (position / fs) if fs > 0 else float(position)
            frame = {"t": t_value, "i": position, "fs": fs, "ts_unix": ts_unix, **signals}
            if isinstance(payload.get("timestamp"), str):
                frame["timestamp"] = payload["timestamp"]
        else:
            frame.setdefault("i", position)
            frame.setdefault("fs", fs)
            frame.setdefault("ts_unix", ts_unix)
            if isinstance(payload.get("timestamp"), str):
                frame.setdefault("timestamp", payload["timestamp"])
            if "t" not in frame:
                frame["t"] = (position / fs) if fs > 0 else float(position)
            for key, value in signals.items():
                existing = frame.get(key)
                if not isinstance(existing, (int, float)) or isinstance(existing, bool):
                    frame[key] = value

        envelope = FrameEnvelope(
            kind=str(payload.get("kind") or "tag_sample"),
            session_id=session_id,
            ts_unix=ts_unix,
            position=position,
            fs=fs,
            source=self.name,
            signals=signals,
            frame=frame,
            metadata=metadata,
        )
        return envelope, frame

    def _append_signals(self, session: Dict[str, Any], signals: Dict[str, float]) -> None:
        data = session.setdefault("data", {})
        position = int(session.get("position", 0) or 0)

        for channel, values in list(data.items()):
            if channel in signals:
                values.append(float(signals[channel]))
            else:
                previous = float(values[-1]) if values else 0.0
                values.append(previous)

        for channel, value in signals.items():
            if channel in data:
                continue
            data[channel] = [0.0] * position + [float(value)]

    async def run(self, session_id: str) -> None:
        session = self._sessions[session_id]
        session["source_name"] = self.name
        session["_stream_source"] = self
        session.pop("last_error", None)
        status = self._status_block(session_id)
        message_queue: asyncio.Queue[MqttMessage] = asyncio.Queue(maxsize=self._queue_maxsize)

        def _handle_message(message: MqttMessage) -> None:
            try:
                message_queue.put_nowait(message)
            except asyncio.QueueFull:
                status["dropped"] = int(status.get("dropped", 0)) + 1

        client = self._make_client(session_id)
        self._clients[session_id] = client
        client.set_message_handler(_handle_message)

        try:
            await client.connect()
            await client.subscribe(self._topic, qos=self._qos)
            status["connected"] = True

            metadata = session.setdefault("metadata", {})
            metadata["sample_frequency"] = self._sample_frequency
            metadata.setdefault("source", self.name)
            metadata.setdefault(
                "mqtt",
                {
                    "topic": self._topic,
                    "broker_host": self._broker_host,
                    "broker_port": self._broker_port,
                    "qos": self._qos,
                },
            )

            while session.get("running", False):
                try:
                    message = await asyncio.wait_for(message_queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue

                position = int(session.get("position", 0) or 0)
                envelope, frame = self._normalize_message(session_id, message, position)
                self._append_signals(session, envelope.signals)
                session_metadata = _merge_metadata(dict(session.get("metadata") or {}), envelope.metadata)
                session_metadata["sample_frequency"] = envelope.fs
                session_metadata.setdefault("source", self.name)
                session["metadata"] = session_metadata
                session["position"] = position + 1

                for queue in list(session.get("subscribers", [])):
                    await queue.put(frame)

                await publish_feature(session_id, envelope)
                status["last_frame_ts"] = envelope.ts_unix
                status["lag_ms"] = max(0.0, (time.time() - envelope.ts_unix) * 1000.0)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            session["last_error"] = str(exc)
            logger.exception("MqttStreamSource crashed for session %s", session_id)
        finally:
            status["connected"] = False
            session["running"] = False
            client = self._clients.pop(session_id, None)
            if client is not None:
                try:
                    await client.disconnect()
                except Exception:
                    logger.debug("MQTT disconnect failed for session %s", session_id, exc_info=True)
            eos = {"eos": True, "fs": session.get("metadata", {}).get("sample_frequency", self._sample_frequency), "final_i": int(session.get("position", 0) or 0)}
            for queue in list(session.get("subscribers", [])):
                try:
                    await queue.put(eos)
                except Exception:
                    pass
            session["task"] = None
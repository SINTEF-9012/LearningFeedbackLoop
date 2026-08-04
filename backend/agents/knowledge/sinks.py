"""Upstream-sink boilerplate — Agent H.

Defines the :class:`UpstreamSink` protocol plus two reference
implementations:

- :class:`FileSink` — writes the pack to a local JSON file. Used in
  tests and as an offline fallback when no upstream is configured.
- :class:`MqttSink` — stub that **never auto-activates**. The
  constructor takes ``enabled=False`` by default so misconfigured
  deployments can't accidentally talk to an MQTT broker. A real
  implementation would use ``paho-mqtt`` / ``asyncio-mqtt`` to publish
  to a delta-or-snapshot topic; here we only ship the contract and a
  no-op that records would-be publishes so the rest of the system can
  wire against a stable surface.

Delta vs snapshot is selected by the caller; the sink doesn't care.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@runtime_checkable
class UpstreamSink(Protocol):
    """Minimal surface every sink must implement."""

    name: str

    async def push(self, payload: Dict[str, Any]) -> bool: ...


# ── FileSink ───────────────────────────────────────────────────────────


@dataclass
class FileSink:
    """Write each payload to ``<dir>/<prefix>_<timestamp>.json``.

    Uses an atomic ``.tmp`` → ``os.replace`` staging pattern so
    partially-written files are never visible.
    """

    directory: Path
    prefix: str = "knowledge_pack"
    name: str = "file"

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    async def push(self, payload: Dict[str, Any]) -> bool:
        try:
            built_at = payload.get("built_at") or ""
            stamp = built_at.replace(":", "-").replace("+", "_") or "now"
            target = self.directory / f"{self.prefix}_{stamp}.json"
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
            os.replace(tmp, target)
            logger.info("FileSink: wrote %s", target)
            return True
        except Exception:
            logger.exception("FileSink: push failed")
            return False


# ── MqttSink (disabled stub) ───────────────────────────────────────────


@dataclass
class MqttSink:
    """Disabled-by-default MQTT sink.

    Real broker wiring is intentionally out of scope for this agent —
    this stub preserves the :class:`UpstreamSink` contract and lets
    downstream tests verify that code paths route through it without
    pulling an MQTT client dependency.

    Configuration:

    - ``broker_url``: e.g. ``"tcp://broker.local:1883"``.
    - ``topic``: e.g. ``"lfl/knowledge/site-a"``.
    - ``mode``: ``"delta"`` (default, MQTT-friendly) or ``"snapshot"``.
    - ``enabled``: must be ``True`` to attempt a publish; defaults to
      ``False`` so misconfigured deployments stay inert.
    """

    broker_url: str = ""
    topic: str = ""
    mode: str = "delta"
    enabled: bool = False
    name: str = "mqtt"
    # Test/inspection aid — records would-be publishes when disabled.
    last_payloads: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in {"delta", "snapshot"}:
            raise ValueError(f"MqttSink.mode must be 'delta' or 'snapshot', got {self.mode!r}")

    async def push(self, payload: Dict[str, Any]) -> bool:
        if not self.enabled:
            logger.info(
                "MqttSink: disabled — would-be publish to %s (topic=%s mode=%s, %d keys)",
                self.broker_url or "<unset>",
                self.topic or "<unset>",
                self.mode,
                len(payload),
            )
            self.last_payloads.append(payload)
            return False
        if not self.broker_url or not self.topic:
            logger.warning("MqttSink: enabled but broker_url/topic missing; refusing to publish")
            return False
        # Intentionally not implemented — real broker code lives outside
        # this agent's scope.
        raise NotImplementedError(
            "MqttSink transport is a stub; provide a concrete client in a follow-up agent."
        )


# ── HttpSink ──────────────────────────────────────────────────────────


@dataclass
class HttpSink:
    """POST a knowledge-pack payload to an upstream HTTP endpoint."""

    url: str = ""
    timeout_seconds: float = 10.0
    headers: Dict[str, str] = field(default_factory=dict)
    name: str = "http"

    async def push(self, payload: Dict[str, Any]) -> bool:
        if not self.url:
            logger.warning("HttpSink: url missing; refusing to publish")
            return False
        try:
            body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            merged_headers = {"Content-Type": "application/json", **self.headers}
            request = Request(self.url, data=body, headers=merged_headers, method="POST")
            with urlopen(request, timeout=float(self.timeout_seconds)) as response:
                status = int(getattr(response, "status", 0) or 0)
            if 200 <= status < 300:
                logger.info("HttpSink: posted knowledge pack to %s (status=%s)", self.url, status)
                return True
            logger.warning("HttpSink: unexpected response status=%s url=%s", status, self.url)
            return False
        except Exception:
            logger.exception("HttpSink: push failed")
            return False


# ── Helper: fan-out push ──────────────────────────────────────────────


async def push_to_sinks(
    sinks: List[UpstreamSink],
    payload: Dict[str, Any],
) -> Dict[str, bool]:
    """Run ``sink.push(payload)`` for each sink sequentially.

    Returns ``{sink.name: ok}``. Exceptions in one sink never stop
    others; they count as ``False``.
    """
    results: Dict[str, bool] = {}
    for sink in sinks:
        try:
            results[sink.name] = bool(await sink.push(payload))
        except NotImplementedError:
            raise
        except Exception:
            logger.exception("push_to_sinks: sink=%s failed", getattr(sink, "name", "?"))
            results[getattr(sink, "name", "?")] = False
    return results

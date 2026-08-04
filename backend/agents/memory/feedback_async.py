"""Async feedback outbox + broadcaster — Agent M (2026-04-24).

Plan §13 asks for three things:

1. **Operator history** — who confirmed/dismissed what, surfaced
   explicitly so UI can show per-operator attribution.
2. **Outbox pattern** — deferred feedback processing with durable
   queue: incoming feedback is appended to a JSONL file *before*
   handler-side side-effects fire, so we never lose an operator
   action on a crash.
3. **Live broadcast** — other connected clients see the action in
   real time via websocket so two operators on different screens
   don't duplicate work.

This module wires on top of the existing
:class:`backend.agents.memory.feedback.MemoryFeedbackHandler` via its
``register_callback`` hook — no changes needed to the handler itself.

Everything is fail-open: broadcaster exceptions never break the
feedback pipeline; outbox write errors are logged but the handler's
main path still runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


# ── Event model ───────────────────────────────────────────────────────


@dataclass
class FeedbackEvent:
    """Wire-format for an operator feedback event.

    Kept as a plain dataclass so it can be JSON-dumped and replayed
    without Pydantic at the outbox layer.
    """

    memory_id: str
    action: str
    operator_id: str
    created_at: str = field(default_factory=_now_iso)
    data: Dict[str, Any] = field(default_factory=dict)
    sequence: int = 0  # Assigned by the outbox; 0 before enqueue.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeedbackEvent":
        return cls(
            memory_id=str(d.get("memory_id", "")),
            action=str(d.get("action", "")),
            operator_id=str(d.get("operator_id", "unknown")),
            created_at=str(d.get("created_at") or _now_iso()),
            data=dict(d.get("data") or {}),
            sequence=int(d.get("sequence") or 0),
        )


# ── Outbox ────────────────────────────────────────────────────────────


class FeedbackOutbox:
    """Append-only JSONL outbox for feedback events.

    Writes are atomic per-line (single ``write()`` call on O_APPEND) —
    small enough to fit in a PIPE_BUF (<4 KiB) on Linux so concurrent
    writers don't interleave partial lines. For large ``data``
    payloads we fall back to an in-memory lock.

    The outbox is *not* a message bus: it's a crash-survival log.
    Consumers replay on startup via :meth:`iter_pending` and mark
    acked with :meth:`mark_acked` (writes a sidecar cursor file).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.cursor_path = self.path.with_suffix(self.path.suffix + ".cursor")
        self._lock = asyncio.Lock()
        self._sequence = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Recover current high-water sequence from the existing file.
        if self.path.exists():
            try:
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        self._sequence = max(self._sequence, int(rec.get("sequence", 0)))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
            except OSError:
                logger.exception("FeedbackOutbox: could not read %s", self.path)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    async def append(self, event: FeedbackEvent) -> FeedbackEvent:
        """Persist *event*; returns the stored copy with ``sequence`` set."""
        async with self._lock:
            self._sequence += 1
            event.sequence = self._sequence
            payload = json.dumps(event.to_dict(), separators=(",", ":"), default=str)
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(payload + "\n")
            except OSError:
                logger.exception("FeedbackOutbox: append failed for memory=%s", event.memory_id)
            return event

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def _read_cursor(self) -> int:
        try:
            return int(self.cursor_path.read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            return 0

    def _write_cursor(self, sequence: int) -> None:
        try:
            tmp = self.cursor_path.with_suffix(self.cursor_path.suffix + ".tmp")
            tmp.write_text(str(int(sequence)), encoding="utf-8")
            os.replace(tmp, self.cursor_path)
        except OSError:
            logger.exception("FeedbackOutbox: cursor write failed")

    def iter_pending(self) -> Iterable[FeedbackEvent]:
        """Yield events that haven't been acknowledged yet."""
        if not self.path.exists():
            return
        cursor = self._read_cursor()
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    seq = int(rec.get("sequence", 0))
                    if seq > cursor:
                        yield FeedbackEvent.from_dict(rec)
        except OSError:
            logger.exception("FeedbackOutbox: read failed for %s", self.path)

    def mark_acked(self, sequence: int) -> None:
        """Persist that everything up to *sequence* has been processed."""
        if sequence > 0:
            self._write_cursor(sequence)

    def pending_count(self) -> int:
        return sum(1 for _ in self.iter_pending())

    def reset(self) -> None:
        """Test helper: drop the outbox + cursor files."""
        for p in (self.path, self.cursor_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        self._sequence = 0


# ── Broadcaster ───────────────────────────────────────────────────────


SubscriberCallback = Callable[[Dict[str, Any]], Awaitable[None]]


class FeedbackBroadcaster:
    """Fan-out feedback events to registered async subscribers.

    Typical wiring: the FastAPI WebSocket endpoint registers a
    subscriber whose body is ``ws.send_json(payload)``. When feedback
    fires, every connected ws sees the payload.

    Subscribers are called concurrently with ``asyncio.gather`` and
    any exception is swallowed with a log — one misbehaving client
    must not block the rest. Subscribers that raise on ``broadcast``
    are **not** auto-unregistered; call :meth:`unsubscribe` explicitly
    (typically in the ws disconnect handler).

    Agent Q (2026-04-24): per-subscriber ``asyncio.wait_for`` wraps
    each callback with a configurable timeout (default 0.5s) so a
    stuck client cannot block the broadcast gather. Timed-out
    subscribers are logged but not auto-removed (caller handles
    disconnect on their own channel).
    """

    DEFAULT_SUBSCRIBER_TIMEOUT_S: float = 0.5

    def __init__(self, *, subscriber_timeout_s: Optional[float] = None) -> None:
        self._subs: Set[SubscriberCallback] = set()
        self._lock = asyncio.Lock()
        self._timeout = (
            float(subscriber_timeout_s)
            if subscriber_timeout_s is not None
            else self.DEFAULT_SUBSCRIBER_TIMEOUT_S
        )

    async def subscribe(self, callback: SubscriberCallback) -> None:
        async with self._lock:
            self._subs.add(callback)

    async def unsubscribe(self, callback: SubscriberCallback) -> None:
        async with self._lock:
            self._subs.discard(callback)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)

    async def broadcast(self, event: FeedbackEvent) -> int:
        """Dispatch to every subscriber concurrently.

        Returns the number of subscribers successfully notified.
        """
        async with self._lock:
            targets = list(self._subs)
        if not targets:
            return 0
        payload = event.to_dict()

        async def _one(cb: SubscriberCallback) -> bool:
            try:
                await asyncio.wait_for(cb(payload), timeout=self._timeout)
                return True
            except asyncio.TimeoutError:
                logger.warning(
                    "FeedbackBroadcaster: subscriber timed out after %.2fs",
                    self._timeout,
                )
                return False
            except Exception:
                logger.exception("FeedbackBroadcaster: subscriber raised")
                return False

        results = await asyncio.gather(*(_one(cb) for cb in targets))
        return sum(1 for ok in results if ok)


# ── Operator history ──────────────────────────────────────────────────


class OperatorFeedbackHistory:
    """Simple in-memory index of ``operator_id → [events]``.

    Fed from the outbox on startup and from the live callback on the
    hot path. Exposed via :meth:`for_operator` and :meth:`summary`.
    """

    def __init__(self, *, max_per_operator: int = 1000):
        self._by_operator: Dict[str, List[FeedbackEvent]] = defaultdict(list)
        self._max = max(1, int(max_per_operator))

    def record(self, event: FeedbackEvent) -> None:
        bucket = self._by_operator[event.operator_id]
        bucket.append(event)
        if len(bucket) > self._max:
            # Drop oldest; O(n) — acceptable for the size we care about.
            del bucket[: len(bucket) - self._max]

    def load_from_outbox(self, outbox: FeedbackOutbox) -> int:
        """Replay every event in the outbox. Returns events loaded."""
        count = 0
        # iter_pending respects cursor; history wants the whole log.
        if not outbox.path.exists():
            return 0
        try:
            with outbox.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.record(FeedbackEvent.from_dict(rec))
                    count += 1
        except OSError:
            logger.exception("OperatorFeedbackHistory: replay failed")
        return count

    def for_operator(self, operator_id: str) -> List[FeedbackEvent]:
        return list(self._by_operator.get(operator_id, []))

    def operators(self) -> List[str]:
        return sorted(self._by_operator.keys())

    def summary(self) -> Dict[str, Dict[str, int]]:
        """Return ``{operator_id: {action: count}}``."""
        out: Dict[str, Dict[str, int]] = {}
        for operator_id, events in self._by_operator.items():
            bucket: Dict[str, int] = defaultdict(int)
            for e in events:
                bucket[e.action] += 1
            out[operator_id] = dict(bucket)
        return out


# ── Pipeline glue ─────────────────────────────────────────────────────


@dataclass
class FeedbackPipeline:
    """Ties outbox + broadcaster + operator history together.

    Install by passing :meth:`callback` to
    ``MemoryFeedbackHandler.register_callback``.
    """

    outbox: FeedbackOutbox
    broadcaster: FeedbackBroadcaster
    history: OperatorFeedbackHistory

    async def callback(self, memory_id: str, action: Any, data: Dict[str, Any]) -> None:
        """Handler-compatible callback.

        Signature matches ``FeedbackCallback`` from
        :mod:`backend.agents.memory.feedback`. Converts the action
        Enum (or string) to its ``.value`` for JSON-friendliness.
        """
        action_str = getattr(action, "value", None) or str(action)
        operator_id = str(data.get("user_id") or data.get("operator_id") or "operator")
        event = FeedbackEvent(
            memory_id=memory_id,
            action=action_str,
            operator_id=operator_id,
            data={k: v for k, v in data.items() if k not in {"user_id", "operator_id"}},
        )
        persisted = await self.outbox.append(event)
        self.history.record(persisted)
        try:
            await self.broadcaster.broadcast(persisted)
        except Exception:
            logger.exception("FeedbackPipeline: broadcast failed (non-fatal)")

    def replay_history(self) -> int:
        """Replay the on-disk outbox into the in-memory history."""
        return self.history.load_from_outbox(self.outbox)


def build_default_pipeline(
    *,
    outbox_path: str | Path = "data/feedback_outbox.jsonl",
    max_per_operator: int = 1000,
) -> FeedbackPipeline:
    """Convenience constructor — wires a fresh pipeline and replays."""
    outbox = FeedbackOutbox(outbox_path)
    broadcaster = FeedbackBroadcaster()
    history = OperatorFeedbackHistory(max_per_operator=max_per_operator)
    pipeline = FeedbackPipeline(outbox=outbox, broadcaster=broadcaster, history=history)
    pipeline.replay_history()
    return pipeline

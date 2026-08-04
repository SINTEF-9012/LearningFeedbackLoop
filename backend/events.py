"""Lightweight event bus for streaming features and events.

Provides an asyncio-based in-memory pub/sub with optional hooks
to plug a Redis-backed implementation later.
"""
import asyncio
from typing import Any, AsyncIterator, Callable, Dict, List, Optional


class PubSub:
    def __init__(self):
        self._subs: Dict[str, List[asyncio.Queue]] = {}

    async def publish(self, channel: str, message: Any):
        queues = list(self._subs.get(channel, []))
        for q in queues:
            # don't await put to all at once; use create_task to avoid blocking
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                # drop if subscriber is slow
                pass

    def subscribe(self, channel: str, maxsize: int = 128) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subs.setdefault(channel, []).append(q)
        return q

    def unsubscribe(self, channel: str, q: asyncio.Queue):
        subs = self._subs.get(channel)
        if not subs:
            return
        try:
            subs.remove(q)
        except ValueError:
            pass


class Subscription:
    """Async context manager for automatic PubSub unsubscription.

    Usage::

        async with Subscription(bus, "features.session1") as q:
            msg = await q.get()
            ...
        # automatically unsubscribes when leaving the block
    """

    def __init__(self, pubsub: PubSub, channel: str, maxsize: int = 128):
        self._bus = pubsub
        self._channel = channel
        self._queue = pubsub.subscribe(channel, maxsize=maxsize)

    @property
    def queue(self) -> asyncio.Queue:
        return self._queue

    async def __aenter__(self) -> asyncio.Queue:
        return self._queue

    async def __aexit__(self, *exc: Any) -> None:
        self._bus.unsubscribe(self._channel, self._queue)


# singleton bus
bus = PubSub()


async def publish_feature(session_id: str, payload: Dict[str, Any]):
    """Publish a feature event for a session on channel 'features.{session_id}' and 'features' global."""
    await bus.publish(f"features.{session_id}", payload)
    await bus.publish("features", payload)


def subscribe_features(session_id: Optional[str] = None) -> asyncio.Queue:
    if session_id:
        return bus.subscribe(f"features.{session_id}")
    return bus.subscribe("features")


async def publish_learning(payload: Any) -> None:
    """Publish a learning event on the global and optional session-scoped channels."""

    session_id = getattr(payload, "session_id", None)
    if session_id is None and isinstance(payload, dict):
        session_id = payload.get("session_id")

    if session_id:
        await bus.publish(f"learnings.{session_id}", payload)
    await bus.publish("learnings", payload)


def subscribe_learnings(session_id: Optional[str] = None) -> asyncio.Queue:
    if session_id:
        return bus.subscribe(f"learnings.{session_id}")
    return bus.subscribe("learnings")

"""Lightweight event bus for streaming features and events.

Provides an asyncio-based in-memory pub/sub with optional hooks
to plug a Redis-backed implementation later.
"""
import asyncio
from typing import Any, AsyncIterator, Callable, Dict, List


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


# singleton bus
bus = PubSub()


async def publish_feature(session_id: str, payload: Dict[str, Any]):
    """Publish a feature event for a session on channel 'features.{session_id}' and 'features' global."""
    await bus.publish(f"features.{session_id}", payload)
    await bus.publish("features", payload)


def subscribe_features(session_id: str = None) -> asyncio.Queue:
    if session_id:
        return bus.subscribe(f"features.{session_id}")
    return bus.subscribe("features")

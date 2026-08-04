"""Tests for Agent Q — FeedbackBroadcaster per-subscriber timeout."""

from __future__ import annotations

import asyncio

import pytest

from backend.agents.memory.feedback_async import FeedbackBroadcaster, FeedbackEvent


@pytest.mark.asyncio
async def test_slow_subscriber_times_out():
    bc = FeedbackBroadcaster(subscriber_timeout_s=0.05)

    fast_calls = 0
    slow_calls = 0

    async def fast(payload):
        nonlocal fast_calls
        fast_calls += 1

    async def slow(payload):
        nonlocal slow_calls
        await asyncio.sleep(1.0)  # Well above timeout
        slow_calls += 1

    await bc.subscribe(fast)
    await bc.subscribe(slow)

    event = FeedbackEvent(memory_id="m1", action="confirm", operator_id="op1")
    delivered = await bc.broadcast(event)

    # Only the fast subscriber should have completed within timeout.
    assert fast_calls == 1
    assert slow_calls == 0
    assert delivered == 1


@pytest.mark.asyncio
async def test_broadcast_completes_within_timeout_budget():
    bc = FeedbackBroadcaster(subscriber_timeout_s=0.05)

    async def slow(payload):
        await asyncio.sleep(0.5)

    await bc.subscribe(slow)
    event = FeedbackEvent(memory_id="m1", action="confirm", operator_id="op1")

    start = asyncio.get_event_loop().time()
    delivered = await bc.broadcast(event)
    elapsed = asyncio.get_event_loop().time() - start

    assert delivered == 0
    # Timeout is 50 ms; broadcast should return well before the 500 ms sleep.
    assert elapsed < 0.3


@pytest.mark.asyncio
async def test_timeout_does_not_unsubscribe():
    bc = FeedbackBroadcaster(subscriber_timeout_s=0.05)

    async def slow(payload):
        await asyncio.sleep(0.5)

    await bc.subscribe(slow)
    event = FeedbackEvent(memory_id="m1", action="confirm", operator_id="op1")
    await bc.broadcast(event)
    # Subscriber remains registered (caller handles disconnect).
    assert bc.subscriber_count == 1


@pytest.mark.asyncio
async def test_default_timeout_present():
    bc = FeedbackBroadcaster()
    # Default should be >0 and <= 1 s (current default is 0.5).
    assert 0.0 < FeedbackBroadcaster.DEFAULT_SUBSCRIBER_TIMEOUT_S <= 1.0
    # Smoke test: broadcast with no subscribers returns 0.
    event = FeedbackEvent(memory_id="m1", action="confirm", operator_id="op1")
    assert await bc.broadcast(event) == 0

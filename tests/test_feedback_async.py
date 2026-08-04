"""Tests for backend.agents.memory.feedback_async — Agent M."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from backend.agents.memory.feedback_async import (
    FeedbackBroadcaster,
    FeedbackEvent,
    FeedbackOutbox,
    FeedbackPipeline,
    OperatorFeedbackHistory,
    build_default_pipeline,
)


# ── FeedbackEvent ──────────────────────────────────────────────────────


def test_feedback_event_round_trip() -> None:
    e = FeedbackEvent(memory_id="m1", action="confirm", operator_id="op-1", data={"note": "hi"})
    d = e.to_dict()
    back = FeedbackEvent.from_dict(d)
    assert back.memory_id == "m1"
    assert back.action == "confirm"
    assert back.operator_id == "op-1"
    assert back.data["note"] == "hi"


def test_feedback_event_from_partial_dict_uses_defaults() -> None:
    back = FeedbackEvent.from_dict({"memory_id": "m", "action": "dismiss"})
    assert back.operator_id == "unknown"
    assert back.sequence == 0
    assert back.data == {}


# ── FeedbackOutbox ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_outbox_append_assigns_monotonic_sequence(tmp_path: Path) -> None:
    box = FeedbackOutbox(tmp_path / "outbox.jsonl")
    e1 = await box.append(FeedbackEvent("m1", "confirm", "op-1"))
    e2 = await box.append(FeedbackEvent("m2", "dismiss", "op-1"))
    e3 = await box.append(FeedbackEvent("m3", "comment", "op-2"))
    assert (e1.sequence, e2.sequence, e3.sequence) == (1, 2, 3)

    lines = (tmp_path / "outbox.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    # Each line is valid JSON.
    for line in lines:
        json.loads(line)


@pytest.mark.asyncio
async def test_outbox_resumes_sequence_across_restart(tmp_path: Path) -> None:
    path = tmp_path / "outbox.jsonl"
    box1 = FeedbackOutbox(path)
    await box1.append(FeedbackEvent("m1", "confirm", "op"))
    await box1.append(FeedbackEvent("m2", "confirm", "op"))

    # Simulate process restart.
    box2 = FeedbackOutbox(path)
    e3 = await box2.append(FeedbackEvent("m3", "confirm", "op"))
    assert e3.sequence == 3


@pytest.mark.asyncio
async def test_outbox_iter_pending_respects_cursor(tmp_path: Path) -> None:
    box = FeedbackOutbox(tmp_path / "outbox.jsonl")
    for i in range(5):
        await box.append(FeedbackEvent(f"m{i}", "confirm", "op"))
    pending = list(box.iter_pending())
    assert len(pending) == 5
    box.mark_acked(3)
    pending = list(box.iter_pending())
    assert [e.sequence for e in pending] == [4, 5]


@pytest.mark.asyncio
async def test_outbox_pending_count(tmp_path: Path) -> None:
    box = FeedbackOutbox(tmp_path / "outbox.jsonl")
    assert box.pending_count() == 0
    await box.append(FeedbackEvent("m", "confirm", "op"))
    assert box.pending_count() == 1


def test_outbox_ignores_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "outbox.jsonl"
    path.write_text('{"sequence":1,"memory_id":"m","action":"a","operator_id":"op"}\n'
                    "not json at all\n"
                    '{"sequence":2,"memory_id":"m2","action":"b","operator_id":"op"}\n')
    box = FeedbackOutbox(path)
    pending = list(box.iter_pending())
    assert len(pending) == 2
    assert {e.memory_id for e in pending} == {"m", "m2"}


# ── Broadcaster ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_broadcaster_fans_out_concurrently() -> None:
    bc = FeedbackBroadcaster()
    received: list[list[dict]] = [[], [], []]

    async def make_sub(idx: int):
        async def sub(payload):
            received[idx].append(payload)
        return sub

    for i in range(3):
        await bc.subscribe(await make_sub(i))

    assert bc.subscriber_count == 3

    event = FeedbackEvent("m1", "confirm", "op")
    n = await bc.broadcast(event)
    assert n == 3
    for bucket in received:
        assert bucket[0]["memory_id"] == "m1"


@pytest.mark.asyncio
async def test_broadcaster_isolates_failing_subscriber() -> None:
    bc = FeedbackBroadcaster()
    good_received: list[dict] = []

    async def bad(payload):
        raise RuntimeError("boom")

    async def good(payload):
        good_received.append(payload)

    await bc.subscribe(bad)
    await bc.subscribe(good)

    n = await bc.broadcast(FeedbackEvent("m", "confirm", "op"))
    assert n == 1  # good succeeded, bad counted as fail
    assert len(good_received) == 1


@pytest.mark.asyncio
async def test_broadcaster_unsubscribe() -> None:
    bc = FeedbackBroadcaster()

    async def sub(payload):
        pass

    await bc.subscribe(sub)
    assert bc.subscriber_count == 1
    await bc.unsubscribe(sub)
    assert bc.subscriber_count == 0
    # Broadcasting to no-one returns 0.
    n = await bc.broadcast(FeedbackEvent("m", "confirm", "op"))
    assert n == 0


# ── Operator history ──────────────────────────────────────────────────


def test_operator_history_summary_counts_actions() -> None:
    hist = OperatorFeedbackHistory()
    hist.record(FeedbackEvent("m1", "confirm", "op-1"))
    hist.record(FeedbackEvent("m2", "confirm", "op-1"))
    hist.record(FeedbackEvent("m3", "dismiss", "op-1"))
    hist.record(FeedbackEvent("m4", "confirm", "op-2"))

    summary = hist.summary()
    assert summary["op-1"] == {"confirm": 2, "dismiss": 1}
    assert summary["op-2"] == {"confirm": 1}
    assert hist.operators() == ["op-1", "op-2"]


def test_operator_history_bounded() -> None:
    hist = OperatorFeedbackHistory(max_per_operator=3)
    for i in range(5):
        hist.record(FeedbackEvent(f"m{i}", "confirm", "op"))
    assert len(hist.for_operator("op")) == 3
    # Oldest dropped: m0, m1 gone.
    ids = [e.memory_id for e in hist.for_operator("op")]
    assert ids == ["m2", "m3", "m4"]


def test_operator_history_replays_from_outbox(tmp_path: Path) -> None:
    path = tmp_path / "outbox.jsonl"
    path.write_text(
        '{"sequence":1,"memory_id":"m1","action":"confirm","operator_id":"op-1","created_at":"t"}\n'
        '{"sequence":2,"memory_id":"m2","action":"dismiss","operator_id":"op-2","created_at":"t"}\n'
    )
    box = FeedbackOutbox(path)
    hist = OperatorFeedbackHistory()
    loaded = hist.load_from_outbox(box)
    assert loaded == 2
    assert set(hist.operators()) == {"op-1", "op-2"}


# ── Pipeline glue ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_callback_persists_records_and_broadcasts(tmp_path: Path) -> None:
    pipeline = build_default_pipeline(outbox_path=tmp_path / "ob.jsonl")
    received: list[dict] = []

    async def sub(payload):
        received.append(payload)

    await pipeline.broadcaster.subscribe(sub)

    # Mimic the signature of the memory feedback callback.
    class _FakeAction:
        value = "confirm"

    await pipeline.callback(
        memory_id="m1",
        action=_FakeAction(),
        data={"user_id": "op-7", "reason": "real event"},
    )

    # Persisted.
    pending = list(pipeline.outbox.iter_pending())
    assert len(pending) == 1
    assert pending[0].operator_id == "op-7"
    assert pending[0].action == "confirm"
    assert "user_id" not in pending[0].data  # Stripped — promoted to operator_id.
    assert pending[0].data["reason"] == "real event"

    # Broadcast.
    assert len(received) == 1
    assert received[0]["memory_id"] == "m1"

    # Recorded in history.
    assert pipeline.history.operators() == ["op-7"]


@pytest.mark.asyncio
async def test_pipeline_tolerates_string_action(tmp_path: Path) -> None:
    pipeline = build_default_pipeline(outbox_path=tmp_path / "ob.jsonl")
    await pipeline.callback("m1", "dismiss", {"user_id": "op"})
    pending = list(pipeline.outbox.iter_pending())
    assert pending[0].action == "dismiss"


@pytest.mark.asyncio
async def test_pipeline_survives_broadcast_failure(tmp_path: Path) -> None:
    pipeline = build_default_pipeline(outbox_path=tmp_path / "ob.jsonl")

    async def bad(payload):
        raise RuntimeError("ws died")

    await pipeline.broadcaster.subscribe(bad)
    # Callback must not raise.
    await pipeline.callback("m1", "confirm", {"user_id": "op"})
    assert pipeline.outbox.pending_count() == 1


def test_build_default_pipeline_replays_existing_outbox(tmp_path: Path) -> None:
    path = tmp_path / "ob.jsonl"
    path.write_text(
        '{"sequence":1,"memory_id":"m1","action":"confirm","operator_id":"op-1","created_at":"t"}\n'
    )
    pipeline = build_default_pipeline(outbox_path=path)
    assert pipeline.history.operators() == ["op-1"]

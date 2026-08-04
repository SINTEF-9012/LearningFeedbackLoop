"""Integration test: SINDIT runtime state tracker keeps node count bounded.

Per plan point 7 (Agent G refined): "current state" discipline depends
on writers behaving. This test asserts that over a simulated long
session with many successive experiment/phase/operation cycles the
number of active SINDIT assets stays bounded — it must NOT grow like a
slow event log.

Uses a stub :class:`FakeSinditWriteClient` so the test is hermetic; the
real SINDIT write client is not exercised.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from backend.agents.sindit.runtime_state import (
    ActiveEntity,
    RuntimeStateTracker,
    experiment,
    operation,
    pattern,
    phase,
)


class FakeSinditWriteClient:
    """In-memory stand-in for the SINDIT write client.

    Tracks every URI ever upserted and the current "active" set
    (``active=False`` removes from the current set). Satisfies the
    :class:`~backend.agents.sindit.runtime_state._SinditWriteClient`
    protocol with ``post_asset`` + ``update_node``.
    """

    def __init__(self) -> None:
        self.upsert_calls: int = 0
        self.update_calls: int = 0
        self._nodes: Dict[str, Dict[str, Any]] = {}

    async def post_asset(self, payload: Dict[str, Any]) -> bool:
        self.upsert_calls += 1
        uri = payload.get("uri") or ""
        if not uri:
            return False
        self._nodes[uri] = {**payload, "active": True}
        return True

    async def update_node(self, node_uri: str, fields: Dict[str, Any]) -> bool:
        self.update_calls += 1
        if node_uri not in self._nodes:
            return False
        self._nodes[node_uri].update(fields)
        return True

    # Helpers for test assertions (not part of the protocol).

    def total_nodes_ever_written(self) -> int:
        return len(self._nodes)

    def currently_active_nodes(self) -> List[str]:
        return sorted(
            uri for uri, p in self._nodes.items() if p.get("active") is True
        )


@pytest.mark.asyncio
async def test_single_cycle_upserts_all_entities():
    client = FakeSinditWriteClient()
    tracker = RuntimeStateTracker()

    result = await tracker.sync_active(
        client,
        [
            experiment("expt-01"),
            phase("roughing"),
            operation("op-1"),
        ],
    )

    assert result["upserts"] == 3
    assert result["deactivations"] == 0
    assert client.upsert_calls == 3
    assert len(client.currently_active_nodes()) == 3


@pytest.mark.asyncio
async def test_repeated_sync_does_not_inflate_active_set():
    """Calling sync with the same active list 10 times must NOT grow the
    active node count (upserts are idempotent)."""
    client = FakeSinditWriteClient()
    tracker = RuntimeStateTracker()

    active = [
        experiment("expt-02"),
        phase("finishing"),
        operation("op-2"),
        pattern("chatter:high_ratio"),
    ]
    for _ in range(10):
        await tracker.sync_active(client, active)

    assert len(client.currently_active_nodes()) == len(active)
    # Upserts accumulate (10 cycles * 4 entries) but active set stays bounded.
    assert client.upsert_calls == 10 * len(active)


@pytest.mark.asyncio
async def test_long_session_bounded_active_nodes():
    """Simulate 50 successive "phases" where each new phase replaces the
    previous one. The active set size must stay at the configured cap
    throughout (=1 experiment + 1 phase + 1 operation = 3 active nodes),
    not accumulate to 50."""
    client = FakeSinditWriteClient()
    tracker = RuntimeStateTracker()

    max_active_observed = 0
    for i in range(50):
        await tracker.sync_active(
            client,
            [
                experiment("expt-long"),
                phase(f"phase-{i}"),
                operation(f"op-{i}"),
            ],
        )
        active_count = len(client.currently_active_nodes())
        max_active_observed = max(max_active_observed, active_count)

    # Across a 50-phase session the simultaneously-active set must never
    # exceed 3. (Deactivated nodes remain in the store marked active=False,
    # but they're out of the "current state" view — that's the whole
    # point of the runtime-state tracker.)
    assert max_active_observed == 3
    assert len(client.currently_active_nodes()) == 3

    # Final check: 50 deactivations happened (49 phase + 49 operation +
    # the 50th snapshot), i.e. the tracker did its job of cleaning up.
    assert client.update_calls >= 49 * 2


@pytest.mark.asyncio
async def test_session_end_clear_removes_all_active():
    client = FakeSinditWriteClient()
    tracker = RuntimeStateTracker()

    await tracker.sync_active(
        client,
        [experiment("expt-end"), phase("ph-end"), operation("op-end")],
    )
    assert len(client.currently_active_nodes()) == 3

    summary = await tracker.clear(client)

    assert summary["deactivations"] == 3
    assert summary["upserts"] == 0
    assert client.currently_active_nodes() == []


@pytest.mark.asyncio
async def test_pattern_transitions_do_not_leak_old_patterns():
    """Per plan point 7: when a pattern is no longer currently detected,
    it must be deactivated, not left hanging."""
    client = FakeSinditWriteClient()
    tracker = RuntimeStateTracker()

    await tracker.sync_active(client, [pattern("chatter:high_ratio")])
    assert "urn:lfl:pattern:chatter-high_ratio" in client.currently_active_nodes()

    # Next tick the pattern is no longer detected.
    await tracker.sync_active(client, [])
    assert client.currently_active_nodes() == []


@pytest.mark.asyncio
async def test_plan_sync_is_pure_diff_without_client():
    """``plan_sync`` must compute the upsert/deactivate lists without
    hitting the client — useful for offline assertions / dry-runs."""
    tracker = RuntimeStateTracker()

    to_upsert, to_deactivate = tracker.plan_sync(
        [experiment("expt-pure"), phase("ph-pure")]
    )
    assert len(to_upsert) == 2
    assert to_deactivate == []

    to_upsert, to_deactivate = tracker.plan_sync([experiment("expt-pure")])
    assert len(to_upsert) == 1
    assert len(to_deactivate) == 1  # phase dropped

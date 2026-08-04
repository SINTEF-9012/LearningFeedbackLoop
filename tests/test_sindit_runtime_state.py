"""Tests for backend.agents.sindit.runtime_state — Agent G."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from backend.agents.sindit.runtime_state import (
    ActiveEntity,
    RuntimeStateTracker,
    build_runtime_asset,
    experiment,
    operation,
    pattern,
    phase,
)


# ── Helpers ────────────────────────────────────────────────────────────


class _FakeClient:
    """Minimal async client recording every call."""

    def __init__(self, *, fail_posts: bool = False, fail_updates: bool = False):
        self.fail_posts = fail_posts
        self.fail_updates = fail_updates
        self.posted: List[Dict[str, Any]] = []
        self.updated: List[Dict[str, Any]] = []

    async def post_asset(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.posted.append(payload)
        if self.fail_posts:
            return None
        return {"ok": True, "uri": payload["uri"]}

    async def update_node(
        self, node_uri: str, fields: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        self.updated.append({"uri": node_uri, "fields": fields})
        if self.fail_updates:
            return None
        return {"ok": True}


# ── build_runtime_asset ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kind,prefix",
    [
        ("experiment", "urn:lfl:experiment:"),
        ("phase", "urn:lfl:phase:"),
        ("operation", "urn:lfl:operation:"),
        ("pattern", "urn:lfl:pattern:"),
    ],
)
def test_build_runtime_asset_shape(kind: str, prefix: str) -> None:
    payload = build_runtime_asset(kind, "Some Weird/ID 01", {"trigger": "tick"})
    assert payload["uri"].startswith(prefix)
    # Slugification must drop weird chars.
    assert "/" not in payload["uri"]
    assert " " not in payload["uri"]
    assert payload["assetType"].endswith("#AbstractAsset")
    assert payload["lflAssetKind"] in {"Experiment", "Phase", "Operation", "Pattern"}
    assert payload["metadata"]["active"] is True
    assert "lastSeenAt" in payload["metadata"]
    assert payload["metadata"]["trigger"] == "tick"


def test_build_runtime_asset_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        build_runtime_asset("machine", "cnc-1")


def test_build_runtime_asset_respects_active_override() -> None:
    payload = build_runtime_asset("experiment", "e1", active=False)
    assert payload["metadata"]["active"] is False


# ── Pure diff ──────────────────────────────────────────────────────────


def test_plan_sync_marks_new_entries_for_upsert() -> None:
    tracker = RuntimeStateTracker()
    to_upsert, to_deactivate = tracker.plan_sync(
        [experiment("expt_a"), phase("roughing")]
    )
    assert len(to_upsert) == 2
    assert to_deactivate == []
    assert sorted(tracker.active_keys()) == [("experiment", "expt_a"), ("phase", "roughing")]


def test_plan_sync_deactivates_dropped_entries() -> None:
    tracker = RuntimeStateTracker()
    tracker.plan_sync([experiment("expt_a"), phase("roughing")])

    to_upsert, to_deactivate = tracker.plan_sync([experiment("expt_a")])
    # Still upserts the surviving one (refreshes lastSeenAt).
    assert len(to_upsert) == 1
    assert to_upsert[0]["uri"] == "urn:lfl:experiment:expt_a"
    # Dropped phase gets flagged for deactivation.
    assert len(to_deactivate) == 1
    assert to_deactivate[0]["uri"] == "urn:lfl:phase:roughing"


def test_plan_sync_idempotent_when_nothing_changed() -> None:
    tracker = RuntimeStateTracker()
    tracker.plan_sync([experiment("expt_a")])
    to_upsert, to_deactivate = tracker.plan_sync([experiment("expt_a")])
    assert len(to_upsert) == 1  # refresh payload
    assert to_deactivate == []


# ── Async sync via FakeClient ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_sync_active_upserts_all_on_first_call() -> None:
    tracker = RuntimeStateTracker()
    client = _FakeClient()

    result = await tracker.sync_active(
        client,
        [experiment("expt_a"), pattern("chatter:high_ratio")],
    )

    assert result["upserts"] == 2
    assert result["deactivations"] == 0
    assert len(client.posted) == 2
    assert client.updated == []
    stats = result["stats"]
    assert stats["upserts_ok"] == 2
    assert stats["upserts_fail"] == 0
    assert stats["active_by_kind"] == {"experiment": 1, "pattern": 1}


@pytest.mark.asyncio
async def test_sync_active_deactivates_dropped() -> None:
    tracker = RuntimeStateTracker()
    client = _FakeClient()

    await tracker.sync_active(client, [experiment("expt_a"), phase("roughing")])
    client.posted.clear()
    client.updated.clear()

    # Second cycle: phase gone, new operation added.
    result = await tracker.sync_active(
        client,
        [experiment("expt_a"), operation("op_42")],
    )

    posted_uris = {p["uri"] for p in client.posted}
    assert posted_uris == {
        "urn:lfl:experiment:expt_a",
        "urn:lfl:operation:op_42",
    }
    assert len(client.updated) == 1
    drop = client.updated[0]
    assert drop["uri"] == "urn:lfl:phase:roughing"
    assert drop["fields"]["active"] is False
    assert "endedAt" in drop["fields"]
    assert result["deactivations"] == 1


@pytest.mark.asyncio
async def test_sync_active_records_failures_without_raising() -> None:
    tracker = RuntimeStateTracker()
    client = _FakeClient(fail_posts=True, fail_updates=True)

    await tracker.sync_active(client, [experiment("expt_a")])
    # Pre-populate tracker so we have something to deactivate next call.
    # (On first call post failed but tracker still holds the key locally.)
    result = await tracker.sync_active(client, [])

    stats = result["stats"]
    assert stats["upserts_fail"] >= 1
    assert stats["deactivations_fail"] >= 1
    # Never raises even when client keeps failing.


@pytest.mark.asyncio
async def test_clear_deactivates_every_entity() -> None:
    tracker = RuntimeStateTracker()
    client = _FakeClient()

    await tracker.sync_active(
        client,
        [experiment("e1"), phase("p1"), pattern("chatter")],
    )
    client.posted.clear()

    result = await tracker.clear(client)

    assert result["upserts"] == 0
    assert result["deactivations"] == 3
    deactivated = {u["uri"] for u in client.updated}
    assert deactivated == {
        "urn:lfl:experiment:e1",
        "urn:lfl:phase:p1",
        "urn:lfl:pattern:chatter",
    }
    assert tracker.active_keys() == []


# ── Integration-style scenario (plan §7 acceptance) ───────────────────


@pytest.mark.asyncio
async def test_short_session_lifecycle_matches_plan_acceptance() -> None:
    """Plan §7: after a short session, SINDIT has ≥1 active experiment
    during the run, and the experiment is marked inactive after end."""
    tracker = RuntimeStateTracker()
    client = _FakeClient()

    # --- During session ---
    await tracker.sync_active(
        client,
        [
            experiment("expt_2026_04_24_01", site="lab-a"),
            phase("roughing"),
            operation("op_1"),
            pattern("chatter:high_ratio"),
        ],
    )
    active_kinds = tracker.stats.active_by_kind
    assert active_kinds.get("experiment", 0) >= 1
    # ≥N properties → in our tracker each active entity carries ≥2 metadata keys (active + lastSeenAt).
    for payload in client.posted:
        assert len(payload["metadata"]) >= 2

    # --- Session ends: one final clean call with empty set. ---
    await tracker.clear(client)

    # Every previously-posted URI must now have an update_node(active=False).
    deactivated = {u["uri"]: u["fields"] for u in client.updated}
    for payload in client.posted:
        assert payload["uri"] in deactivated
        assert deactivated[payload["uri"]]["active"] is False
    assert tracker.active_keys() == []

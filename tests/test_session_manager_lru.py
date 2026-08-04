"""Tests for Agent Q — SessionManager LRU cap."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytest

from backend.session_manager import SessionManager


def test_unbounded_by_default():
    store: Dict[str, Dict] = {}
    mgr = SessionManager(store)
    for i in range(20):
        mgr.register(f"s{i}", {"i": i})
    assert len(mgr) == 20
    assert mgr.max_sessions == 0


def test_lru_eviction_on_register_over_cap():
    store: Dict[str, Dict] = {}
    evicted: List[Tuple[str, Dict]] = []

    def on_evict(sid, state):
        evicted.append((sid, state))

    mgr = SessionManager(store, max_sessions=3, on_evict=on_evict)
    mgr.register("a", {"v": 1})
    mgr.register("b", {"v": 2})
    mgr.register("c", {"v": 3})
    assert len(mgr) == 3
    # Insert 4th → 'a' (oldest) evicted.
    mgr.register("d", {"v": 4})
    assert len(mgr) == 3
    assert "a" not in store
    assert [sid for sid, _ in evicted] == ["a"]


def test_get_touches_lru_order():
    store: Dict[str, Dict] = {}
    mgr = SessionManager(store, max_sessions=3)
    mgr.register("a", {"v": 1})
    mgr.register("b", {"v": 2})
    mgr.register("c", {"v": 3})
    # Touch 'a' so it's no longer the LRU.
    mgr.get("a")
    mgr.register("d", {"v": 4})
    # 'b' should be evicted instead of 'a'.
    assert "a" in store
    assert "b" not in store


def test_get_or_404_touches_lru():
    store: Dict[str, Dict] = {}
    mgr = SessionManager(store, max_sessions=2)
    mgr.register("a", {"v": 1})
    mgr.register("b", {"v": 2})
    mgr.get_or_404("a")
    mgr.register("c", {"v": 3})
    assert "a" in store
    assert "b" not in store


def test_remove_cleans_lru():
    store: Dict[str, Dict] = {}
    mgr = SessionManager(store, max_sessions=2)
    mgr.register("a", {"v": 1})
    mgr.register("b", {"v": 2})
    mgr.remove("a")
    mgr.register("c", {"v": 3})
    # 'b' should remain; the cap permitted adding 'c' without eviction.
    assert {"b", "c"} == set(store.keys())


def test_update_existing_does_not_trigger_eviction():
    store: Dict[str, Dict] = {}
    mgr = SessionManager(store, max_sessions=2)
    mgr.register("a", {"v": 1})
    mgr.register("b", {"v": 2})
    # Re-registering an existing id refreshes the slot — no eviction needed.
    mgr.register("a", {"v": 99})
    assert len(mgr) == 2
    assert store["a"]["v"] == 99
    assert {"a", "b"} == set(store.keys())


def test_on_evict_failure_is_swallowed():
    store: Dict[str, Dict] = {}

    def boom(sid, state):  # noqa: ARG001
        raise RuntimeError("boom")

    mgr = SessionManager(store, max_sessions=1, on_evict=boom)
    mgr.register("a", {"v": 1})
    # Should not raise even though on_evict blows up.
    mgr.register("b", {"v": 2})
    assert "a" not in store
    assert "b" in store


def test_max_sessions_from_env(monkeypatch):
    monkeypatch.setenv("LFL_MAX_SESSIONS", "4")
    mgr = SessionManager({})
    assert mgr.max_sessions == 4


def test_invalid_env_ignored(monkeypatch):
    monkeypatch.setenv("LFL_MAX_SESSIONS", "not-an-int")
    mgr = SessionManager({})
    assert mgr.max_sessions == 0


def test_evict_to_cap_bulk():
    store = {f"s{i}": {"v": i} for i in range(10)}
    mgr = SessionManager(store, max_sessions=3)
    evicted = mgr.evict_to_cap()
    assert len(evicted) == 7
    assert len(mgr) == 3

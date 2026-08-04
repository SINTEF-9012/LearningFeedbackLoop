"""Tests for durable per-session signal binding (Agent N)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.agents.storage.session_signals import (
    compute_signal_digest,
    extract_signal_window,
    list_persisted_sessions,
    load_session_signal,
    save_session_signal,
    session_signal_path,
)


# ──────────────────────────────────────────────────────────────────────
# Save / load round-trip
# ──────────────────────────────────────────────────────────────────────


def test_save_and_load_roundtrip(tmp_path):
    data = {"Fx": [1.0, 2.0, 3.0, 4.0], "Fy": [0.5, 0.6, 0.7, 0.8]}
    meta = {"fs": 100.0, "source": "test"}

    path = save_session_signal("sess-123", data, meta, fs=100.0, sessions_dir=tmp_path)
    assert path is not None and path.exists()
    assert path == session_signal_path("sess-123", sessions_dir=tmp_path)

    loaded = load_session_signal("sess-123", sessions_dir=tmp_path)
    assert loaded is not None
    loaded_data, loaded_meta = loaded
    assert set(loaded_data.keys()) == {"Fx", "Fy"}
    np.testing.assert_allclose(loaded_data["Fx"], [1.0, 2.0, 3.0, 4.0])
    assert loaded_meta["fs"] == 100.0
    assert loaded_meta["_session_id"] == "sess-123"
    assert loaded_meta["_channels"] == ["Fx", "Fy"]


def test_load_missing_returns_none(tmp_path):
    assert load_session_signal("does-not-exist", sessions_dir=tmp_path) is None


def test_save_empty_returns_none(tmp_path):
    assert save_session_signal("empty", {}, {}, sessions_dir=tmp_path) is None
    assert save_session_signal("all-none", {"a": None, "b": []}, {}, sessions_dir=tmp_path) is None


def test_save_skips_non_numeric_channels(tmp_path):
    path = save_session_signal(
        "mixed",
        {"Fx": [1.0, 2.0], "bad": ["a", "b"], "Fy": [3.0, 4.0]},
        {},
        fs=10.0,
        sessions_dir=tmp_path,
    )
    assert path is not None
    loaded = load_session_signal("mixed", sessions_dir=tmp_path)
    assert loaded is not None
    data, _ = loaded
    assert set(data.keys()) == {"Fx", "Fy"}


def test_list_persisted_sessions(tmp_path):
    save_session_signal("a", {"x": [1.0]}, {}, fs=1.0, sessions_dir=tmp_path)
    save_session_signal("b", {"x": [2.0]}, {}, fs=1.0, sessions_dir=tmp_path)
    assert list_persisted_sessions(sessions_dir=tmp_path) == ["a", "b"]


def test_atomic_write_no_tmp_left_behind(tmp_path):
    save_session_signal("atomic", {"x": [1.0, 2.0]}, {}, fs=1.0, sessions_dir=tmp_path)
    # no ".tmp" should remain
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


# ──────────────────────────────────────────────────────────────────────
# Window extraction
# ──────────────────────────────────────────────────────────────────────


def test_extract_window_basic():
    data = {"Fx": list(range(100)), "Fy": [v * 0.1 for v in range(100)]}
    w = extract_signal_window(data, ["Fx"], 10, 20, fs=100.0)
    assert w["i0"] == 10 and w["i1"] == 20
    assert w["fs"] == 100.0
    assert len(w["channels"]["Fx"]) == 10
    assert w["channels"]["Fx"] == list(float(v) for v in range(10, 20))
    assert w["requested_range"] == {"i0": 10, "i1": 20}


def test_extract_window_margin_expands_symmetrically():
    data = {"Fx": list(range(100))}
    w = extract_signal_window(data, ["Fx"], 50, 60, fs=100.0, margin_s=0.05)
    # 0.05s * 100Hz = 5 samples each side → [45, 65)
    assert w["i0"] == 45 and w["i1"] == 65
    assert w["margin_samples"] == 5
    assert len(w["channels"]["Fx"]) == 20


def test_extract_window_clamps_to_bounds():
    data = {"Fx": list(range(10))}
    w = extract_signal_window(data, None, -5, 1000, fs=10.0)
    assert w["i0"] == 0
    # left clamp sets i0 to 0; right stays at 1000 but slice is clamped internally
    assert len(w["channels"]["Fx"]) == 10


def test_extract_window_all_channels_when_none():
    data = {"Fx": [1.0, 2.0, 3.0], "Fy": [4.0, 5.0, 6.0], "_private": [9.0]}
    w = extract_signal_window(data, None, 0, 3, fs=1.0)
    # underscore-prefixed keys are skipped when channels=None
    assert set(w["channels"].keys()) == {"Fx", "Fy"}


def test_extract_window_invalid_fs():
    w = extract_signal_window({"Fx": [1.0, 2.0]}, ["Fx"], 0, 2, fs=0.0, margin_s=1.0)
    # fs=0 → margin_samples=0 and times are 0
    assert w["margin_samples"] == 0
    assert w["t0"] == 0.0 and w["t1"] == 0.0


def test_extract_window_swapped_indices():
    data = {"Fx": list(range(20))}
    w = extract_signal_window(data, ["Fx"], 15, 5, fs=10.0)
    # swaps i0, i1
    assert w["i0"] == 5 and w["i1"] == 15
    assert len(w["channels"]["Fx"]) == 10


# ──────────────────────────────────────────────────────────────────────
# Digest
# ──────────────────────────────────────────────────────────────────────


def test_digest_stable_and_detects_mutation():
    data = {"Fx": [1.0, 2.0, 3.0, 4.0]}
    d1 = compute_signal_digest(data, 0, 4)
    d2 = compute_signal_digest(data, 0, 4)
    assert d1 == d2 and len(d1) == 40

    mutated = {"Fx": [1.0, 2.0, 3.0, 5.0]}
    d3 = compute_signal_digest(mutated, 0, 4)
    assert d3 != d1


def test_digest_empty_range_is_empty_string():
    assert compute_signal_digest({"Fx": [1.0, 2.0]}, 1, 1) == ""
    assert compute_signal_digest({}, 0, 10) == ""


def test_digest_channel_order_independent():
    data = {"Fx": [1.0, 2.0], "Fy": [3.0, 4.0]}
    d1 = compute_signal_digest(data, 0, 2, channels=["Fx", "Fy"])
    d2 = compute_signal_digest(data, 0, 2, channels=["Fy", "Fx"])
    assert d1 == d2

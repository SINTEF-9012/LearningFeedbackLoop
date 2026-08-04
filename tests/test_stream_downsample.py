"""Tests for Agent Q Round 20 — WS frame downsampling."""

from __future__ import annotations

import numpy as np
import pytest

from backend.routers._stream_downsample import (
    DEFAULT_STREAM_DOWNSAMPLE_THRESHOLD,
    maybe_downsample_frame,
)


# ── Passthrough cases ───────────────────────────────────────────────


def test_passthrough_when_threshold_zero():
    frame = {"t0": 0.0, "t1": 1.0, "a": list(range(5000))}
    assert maybe_downsample_frame(frame, 0) is frame


def test_passthrough_when_threshold_too_small():
    frame = {"t0": 0.0, "t1": 1.0, "a": list(range(5000))}
    # threshold <= 2 disables downsampling.
    assert maybe_downsample_frame(frame, 2) is frame


def test_passthrough_on_non_dict():
    assert maybe_downsample_frame("not a frame", 500) == "not a frame"
    assert maybe_downsample_frame(None, 500) is None


def test_passthrough_when_frame_too_short():
    frame = {"t0": 0.0, "t1": 1.0, "fs": 1000.0, "a": list(range(100))}
    out = maybe_downsample_frame(frame, 500)
    # No per-channel array met the threshold; frame returned unchanged
    # (content-wise — same dict since helper only copies when it mutates).
    assert out == frame


def test_passthrough_on_per_sample_frame():
    # Per-sample frames have scalar channel values (no t0/t1).
    frame = {"t": 0.1, "i": 100, "fs": 1000.0, "a": 0.5, "b": -0.3}
    out = maybe_downsample_frame(frame, 500)
    assert out == frame


def test_passthrough_on_eos_frame():
    frame = {"eos": True, "fs": 1000.0, "final_i": 5000}
    out = maybe_downsample_frame(frame, 500)
    assert out == frame


# ── Time-domain chunk downsampling ──────────────────────────────────


def test_time_chunk_downsample_per_channel():
    n = 5000
    xs = np.linspace(0.0, 1.0, n)
    a = np.sin(xs * 2 * np.pi * 5).tolist()
    b = np.cos(xs * 2 * np.pi * 5).tolist()
    frame = {"t0": 0.0, "t1": 1.0, "i0": 0, "i1": n, "fs": 5000.0, "a": a, "b": b}

    out = maybe_downsample_frame(frame, 200)

    assert out is not frame  # shallow copy when mutated
    assert out["downsampled"] is True
    assert out["downsample_threshold"] == 200
    assert len(out["a"]) == 200
    assert len(out["b"]) == 200
    assert "t_downsampled" in out
    assert len(out["t_downsampled"]) == 200
    # Endpoints of the downsampled x-axis match the frame window.
    assert out["t_downsampled"][0] == pytest.approx(0.0)
    assert out["t_downsampled"][-1] == pytest.approx(1.0)
    # Downsampled y retains LTTB endpoint guarantee per channel.
    assert out["a"][0] == pytest.approx(a[0])
    assert out["a"][-1] == pytest.approx(a[-1])
    assert out["b"][0] == pytest.approx(b[0])
    assert out["b"][-1] == pytest.approx(b[-1])


def test_time_chunk_preserves_reserved_keys():
    n = 5000
    frame = {
        "t0": 0.0, "t1": 2.0, "i0": 100, "i1": 100 + n, "fs": 2500.0,
        "a": list(range(n)),
    }
    out = maybe_downsample_frame(frame, 250)
    assert out["t0"] == 0.0 and out["t1"] == 2.0
    assert out["i0"] == 100 and out["i1"] == 100 + n
    assert out["fs"] == 2500.0


def test_time_chunk_numpy_array_channel():
    n = 5000
    a = np.random.default_rng(0).standard_normal(n)
    frame = {"t0": 0.0, "t1": 1.0, "i0": 0, "i1": n, "fs": 5000.0, "a": a}
    out = maybe_downsample_frame(frame, 300)
    assert len(out["a"]) == 300
    assert isinstance(out["a"], list)  # serialised to list for JSON


def test_time_chunk_mixed_channel_types_skips_non_numeric():
    # Strings in a channel should be left alone (defensive fail-open).
    n = 5000
    frame = {
        "t0": 0.0, "t1": 1.0, "i0": 0, "i1": n, "fs": 5000.0,
        "a": list(range(n)),
        "label": "some_text",
    }
    out = maybe_downsample_frame(frame, 200)
    assert len(out["a"]) == 200
    assert out["label"] == "some_text"


# ── FFT frame downsampling ──────────────────────────────────────────


def test_fft_frame_downsample_shared_freqs():
    n = 5000
    freqs = np.linspace(0.0, 1000.0, n)
    ch_a = np.abs(np.sin(freqs / 50.0))
    ch_b = np.abs(np.cos(freqs / 30.0))
    frame = {
        "freqs": freqs.tolist(),
        "channels": {"a": ch_a.tolist(), "b": ch_b.tolist()},
        "fs": 2000.0,
        "nfft": 1024,
    }
    out = maybe_downsample_frame(frame, 300)
    assert out["downsampled"] is True
    assert out["downsample_threshold"] == 300
    assert len(out["freqs"]) == 300
    assert len(out["channels"]["a"]) == 300
    assert len(out["channels"]["b"]) == 300
    # Preserved peer-keys
    assert out["fs"] == 2000.0
    assert out["nfft"] == 1024


def test_fft_frame_short_passthrough():
    n = 100
    freqs = np.linspace(0.0, 1000.0, n).tolist()
    frame = {
        "freqs": freqs,
        "channels": {"a": [0.0] * n},
    }
    out = maybe_downsample_frame(frame, 300)
    assert out == frame


def test_fft_frame_mismatched_channel_length_skipped():
    n = 5000
    freqs = np.linspace(0.0, 1000.0, n).tolist()
    frame = {
        "freqs": freqs,
        "channels": {
            "a": [0.0] * n,
            # b has wrong length — must pass through untouched.
            "b": [1.0] * (n - 10),
        },
    }
    out = maybe_downsample_frame(frame, 300)
    assert len(out["channels"]["a"]) == 300
    assert out["channels"]["b"] == [1.0] * (n - 10)


# ── Fail-open / robustness ──────────────────────────────────────────


def test_invalid_t_bounds_passes_through():
    frame = {"t0": "oops", "t1": None, "a": list(range(5000))}
    out = maybe_downsample_frame(frame, 200)
    assert out == frame


def test_fft_non_mapping_channels_passes_through():
    frame = {"freqs": list(range(5000)), "channels": "not-a-mapping"}
    out = maybe_downsample_frame(frame, 200)
    assert out == frame


def test_default_threshold_sane():
    assert 500 <= DEFAULT_STREAM_DOWNSAMPLE_THRESHOLD <= 10000

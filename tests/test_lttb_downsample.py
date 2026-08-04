"""Tests for Agent Q — LTTB downsampling utility."""

from __future__ import annotations

import numpy as np
import pytest

from backend.agents.processing.downsample import lttb


def test_lttb_short_input_returned_unchanged():
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([10.0, 20.0, 30.0])
    xo, yo = lttb(xs, ys, threshold=5)
    np.testing.assert_array_equal(xo, xs)
    np.testing.assert_array_equal(yo, ys)


def test_lttb_threshold_leq_2_returns_unchanged():
    xs = np.arange(100, dtype=np.float64)
    ys = np.sin(xs)
    xo, yo = lttb(xs, ys, threshold=2)
    assert xo.shape == xs.shape


def test_lttb_produces_exact_threshold_count():
    xs = np.linspace(0.0, 10.0, 1000)
    ys = np.sin(xs)
    xo, yo = lttb(xs, ys, threshold=100)
    assert xo.shape == (100,)
    assert yo.shape == (100,)


def test_lttb_preserves_endpoints():
    xs = np.linspace(0.0, 10.0, 500)
    ys = np.cos(xs) + 0.1 * np.sin(10 * xs)
    xo, yo = lttb(xs, ys, threshold=50)
    assert xo[0] == xs[0]
    assert xo[-1] == xs[-1]
    assert yo[0] == ys[0]
    assert yo[-1] == ys[-1]


def test_lttb_preserves_extremes_of_sine():
    xs = np.linspace(0.0, 2 * np.pi, 1000)
    ys = np.sin(xs)
    xo, yo = lttb(xs, ys, threshold=30)
    # LTTB should retain points close to the peaks (±1) and troughs.
    assert yo.max() > 0.95
    assert yo.min() < -0.95


def test_lttb_length_mismatch_raises():
    with pytest.raises(ValueError):
        lttb(np.arange(10), np.arange(9), threshold=5)


def test_lttb_does_not_mutate_inputs():
    xs = np.linspace(0.0, 10.0, 500)
    ys = np.sin(xs)
    xs_copy = xs.copy()
    ys_copy = ys.copy()
    lttb(xs, ys, threshold=50)
    np.testing.assert_array_equal(xs, xs_copy)
    np.testing.assert_array_equal(ys, ys_copy)


def test_lttb_monotonic_x_output():
    xs = np.linspace(0.0, 10.0, 1000)
    ys = np.random.default_rng(0).standard_normal(1000)
    xo, _ = lttb(xs, ys, threshold=100)
    assert np.all(np.diff(xo) >= 0.0)

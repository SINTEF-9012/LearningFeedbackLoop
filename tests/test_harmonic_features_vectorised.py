"""Tests for Agent Q — vectorised ``compute_harmonics`` equivalence.

The vectorised path must produce numerically identical output to the
loop implementation (up to float32 rounding) for a variety of input
shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.agents.processing.harmonic_features import compute_harmonics


def _compute_harmonics_loop_reference(
    signals: np.ndarray,
    fg: float,
    harm_mults,
    fft_win: int,
    fft_step: int,
    sample_rate: float,
) -> np.ndarray:
    """Reference implementation — literal port of the pre-Q loop."""
    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)
    n_samples, n_channels = signals.shape
    n_harm = len(harm_mults)
    n_steps = max(0, (n_samples - fft_win) // fft_step + 1)
    result = np.zeros((n_steps, n_harm * n_channels), dtype=np.float32)
    if n_steps == 0 or fg <= 0:
        return result
    bin_scale = fft_win / sample_rate
    for t in range(n_steps):
        start = t * fft_step
        for ch in range(n_channels):
            seg = signals[start: start + fft_win, ch]
            spectrum = np.abs(np.fft.rfft(seg))
            for hi, mult in enumerate(harm_mults):
                bin_idx = int(round(fg * mult * bin_scale))
                col = ch * n_harm + hi
                if 0 < bin_idx < len(spectrum):
                    result[t, col] = spectrum[bin_idx]
    return result


@pytest.mark.parametrize(
    "n_samples,n_channels,fft_win,fft_step,fg,harm_mults,sample_rate",
    [
        (8192, 1, 1024, 256, 50.0, [1, 2, 3, 4], 20000.0),
        (8192, 3, 2048, 512, 120.0, [1, 2, 3, 4, 6, 8, 10], 20000.0),
        (16384, 4, 4096, 1024, 80.0, [1, 2, 3, 4, 6, 8, 10], 20000.0),
        (5000, 2, 1024, 256, 100.0, [1, 2], 10000.0),
    ],
)
def test_vectorised_matches_loop(
    n_samples, n_channels, fft_win, fft_step, fg, harm_mults, sample_rate
):
    rng = np.random.default_rng(42)
    signals = rng.standard_normal((n_samples, n_channels)).astype(np.float32)

    vec = compute_harmonics(
        signals, fg=fg, harm_mults=harm_mults, fft_win=fft_win,
        fft_step=fft_step, sample_rate=sample_rate,
    )
    ref = _compute_harmonics_loop_reference(
        signals, fg=fg, harm_mults=harm_mults, fft_win=fft_win,
        fft_step=fft_step, sample_rate=sample_rate,
    )

    assert vec.shape == ref.shape
    # float32 FFT + batched vs per-window paths can diverge by a few ULPs.
    np.testing.assert_allclose(vec, ref, rtol=1e-4, atol=1e-4)


def test_compute_harmonics_empty_for_short_signal():
    signals = np.zeros((100, 2), dtype=np.float32)
    out = compute_harmonics(signals, fg=50.0, harm_mults=[1, 2], fft_win=1024, fft_step=256)
    assert out.shape == (0, 4)


def test_compute_harmonics_zero_fg_returns_zeros():
    signals = np.random.default_rng(0).standard_normal((4096, 2)).astype(np.float32)
    out = compute_harmonics(signals, fg=0.0, harm_mults=[1, 2], fft_win=1024, fft_step=256)
    assert out.shape == ((4096 - 1024) // 256 + 1, 4)
    assert np.all(out == 0.0)


def test_compute_harmonics_1d_input_reshapes_to_single_channel():
    rng = np.random.default_rng(7)
    sig = rng.standard_normal(8192).astype(np.float32)
    out = compute_harmonics(sig, fg=50.0, harm_mults=[1, 2, 3], fft_win=1024, fft_step=256, sample_rate=20000.0)
    ref = _compute_harmonics_loop_reference(
        sig, fg=50.0, harm_mults=[1, 2, 3], fft_win=1024, fft_step=256, sample_rate=20000.0,
    )
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_compute_harmonics_all_bins_out_of_range_returns_zeros():
    # Very low fg with small harm_mults means bin index 0 → marked invalid.
    signals = np.random.default_rng(1).standard_normal((4096, 1)).astype(np.float32)
    out = compute_harmonics(
        signals, fg=0.001, harm_mults=[1], fft_win=1024, fft_step=256, sample_rate=20000.0,
    )
    assert out.shape == ((4096 - 1024) // 256 + 1, 1)
    assert np.all(out == 0.0)

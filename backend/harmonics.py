"""Frequency-domain feature extraction.

The new pipeline ("pair-input-model" branch) replaces the fixed-bin harmonic
amplitudes with **(amplitude, frequency)** pairs: per FFT window, per channel
we pick the top-K spectral peaks. The model consumes these pairs directly,
which lets it cope with peaks that drift in frequency (variable spindle speed)
and with deployment-time inputs that already arrive as peak triplets rather
than raw acceleration.

Channels used: X (Channel_1) and Y (Channel_2). Z is intentionally ignored
per current configuration.

Output of `compute_peak_pairs`:
    pairs: float32 array of shape (T, n_channels, K, 2)
        last dim = (f_rel, amp) where f_rel = f_hz / fg.
        Pad rows are (0, 0) when fewer than K peaks are present.

Raw frequency in Hz can be recovered as ``pairs[..., 0] * fg``. Storing the
spindle-relative frequency keeps the model insensitive to the absolute RPM,
which is what we want when streaming with a varying spindle speed.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


PAIR_FEATURE_DIM = 2  # (f_rel, amp)
DEFAULT_CHANNELS: tuple[int, ...] = (0, 1)  # X, Y
CHANNEL_NAMES = ["X", "Y"]


def compute_peak_pairs(
    accel: np.ndarray,
    fg: float,
    fft_win: int = 4096,
    fft_step: int = 4096,
    k_peaks: int = 5,
    sample_rate: float = 4096.0,
    channels: tuple[int, ...] = DEFAULT_CHANNELS,
    f_max_rel: float | None = None,
) -> np.ndarray:
    """Sliding-window FFT then top-K peak picking per channel.

    Args:
        accel: (N, C) accelerometer signal. Columns are X, Y, Z.
        fg: spindle frequency in Hz. Used only for normalising the reported
            peak frequencies; FFT itself is independent of it.
        fft_win, fft_step: window length and stride in samples.
        k_peaks: number of peaks to keep per channel per window.
        sample_rate: signal sample rate in Hz; sets FFT bin spacing.
        channels: which channel indices to process (default X, Y).
        f_max_rel: optional cap on the reported relative frequency; peaks
            above ``f_max_rel * fg`` are ignored. ``None`` means no cap.

    Returns:
        (T, len(channels), k_peaks, 2) float32. Last dim = (f_rel, amp).
        Empty slots (fewer than K real peaks) are zero-padded.
    """
    n_samples, n_channels = accel.shape
    chans = tuple(c for c in channels if c < n_channels)
    n_steps = max(0, (n_samples - fft_win) // fft_step + 1)
    out = np.zeros((n_steps, len(chans), k_peaks, PAIR_FEATURE_DIM), dtype=np.float32)

    if n_steps == 0 or fg <= 0:
        return out

    bin_freqs = np.fft.rfftfreq(fft_win, d=1.0 / sample_rate)
    f_max_hz = (f_max_rel * fg) if f_max_rel is not None else None

    for t in range(n_steps):
        start = t * fft_step
        for ci, ch in enumerate(chans):
            seg = accel[start : start + fft_win, ch]
            spectrum = np.abs(np.fft.rfft(seg)).astype(np.float32)

            if f_max_hz is not None:
                cutoff = int(np.searchsorted(bin_freqs, f_max_hz))
                spec_for_peaks = spectrum[:cutoff] if cutoff > 0 else spectrum
            else:
                spec_for_peaks = spectrum

            # find_peaks returns strict local maxima only — exactly what we
            # want so that monotonic slopes don't get reported as "peaks".
            peak_idx, _ = find_peaks(spec_for_peaks)
            if peak_idx.size == 0:
                continue

            amps = spec_for_peaks[peak_idx]
            top = peak_idx[np.argsort(-amps)[:k_peaks]]
            # Sort kept peaks by frequency for human-readable display; the
            # set-based encoder is order-invariant either way.
            top = np.sort(top)

            for j, bin_idx in enumerate(top):
                f_hz = float(bin_freqs[bin_idx])
                out[t, ci, j, 0] = f_hz / fg
                out[t, ci, j, 1] = float(spectrum[bin_idx])

    return out

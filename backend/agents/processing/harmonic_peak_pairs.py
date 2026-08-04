"""Peak-pair feature extraction for pair-input harmonic models.

Supports two data contracts:

1. Raw multi-channel vibration windows -> sliding top-K FFT peaks.
2. DataFrames that already contain per-row FFT peak frequency/amplitude columns.

Both routes produce the pair-input tensor expected by the pair model:
``(T, C, K, 2)`` where the last axis is ``(f_rel, amplitude)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


_PAIR_COL_RE = re.compile(
    r"Accel_FFT_Acc(?P<channel>\d+)_range\d+_(?P<kind>Frequencies|Amplitudes)_(?P<peak>\d+)"
)
_CASEDATA_PAIR_COL_RE = re.compile(
    r"Vibration_Peak_(?P<peak>\d+)_(?P<axis>[XY])_(?P<kind>Amplitude|Frequency)"
)
_AXIS_TO_CHANNEL_ID = {"X": 0, "Y": 1}


@dataclass(frozen=True)
class PeakPairColumnSpec:
    """Resolved frequency/amplitude column pair for one channel peak."""

    channel_id: int
    peak_index: int
    frequency_col: str
    amplitude_col: str
    channel_label: str = ""


def compute_peak_pairs(
    signals: np.ndarray,
    fg: float,
    sample_rate: float,
    *,
    k_peaks: int = 5,
    fft_win: int = 4096,
    fft_step: int = 1024,
    f_max_rel: float = 12.0,
) -> np.ndarray:
    """Extract top-K peak pairs from raw signals.

    Args:
        signals: ``(N,)`` or ``(N, C)`` raw accelerometer window.
        fg: Spindle frequency in Hz.
        sample_rate: Signal sample rate in Hz.

    Returns:
        ``(T, C, K, 2)`` where the last axis is ``(f_rel, amplitude)``.
    """
    arr = np.asarray(signals, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]

    if arr.ndim != 2 or arr.shape[0] < fft_win or sample_rate <= 0:
        return np.zeros((0, 0, max(1, int(k_peaks)), 2), dtype=np.float32)

    n_samples, n_channels = arr.shape
    n_steps = 1 + max(0, (n_samples - fft_win) // max(1, fft_step))
    pairs = np.zeros((n_steps, n_channels, max(1, int(k_peaks)), 2), dtype=np.float32)

    freqs = np.fft.rfftfreq(fft_win, d=1.0 / float(sample_rate)).astype(np.float32)
    max_freq = float(f_max_rel) * float(fg) if fg > 0 and f_max_rel > 0 else np.inf

    for step_idx, start in enumerate(range(0, n_samples - fft_win + 1, max(1, fft_step))):
        window = arr[start : start + fft_win]
        for channel_idx in range(n_channels):
            channel = window[:, channel_idx]
            spectrum = np.abs(np.fft.rfft(channel)).astype(np.float32)
            if spectrum.size <= 1:
                continue
            spectrum[0] = 0.0
            valid_mask = np.isfinite(freqs) & (freqs > 0)
            if np.isfinite(max_freq):
                valid_mask &= freqs <= max_freq

            valid_idx = np.flatnonzero(valid_mask)
            if valid_idx.size == 0:
                continue

            valid_mag = spectrum[valid_idx]
            top_n = min(max(1, int(k_peaks)), valid_idx.size)
            top_local = np.argpartition(valid_mag, -top_n)[-top_n:]
            top_idx = valid_idx[top_local]
            top_idx = top_idx[np.argsort(spectrum[top_idx])[::-1]]

            for peak_slot, spec_idx in enumerate(top_idx[:top_n]):
                peak_freq = float(freqs[spec_idx])
                peak_amp = float(spectrum[spec_idx])
                pairs[step_idx, channel_idx, peak_slot, 0] = (
                    peak_freq / float(fg) if fg > 0 else 0.0
                )
                pairs[step_idx, channel_idx, peak_slot, 1] = peak_amp

    if f_max_rel > 0:
        pairs[..., 0] = np.clip(pairs[..., 0], 0.0, float(f_max_rel))
    return pairs


def discover_peak_pair_columns(
    columns: Sequence[str],
    *,
    frequency_patterns: Optional[Sequence[str]] = None,
    amplitude_patterns: Optional[Sequence[str]] = None,
    k_peaks: int = 5,
) -> List[PeakPairColumnSpec]:
    """Resolve matching frequency/amplitude columns into channel/peak specs."""
    freq_patterns = [re.compile(p) for p in (frequency_patterns or [])]
    amp_patterns = [re.compile(p) for p in (amplitude_patterns or [])]

    freq_lookup = {}
    amp_lookup = {}
    for col in columns:
        parsed = _parse_pair_column(str(col), k_peaks=max(1, int(k_peaks)))
        if parsed is None:
            continue
        channel_id, peak_index, kind, channel_label = parsed
        key = (channel_id, peak_index)

        if kind == "frequency":
            if freq_patterns and not _matches_any(col, freq_patterns):
                continue
            freq_lookup[key] = (col, channel_label)
        else:
            if amp_patterns and not _matches_any(col, amp_patterns):
                continue
            amp_lookup[key] = (col, channel_label)

    specs: List[PeakPairColumnSpec] = []
    for key in sorted(set(freq_lookup) & set(amp_lookup)):
        freq_col, freq_label = freq_lookup[key]
        amp_col, amp_label = amp_lookup[key]
        specs.append(
            PeakPairColumnSpec(
                channel_id=key[0],
                peak_index=key[1],
                frequency_col=freq_col,
                amplitude_col=amp_col,
                channel_label=freq_label or amp_label,
            )
        )
    return specs


def extract_peak_pairs_from_df(
    df: any,
    specs: Sequence[PeakPairColumnSpec],
    *,
    spindle_speed_col: Optional[str],
    k_peaks: int = 5,
    f_max_rel: float = 12.0,
) -> np.ndarray:
    """Build a ``(T, C, K, 2)`` pair tensor from FFT peak columns."""
    if df is None or len(df) == 0 or not specs:
        return np.zeros((0, 0, max(1, int(k_peaks)), 2), dtype=np.float32)

    channel_ids = sorted({spec.channel_id for spec in specs})
    channel_to_idx = {channel_id: idx for idx, channel_id in enumerate(channel_ids)}

    pairs = np.zeros(
        (len(df), len(channel_ids), max(1, int(k_peaks)), 2),
        dtype=np.float32,
    )

    spindle_rpm = (
        df[spindle_speed_col].fillna(0).to_numpy(dtype=np.float32)
        if spindle_speed_col and spindle_speed_col in df.columns
        else np.zeros(len(df), dtype=np.float32)
    )
    fg = spindle_rpm / 60.0
    valid_fg = fg > 1e-6
    safe_fg = np.where(valid_fg, fg, 1.0).astype(np.float32)

    for spec in specs:
        freq = df[spec.frequency_col].fillna(0).to_numpy(dtype=np.float32)
        amp = df[spec.amplitude_col].fillna(0).to_numpy(dtype=np.float32)
        f_rel = np.where(valid_fg, freq / safe_fg, 0.0)
        if f_max_rel > 0:
            f_rel = np.clip(f_rel, 0.0, float(f_max_rel))
        amp = np.where(valid_fg, amp, 0.0)
        chan_idx = channel_to_idx[spec.channel_id]
        peak_idx = spec.peak_index
        pairs[:, chan_idx, peak_idx, 0] = f_rel.astype(np.float32)
        pairs[:, chan_idx, peak_idx, 1] = amp.astype(np.float32)

    return pairs


def build_pair_feature_labels(specs: Iterable[PeakPairColumnSpec]) -> List[str]:
    """Build stable labels for pair features, grouped by channel and peak slot."""
    labels = []
    for spec in sorted(specs, key=lambda item: (item.channel_id, item.peak_index)):
        channel_label = spec.channel_label or f"Acc{spec.channel_id}"
        labels.append(f"{channel_label}·P{spec.peak_index + 1}")
    return labels


def _parse_pair_column(
    column_name: str,
    *,
    k_peaks: int,
) -> Optional[tuple[int, int, str, str]]:
    raw_match = _PAIR_COL_RE.fullmatch(column_name)
    if raw_match is not None:
        peak_index = int(raw_match.group("peak"))
        if peak_index >= k_peaks:
            return None
        channel_id = int(raw_match.group("channel"))
        kind = raw_match.group("kind")
        kind_norm = "frequency" if kind == "Frequencies" else "amplitude"
        return channel_id, peak_index, kind_norm, f"Acc{channel_id}"

    casedata_match = _CASEDATA_PAIR_COL_RE.fullmatch(column_name)
    if casedata_match is not None:
        peak_number = int(casedata_match.group("peak"))
        peak_index = peak_number - 1
        if peak_index < 0 or peak_index >= k_peaks:
            return None
        axis = str(casedata_match.group("axis") or "").upper()
        channel_id = _AXIS_TO_CHANNEL_ID.get(axis)
        if channel_id is None:
            return None
        kind = casedata_match.group("kind")
        kind_norm = "frequency" if kind == "Frequency" else "amplitude"
        return channel_id, peak_index, kind_norm, axis

    return None


def _matches_any(name: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(name) for pattern in patterns)
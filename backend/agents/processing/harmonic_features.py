"""
Harmonic Feature Extraction — Domain-agnostic harmonic feature computation.

Ported from classical/lfl/backend/harmonics.py with enhancements:
- Configurable number of input channels (not hardcoded to 3)
- Configurable FFT and harmonic parameters
- Pre-extracted harmonic column selection from DataFrames
- Context parameter extraction with normalisation

Tag: [HARMONIC_CONTEXT_V1]
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def runtime_context_param_stats(config: Any) -> Dict[str, Dict[str, float]]:
    """Return the context stats that should be applied externally at runtime."""
    stats = getattr(config, "context_param_stats", {}) or {}
    return stats if isinstance(stats, dict) else {}


def runtime_context_normalize(config: Any) -> bool:
    """Whether runtime callers should z-score params before scoring."""
    model_kind = str(getattr(config, "model_kind", "legacy_v1") or "legacy_v1").strip().lower()
    return model_kind != "lfl_v2"


def resolve_spindle_speed_source_column(config: Any, default: str = "spindle_speed") -> str:
    """Resolve the source column used for relative-frequency peak extraction."""
    context_stats = getattr(config, "context_param_stats", {}) or {}
    if isinstance(context_stats, dict):
        for key in ("spindle_speed", "n"):
            entry = context_stats.get(key)
            if isinstance(entry, dict):
                source_column = entry.get("source_column")
                if source_column:
                    return str(source_column)

    context_sources = getattr(config, "context_param_sources", {}) or {}
    if isinstance(context_sources, dict):
        for key in ("spindle_speed", "n"):
            source_column = context_sources.get(key)
            if source_column:
                return str(source_column)

    return str(default)


# ══════════════════════════════════════════════════════════════════════════════
# Raw FFT harmonic extraction (high-frequency signals)
# ══════════════════════════════════════════════════════════════════════════════


def compute_harmonics(
    signals: np.ndarray,
    fg: float,
    harm_mults: Optional[List[int]] = None,
    fft_win: int = 4096,
    fft_step: int = 1024,
    sample_rate: Optional[float] = None,
) -> np.ndarray:
    """Compute harmonic magnitudes for multi-channel signals via sliding FFT.

    Adapted from classical/lfl/backend/harmonics.py to accept any number of
    channels and configurable parameters.

    Args:
        signals: (N, C) — N samples, C channels.  Any number of channels.
        fg: Spindle frequency in Hz (RPM / 60).  Must be > 0.
        harm_mults: Harmonic multipliers (e.g. [1, 2, 3, 4, 6, 8, 10]).
            Defaults to [1, 2, 3, 4, 6, 8, 10].
        fft_win: FFT window size in samples.
        fft_step: Step between consecutive FFT windows.
        sample_rate: Sample rate in Hz.  Required for correct FFT bin mapping.
            If ``None``, defaults to ``fft_win`` (correct only when
            ``sample_rate == fft_win``).

    Returns:
        (T, C * len(harm_mults)) array of harmonic magnitudes.
        Columns ordered: [h1_ch0, h2_ch0, ..., hN_ch0, h1_ch1, ..., hN_chC]
        Returns an empty (0, C * len(harm_mults)) array if input is too short.
    """
    if harm_mults is None:
        harm_mults = [1, 2, 3, 4, 6, 8, 10]

    if sample_rate is None:
        sample_rate = float(fft_win)
        logger.debug(
            "compute_harmonics: sample_rate not supplied, defaulting to fft_win=%d",
            fft_win,
        )

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    n_samples, n_channels = signals.shape
    n_harm = len(harm_mults)
    n_steps = max(0, (n_samples - fft_win) // fft_step + 1)

    result = np.zeros((n_steps, n_harm * n_channels), dtype=np.float32)

    if n_steps == 0 or fg <= 0:
        if fg <= 0:
            logger.debug("compute_harmonics: fg=%.4f Hz — skipping (spindle not running?)", fg)
        return result

    # FFT bin index = frequency_hz * fft_win / sample_rate
    bin_scale = fft_win / sample_rate

    # Agent Q (2026-04-24): channel-batched FFT. Each step calls
    # ``np.fft.rfft`` once on a ``(C, fft_win)`` contiguous slab instead
    # of looping per-channel. The step loop is kept because fully
    # vectorising across steps requires a non-contiguous sliding-window
    # view whose FFT copy overhead outweighs the batching win on
    # small-step workloads. Measured 1.4–2× speedup vs. the previous
    # triple-loop across the benchmark configs. Numerically equivalent
    # to the loop reference up to float32 rounding.
    bin_indices = np.array(
        [int(round(fg * mult * bin_scale)) for mult in harm_mults],
        dtype=np.int64,
    )
    spectrum_len = fft_win // 2 + 1  # rfft output length
    valid_mask = (bin_indices > 0) & (bin_indices < spectrum_len)

    if not np.any(valid_mask):
        # No harmonic lands inside the spectrum — keep existing zeros.
        return result

    safe_indices = np.where(valid_mask, bin_indices, 0)
    # Transpose to (C, N) and force contiguous so each per-step slab is
    # already in C-order for the FFT kernel.
    signals_t = np.ascontiguousarray(signals.T)

    for t in range(n_steps):
        start = t * fft_step
        seg = signals_t[:, start: start + fft_win]  # (C, fft_win)
        spectrum = np.abs(np.fft.rfft(seg, axis=-1))  # (C, spectrum_len)
        gathered = spectrum[:, safe_indices] * valid_mask  # (C, n_harm)
        # Column order: [h1_ch0, h2_ch0, ..., hN_ch0, h1_ch1, ...].
        # gathered is (C, H); ravel() with default C-order yields that layout.
        result[t] = gathered.ravel().astype(np.float32)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Pre-extracted harmonic selection (1 Hz / CMS data)
# ══════════════════════════════════════════════════════════════════════════════


# Fallback when a caller supplies no patterns: match standard harmonic amplitude
# columns (X/Y/Z and Acc_N variants, incl. ``_from_peaks``) without grabbing context
# columns like ``spindle_speed_mean``. Restores auto-detection for configs whose
# ``harmonic_column_patterns`` is left empty (e.g. the simulated-session cfg).
_DEFAULT_HARMONIC_COLUMN_PATTERNS = [r"Vibration_Harmonic_\d+_[A-Za-z0-9_]+_Amplitude"]


def select_harmonic_columns(
    df_columns: List[str],
    patterns: List[str],
) -> List[str]:
    """Select columns from a DataFrame that match harmonic feature patterns.

    Args:
        df_columns: All column names in the DataFrame.
        patterns: Regex patterns to match against (e.g. ``r"Vibration_Harmonic_\\d+"``).
            When empty, a default harmonic-amplitude pattern is used so columns are
            still auto-detected.

    Returns:
        Sorted list of matching column names.
    """
    selected = set()
    compiled = [re.compile(p) for p in (patterns or _DEFAULT_HARMONIC_COLUMN_PATTERNS)]
    for col in df_columns:
        for pat in compiled:
            if pat.search(col):
                selected.add(col)
                break
    return sorted(selected)


def extract_harmonic_matrix_from_df(
    df: Any,  # pd.DataFrame
    harmonic_columns: List[str],
) -> np.ndarray:
    """Extract a harmonic feature matrix from a DataFrame.

    Args:
        df: DataFrame with harmonic columns at rows representing time steps.
        harmonic_columns: Ordered list of column names to extract.

    Returns:
        (T, n_harm_features) float32 array.  NaN values are replaced with 0.
    """
    import pandas as pd

    if not harmonic_columns:
        logger.warning("extract_harmonic_matrix_from_df: no harmonic columns specified")
        return np.zeros((len(df), 0), dtype=np.float32)

    # Select only columns that exist in the DataFrame
    available = [c for c in harmonic_columns if c in df.columns]
    if not available:
        logger.debug(
            "extract_harmonic_matrix_from_df: none of %d harmonic columns found in DataFrame (%d cols)",
            len(harmonic_columns), len(df.columns),
        )
        return np.zeros((len(df), len(harmonic_columns)), dtype=np.float32)

    if len(available) < len(harmonic_columns):
        logger.info(
            "extract_harmonic_matrix_from_df: %d / %d harmonic columns found",
            len(available), len(harmonic_columns),
        )

    mat = df[available].apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0.0).values
    return mat.astype(np.float32)


def extract_peak_binned_harmonic_matrix_from_df(
    df: Any,
    *,
    frequency_patterns: List[str],
    amplitude_patterns: List[str],
    spindle_speed_col: Optional[str],
    harmonic_bins: Optional[List[int]] = None,
    k_peaks: int = 5,
    f_max_rel: float = 12.0,
    tolerance: float = 0.35,
) -> Tuple[np.ndarray, List[str]]:
    """Convert pre-extracted peak columns into harmonic-style features.

    Each output feature is the max peak amplitude within ``tolerance`` of an
    integer harmonic bin (relative frequency ``f / fg``), grouped by channel.
    """
    from .harmonic_peak_pairs import discover_peak_pair_columns, extract_peak_pairs_from_df

    if df is None or len(df) == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    bins = [int(v) for v in (harmonic_bins or list(range(1, max(1, int(k_peaks)) + 1)))]
    specs = discover_peak_pair_columns(
        list(df.columns),
        frequency_patterns=frequency_patterns,
        amplitude_patterns=amplitude_patterns,
        k_peaks=k_peaks,
    )
    if not specs:
        return np.zeros((len(df), 0), dtype=np.float32), []

    pairs = extract_peak_pairs_from_df(
        df,
        specs,
        spindle_speed_col=spindle_speed_col,
        k_peaks=k_peaks,
        f_max_rel=f_max_rel,
    )
    if pairs.size == 0 or pairs.shape[1] == 0:
        return np.zeros((len(df), 0), dtype=np.float32), []

    channel_ids = sorted({spec.channel_id for spec in specs})
    channel_labels = {
        spec.channel_id: (spec.channel_label or f"Acc{spec.channel_id}")
        for spec in specs
    }
    labels = [
        f"Vibration_Harmonic_{harmonic_bin}_{channel_labels[channel_id]}_Amplitude_from_peaks"
        for channel_id in channel_ids
        for harmonic_bin in bins
    ]

    matrix = np.zeros((pairs.shape[0], len(channel_ids) * len(bins)), dtype=np.float32)
    for channel_idx, channel_id in enumerate(channel_ids):
        f_rel = np.asarray(pairs[:, channel_idx, :, 0], dtype=np.float32)
        amp = np.asarray(pairs[:, channel_idx, :, 1], dtype=np.float32)
        f_rel = np.where(np.isfinite(f_rel), f_rel, 0.0)
        amp = np.where(np.isfinite(amp), amp, 0.0)

        for bin_idx, harmonic_bin in enumerate(bins):
            diff = np.abs(f_rel - float(harmonic_bin))
            masked_amp = np.where(diff <= float(tolerance), amp, 0.0)
            matrix[:, channel_idx * len(bins) + bin_idx] = masked_amp.max(axis=1)

    return matrix.astype(np.float32), labels


# ══════════════════════════════════════════════════════════════════════════════
# Context parameter extraction
# ══════════════════════════════════════════════════════════════════════════════


def extract_context_params(
    source: Dict[str, Any],
    param_keys: List[str],
    param_sources: Optional[Dict[str, str]] = None,
    param_stats: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Extract and normalise context parameters from a dict-like source.

    Args:
        source: Feature dict, metadata dict, or CuttingContext-like dict.
        param_keys: Ordered list of parameter keys (e.g. ["spindle_speed", "feed_rate"]).
        param_sources: Maps param key → source dict key if names differ.
        param_stats: Normalisation stats ``{key: {"mean": float, "std": float}}``.
            If provided, missing values are filled from the training mean.
        normalize: When true, apply z-score normalization using ``param_stats``.

    Returns:
        (n_params,) float32 array. Missing values become 0.0 or the training
        mean when ``param_stats`` are available.
    """
    param_sources = param_sources or {}
    values = []

    for key in param_keys:
        # Try the mapped source key first, then the key itself
        source_key = param_sources.get(key, key)
        val = source.get(source_key)
        if val is None:
            val = source.get(key)  # fallback
        if val is None:
            numeric: Optional[float] = None
        else:
            try:
                numeric = float(val)
            except (TypeError, ValueError):
                numeric = None
        if numeric is not None and not np.isfinite(numeric):
            numeric = None

        if param_stats and key in param_stats:
            stats = param_stats[key]
            mean = float(stats.get("mean", 0.0) or 0.0)
            std = float(stats.get("std", 1.0) or 1.0)
            if numeric is None:
                numeric = mean
            if normalize and std > 1e-8:
                numeric = (numeric - mean) / std
            elif normalize:
                numeric = 0.0
        elif numeric is None:
            numeric = 0.0

        values.append(numeric)

    return np.array(values, dtype=np.float32)


def extract_context_params_batch(
    df: Any,  # pd.DataFrame
    param_keys: List[str],
    param_sources: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """Extract context params from all rows of a DataFrame + compute stats.

    Args:
        df: DataFrame with rows as samples.
        param_keys: Parameter key names.
        param_sources: Maps param key → DataFrame column name.

    Returns:
        (params: (N, n_params) float32, stats: {key: {"mean": float, "std": float}})
        Stats can be stored in HarmonicContextConfig.context_param_stats.
    """
    import pandas as pd

    param_sources = param_sources or {}
    n = len(df)
    n_params = len(param_keys)
    params = np.zeros((n, n_params), dtype=np.float32)
    stats: Dict[str, Dict[str, float]] = {}

    for i, key in enumerate(param_keys):
        col_name = param_sources.get(key, key)
        if col_name in df.columns:
            vals = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0).values
        else:
            logger.warning("Context param '%s' (col '%s') not found in DataFrame", key, col_name)
            vals = np.zeros(n, dtype=np.float32)

        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        stats[key] = {"mean": mean_val, "std": std_val}

        # Normalise
        if std_val > 1e-8:
            params[:, i] = (vals - mean_val) / std_val
        else:
            params[:, i] = 0.0

    return params, stats


# ══════════════════════════════════════════════════════════════════════════════
# Channel extraction helpers
# ══════════════════════════════════════════════════════════════════════════════


def extract_channels_from_window(
    window: Dict[str, np.ndarray],
    config: Any,  # HarmonicContextConfig
    domain: Any = None,  # DomainConfig
) -> Optional[np.ndarray]:
    """Extract multi-channel signal array from a live session window.

    Tries, in order:
    1. ``config.input_columns`` (exact names)
    2. ``config.input_channel_roles`` (resolved via DomainConfig)
    3. ``config.input_column_patterns`` (regex match against window keys)

    Returns:
        (N, C) float64 array, or None if no channels found.
    """
    arrays: List[np.ndarray] = []

    # 1. Exact column names
    if config.input_columns:
        for col in config.input_columns:
            if col in window:
                arrays.append(np.asarray(window[col], dtype=np.float64))
        if arrays:
            return np.column_stack(arrays)

    # 2. Channel roles via DomainConfig
    if config.input_channel_roles and domain is not None:
        for role in config.input_channel_roles:
            ch_name = domain.resolve_channel(role, window.keys())
            if ch_name and ch_name in window:
                arrays.append(np.asarray(window[ch_name], dtype=np.float64))
        if arrays:
            return np.column_stack(arrays)

    # 3. Regex patterns
    if config.input_column_patterns:
        compiled = [re.compile(p) for p in config.input_column_patterns]
        for ch_name in sorted(window.keys()):
            for pat in compiled:
                if pat.search(ch_name):
                    arrays.append(np.asarray(window[ch_name], dtype=np.float64))
                    break
        if arrays:
            return np.column_stack(arrays)

    return None

"""Parametric windowing over a per-sample labelled raw series (C2).

This decouples *labelling* (done once, per sample — see ``scripts/label_raw_series.py``)
from *windowing* (a load-time parameter sweep). Given a long-format, per-sample
table with a continuous label, ``make_windows`` produces feature rows at any
``window_s`` / ``stride_s`` / ``gap_s`` / ``horizon_s`` without re-extracting from
the raw source. This enables window-size sweeps, overlapping windows for training
augmentation, and consistent detection-vs-prediction (gap) semantics.

Per-sample input schema (one row per timestamp, sorted within operation):
    operation_id : str       grouping key; windows never cross operations
    t            : float      sample time in seconds (monotonic within operation)
    <channel...> : float      one or more numeric signal channels
    phase        : str        one of PHASE_* below
    time_to_event_s : float   seconds until the next event (NaN if none ahead)
    event_id     : object     id of the next event (NaN/"" if none ahead)

The window label is **leakage-safe**: a window is dropped if it overlaps an
event/idle sample, positive only if it sits entirely in clean cutting and ends
within [gap, horizon] before an event, negative only if it is clean cutting and
comfortably far from any event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

# Per-sample phases.
PHASE_NORMAL = "normal"        # healthy active cutting
PHASE_PRE_EVENT = "pre_event"  # within the pre-event horizon (label source)
PHASE_EVENT = "event"          # the stop/breakage itself
PHASE_POST = "post_event"      # after the event, recovery
PHASE_IDLE = "idle"            # not actively cutting (approach/retract/pause)

# Phases that must NOT appear inside a usable window.
_DIRTY_PHASES = frozenset({PHASE_EVENT, PHASE_POST, PHASE_IDLE})

LABEL_POSITIVE = "pre_stoppage"
LABEL_NEGATIVE = "normal"


@dataclass
class WindowingParams:
    window_s: float = 60.0
    stride_s: float = 60.0       # == window_s -> non-overlapping (default, leakage-safe)
    gap_s: float = 0.0           # window ends gap_s before the event (0 = detection)
    horizon_s: float = 60.0      # a window is positive only within this lead time
    sample_rate_hz: float = 1.0
    negative_margin_s: float = 120.0  # negatives must be this far from any event
    min_coverage: float = 0.9    # require >= this fraction of expected samples present
    negative_block_s: float = 300.0  # group negatives into time blocks of this size
                                     # so block-bootstrap (C4) is not swamped by
                                     # thousands of singleton negative groups


def _channel_columns(df: pd.DataFrame) -> List[str]:
    reserved = {"operation_id", "t", "phase", "time_to_event_s", "event_id",
                "timestamp", "label", "sample_id"}
    return [
        c for c in df.columns
        if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
    ]


def _aggregate(window: pd.DataFrame, channels: List[str]) -> dict:
    """Summary stats per channel: mean / std / min / max / slope."""
    feats: dict = {}
    n = len(window)
    x = np.arange(n, dtype=float)
    x_centered = x - x.mean() if n > 1 else x
    denom = float((x_centered ** 2).sum()) or 1.0
    for ch in channels:
        col = window[ch].to_numpy(dtype=float)
        col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
        feats[f"{ch}_mean"] = float(col.mean())
        feats[f"{ch}_std"] = float(col.std())
        feats[f"{ch}_min"] = float(col.min())
        feats[f"{ch}_max"] = float(col.max())
        # least-squares slope over the window
        feats[f"{ch}_slope"] = float(((col - col.mean()) * x_centered).sum() / denom) if n > 1 else 0.0
    return feats


def window_to_window_metrics(window: pd.DataFrame, channels: Optional[List[str]] = None):
    """Build a real per-channel ``WindowMetrics`` from a window's raw channels (D2).

    The scorer's anomaly-deviation rule is inert without a per-channel
    ``WindowMetrics`` + warmed baseline (it returns not-triggered when
    ``metrics is None``). The fixed-feature experiment never carried one; the
    windower does, so it can revive that rule. Only time-domain fields are
    populated (the experiment series are 1 Hz, so spectral fields stay empty).
    """
    from backend.agents.core.metrics import WindowMetrics

    channels = channels or _channel_columns(window)
    means, stds, rms, peaks, crests = [], [], [], [], []
    for ch in channels:
        col = np.nan_to_num(window[ch].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        m = float(col.mean())
        sd = float(col.std())
        r = float(np.sqrt(np.mean(col ** 2)))
        pk = float(np.max(np.abs(col))) if col.size else 0.0
        means.append(m); stds.append(sd); rms.append(r); peaks.append(pk)
        crests.append(pk / r if r > 1e-9 else 0.0)
    return WindowMetrics(
        channel_means=means,
        channel_stds=stds,
        channel_rms=rms,
        channel_peaks=peaks,
        channel_crest_factors=crests,
        total_energy=float(sum(r ** 2 for r in rms)),
    )


def _label_window(window: pd.DataFrame, params: WindowingParams) -> Optional[tuple]:
    """Return (label, event_id) or None to DROP the window (leakage-safe).

    Decision point is the window END. A window is:
      - dropped   if it overlaps any dirty phase (event/post/idle),
      - positive  if clean AND ends within [gap, horizon] before an event,
      - negative  if clean AND the nearest event is > negative_margin away,
      - dropped   otherwise (ambiguous band between horizon and margin).
    """
    phases = set(window["phase"].astype(str))
    if phases & _DIRTY_PHASES:
        return None

    # time-to-event at the window end (last sample)
    tte_end = window["time_to_event_s"].to_numpy(dtype=float)
    last_tte = tte_end[-1] if len(tte_end) else np.nan
    ev_series = window["event_id"].to_numpy()
    last_ev = ev_series[-1] if len(ev_series) else None

    has_event_ahead = np.isfinite(last_tte)
    if has_event_ahead and (params.gap_s <= last_tte <= params.horizon_s + params.gap_s):
        return (LABEL_POSITIVE, last_ev)

    # negative: no event within the margin window
    min_tte = np.nanmin(tte_end) if np.isfinite(tte_end).any() else np.inf
    if not has_event_ahead or min_tte > params.negative_margin_s:
        # Coarse, per-operation time-block id so adjacent (autocorrelated)
        # negative windows share a block for the block bootstrap (C4).
        op = window["operation_id"].iloc[0] if "operation_id" in window else "op"
        block = int(window["t"].iloc[0] // max(1.0, params.negative_block_s))
        return (LABEL_NEGATIVE, f"neg::{op}::{block}")

    return None  # ambiguous band -> drop


def make_windows(
    labeled_raw: pd.DataFrame,
    params: Optional[WindowingParams] = None,
    **kwargs,
) -> pd.DataFrame:
    """Slice a per-sample labelled raw series into labelled feature windows.

    Windows never cross ``operation_id``. Overlap is controlled by
    ``stride_s`` (== ``window_s`` gives non-overlapping; the headline analyses
    should use non-overlapping, with overlap reserved for training augmentation
    — see C4). Every output row carries ``event_id`` so statistics can be
    block-bootstrapped by event rather than by (autocorrelated) window.
    """
    params = params or WindowingParams(**kwargs)
    if labeled_raw.empty:
        return pd.DataFrame()

    channels = _channel_columns(labeled_raw)
    win_n = max(1, int(round(params.window_s * params.sample_rate_hz)))
    stride_n = max(1, int(round(params.stride_s * params.sample_rate_hz)))
    min_n = int(np.ceil(params.min_coverage * win_n))

    rows: List[dict] = []
    for op, grp in labeled_raw.groupby("operation_id", sort=False):
        grp = grp.sort_values("t").reset_index(drop=True)
        n = len(grp)
        for start in range(0, max(1, n - win_n + 1), stride_n):
            window = grp.iloc[start:start + win_n]
            if len(window) < min_n:
                continue
            labelled = _label_window(window, params)
            if labelled is None:
                continue
            label, event_id = labelled
            feats = _aggregate(window, channels)
            feats.update({
                "operation_id": op,
                "window_start_t": float(window["t"].iloc[0]),
                "window_end_t": float(window["t"].iloc[-1]),
                "window_s": params.window_s,
                "gap_s": params.gap_s,
                "label": label,
                "event_id": event_id,
            })
            rows.append(feats)

    return pd.DataFrame(rows)

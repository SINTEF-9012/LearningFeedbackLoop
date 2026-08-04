#!/usr/bin/env python3
"""
Extract pre-stoppage data patterns as ground truth for tool breakage detection.

Uses the anomalous stops detected by detect_premature_stoppage.py as positive
labels — the operator stopped the machine because something went wrong.
Extracts multi-channel time-series windows *preceding* each stop event, then
builds a labelled dataset of "pre-break" vs "normal" cutting for downstream
ML training.

Data channels used:
  TYZBPS  – Spindle speed, feed rate, override, tool number, positions, temps
  BXCZ3M  – Operation status, axis power (spindle, X1, X2, Y, Z)
  7DTZHE  – Vibration harmonics (8 × X/Y), chatter detection, severity
  92SQBY  – Total energy, active/reactive power, power factor

Extraction strategies:
  1. Feature windows  – statistical features over 60s/30s/10s pre-event windows
                        → tabular dataset for classical ML (RF, XGBoost)
  2. Raw time-series  – full multi-channel 60s segments as numpy arrays
                        → ready for 1D-CNN / LSTM / transformer models
  3. Delta features   – rate-of-change comparing last 10s vs preceding 50s
                        → captures deviation from the tool's own baseline

Negative (normal) labels are drawn from the *same tool, same program* in
operations where no stop occurred, matched by cutting regime (similar spindle
speed and feed rate).

Usage:
    .venv/bin/python scripts/extract_pre_stoppage_patterns.py \\
        [--data-dir data/casedata] [--window 60] [--output-dir data/breakage_patterns]

    .venv/bin/python scripts/extract_pre_stoppage_patterns.py \
        --data-dir data/site_a \
        --case 'Site_a - MACHINE_A1 - CASE_A1' \
        --weak-label-report "$TMPDIR/site_a_unexpected_stops.json" \
        [--include-candidate-labels]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agents.processing.archive_staging import prepare_analysis_root


# ── Channel suffixes ──────────────────────────────────────────────────
MACHINE_STATE_SUFFIX = "TYZBPS"
AXIS_POWER_SUFFIX    = "BXCZ3M"
VIBRATION_SUFFIX     = "7DTZHE"
ENERGY_SUFFIX        = "92SQBY"


# ── Columns to load per channel ──────────────────────────────────────
TYZBPS_COLS = [
    "Spindle_Speed_Actual", "Spindle_Speed_Commanded",
    "Feed_Rate_Actual", "Feed_Rate_Commanded",
    "Feed_Override", "Spindle_Speed_Override",
    "Tool_Number", "Program_Name",
    "Position_MCS_X", "Position_MCS_Y", "Position_MCS_Z",
    "Temperature_Head", "Temperature_Room",
]

BXCZ3M_COLS = [
    "Operation_Status", "Power_Spindle",
    "Power_X1", "Power_X2", "Power_Y", "Power_Z",
]

VIBRATION_COLS = [
    "Vibration_Severity_X", "Vibration_Severity_Y",
    "Chatter_Detection_Amplitude_X", "Chatter_Detection_Amplitude_Y",
    "Chatter_Detection_Frequency_X", "Chatter_Detection_Frequency_Y",
    # Harmonics 1-4 (most energy), amplitudes only to keep feature count sane
    "Vibration_Harmonic_1_X_Amplitude", "Vibration_Harmonic_1_Y_Amplitude",
    "Vibration_Harmonic_2_X_Amplitude", "Vibration_Harmonic_2_Y_Amplitude",
    "Vibration_Harmonic_3_X_Amplitude", "Vibration_Harmonic_3_Y_Amplitude",
    "Vibration_Harmonic_4_X_Amplitude", "Vibration_Harmonic_4_Y_Amplitude",
    # And their frequencies
    "Vibration_Harmonic_1_X_Frequency", "Vibration_Harmonic_1_Y_Frequency",
    "Vibration_Harmonic_2_X_Frequency", "Vibration_Harmonic_2_Y_Frequency",
]

ENERGY_COLS = [
    "Energy_Total", "Power_Active", "Power_Reactive",
    "Power_Apparent", "Power_Factor",
]


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class StopEvent:
    """An operator-initiated stop during active cutting."""
    operation_id: str
    timestamp: pd.Timestamp
    tool_number: float
    severity: str
    stop_type: str
    spindle_rpm: float
    feed_rate: float
    feed_override: float
    stop_duration_s: float
    case_dir: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternSample:
    """An extracted pre-stoppage (or normal) pattern with features."""
    sample_id: str
    label: str                     # "pre_stoppage" or "normal"
    operation_id: str
    case_dir: str
    tool_number: float
    event_timestamp: str
    severity: str                  # from the stop event (or "none")
    stop_type: str
    window_seconds: float
    features: Dict[str, float]     # statistical features
    gap_seconds: float = 0.0       # prediction gap: window ends gap_s before event
    sample_rate_hz: float = 1.0    # sampling frequency (1 Hz for casedata CSVs)
    raw_series: Optional[Dict[str, np.ndarray]] = None   # multi-channel arrays

    @property
    def window_entries(self) -> int:
        """Number of data-points in the window: window_seconds × sample_rate_hz."""
        return int(self.window_seconds * self.sample_rate_hz)


# ── Helpers ───────────────────────────────────────────────────────────

def _operation_key(case_dir: str, operation_id: str) -> str:
    return f"{case_dir} / {operation_id}" if case_dir else operation_id

def find_csv(op_dir: Path, suffix: str) -> Optional[Path]:
    """Find the CSV for a given channel suffix in an operation directory."""
    for f in op_dir.glob("*.csv"):
        if suffix in f.name and not f.name.endswith("Zone.Identifier"):
            return f
    return None


def load_channel(csv_path: Path, usecols: list) -> pd.DataFrame:
    """Load a channel CSV with selected columns + timestamp."""
    header = pd.read_csv(csv_path, nrows=0)
    available = set(str(column) for column in header.columns)
    cols = ["timestamp"]
    cols.extend(c for c in usecols if c != "timestamp" and c in available)
    df = pd.read_csv(csv_path, usecols=cols, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ── 1. Detect operator stops (lightweight re-implementation) ──────────
#    We re-detect rather than import to keep this script self-contained.

def detect_operator_stops(
    ms: pd.DataFrame,
    ap: pd.DataFrame,
    operation_id: str,
    *,
    case_dir: str = "",
    spindle_threshold: float = 100.0,
    feed_threshold: float = 50.0,
    lookback_s: float = 10.0,
    min_cluster_gap_s: float = 120.0,
) -> List[StopEvent]:
    """Detect AUTO→STOPPED and AUTO→FEED_HOLD during active cutting.

    Events within ``min_cluster_gap_s`` of each other are treated as a
    single physical stop.  The most-severe event in each cluster is kept
    (preferring ``mid_cut_stop`` over ``feed_hold_mid_cut`` at equal
    severity).
    """
    raw_events: List[StopEvent] = []
    ap = ap.copy()
    ap["prev_status"] = ap["Operation_Status"].shift(1)

    transitions = [
        ("mid_cut_stop",      3.0, 0.0),
        ("feed_hold_mid_cut", 3.0, 2.0),
    ]
    for stop_type, from_s, to_s in transitions:
        mask = (ap["prev_status"] == from_s) & (ap["Operation_Status"] == to_s)
        for _, row in ap[mask].iterrows():
            t_stop = row["timestamp"]
            window = ms[
                (ms["timestamp"] >= t_stop - pd.Timedelta(seconds=lookback_s))
                & (ms["timestamp"] <= t_stop)
            ]
            if len(window) == 0:
                continue
            avg_spindle = float(window["Spindle_Speed_Actual"].mean())
            avg_feed    = float(window["Feed_Rate_Actual"].mean())
            avg_over    = float(window["Feed_Override"].mean())
            tool        = float(window["Tool_Number"].iloc[-1])

            if avg_spindle < spindle_threshold or avg_feed < feed_threshold:
                continue

            # How long was the stop?
            future = ap[ap["timestamp"] > t_stop]
            resumed = future[future["Operation_Status"].isin([2.0, 3.0])]
            dur = (
                (resumed["timestamp"].iloc[0] - t_stop).total_seconds()
                if len(resumed) > 0 else 0.0
            )

            if avg_spindle > 500 and avg_feed > 200:
                severity = "critical"
            elif avg_spindle > 200 or avg_feed > 100:
                severity = "high"
            else:
                severity = "medium"

            raw_events.append(StopEvent(
                operation_id=operation_id,
                timestamp=t_stop,
                tool_number=tool,
                severity=severity,
                stop_type=stop_type,
                spindle_rpm=avg_spindle,
                feed_rate=avg_feed,
                feed_override=avg_over,
                stop_duration_s=dur,
                case_dir=case_dir,
            ))

    # ── Cluster de-duplication ────────────────────────────────────────
    # Sort by time, then merge events within min_cluster_gap_s.
    # Keep the most-severe event per cluster (mid_cut_stop > feed_hold).
    if not raw_events:
        return []

    raw_events.sort(key=lambda e: e.timestamp)
    severity_rank = {"critical": 0, "high": 1, "medium": 2}
    type_rank = {"mid_cut_stop": 0, "feed_hold_mid_cut": 1}

    def _pick_best(cluster: List[StopEvent]) -> StopEvent:
        """Select the single most representative event from a cluster."""
        cluster.sort(key=lambda e: (
            severity_rank.get(e.severity, 9),
            type_rank.get(e.stop_type, 9),
        ))
        best = cluster[0]
        # Use the longest recorded stop duration from the cluster
        best.stop_duration_s = max(e.stop_duration_s for e in cluster)
        return best

    events: List[StopEvent] = []
    current_cluster: List[StopEvent] = [raw_events[0]]
    for ev in raw_events[1:]:
        gap = (ev.timestamp - current_cluster[-1].timestamp).total_seconds()
        if gap <= min_cluster_gap_s:
            current_cluster.append(ev)
        else:
            events.append(_pick_best(current_cluster))
            current_cluster = [ev]
    events.append(_pick_best(current_cluster))

    return events


# ── 2. Load all channels for an operation ─────────────────────────────

@dataclass
class OpData:
    """All channel data for one operation."""
    op_id: str
    case_dir: str
    ms: pd.DataFrame          # TYZBPS
    ap: pd.DataFrame          # BXCZ3M
    vib: Optional[pd.DataFrame] = None  # 7DTZHE
    energy: Optional[pd.DataFrame] = None  # 92SQBY


def load_operation(op_dir: Path, *, case_dir: str = "") -> Optional[OpData]:
    """Load all channels for one operation."""
    op_id = op_dir.name

    ms_path = find_csv(op_dir, MACHINE_STATE_SUFFIX)
    ap_path = find_csv(op_dir, AXIS_POWER_SUFFIX)
    if not ms_path or not ap_path:
        return None

    ms = load_channel(ms_path, TYZBPS_COLS)
    ap = load_channel(ap_path, BXCZ3M_COLS)

    vib_path = find_csv(op_dir, VIBRATION_SUFFIX)
    vib = load_channel(vib_path, VIBRATION_COLS) if vib_path else None

    en_path = find_csv(op_dir, ENERGY_SUFFIX)
    energy = load_channel(en_path, ENERGY_COLS) if en_path else None

    return OpData(op_id=op_id, case_dir=case_dir, ms=ms, ap=ap, vib=vib, energy=energy)


def _build_report_stop_event(entry: Dict[str, Any], *, candidate: bool) -> Optional[StopEvent]:
    timestamp = entry.get("timestamp")
    operation_id = str(entry.get("operation_id") or "")
    if not timestamp or not operation_id:
        return None
    try:
        parsed_timestamp = pd.Timestamp(timestamp)
    except Exception:
        return None
    stop_type = str(entry.get("event_type") or entry.get("label_source") or "unexpected_stop_candidate")
    if not candidate:
        stop_type = str(entry.get("stop_type") or entry.get("label_source") or "unexpected_stop")
    return StopEvent(
        operation_id=operation_id,
        case_dir=str(entry.get("case_dir") or ""),
        timestamp=parsed_timestamp,
        tool_number=float(entry.get("tool_number") or -1.0),
        severity=str(entry.get("severity") or entry.get("confidence") or "medium"),
        stop_type=stop_type,
        spindle_rpm=float(entry.get("spindle_rpm_before") or 0.0),
        feed_rate=float(entry.get("feed_rate_before") or 0.0),
        feed_override=float(entry.get("feed_override_before") or 0.0),
        stop_duration_s=float(entry.get("time_in_stopped_s") or 0.0),
        context=dict(entry.get("context") or {}),
    )


def load_weak_label_events(
    report_path: Path,
    *,
    include_candidate_labels: bool = False,
) -> Dict[str, List[StopEvent]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    events_by_operation: Dict[str, List[StopEvent]] = defaultdict(list)

    detailed_operator_events = payload.get("operator_stop_events") or []
    detailed_candidate_events = payload.get("baseline_anomalies") or []

    if detailed_operator_events:
        for entry in detailed_operator_events:
            event = _build_report_stop_event(entry, candidate=False)
            if event is not None:
                events_by_operation[_operation_key(event.case_dir, event.operation_id)].append(event)
    else:
        for entry in payload.get("weak_labels") or []:
            if entry.get("label") != "unexpected_stop":
                continue
            event = _build_report_stop_event(entry, candidate=False)
            if event is not None:
                events_by_operation[_operation_key(event.case_dir, event.operation_id)].append(event)

    if include_candidate_labels:
        if detailed_candidate_events:
            for entry in detailed_candidate_events:
                event = _build_report_stop_event(entry, candidate=True)
                if event is not None:
                    events_by_operation[_operation_key(event.case_dir, event.operation_id)].append(event)
        else:
            for entry in payload.get("weak_labels") or []:
                if entry.get("label") != "unexpected_stop_candidate":
                    continue
                event = _build_report_stop_event(entry, candidate=True)
                if event is not None:
                    events_by_operation[_operation_key(event.case_dir, event.operation_id)].append(event)

    for events in events_by_operation.values():
        events.sort(key=lambda event: event.timestamp)
    return dict(events_by_operation)


def _align_report_events_to_loaded_ops(
    ops: Dict[str, OpData],
    report_events: Dict[str, List[StopEvent]],
) -> Tuple[Dict[str, List[StopEvent]], List[str]]:
    aligned: Dict[str, List[StopEvent]] = defaultdict(list)
    skipped: List[str] = []

    for op_key, events in report_events.items():
        if op_key in ops:
            aligned[op_key].extend(events)
            continue

        for event in events:
            fallback_matches = [
                candidate_key
                for candidate_key, op in ops.items()
                if op.op_id == event.operation_id
            ]
            if len(fallback_matches) == 1:
                matched_key = fallback_matches[0]
                event.case_dir = ops[matched_key].case_dir
                aligned[matched_key].append(event)
            else:
                skipped.append(op_key)
                break

    for events in aligned.values():
        events.sort(key=lambda event: event.timestamp)
    return dict(aligned), skipped


# ── 3. Extract multi-channel window around a timestamp ────────────────

def _trim_deceleration(
    slices: Dict[str, pd.DataFrame],
    feed_col: str = "Feed_Rate_Actual",
    threshold_frac: float = 0.5,
    min_consecutive: int = 3,
) -> Tuple[Dict[str, pd.DataFrame], float]:
    """Trim window slices to exclude the deceleration ramp before a stop.

    Scans the machine_state channel's feed rate backwards from the end.
    When it finds the point where feed drops below *threshold_frac × median*
    for at least *min_consecutive* samples, it truncates ALL channels to
    end at that point.

    Returns (trimmed_slices, trimmed_seconds_removed).
    """
    ms = slices.get("machine_state")
    if ms is None or feed_col not in ms.columns or len(ms) < 10:
        return slices, 0.0

    feed = ms[feed_col].values.astype(float)
    # Robust median: use the first 75% of the window (less affected by ramp)
    n_stable = max(10, int(len(feed) * 0.75))
    median_feed = float(np.median(feed[:n_stable]))

    if median_feed < 10.0:  # tool not really cutting — skip trim
        return slices, 0.0

    threshold = threshold_frac * median_feed

    # Scan backwards: find first run of min_consecutive samples below threshold
    below = feed < threshold
    trim_idx = len(feed)  # default: no trim
    run = 0
    for i in range(len(below) - 1, -1, -1):
        if below[i]:
            run += 1
            if run >= min_consecutive:
                trim_idx = i
        else:
            if run >= min_consecutive:
                break  # found the deceleration edge
            run = 0

    if trim_idx >= len(feed) - min_consecutive:
        return slices, 0.0  # nothing meaningful to trim

    # Compute the cut-off timestamp
    t_cut = ms.iloc[trim_idx]["timestamp"]
    secs_removed = (ms.iloc[-1]["timestamp"] - t_cut).total_seconds()

    trimmed: Dict[str, pd.DataFrame] = {}
    for name, df in slices.items():
        mask = df["timestamp"] < t_cut
        sl = df.loc[mask].copy()
        if len(sl) > 0:
            trimmed[name] = sl

    return trimmed, secs_removed


def extract_window(
    op: OpData,
    t_end: pd.Timestamp,
    window_s: float = 60.0,
) -> Dict[str, pd.DataFrame]:
    """Extract a time-aligned window from all channels ending at t_end.

    Returns {channel_name: DataFrame slice} for each available channel.
    """
    t_start = t_end - pd.Timedelta(seconds=window_s)
    slices: Dict[str, pd.DataFrame] = {}

    for name, df in [
        ("machine_state", op.ms),
        ("axis_power", op.ap),
        ("vibration", op.vib),
        ("energy", op.energy),
    ]:
        if df is None:
            continue
        mask = (df["timestamp"] >= t_start) & (df["timestamp"] < t_end)
        sl = df.loc[mask].copy()
        if len(sl) > 0:
            slices[name] = sl

    return slices


# ── 4. Feature extraction from a window ───────────────────────────────

def _safe_slope(arr: np.ndarray) -> float:
    """Compute linear-regression slope over an array; 0 if too short."""
    if len(arr) < 3:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    with np.errstate(invalid="ignore"):
        coeffs = np.polyfit(x, arr, 1)
    return float(coeffs[0])


def _nan_safe(func, arr, default=0.0):
    """Run a numpy nan-aware function, returning *default* for empty / all-NaN."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return default
    return float(func(valid))


def _stat_features(
    arr: np.ndarray, prefix: str,
) -> Dict[str, float]:
    """Compute standard statistical features for a 1D signal."""
    zeros = {f"{prefix}_{s}": 0.0 for s in [
        "mean", "std", "min", "max", "range",
        "slope", "p25", "p75", "iqr",
    ]}
    if len(arr) == 0:
        return zeros
    arr = arr.astype(float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return zeros
    p25, p75 = np.percentile(valid, [25, 75]) if len(valid) >= 4 else (0.0, 0.0)
    vmin = float(np.min(valid))
    vmax = float(np.max(valid))
    return {
        f"{prefix}_mean":  float(np.mean(valid)),
        f"{prefix}_std":   float(np.std(valid)),
        f"{prefix}_min":   vmin,
        f"{prefix}_max":   vmax,
        f"{prefix}_range": vmax - vmin,
        f"{prefix}_slope": _safe_slope(arr),
        f"{prefix}_p25":   float(p25),
        f"{prefix}_p75":   float(p75),
        f"{prefix}_iqr":   float(p75 - p25),
    }


def _delta_features(
    arr: np.ndarray, prefix: str,
    split_frac: float = 1/6,   # last 10s of a 60s window
    window_s: float = 60.0,
) -> Dict[str, float]:
    """Compare the last portion of the window against the earlier portion.

    This captures *change* relative to the tool's own recent baseline.

    The split defaults to "last 10 s" for 60 s windows.  For other window
    sizes the split fraction is recomputed so the late-portion always
    represents ~10 s (clamped to at least 20% of the window).
    """
    # Derive a window-aware split: keep 10 s absolute, but never less than 20%
    if window_s != 60.0:
        ideal_late_frac = min(10.0 / max(window_s, 1.0), 0.8)
        split_frac = max(ideal_late_frac, 0.2)

    if len(arr) < 10:
        return {
            f"{prefix}_delta_mean": 0.0,
            f"{prefix}_delta_std":  0.0,
            f"{prefix}_delta_max":  0.0,
        }
    split = max(3, int(len(arr) * (1 - split_frac)))
    early = arr[:split].astype(float)
    late  = arr[split:].astype(float)
    e_mean = _nan_safe(np.mean, early)
    e_std  = _nan_safe(np.std, early, default=1.0)
    l_mean = _nan_safe(np.mean, late)
    l_std  = _nan_safe(np.std, late)
    l_max  = _nan_safe(np.max, late)
    return {
        f"{prefix}_delta_mean": l_mean - e_mean,
        f"{prefix}_delta_std":  l_std  - e_std,
        f"{prefix}_delta_max":  l_max  - e_mean,
    }


def extract_features(
    slices: Dict[str, pd.DataFrame],
    window_s: float = 60.0,
) -> Dict[str, float]:
    """Extract a flat dictionary of features from a multi-channel window.

    Returns ~150+ features covering:
      - Spindle & feed dynamics (mean, std, slope, range, deltas)
      - Axis power draws (all 5 axes)
      - Vibration severity & harmonics
      - Energy / power consumption
      - Cross-signal: power-vibration, spindle-feed correlations
    """
    feats: Dict[str, float] = {"window_seconds": window_s}

    # ── Machine state (TYZBPS) ────────────────────────────────────────
    ms = slices.get("machine_state")
    if ms is not None and len(ms) > 0:
        for col, prefix in [
            ("Spindle_Speed_Actual",    "spindle_actual"),
            ("Spindle_Speed_Commanded",  "spindle_cmd"),
            ("Feed_Rate_Actual",         "feed_actual"),
            ("Feed_Rate_Commanded",      "feed_cmd"),
            ("Feed_Override",            "feed_override"),
            ("Spindle_Speed_Override",   "spindle_override"),
        ]:
            if col in ms.columns:
                arr = ms[col].values
                feats.update(_stat_features(arr, prefix))
                feats.update(_delta_features(arr, prefix))

        # Spindle error (actual - commanded)
        if "Spindle_Speed_Actual" in ms.columns and "Spindle_Speed_Commanded" in ms.columns:
            err = ms["Spindle_Speed_Actual"].values - ms["Spindle_Speed_Commanded"].values
            feats.update(_stat_features(err, "spindle_error"))
            feats.update(_delta_features(err, "spindle_error"))

        # Feed error
        if "Feed_Rate_Actual" in ms.columns and "Feed_Rate_Commanded" in ms.columns:
            err = ms["Feed_Rate_Actual"].values - ms["Feed_Rate_Commanded"].values
            feats.update(_stat_features(err, "feed_error"))
            feats.update(_delta_features(err, "feed_error"))

        # Positional velocity (rate of tool movement)
        for axis in ["X", "Y", "Z"]:
            col = f"Position_MCS_{axis}"
            if col in ms.columns:
                pos = ms[col].values.astype(float)
                vel = np.diff(pos)  # position change per second (1Hz sampling)
                feats.update(_stat_features(vel, f"velocity_{axis.lower()}"))

        # Temperature trends
        for col in ["Temperature_Head", "Temperature_Room"]:
            if col in ms.columns:
                arr = ms[col].values
                feats.update(_stat_features(arr, col.lower()))
                feats.update(_delta_features(arr, col.lower()))

    # ── Axis power (BXCZ3M) ──────────────────────────────────────────
    ap = slices.get("axis_power")
    if ap is not None and len(ap) > 0:
        for col, prefix in [
            ("Power_Spindle", "power_spindle"),
            ("Power_X1",      "power_x1"),
            ("Power_X2",      "power_x2"),
            ("Power_Y",       "power_y"),
            ("Power_Z",       "power_z"),
        ]:
            if col in ap.columns:
                arr = ap[col].values
                feats.update(_stat_features(arr, prefix))
                feats.update(_delta_features(arr, prefix))

        # Total axis power
        power_cols = [c for c in ["Power_Spindle", "Power_X1", "Power_X2",
                                   "Power_Y", "Power_Z"] if c in ap.columns]
        if power_cols:
            total_power = ap[power_cols].sum(axis=1).values
            feats.update(_stat_features(total_power, "power_total_axes"))
            feats.update(_delta_features(total_power, "power_total_axes"))

    # ── Vibration (7DTZHE) ────────────────────────────────────────────
    vib = slices.get("vibration")
    if vib is not None and len(vib) > 0:
        for col, prefix in [
            ("Vibration_Severity_X",            "vib_severity_x"),
            ("Vibration_Severity_Y",            "vib_severity_y"),
            ("Chatter_Detection_Amplitude_X",   "chatter_amp_x"),
            ("Chatter_Detection_Amplitude_Y",   "chatter_amp_y"),
            ("Chatter_Detection_Frequency_X",   "chatter_freq_x"),
            ("Chatter_Detection_Frequency_Y",   "chatter_freq_y"),
        ]:
            if col in vib.columns:
                arr = vib[col].values
                feats.update(_stat_features(arr, prefix))
                feats.update(_delta_features(arr, prefix))

        # Harmonic amplitudes (1-4, X and Y)
        for h in range(1, 5):
            for axis in ["X", "Y"]:
                amp_col  = f"Vibration_Harmonic_{h}_{axis}_Amplitude"
                freq_col = f"Vibration_Harmonic_{h}_{axis}_Frequency"
                if amp_col in vib.columns:
                    arr = vib[amp_col].values
                    feats.update(_stat_features(arr, f"vib_h{h}_{axis.lower()}_amp"))
                    feats.update(_delta_features(arr, f"vib_h{h}_{axis.lower()}_amp"))
                if freq_col in vib.columns:
                    arr = vib[freq_col].values
                    feats.update(_stat_features(arr, f"vib_h{h}_{axis.lower()}_freq"))

        # Combined severity
        sev_x = vib.get("Vibration_Severity_X")
        sev_y = vib.get("Vibration_Severity_Y")
        if sev_x is not None and sev_y is not None:
            combined = np.sqrt(sev_x.values**2 + sev_y.values**2)
            feats.update(_stat_features(combined, "vib_severity_combined"))
            feats.update(_delta_features(combined, "vib_severity_combined"))

    # ── Energy (92SQBY) ───────────────────────────────────────────────
    en = slices.get("energy")
    if en is not None and len(en) > 0:
        for col, prefix in [
            ("Power_Active",   "power_active"),
            ("Power_Reactive", "power_reactive"),
            ("Power_Apparent", "power_apparent"),
            ("Power_Factor",   "power_factor"),
        ]:
            if col in en.columns:
                arr = en[col].values
                feats.update(_stat_features(arr, prefix))
                feats.update(_delta_features(arr, prefix))

    # ── Cross-signal correlations ─────────────────────────────────────
    # These capture coupling between subsystems — breaks often show
    # decorrelation (vibration increases while power doesn't, etc.)
    feats.update(_cross_correlations(slices))

    return feats


def _cross_correlations(slices: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Compute cross-signal correlations between channels."""
    cc: Dict[str, float] = {}

    def _corr(a: np.ndarray, b: np.ndarray, name: str) -> None:
        min_len = min(len(a), len(b))
        if min_len < 5:
            cc[name] = 0.0
            return
        a, b = a[:min_len].astype(float), b[:min_len].astype(float)
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            cc[name] = 0.0
            return
        cc[name] = float(np.corrcoef(a, b)[0, 1])

    ms  = slices.get("machine_state")
    ap  = slices.get("axis_power")
    vib = slices.get("vibration")
    en  = slices.get("energy")

    if ms is not None and ap is not None:
        if "Spindle_Speed_Actual" in ms.columns and "Power_Spindle" in ap.columns:
            _corr(ms["Spindle_Speed_Actual"].values,
                   ap["Power_Spindle"].values,
                   "corr_spindle_speed_power")
        if "Feed_Rate_Actual" in ms.columns and "Power_Y" in ap.columns:
            _corr(ms["Feed_Rate_Actual"].values,
                   ap["Power_Y"].values,
                   "corr_feed_power_y")

    if ap is not None and vib is not None:
        if "Power_Spindle" in ap.columns and "Vibration_Severity_X" in vib.columns:
            _corr(ap["Power_Spindle"].values,
                   vib["Vibration_Severity_X"].values,
                   "corr_spindle_power_vib_x")
        if "Power_Spindle" in ap.columns and "Vibration_Severity_Y" in vib.columns:
            _corr(ap["Power_Spindle"].values,
                   vib["Vibration_Severity_Y"].values,
                   "corr_spindle_power_vib_y")

    if ms is not None and vib is not None:
        if "Spindle_Speed_Actual" in ms.columns and "Vibration_Severity_X" in vib.columns:
            _corr(ms["Spindle_Speed_Actual"].values,
                   vib["Vibration_Severity_X"].values,
                   "corr_spindle_speed_vib_x")

    if ms is not None and en is not None:
        if "Feed_Rate_Actual" in ms.columns and "Power_Active" in en.columns:
            _corr(ms["Feed_Rate_Actual"].values,
                   en["Power_Active"].values,
                   "corr_feed_power_active")

    return cc


# ── 5. Extract raw time-series windows (for deep learning) ───────────

# Columns to include in the raw multi-channel tensor
RAW_SERIES_COLUMNS: Dict[str, List[str]] = {
    "machine_state": [
        "Spindle_Speed_Actual", "Feed_Rate_Actual",
        "Feed_Override", "Spindle_Speed_Override",
    ],
    "axis_power": [
        "Power_Spindle", "Power_X1", "Power_X2", "Power_Y", "Power_Z",
    ],
    "vibration": [
        "Vibration_Severity_X", "Vibration_Severity_Y",
        "Chatter_Detection_Amplitude_X", "Chatter_Detection_Amplitude_Y",
        "Vibration_Harmonic_1_X_Amplitude", "Vibration_Harmonic_1_Y_Amplitude",
        "Vibration_Harmonic_2_X_Amplitude", "Vibration_Harmonic_2_Y_Amplitude",
    ],
    "energy": [
        "Power_Active", "Power_Reactive", "Power_Factor",
    ],
}


def extract_raw_series(
    slices: Dict[str, pd.DataFrame],
    window_s: float = 60.0,
    sample_rate_hz: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Extract raw column arrays from window slices for deep-learning use.

    Returns {column_name: 1D array}, each of length ≈ window_s × sample_rate_hz.
    All arrays are padded/truncated to exactly ``int(window_s * sample_rate_hz)``
    entries.  At 1 Hz this equals *window_s* samples; at 1 kHz this equals
    ``1000 * window_s`` samples.
    """
    target_len = int(window_s * sample_rate_hz)
    series: Dict[str, np.ndarray] = {}

    for channel_name, cols in RAW_SERIES_COLUMNS.items():
        df = slices.get(channel_name)
        if df is None:
            for col in cols:
                series[col] = np.zeros(target_len, dtype=np.float32)
            continue
        for col in cols:
            if col in df.columns:
                arr = df[col].values.astype(np.float32)
            else:
                arr = np.zeros(0, dtype=np.float32)
            # Pad or truncate to target length
            if len(arr) >= target_len:
                arr = arr[-target_len:]   # take the latest window_s seconds
            else:
                arr = np.pad(arr, (target_len - len(arr), 0),
                             mode="constant", constant_values=0.0)
            series[col] = arr

    return series


# ── 6. Find normal (negative) windows ────────────────────────────────

def find_normal_windows(
    ops: Dict[str, OpData],
    stop_events: Dict[str, List[StopEvent]],
    window_s: float = 60.0,
    min_gap_s: float = 120.0,
    samples_per_tool: int = 3,           # fallback per-op count when no target given
    sample_rate_hz: float = 1.0,
    target_per_op: Optional[Dict[str, int]] = None,
    random_seed: int = 42,
) -> List[Tuple[str, pd.Timestamp, float]]:
    """Find normal cutting windows to serve as the negative class.

    Rebalanced (2026-06-16): the previous version capped negatives at
    ``samples_per_tool`` (=3) per stop-tool and took a single window per
    segment, while the positive class is *every* detected stop. With many
    stops this produced a heavily positive-imbalanced dataset (and zero
    negatives for operations whose stop-tools never cut cleanly elsewhere),
    which makes F1 meaningless (an always-positive classifier wins).

    This version instead **tiles all clean cutting** with non-overlapping
    windows (outside the ±``min_gap_s`` no-go zone around every stop) and
    samples, per operation, ``target_per_op`` windows — set to that operation's
    positive count for a ~1:1 class balance. Sampling is seeded for
    reproducibility. Operations with genuinely no clean cutting away from stops
    yield no negatives (and should be dropped downstream — they cannot
    discriminate).

    Returns list of (op_key, window_end_timestamp, tool_number).
    """
    rng = np.random.RandomState(random_seed)

    # Build "no-go" ranges (around every stop ± min_gap) so negatives are not
    # secretly pre-stop windows.
    no_go: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(list)
    for op_key, events in stop_events.items():
        for e in events:
            no_go[op_key].append((
                e.timestamp - pd.Timedelta(seconds=min_gap_s),
                e.timestamp + pd.Timedelta(seconds=min_gap_s),
            ))

    window_entries = int(window_s * sample_rate_hz)
    stride = max(1, window_entries)  # non-overlapping tiles

    normals: List[Tuple[str, pd.Timestamp, float]] = []

    for op_key, op in ops.items():
        ms = op.ms
        if ms is None or ms.empty or "Spindle_Speed_Actual" not in ms.columns:
            continue

        cutting_mask = (ms["Spindle_Speed_Actual"] > 100) & (ms["Feed_Rate_Actual"] > 50)
        cutting_idx = ms.index[cutting_mask]
        if len(cutting_idx) < window_entries:
            continue

        # Contiguous cutting segments, then tile each with non-overlapping windows.
        breaks = np.where(np.diff(cutting_idx) > 2)[0]
        segments = np.split(cutting_idx, breaks + 1)
        op_no_go = no_go.get(op_key, [])

        candidates: List[Tuple[str, pd.Timestamp, float]] = []
        for seg in segments:
            if len(seg) < window_entries:
                continue
            for start in range(0, len(seg) - window_entries + 1, stride):
                end_idx = seg[start + window_entries - 1]
                t_end = ms.loc[end_idx, "timestamp"]
                if any(ng_s <= t_end <= ng_e for ng_s, ng_e in op_no_go):
                    continue
                tool = float(ms.loc[end_idx, "Tool_Number"]) if "Tool_Number" in ms.columns else 0.0
                candidates.append((op_key, t_end, tool))

        if not candidates:
            continue

        # Take enough to balance this operation's positive count (~1:1).
        want = (target_per_op or {}).get(op_key, samples_per_tool * 4)
        want = min(int(want), len(candidates))
        if want >= len(candidates):
            chosen = candidates
        else:
            sel = sorted(rng.choice(len(candidates), size=want, replace=False))
            chosen = [candidates[i] for i in sel]
        normals.extend(chosen)

    return normals


# ── 7. Main pipeline ─────────────────────────────────────────────────

def run_extraction(
    data_dir: Path,
    window_s: float = 60.0,
    output_dir: Path = Path("data/breakage_patterns"),
    save_raw: bool = True,
    gap_s: float = 0.0,
    preloaded: Optional[Tuple[Dict[str, "OpData"], Dict[str, List["StopEvent"]]]] = None,
    sample_rate_hz: float = 1.0,
    min_cluster_gap_s: float = 120.0,
    case: Optional[str] = None,
    weak_label_report: Optional[Path] = None,
    include_candidate_labels: bool = False,
    include_archives: bool = True,
) -> Tuple[List[PatternSample], Dict[str, OpData], Dict[str, List[StopEvent]]]:
    """Run full extraction pipeline.

    1. Load all operations (all 4 channels)
    2. Detect operator stops (positive labels)
    3. Find matched normal windows (negative labels)
    4. Extract features + raw series for every window
    5. Save to disk

    Parameters
    ----------
    window_s : float
        Window duration in seconds.
    sample_rate_hz : float
        Sampling frequency in Hz.  Determines how many data-points
        (entries) correspond to ``window_s`` seconds:
        ``window_entries = int(window_s * sample_rate_hz)``.
        Default 1.0 (casedata CSVs are 1 Hz).
    gap_s : float
        Prediction gap in seconds.  When > 0, the pre-break feature
        window ends gap_s seconds **before** the stop event, turning
        the task from stop *detection* to stop *prediction*.
        Normal windows are unaffected (they are already far from stops).
    preloaded : tuple, optional
        (ops, all_stops) to skip data loading and stop detection
        when sweeping multiple gap values.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_data_dir, temp_root, staging_summary = prepare_analysis_root(
        data_dir,
        case=case,
        include_archives=include_archives,
    )

    try:
        if staging_summary["extracted_archives"] > 0:
            print(
                f"Prepared archive view: linked {staging_summary['linked_dirs']} unpacked operations, "
                f"extracted {staging_summary['extracted_archives']} archived operations"
            )

        # ── Load data (or use preloaded) ──────────────────────────────
        if preloaded is not None:
            ops, all_stops = preloaded
            total_stops = sum(len(v) for v in all_stops.values())
            print(f"  Using preloaded data: {len(ops)} operations, {total_stops} stops")
        else:
            ops = {}
            if case is not None:
                case_dirs = [prepared_data_dir / case] if (prepared_data_dir / case).is_dir() else []
            else:
                case_dirs = sorted(
                    d for d in prepared_data_dir.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                )

            for case_dir_path in case_dirs:
                op_dirs = sorted(
                    d for d in case_dir_path.iterdir()
                    if d.is_dir() and d.name.startswith("OF")
                )
                print(f"\n{'='*70}")
                print(f"Case: {case_dir_path.name}")
                print(f"Operations: {[d.name for d in op_dirs]}")
                print(f"{'='*70}")

                for op_dir in op_dirs:
                    op_id = op_dir.name
                    print(f"\n  Loading {op_id}... ", end="", flush=True)
                    op = load_operation(op_dir, case_dir=case_dir_path.name)
                    if op is None:
                        print("SKIP (missing channels)")
                        continue
                    op_key = _operation_key(case_dir_path.name, op_id)
                    ops[op_key] = op
                    n_vib = len(op.vib) if op.vib is not None else 0
                    n_en = len(op.energy) if op.energy is not None else 0
                    print(f"OK  ms={len(op.ms):,}  ap={len(op.ap):,}  vib={n_vib:,}  energy={n_en:,}")

            # ── Detect or load positives ──────────────────────────────
            print(f"\n{'='*70}")
            if weak_label_report is not None:
                print("Loading weak-label report (positive labels)")
            else:
                print("Detecting operator-initiated stops (positive labels)")
            print(f"{'='*70}")

            if weak_label_report is not None:
                report_events = load_weak_label_events(
                    weak_label_report,
                    include_candidate_labels=include_candidate_labels,
                )
                all_stops, skipped_ops = _align_report_events_to_loaded_ops(ops, report_events)
                total_stops = sum(len(v) for v in all_stops.values())
                for op_key, stops in sorted(all_stops.items()):
                    n_crit = sum(1 for s in stops if s.severity == "critical")
                    n_high = sum(1 for s in stops if s.severity == "high")
                    print(f"  {op_key}: {len(stops)} report labels (🔴 {n_crit} critical, 🟠 {n_high} high)")
                if skipped_ops:
                    print(f"  ⚠ Skipped labels for unloaded operations: {sorted(set(skipped_ops))}")
                print(f"\n  Total positive events from report: {total_stops}")
            else:
                all_stops = {}
                total_stops = 0
                for op_key, op in ops.items():
                    stops = detect_operator_stops(
                        op.ms,
                        op.ap,
                        op.op_id,
                        case_dir=op.case_dir,
                        min_cluster_gap_s=min_cluster_gap_s,
                    )
                    all_stops[op_key] = stops
                    total_stops += len(stops)
                    n_crit = sum(1 for s in stops if s.severity == "critical")
                    n_high = sum(1 for s in stops if s.severity == "high")
                    print(f"  {op_key}: {len(stops)} stops (🔴 {n_crit} critical, 🟠 {n_high} high)")

                print(f"\n  Total positive events: {total_stops}")

        # ── Find normal windows ───────────────────────────────────────
        print(f"\n{'='*70}")
        print("Finding matched normal windows (negative labels)")
        print(f"{'='*70}")

        # Target a ~1:1 class balance: sample, per operation, as many normal
        # windows as that operation has stops. Avoids the positive-imbalance
        # that makes F1 meaningless.
        target_per_op = {op_key: len(stops) for op_key, stops in all_stops.items()}
        normals = find_normal_windows(
            ops, all_stops,
            window_s=window_s,
            samples_per_tool=3,
            sample_rate_hz=sample_rate_hz,
            target_per_op=target_per_op,
        )
        print(f"  Normal windows found: {len(normals)} "
              f"(target ~{sum(target_per_op.values())} for 1:1 balance)")

        # ── Extract features ──────────────────────────────────────────
        print(f"\n{'='*70}")
        window_entries = int(window_s * sample_rate_hz)
        print(f"Extracting features (window={window_s}s × {sample_rate_hz} Hz = {window_entries} entries)")
        print(f"{'='*70}")

        samples: List[PatternSample] = []
        raw_arrays: Dict[str, Dict[str, np.ndarray]] = {}

        sample_idx = 0
        skipped_gap = 0
        for op_key, stops in all_stops.items():
            op = ops.get(op_key)
            if op is None:
                continue
            for ev in stops:
                sample_id = f"break_{sample_idx:04d}"
                sample_idx += 1

                t_end = ev.timestamp - pd.Timedelta(seconds=gap_s)
                slices = extract_window(op, t_end, window_s)
                if not slices or all(len(v) < 5 for v in slices.values()):
                    skipped_gap += 1
                    continue

                slices, trim_secs = _trim_deceleration(slices)
                if not slices or all(len(v) < 5 for v in slices.values()):
                    skipped_gap += 1
                    continue

                feats = extract_features(slices, window_s)
                feats["trim_seconds_removed"] = trim_secs
                feats["sample_rate_hz"] = sample_rate_hz
                feats["window_entries"] = int(window_s * sample_rate_hz)
                feats["event_spindle_rpm"] = ev.spindle_rpm
                feats["event_feed_rate"] = ev.feed_rate
                feats["event_feed_override"] = ev.feed_override
                feats["event_stop_duration_s"] = ev.stop_duration_s

                samples.append(PatternSample(
                    sample_id=sample_id,
                    label="pre_stoppage",
                    operation_id=op.op_id,
                    case_dir=ev.case_dir or op.case_dir,
                    tool_number=ev.tool_number,
                    event_timestamp=str(ev.timestamp),
                    severity=ev.severity,
                    stop_type=ev.stop_type,
                    window_seconds=window_s,
                    gap_seconds=gap_s,
                    sample_rate_hz=sample_rate_hz,
                    features=feats,
                ))

                if save_raw:
                    raw_arrays[sample_id] = extract_raw_series(slices, window_s, sample_rate_hz=sample_rate_hz)

        for op_key, t_end, tool_num in normals:
            sample_id = f"normal_{sample_idx:04d}"
            sample_idx += 1

            op = ops[op_key]
            slices = extract_window(op, t_end, window_s)
            feats = extract_features(slices, window_s)
            feats["event_spindle_rpm"] = 0.0
            feats["event_feed_rate"] = 0.0
            feats["event_feed_override"] = 0.0
            feats["event_stop_duration_s"] = 0.0
            feats["trim_seconds_removed"] = 0.0
            feats["sample_rate_hz"] = sample_rate_hz
            feats["window_entries"] = int(window_s * sample_rate_hz)

            samples.append(PatternSample(
                sample_id=sample_id,
                label="normal",
                operation_id=op.op_id,
                case_dir=op.case_dir,
                tool_number=tool_num,
                event_timestamp=str(t_end),
                severity="none",
                stop_type="none",
                window_seconds=window_s,
                gap_seconds=gap_s,
                sample_rate_hz=sample_rate_hz,
                features=feats,
            ))

            if save_raw:
                raw_arrays[sample_id] = extract_raw_series(slices, window_s, sample_rate_hz=sample_rate_hz)

        n_pos = sum(1 for s in samples if s.label == "pre_stoppage")
        n_neg = sum(1 for s in samples if s.label == "normal")
        n_feat = len(samples[0].features) if samples else 0

        print(f"\n  Samples extracted: {len(samples)} ({n_pos} pre_stoppage, {n_neg} normal)")
        print(f"  Features per sample: {n_feat}")
        if gap_s > 0:
            print(f"  Prediction gap: {gap_s}s (window ends {gap_s}s before stop)")
            if skipped_gap > 0:
                print(f"  ⚠️  {skipped_gap} pre_stoppage samples skipped (insufficient data in shifted window)")

        print(f"\n{'='*70}")
        print(f"Saving to {output_dir}")
        print(f"{'='*70}")

        rows = []
        for sample in samples:
            row = {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "case_dir": sample.case_dir,
                "operation_id": sample.operation_id,
                "tool_number": sample.tool_number,
                "event_timestamp": sample.event_timestamp,
                "severity": sample.severity,
                "stop_type": sample.stop_type,
                "gap_seconds": sample.gap_seconds,
            }
            row.update(sample.features)
            rows.append(row)

        df = pd.DataFrame(rows)
        win_tag = f"_w{int(window_s)}s" if window_s != 60.0 else ""
        hz_tag = f"_{int(sample_rate_hz)}hz" if sample_rate_hz != 1.0 else ""
        gap_tag = f"_gap{int(gap_s)}s" if gap_s > 0 else ""
        csv_path = output_dir / f"stoppage_features{win_tag}{hz_tag}{gap_tag}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  ✅ Features CSV: {csv_path}  ({len(df)} rows × {len(df.columns)} cols)")

        if save_raw and raw_arrays:
            npz_path = output_dir / f"stoppage_raw_series{win_tag}{hz_tag}.npz"
            channel_names = sorted(next(iter(raw_arrays.values())).keys())
            sample_ids = [sample.sample_id for sample in samples]
            labels = [sample.label for sample in samples]
            n_samples = len(sample_ids)
            n_channels = len(channel_names)
            window_len = int(window_s * sample_rate_hz)

            tensor = np.zeros((n_samples, n_channels, window_len), dtype=np.float32)
            for i, sample_id in enumerate(sample_ids):
                series = raw_arrays.get(sample_id, {})
                for j, channel_name in enumerate(channel_names):
                    arr = series.get(channel_name, np.zeros(window_len, dtype=np.float32))
                    tensor[i, j, :len(arr)] = arr[:window_len]

            np.savez_compressed(
                npz_path,
                data=tensor,
                sample_ids=np.array(sample_ids),
                labels=np.array(labels),
                channel_names=np.array(channel_names),
                sample_rate_hz=np.float64(sample_rate_hz),
            )
            print(
                f"  ✅ Raw series NPZ: {npz_path}  shape={tensor.shape} "
                f"({n_samples} samples × {n_channels} channels × {window_len} entries [{window_s}s @ {sample_rate_hz} Hz])"
            )

        meta_path = output_dir / f"extraction_metadata{win_tag}{hz_tag}{gap_tag}.json"
        meta = {
            "extraction_date": str(pd.Timestamp.now()),
            "data_dir": str(data_dir),
            "case_filter": case,
            "weak_label_report": str(weak_label_report) if weak_label_report is not None else None,
            "include_candidate_labels": include_candidate_labels,
            "window_seconds": window_s,
            "sample_rate_hz": sample_rate_hz,
            "window_entries": int(window_s * sample_rate_hz),
            "gap_seconds": gap_s,
            "mode": "prediction" if gap_s > 0 else "detection",
            "n_samples": len(samples),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_features": n_feat,
            "operations": [
                {"case_dir": op.case_dir, "operation_id": op.op_id}
                for op in ops.values()
            ],
            "cases": sorted({op.case_dir for op in ops.values()}),
            "tool_numbers_with_stops": sorted(set(
                event.tool_number
                for events in all_stops.values()
                for event in events
            )),
            "channel_names": sorted(next(iter(raw_arrays.values())).keys()) if raw_arrays else [],
            "feature_names": sorted(samples[0].features.keys()) if samples else [],
            "severity_distribution": {
                severity: sum(1 for sample in samples if sample.severity == severity)
                for severity in ["critical", "high", "medium", "none"]
            },
        }
        with open(meta_path, "w") as handle:
            json.dump(meta, handle, indent=2, default=str)
        print(f"  ✅ Metadata JSON: {meta_path}")

        print(f"\n{'='*70}")
        print("Feature variance preview (top 20 by |mean(pre_stoppage) - mean(normal)|)")
        print(f"{'='*70}")

        if n_pos > 0 and n_neg > 0:
            pos_feats = df[df["label"] == "pre_stoppage"].select_dtypes(include="number")
            neg_feats = df[df["label"] == "normal"].select_dtypes(include="number")
            diffs = []
            for col in pos_feats.columns:
                if col in neg_feats.columns:
                    p_mean = pos_feats[col].mean()
                    n_mean = neg_feats[col].mean()
                    p_std = pos_feats[col].std()
                    n_std = neg_feats[col].std()
                    pooled_std = np.sqrt((p_std**2 + n_std**2) / 2) if (p_std + n_std) > 0 else 1
                    cohens_d = abs(p_mean - n_mean) / pooled_std
                    diffs.append((col, p_mean, n_mean, cohens_d))

            diffs.sort(key=lambda x: -x[3])
            print(f"\n  {'Feature':<45s}  {'Pre_stoppage':>13s}  {'Normal':>10s}  {'Cohen_d':>8s}")
            print(f"  {'─'*45}  {'─'*10}  {'─'*10}  {'─'*8}")
            for col, pm, nm, cd in diffs[:20]:
                print(f"  {col:<45s}  {pm:>10.2f}  {nm:>10.2f}  {cd:>8.2f}")

        return samples, ops, all_stops
    finally:
        if temp_root is not None:
            temp_root.cleanup()


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract pre-stoppage patterns as ground truth for "
                    "tool breakage detection."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/casedata"),
        help="Path to casedata directory (default: data/casedata)",
    )
    parser.add_argument(
        "--case", type=str, default=None,
        help="Optional case directory filter for casedata/Site_a roots",
    )
    parser.add_argument(
        "--window", type=float, default=60.0,
        help="Pre-event window in seconds (default: 60)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/breakage_patterns"),
        help="Output directory (default: data/breakage_patterns)",
    )
    parser.add_argument(
        "--no-raw", action="store_true",
        help="Skip saving raw time-series arrays (saves disk space)",
    )
    parser.add_argument(
        "--gap", type=float, default=0.0,
        help="Prediction gap in seconds: window ends gap_s before the stop "
             "event, shifting from detection to prediction (default: 0 = detection)",
    )
    parser.add_argument(
        "--gap-sweep", nargs="+", type=float, default=None,
        help="Run extractions for multiple gap values, e.g. --gap-sweep 0 5 10 30",
    )
    parser.add_argument(
        "--window-sweep", nargs="+", type=float, default=None,
        help="Run extractions for multiple window sizes, e.g. --window-sweep 60 30 10",
    )
    parser.add_argument(
        "--hz", type=float, default=1.0,
        help="Sampling frequency in Hz. Determines how many data-points "
             "(entries) are in a window: entries = window_s × Hz.  "
             "For 1 Hz CSV data (default): 60s = 60 entries.  "
             "For 1 kHz sensor data: 60s = 60000 entries. (default: 1.0)",
    )
    parser.add_argument(
        "--min-cluster-gap", type=float, default=120.0,
        help="Minimum gap in seconds between independent stop events. "
             "Events closer than this are clustered and only the most "
             "severe event per cluster is kept. (default: 120)",
    )
    parser.add_argument(
        "--weak-label-report", type=Path, default=None,
        help="Optional JSON report written by detect_premature_stoppage.py to use as positive labels",
    )
    parser.add_argument(
        "--include-candidate-labels", action="store_true",
        help="When using --weak-label-report, include baseline anomaly candidates as positives too",
    )
    parser.add_argument(
        "--skip-archives", action="store_true",
        help="Ignore archived OF*.tar.gz operations and use only unpacked directories",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: {args.data_dir} not found", file=sys.stderr)
        sys.exit(1)
    if args.weak_label_report is not None and not args.weak_label_report.exists():
        print(f"Error: weak-label report not found: {args.weak_label_report}", file=sys.stderr)
        sys.exit(1)

    gap_values = args.gap_sweep if args.gap_sweep else [args.gap]
    window_values = args.window_sweep if args.window_sweep else [args.window]

    # For sweeps with >1 gap value, load data once with the first gap,
    # then reuse preloaded data for subsequent gaps.
    preloaded = None

    for window_s in window_values:
        for i, gap_s in enumerate(gap_values):
            mode_str = f"PREDICTION (gap={gap_s}s)" if gap_s > 0 else "DETECTION (gap=0)"
            w_entries = int(window_s * args.hz)
            print("\n" + "=" * 70)
            print(f"  PRE-STOPPAGE PATTERN EXTRACTION — {mode_str}")
            print(f"  Data:   {args.data_dir.resolve()}")
            if args.case:
                print(f"  Case:   {args.case}")
            print(f"  Window: {window_s}s × {args.hz} Hz = {w_entries} entries")
            if gap_s > 0:
                print(f"  Gap:    {gap_s}s (window ends {gap_s}s before stop)")
            if args.weak_label_report is not None:
                print(f"  Labels: {args.weak_label_report.resolve()}")
            print(f"  Output: {args.output_dir.resolve()}")
            print("=" * 70)

            samples, ops_loaded, stops_loaded = run_extraction(
                data_dir=args.data_dir,
                window_s=window_s,
                output_dir=args.output_dir,
                save_raw=not args.no_raw,
                gap_s=gap_s,
                preloaded=preloaded,
                sample_rate_hz=args.hz,
                min_cluster_gap_s=args.min_cluster_gap,
                case=args.case,
                weak_label_report=args.weak_label_report,
                include_candidate_labels=args.include_candidate_labels,
                include_archives=not args.skip_archives,
            )

            # Reuse loaded data for subsequent gap values (avoids re-loading CSVs)
            if preloaded is None:
                preloaded = (ops_loaded, stops_loaded)

            n_pos = sum(1 for s in samples if s.label == "pre_stoppage")
            n_neg = sum(1 for s in samples if s.label == "normal")
            print(f"\n✅ Done [{mode_str}, window={window_s}s]. {len(samples)} samples "
                  f"({n_pos} pre-stoppage, {n_neg} normal)")
            print(f"   CSVs and arrays saved to {args.output_dir}")


if __name__ == "__main__":
    main()

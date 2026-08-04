#!/usr/bin/env python3
"""
Detect premature stoppage of operations in CNC machining data.

Analyses the data/casedata files (OF00001–OF00004) — four runs of the same
part on the same machine — and builds a **per-tool baseline** from the
cross-operation data.  Only flags events where a tool engagement is
anomalously short relative to the expected duration for that tool.

Detection approach:
  1. Segment each operation into contiguous tool engagements.
  2. Build a per-tool expected-duration baseline (median + MAD across ops).
  3. Flag individual engagements that are significantly shorter than baseline.
  4. For flagged engagements, report what caused the early stop:
     - Spindle crash (speed → 0 while feed was active)
     - Feed-override collapse (operator hit feed-hold / e-stop)
     - Operation-status gap (AUTO → STOPPED)
  5. Also flag tools whose *total* time in an operation is an outlier.

Usage:
    .venv/bin/python scripts/detect_premature_stoppage.py [--data-dir data/casedata]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agents.processing.archive_staging import prepare_analysis_root
from backend.agents.processing.dataset_loader import DatasetLoader
from backend.agents.processing.tool_lookup import resolve_machine_family
from backend.agents.processing.tool_reference_catalog import load_use_case_operation_index


# ── Channel suffixes ──────────────────────────────────────────────────
MACHINE_STATE_SUFFIX = "TYZBPS"
AXIS_POWER_SUFFIX = "BXCZ3M"


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class ToolEngagement:
    """A contiguous period where a single tool is active."""
    operation_id: str
    tool_number: float
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    duration_s: float
    n_samples: int
    case_dir: str = ""
    mean_spindle_speed: float = 0.0
    mean_feed_rate: float = 0.0
    max_feed_override: float = 0.0
    programs: list = field(default_factory=list)


@dataclass
class ToolBaseline:
    """Expected behaviour of a tool across operations."""
    tool_number: float
    n_operations: int  # how many ops this tool appeared in
    # Total time per operation
    total_durations: Dict[str, float] = field(default_factory=dict)
    median_total_s: float = 0.0
    mad_total_s: float = 0.0  # median absolute deviation
    # Per-engagement stats
    engagement_durations: List[float] = field(default_factory=list)
    median_engagement_s: float = 0.0
    mad_engagement_s: float = 0.0


@dataclass
class StoppageEvent:
    """A detected premature stoppage."""
    operation_id: str
    event_type: str
    timestamp: str
    severity: str  # "critical", "high", "medium"
    description: str
    tool_number: float = -1
    actual_duration_s: float = 0.0
    expected_duration_s: float = 0.0
    deficit_pct: float = 0.0  # how much shorter than expected (0–100)
    cause: str = ""  # what caused the stop
    context: Dict[str, Any] = field(default_factory=dict)
    case_dir: str = ""


@dataclass
class OperatorStopEvent:
    """An operator-initiated stop detected from Operation_Status transitions."""
    operation_id: str
    timestamp: str
    tool_number: float
    severity: str  # "critical" if mid-cut, "high" if mid-engagement, "medium" otherwise
    stop_type: str  # "mid_cut_stop", "feed_hold_mid_cut", "auto_to_stopped"
    description: str
    spindle_rpm_before: float = 0.0
    feed_rate_before: float = 0.0
    feed_override_before: float = 0.0
    time_in_stopped_s: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    case_dir: str = ""


# ── Helpers ───────────────────────────────────────────────────────────

MACHINE_STATE_COLUMNS = [
    "Feed_Override", "Feed_Rate_Actual", "Feed_Rate_Commanded",
    "Spindle_Speed_Actual", "Tool_Number",
    "Program_Block_Number", "Program_Name",
]
MACHINE_STATE_REQUIRED_COLUMNS = [
    "Feed_Rate_Actual", "Spindle_Speed_Actual", "Tool_Number",
]
AXIS_POWER_COLUMNS = ["Operation_Status", "Power_Spindle"]

def find_csv(op_dir: Path, suffix: str) -> Optional[Path]:
    for f in op_dir.glob("*.csv"):
        if suffix in f.name and not f.name.endswith("Zone.Identifier"):
            return f
    return None


def load_channel(
    csv_path: Path,
    usecols: Optional[list] = None,
    required_usecols: Optional[list] = None,
) -> pd.DataFrame:
    kwargs: dict = {"parse_dates": ["timestamp"]}
    if usecols:
        header = pd.read_csv(csv_path, nrows=0)
        available = [column for column in usecols if column in header.columns]
        missing = [
            column for column in (required_usecols or []) if column not in available
        ]
        if missing:
            raise ValueError(
                f"Missing required columns {missing} in {csv_path.name}"
            )
        kwargs["usecols"] = list(dict.fromkeys(["timestamp", *available]))
    df = pd.read_csv(csv_path, **kwargs)
    return df.sort_values("timestamp").reset_index(drop=True)


def median_absolute_deviation(values: list) -> float:
    """Robust spread measure: median(|x - median(x)|)."""
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=float)
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def load_axis_power(csv_path: Path) -> pd.DataFrame:
    """Load the axis-power / operation-status channel (BXCZ3M)."""
    df = load_channel(
        csv_path,
        usecols=AXIS_POWER_COLUMNS,
        required_usecols=["Operation_Status"],
    )
    if "Power_Spindle" not in df.columns:
        df["Power_Spindle"] = np.nan
    return df


def load_machine_state(csv_path: Path) -> pd.DataFrame:
    """Load the machine-state channel with optional columns handled safely."""
    df = load_channel(
        csv_path,
        usecols=MACHINE_STATE_COLUMNS,
        required_usecols=MACHINE_STATE_REQUIRED_COLUMNS,
    )
    if "Feed_Override" not in df.columns:
        df["Feed_Override"] = 100.0
    if "Feed_Rate_Commanded" not in df.columns:
        df["Feed_Rate_Commanded"] = df["Feed_Rate_Actual"]
    if "Program_Block_Number" not in df.columns:
        df["Program_Block_Number"] = np.nan
    if "Program_Name" not in df.columns:
        df["Program_Name"] = None
    return df


# ── 1. Segment tool engagements ──────────────────────────────────────

def segment_tool_engagements(
    ms: pd.DataFrame,
    operation_id: str,
    *,
    case_dir: str = "",
    min_duration_s: float = 5.0,
) -> List[ToolEngagement]:
    """Break a machine-state dataframe into contiguous tool engagements."""
    ms = ms.copy()
    ms["tool_change"] = (ms["Tool_Number"] != ms["Tool_Number"].shift(1)).astype(int)
    ms["segment_id"] = ms["tool_change"].cumsum()

    engagements: List[ToolEngagement] = []
    for _, grp in ms.groupby("segment_id"):
        if len(grp) < 2:
            continue
        t0 = grp["timestamp"].iloc[0]
        t1 = grp["timestamp"].iloc[-1]
        dur = (t1 - t0).total_seconds()
        if dur < min_duration_s:
            continue

        tool_num = grp["Tool_Number"].iloc[0]
        if pd.isna(tool_num) or tool_num == 0:
            continue

        programs = []
        if "Program_Name" in grp.columns:
            programs = grp["Program_Name"].dropna().unique().tolist()

        eng = ToolEngagement(
            operation_id=operation_id,
            case_dir=case_dir,
            tool_number=float(tool_num),
            t_start=t0, t_end=t1,
            duration_s=dur, n_samples=len(grp),
            programs=programs,
        )
        if "Spindle_Speed_Actual" in grp.columns:
            eng.mean_spindle_speed = float(grp["Spindle_Speed_Actual"].mean())
        if "Feed_Rate_Actual" in grp.columns:
            eng.mean_feed_rate = float(grp["Feed_Rate_Actual"].mean())
        if "Feed_Override" in grp.columns:
            eng.max_feed_override = float(grp["Feed_Override"].max())

        engagements.append(eng)

    return engagements


# ── 2. Build baselines ───────────────────────────────────────────────

def build_tool_baselines(
    all_engagements: Dict[str, List[ToolEngagement]],
    min_ops: int = 3,
) -> Dict[float, ToolBaseline]:
    """Build a per-tool expected duration baseline from cross-operation data.

    Only tools that appear in at least `min_ops` operations get a baseline.
    Uses median + MAD (robust to outliers) rather than mean + std.
    """
    # Aggregate total duration and individual engagements per tool per op
    tool_total: Dict[float, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    tool_engs: Dict[float, List[float]] = defaultdict(list)

    for op_id, engs in all_engagements.items():
        for e in engs:
            tool_total[e.tool_number][op_id] += e.duration_s
            tool_engs[e.tool_number].append(e.duration_s)

    baselines: Dict[float, ToolBaseline] = {}
    for tool_num, op_durs in tool_total.items():
        if len(op_durs) < min_ops:
            continue

        total_vals = list(op_durs.values())
        eng_vals = tool_engs[tool_num]

        bl = ToolBaseline(
            tool_number=tool_num,
            n_operations=len(op_durs),
            total_durations=dict(op_durs),
            median_total_s=float(np.median(total_vals)),
            mad_total_s=median_absolute_deviation(total_vals),
            engagement_durations=eng_vals,
            median_engagement_s=float(np.median(eng_vals)),
            mad_engagement_s=median_absolute_deviation(eng_vals),
        )
        baselines[tool_num] = bl

    return baselines


# ── 3. Diagnose cause of short engagement ─────────────────────────────

def diagnose_stop_cause(
    eng: ToolEngagement,
    ms_slice: pd.DataFrame,
    ap: Optional[pd.DataFrame] = None,
) -> str:
    """Look at the end of an engagement to determine *why* it stopped early.

    Now also checks the BXCZ3M axis-power channel for Operation_Status
    transitions (AUTO→STOPPED or AUTO→FEED_HOLD) that occurred during
    the engagement — the clearest signal of an operator-initiated stop.
    """
    if len(ms_slice) < 3:
        return "unknown (too few samples)"

    # Look at the last 20% of the engagement
    tail_start = max(0, len(ms_slice) - max(int(len(ms_slice) * 0.2), 5))
    tail = ms_slice.iloc[tail_start:]

    ss = tail["Spindle_Speed_Actual"].values.astype(float)
    fr = tail["Feed_Rate_Actual"].values.astype(float)
    fo = tail["Feed_Override"].values.astype(float)

    causes = []

    # ── Check Operation_Status for operator-initiated stops ──────────
    if ap is not None:
        ap_window = ap[
            (ap["timestamp"] >= eng.t_start) & (ap["timestamp"] <= eng.t_end)
        ]
        if len(ap_window) > 1:
            ap_window = ap_window.copy()
            ap_window["prev"] = ap_window["Operation_Status"].shift(1)
            # AUTO(3) → STOPPED(0): operator hit e-stop / cycle-stop
            e_stops = ap_window[
                (ap_window["prev"] == 3.0) & (ap_window["Operation_Status"] == 0.0)
            ]
            if len(e_stops) > 0:
                causes.append(
                    f"operator stop (AUTO→STOPPED ×{len(e_stops)} "
                    f"at {e_stops['timestamp'].iloc[0]})"
                )
            # AUTO(3) → FEED_HOLD(2): operator hit feed-hold
            f_holds = ap_window[
                (ap_window["prev"] == 3.0) & (ap_window["Operation_Status"] == 2.0)
            ]
            if len(f_holds) > 0:
                causes.append(
                    f"operator feed-hold (AUTO→HOLD ×{len(f_holds)})"
                )

    # ── Check for spindle crash at end ───────────────────────────────
    if len(ss) >= 2:
        running_mask = ss > 100
        crashed_mask = ss < 50
        if running_mask.any() and crashed_mask.any():
            last_running = np.where(running_mask)[0][-1]
            first_crash = np.where(crashed_mask)[0][0]
            if last_running < first_crash:
                causes.append(f"spindle crash (→{ss[-1]:.0f} RPM)")

    # Check for feed override drop at end
    if fo.max() > 50 and fo[-1] < 5:
        causes.append("feed override → 0%")

    # Check if feed rate was still active (mid-cut stop)
    if fr.max() > 100 and fr[-1] < 1:
        causes.append("feed stopped mid-cut")

    if not causes:
        if ss[-1] < 10 and (len(ss) < 3 or ss[-3] < 50):
            causes.append("normal deceleration (possibly expected)")
        else:
            causes.append("undetermined")

    return "; ".join(causes)


# ── 4. Detect anomalous tool engagements (baseline-aware) ────────────

def detect_anomalous_engagements(
    all_engagements: Dict[str, List[ToolEngagement]],
    all_ms: Dict[str, pd.DataFrame],
    baselines: Dict[float, ToolBaseline],
    all_ap: Optional[Dict[str, pd.DataFrame]] = None,
    deficit_threshold: float = 60.0,
    z_threshold: float = 2.5,
) -> List[StoppageEvent]:
    """Compare each tool engagement against its baseline.

    An engagement is flagged if:
      - It has a baseline (tool appears in 3+ operations)
      - Its duration is >deficit_threshold% shorter than the median
      - OR its modified z-score is below -z_threshold
    """
    events: List[StoppageEvent] = []

    for op_key, engs in all_engagements.items():
        ms = all_ms.get(op_key)
        for eng in engs:
            bl = baselines.get(eng.tool_number)
            if bl is None:
                continue

            expected = bl.median_engagement_s
            if expected < 10:
                continue

            # Modified z-score: (x - median) / (1.4826 * MAD)
            mad_scaled = (
                1.4826 * bl.mad_engagement_s
                if bl.mad_engagement_s > 0
                else expected * 0.3
            )
            z_score = (eng.duration_s - expected) / max(mad_scaled, 1.0)

            deficit_pct = max(0.0, (1 - eng.duration_s / expected) * 100)

            if deficit_pct < deficit_threshold and z_score > -z_threshold:
                continue

            # Diagnose cause
            cause = "unknown"
            if ms is not None:
                mask = (
                    (ms["timestamp"] >= eng.t_start)
                    & (ms["timestamp"] <= eng.t_end)
                )
                ms_slice = ms.loc[mask]
                ap = all_ap.get(op_key) if all_ap else None
                if len(ms_slice) > 0:
                    cause = diagnose_stop_cause(eng, ms_slice, ap=ap)

            if deficit_pct > 90 or z_score < -5:
                severity = "critical"
            elif deficit_pct > 75 or z_score < -3:
                severity = "high"
            else:
                severity = "medium"

            events.append(StoppageEvent(
                operation_id=eng.operation_id,
                event_type="short_engagement",
                timestamp=str(eng.t_start),
                severity=severity,
                tool_number=eng.tool_number,
                actual_duration_s=round(eng.duration_s, 1),
                expected_duration_s=round(expected, 1),
                deficit_pct=round(deficit_pct, 1),
                cause=cause,
                description=(
                    f"Tool T{int(eng.tool_number)} engagement lasted {eng.duration_s:.0f}s "
                    f"vs expected {expected:.0f}s ({deficit_pct:.0f}% short, z={z_score:.1f}). "
                    f"Cause: {cause}."
                ),
                context={
                    "programs": eng.programs,
                    "mean_spindle_rpm": eng.mean_spindle_speed,
                    "mean_feed_rate": eng.mean_feed_rate,
                    "z_score": round(z_score, 2),
                    "baseline_median_s": expected,
                    "baseline_mad_s": bl.mad_engagement_s,
                },
                case_dir=eng.case_dir,
            ))

    return events


# ── 5. Detect anomalous total tool time per operation ─────────────────

def detect_anomalous_tool_totals(
    baselines: Dict[float, ToolBaseline],
    operation_context: Dict[str, Dict[str, Any]],
    z_threshold: float = 2.0,
    deficit_threshold: float = 50.0,
) -> List[StoppageEvent]:
    """Flag operations where a tool's *total* time is an outlier (short)."""
    events: List[StoppageEvent] = []

    for tool_num, bl in baselines.items():
        if bl.n_operations < 3:
            continue

        median = bl.median_total_s
        mad = bl.mad_total_s
        if median < 30:
            continue

        mad_scaled = 1.4826 * mad if mad > 0 else median * 0.3

        for op_key, dur in bl.total_durations.items():
            z = (dur - median) / max(mad_scaled, 1.0)
            deficit_pct = max(0.0, (1 - dur / median) * 100)

            if deficit_pct < deficit_threshold and z > -z_threshold:
                continue

            op_context = operation_context.get(op_key, {})
            operation_id = str(op_context.get("operation_id") or op_key)
            case_dir = str(op_context.get("case_dir") or "")

            if deficit_pct > 90 or z < -5:
                severity = "critical"
            elif deficit_pct > 75 or z < -3:
                severity = "high"
            else:
                severity = "medium"

            all_durs_str = ", ".join(
                f"{_operation_label(str(operation_context.get(op, {}).get('case_dir') or ''), str(operation_context.get(op, {}).get('operation_id') or op))}={'▸' if op == op_key else ''}{d:.0f}s"
                for op, d in sorted(bl.total_durations.items())
            )

            events.append(StoppageEvent(
                operation_id=operation_id,
                event_type="short_tool_total",
                timestamp="n/a (aggregate)",
                severity=severity,
                tool_number=tool_num,
                actual_duration_s=round(dur, 1),
                expected_duration_s=round(median, 1),
                deficit_pct=round(deficit_pct, 1),
                description=(
                    f"Tool T{int(tool_num)} total time in {operation_id} is {dur:.0f}s "
                    f"vs median {median:.0f}s ({deficit_pct:.0f}% short, z={z:.1f}). "
                    f"All ops: [{all_durs_str}]"
                ),
                context={
                    "z_score": round(z, 2),
                    "all_op_durations": bl.total_durations,
                },
                case_dir=case_dir,
            ))

    return events


# ── 6. Detect operator-initiated stops (mid-cut) ─────────────────────

def detect_operator_stops(
    all_ms: Dict[str, pd.DataFrame],
    all_ap: Dict[str, pd.DataFrame],
    operation_context: Dict[str, Dict[str, Any]],
    spindle_threshold: float = 100.0,
    feed_threshold: float = 50.0,
    lookback_s: float = 10.0,
    min_cluster_gap_s: float = 120.0,
) -> List[OperatorStopEvent]:
    """Detect operator-initiated stops during active cutting.

    Scans Operation_Status in the BXCZ3M channel for transitions:
      - AUTO (3) → STOPPED (0): operator hit cycle-stop or e-stop
      - AUTO (3) → FEED_HOLD (2): operator hit feed-hold

    Then cross-references with TYZBPS to check whether the spindle was
    running and feed was active at the moment of the stop.  Only flags
    transitions that occurred **during active cutting** — i.e. when
    something was likely going wrong.

    Events within ``min_cluster_gap_s`` seconds are clustered; only the
    most-severe event per cluster is kept.
    """
    events: List[OperatorStopEvent] = []

    for op_key, ap in all_ap.items():
        ms = all_ms.get(op_key)
        if ms is None or len(ap) < 2:
            continue

        op_context = operation_context.get(op_key, {})
        operation_id = str(op_context.get("operation_id") or op_key)
        case_dir = str(op_context.get("case_dir") or "")

        ap = ap.copy()
        ap["prev_status"] = ap["Operation_Status"].shift(1)

        # Detect both hard stops and feed-holds
        transitions = [
            ("mid_cut_stop", 3.0, 0.0),
            ("feed_hold_mid_cut", 3.0, 2.0),
        ]

        for stop_type, from_status, to_status in transitions:
            mask = (
                (ap["prev_status"] == from_status)
                & (ap["Operation_Status"] == to_status)
            )
            stop_rows = ap[mask]

            for _, row in stop_rows.iterrows():
                t_stop = row["timestamp"]

                # Look at machine state in the window before the stop
                window = ms[
                    (ms["timestamp"] >= t_stop - pd.Timedelta(seconds=lookback_s))
                    & (ms["timestamp"] <= t_stop)
                ]
                if len(window) == 0:
                    continue

                avg_spindle = float(window["Spindle_Speed_Actual"].mean())
                avg_feed = float(window["Feed_Rate_Actual"].mean())
                avg_override = float(window["Feed_Override"].mean())
                tool = float(window["Tool_Number"].iloc[-1])

                # Only flag if the machine was actively cutting
                if avg_spindle < spindle_threshold or avg_feed < feed_threshold:
                    continue

                # How long did the stop last?
                future_ap = ap[ap["timestamp"] > t_stop]
                resumed = future_ap[
                    future_ap["Operation_Status"].isin([2.0, 3.0])
                ]
                if len(resumed) > 0:
                    stop_duration = (
                        resumed["timestamp"].iloc[0] - t_stop
                    ).total_seconds()
                else:
                    stop_duration = 0.0

                # Severity based on cutting intensity at time of stop
                if avg_spindle > 500 and avg_feed > 200:
                    severity = "critical"
                elif avg_spindle > 200 or avg_feed > 100:
                    severity = "high"
                else:
                    severity = "medium"

                status_label = (
                    "STOPPED" if to_status == 0.0 else "FEED_HOLD"
                )
                events.append(OperatorStopEvent(
                    operation_id=operation_id,
                    case_dir=case_dir,
                    timestamp=str(t_stop),
                    tool_number=tool,
                    severity=severity,
                    stop_type=stop_type,
                    spindle_rpm_before=round(avg_spindle, 0),
                    feed_rate_before=round(avg_feed, 0),
                    feed_override_before=round(avg_override, 0),
                    time_in_stopped_s=round(stop_duration, 1),
                    description=(
                        f"Operator {status_label} during active cut "
                        f"(T{int(tool)}, spindle={avg_spindle:.0f}rpm, "
                        f"feed={avg_feed:.0f}mm/min, "
                        f"override={avg_override:.0f}%). "
                        f"Stopped for {stop_duration:.0f}s."
                    ),
                    context={
                        "from_status": from_status,
                        "to_status": to_status,
                        "stop_duration_s": stop_duration,
                    },
                ))

    # ── Cluster de-duplication ────────────────────────────────────────
    if not events or min_cluster_gap_s <= 0:
        return events

    from datetime import datetime as _dt
    def _ts(e: OperatorStopEvent):
        t = e.timestamp
        if isinstance(t, str):
            return pd.Timestamp(t)
        return t

    events.sort(key=lambda e: _ts(e))
    severity_rank = {"critical": 0, "high": 1, "medium": 2}
    type_rank = {"mid_cut_stop": 0, "feed_hold_mid_cut": 1}

    def _pick_best(cluster: List[OperatorStopEvent]) -> OperatorStopEvent:
        cluster.sort(key=lambda e: (
            severity_rank.get(e.severity, 9),
            type_rank.get(e.stop_type, 9),
        ))
        best = cluster[0]
        best.time_in_stopped_s = max(e.time_in_stopped_s for e in cluster)
        return best

    deduped: List[OperatorStopEvent] = []
    current_cluster: List[OperatorStopEvent] = [events[0]]
    for ev in events[1:]:
        gap = (_ts(ev) - _ts(current_cluster[-1])).total_seconds()
        if gap <= min_cluster_gap_s:
            current_cluster.append(ev)
        else:
            deduped.append(_pick_best(current_cluster))
            current_cluster = [ev]
    deduped.append(_pick_best(current_cluster))

    return deduped


def _operation_label(case_dir: str, operation_id: str) -> str:
    return f"{case_dir} / {operation_id}" if case_dir else operation_id


def _process_plan_context(case_dir: str, tool_number: float) -> Optional[Dict[str, Any]]:
    if not case_dir or tool_number < 0 or pd.isna(tool_number):
        return None
    family = resolve_machine_family(case_dir)
    if not family:
        return None
    index = load_use_case_operation_index()
    payload = index.get((family, int(tool_number)))
    if payload is None:
        return None
    return {
        "machine_family": family,
        "use_case_ids": list(payload.get("use_case_ids") or []),
        "use_case_titles": list(payload.get("use_case_titles") or []),
        "operation_ids": list(payload.get("operation_ids") or []),
        "setups": list(payload.get("setups") or []),
        "entries": [
            {
                "use_case_id": entry.get("use_case_id"),
                "use_case_title": entry.get("use_case_title"),
                "setup": entry.get("setup"),
                "operation_id": entry.get("operation_id"),
                "head": entry.get("head"),
                "op_type": entry.get("op_type"),
                "description": entry.get("description"),
                "slide_number": entry.get("slide_number"),
                "slide_row_index": entry.get("slide_row_index"),
            }
            for entry in (payload.get("entries") or [])
        ],
    }


def _enrich_event_with_process_plan(event: Any) -> None:
    if not getattr(event, "case_dir", ""):
        return
    event.context.setdefault("case_dir", event.case_dir)
    family = resolve_machine_family(event.case_dir)
    if family:
        event.context.setdefault("machine_family", family)
    plan = _process_plan_context(event.case_dir, float(event.tool_number))
    if plan is not None:
        event.context["process_plan"] = plan


def enrich_events_with_process_plan(
    events: List[StoppageEvent],
    operator_events: List[OperatorStopEvent],
) -> None:
    for event in events:
        _enrich_event_with_process_plan(event)
    for event in operator_events:
        _enrich_event_with_process_plan(event)


# ── Main analysis ─────────────────────────────────────────────────────

def analyse(
    data_dir: Path,
    deficit_threshold: float = 60.0,
    z_threshold: float = 2.5,
    *,
    case: Optional[str] = None,
    include_archives: bool = True,
) -> Tuple[List[StoppageEvent], List[OperatorStopEvent], Dict[float, ToolBaseline]]:
    """Run baseline-aware premature stoppage detection.

    Returns (stoppage_events, operator_stop_events, baselines).
    """
    all_engagements: Dict[str, List[ToolEngagement]] = {}
    all_ms: Dict[str, pd.DataFrame] = {}
    all_ap: Dict[str, pd.DataFrame] = {}
    operation_context: Dict[str, Dict[str, Any]] = {}

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

        loader = DatasetLoader(prepared_data_dir)
        case_dirs = [case] if case else loader.list_cases()
        for case_dir in case_dirs:
            operations = loader.list_operations(case=case_dir)
            print(f"\n{'='*70}")
            print(f"Case: {case_dir}")
            print(f"Operations: {[operation.operation_id for operation in operations]}")
            print(f"{'='*70}")

            for operation in operations:
                op_id = operation.operation_id
                op_key = _operation_label(case_dir, op_id)
                ms_path = operation.channel_files.get("machine_state")
                if ms_path is None:
                    continue

                print(f"\n── Loading {op_id} ──")
                try:
                    ms = load_machine_state(ms_path)
                except ValueError as exc:
                    print(f"   ⚠ Skipping {op_id}: {exc}")
                    continue
                t0, t1 = ms["timestamp"].iloc[0], ms["timestamp"].iloc[-1]
                total_h = (t1 - t0).total_seconds() / 3600
                print(f"   Time: {t0} → {t1} ({total_h:.1f}h, {len(ms):,} rows)")

                # Load axis-power channel for Operation_Status
                ap_path = operation.channel_files.get("axis_power")
                if ap_path:
                    ap = load_axis_power(ap_path)
                    all_ap[op_key] = ap
                    print(f"   Axis-power: {len(ap):,} rows (Operation_Status loaded)")
                else:
                    print(f"   ⚠ No axis-power channel found — operator-stop "
                          f"detection unavailable for {op_id}")

                all_ms[op_key] = ms
                operation_context[op_key] = {
                    "case_dir": case_dir,
                    "operation_id": op_id,
                    "tool_id": operation.tool_id,
                }
                engs = segment_tool_engagements(ms, op_id, case_dir=case_dir)
                all_engagements[op_key] = engs

                unique_tools = set(e.tool_number for e in engs)
                print(f"   Segments: {len(engs)} engagements, "
                      f"{len(unique_tools)} unique tools")

        # Build baselines ——————————————————————————————
        print(f"\n{'='*70}")
        print("Building per-tool baselines (cross-operation)")
        print(f"{'='*70}")

        baselines = build_tool_baselines(all_engagements, min_ops=3)
        print(f"Tools with baselines (≥3 operations): {len(baselines)}")
        for tn in sorted(baselines.keys()):
            bl = baselines[tn]
            durs_str = ", ".join(
                f"{op}={d:.0f}s" for op, d in sorted(bl.total_durations.items())
            )
            print(f"  T{int(tn):>5}: median_total={bl.median_total_s:.0f}s  "
                  f"MAD={bl.mad_total_s:.0f}s  [{durs_str}]")

        # Detect anomalies ————————————————————————————
        print(f"\n{'='*70}")
        print("Detecting anomalies vs baseline")
        print(f"Thresholds: deficit ≥{deficit_threshold}%  or  z ≤ -{z_threshold}")
        print(f"{'='*70}")

        events: List[StoppageEvent] = []

        eng_events = detect_anomalous_engagements(
            all_engagements, all_ms, baselines,
            all_ap=all_ap,
            deficit_threshold=deficit_threshold,
            z_threshold=z_threshold,
        )
        print(f"Anomalous engagements: {len(eng_events)}")
        events.extend(eng_events)

        total_events = detect_anomalous_tool_totals(
            baselines,
            operation_context,
            z_threshold=z_threshold,
            deficit_threshold=deficit_threshold - 10,  # slightly more sensitive for totals
        )
        print(f"Anomalous tool totals: {len(total_events)}")
        events.extend(total_events)

        # ── Operator-initiated mid-cut stops ──────────────────────────────
        print(f"\n{'='*70}")
        print("Detecting operator-initiated stops (AUTO→STOPPED / FEED_HOLD")
        print(f"during active cutting)")
        print(f"{'='*70}")

        operator_events: List[OperatorStopEvent] = []
        if all_ap:
            operator_events = detect_operator_stops(all_ms, all_ap, operation_context)
            n_crit = sum(1 for e in operator_events if e.severity == "critical")
            n_high = sum(1 for e in operator_events if e.severity == "high")
            n_med = sum(1 for e in operator_events if e.severity == "medium")
            print(f"Operator mid-cut stops: {len(operator_events)} "
                  f"(🔴 {n_crit} critical, 🟠 {n_high} high, 🟡 {n_med} medium)")
        else:
            print("⚠ No axis-power data loaded — skipping operator-stop detection")

        enrich_events_with_process_plan(events, operator_events)

        return events, operator_events, baselines
    finally:
        if temp_root is not None:
            temp_root.cleanup()


def print_report(
    events: List[StoppageEvent],
    operator_events: List[OperatorStopEvent],
    baselines: Dict[float, ToolBaseline],
) -> None:
    """Print a structured report."""

    # ── Operator-initiated stops section (highest priority) ───────────
    if operator_events:
        severity_order = {"critical": 0, "high": 1, "medium": 2}
        operator_events.sort(key=lambda e: (
            severity_order.get(e.severity, 3), e.operation_id
        ))

        print(f"\n{'='*70}")
        print(f"  ⚠  OPERATOR-INITIATED MID-CUT STOPS")
        print(f"  Detected from Operation_Status transitions (BXCZ3M channel)")
        print(f"  Total: {len(operator_events)}")
        print(f"{'='*70}")

        by_op: Dict[str, int] = defaultdict(int)
        by_type: Dict[str, int] = defaultdict(int)
        for e in operator_events:
            by_op[e.operation_id] += 1
            by_type[e.stop_type] += 1

        print("\n  By operation:")
        for op, n in sorted(by_op.items()):
            print(f"    {op}  {n:3d} stops")

        print("\n  By type:")
        for t, n in sorted(by_type.items()):
            print(f"    {t:25s}  {n:3d}")

        for sev_label, sev_icon in [
            ("critical", "🔴"), ("high", "🟠"), ("medium", "🟡"),
        ]:
            sev_events = [e for e in operator_events if e.severity == sev_label]
            if not sev_events:
                continue
            max_show = 15
            extra = (
                f", showing first {max_show}"
                if len(sev_events) > max_show else ""
            )
            print(f"\n{'─'*70}")
            print(f"  {sev_icon} {sev_label.upper()} OPERATOR STOPS "
                  f"({len(sev_events)}{extra})")
            print(f"{'─'*70}")
            for i, e in enumerate(sev_events[:max_show], 1):
                print(f"\n  [{i}] {e.stop_type} — {_operation_label(e.case_dir, e.operation_id)} "
                      f"— T{int(e.tool_number)}")
                print(f"      Time:      {e.timestamp}")
                print(f"      Spindle:   {e.spindle_rpm_before:.0f} RPM")
                print(f"      Feed:      {e.feed_rate_before:.0f} mm/min")
                print(f"      Override:  {e.feed_override_before:.0f}%")
                print(f"      Down time: {e.time_in_stopped_s:.0f}s")
                process_plan = e.context.get("process_plan") or {}
                if process_plan.get("operation_ids"):
                    print(
                        f"      Planned:   {', '.join(process_plan['operation_ids'][:4])}"
                    )
                print(f"      {e.description}")
    else:
        print("\n✅ No operator-initiated mid-cut stops detected.")

    # ── Baseline-aware stoppage section ───────────────────────────────
    if not events:
        print("\n✅ No anomalous premature stoppages detected against baseline.")
        return

    severity_order = {"critical": 0, "high": 1, "medium": 2}
    events.sort(key=lambda e: (
        severity_order.get(e.severity, 3), e.operation_id
    ))

    print(f"\n{'='*70}")
    print(f"  PREMATURE STOPPAGE REPORT (baseline-aware)")
    print(f"  Baseline built from {len(baselines)} tools across operations")
    print(f"  Total anomalies: {len(events)}")
    print(f"{'='*70}")

    by_type = defaultdict(int)
    by_severity = defaultdict(int)
    by_op = defaultdict(int)
    by_tool = defaultdict(int)
    for e in events:
        by_type[e.event_type] += 1
        by_severity[e.severity] += 1
        by_op[e.operation_id] += 1
        by_tool[e.tool_number] += 1

    print("\n  By event type:")
    for t, n in sorted(by_type.items()):
        print(f"    {t:25s}  {n:3d}")

    print("\n  By severity:")
    for s in ["critical", "high", "medium"]:
        if s in by_severity:
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}[s]
            print(f"    {icon} {s:10s}  {by_severity[s]:3d}")

    print("\n  By operation:")
    for op, n in sorted(by_op.items()):
        print(f"    {op}  {n:3d} anomalies")

    print("\n  Most affected tools:")
    for tool, n in sorted(by_tool.items(), key=lambda x: -x[1])[:10]:
        bl = baselines.get(tool)
        bl_str = (
            f"  (baseline median: {bl.median_total_s:.0f}s total)"
            if bl else ""
        )
        print(f"    T{int(tool):>5}  {n:3d} anomalies{bl_str}")

    # ── Detail sections ──
    for sev_label, sev_icon in [
        ("critical", "🔴"), ("high", "🟠"), ("medium", "🟡"),
    ]:
        sev_events = [e for e in events if e.severity == sev_label]
        if not sev_events:
            continue

        max_show = 20 if sev_label in ("critical", "high") else 10
        extra_note = (
            f", showing first {max_show}"
            if len(sev_events) > max_show else ""
        )
        print(f"\n{'─'*70}")
        print(f"  {sev_icon} {sev_label.upper()} EVENTS "
              f"({len(sev_events)}{extra_note})")
        print(f"{'─'*70}")

        for i, e in enumerate(sev_events[:max_show], 1):
            print(f"\n  [{i}] {e.event_type} — {_operation_label(e.case_dir, e.operation_id)} "
                  f"— T{int(e.tool_number)}")
            print(f"      Time:     {e.timestamp}")
            print(f"      Actual:   {e.actual_duration_s:.0f}s")
            print(f"      Expected: {e.expected_duration_s:.0f}s  "
                  f"(deficit: {e.deficit_pct:.0f}%)")
            if e.cause:
                print(f"      Cause:    {e.cause}")
            process_plan = e.context.get("process_plan") or {}
            if process_plan.get("operation_ids"):
                print(
                    f"      Planned:  {', '.join(process_plan['operation_ids'][:4])}"
                )
            print(f"      {e.description}")
            if e.context.get("programs"):
                print(f"      Programs: {e.context['programs']}")


def _serialize_event(event: Any) -> Dict[str, Any]:
    return {
        "operation_id": event.operation_id,
        "case_dir": getattr(event, "case_dir", "") or None,
        "tool_number": float(event.tool_number),
        "severity": event.severity,
        "description": event.description,
        "context": dict(event.context),
        **({
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "actual_duration_s": float(event.actual_duration_s),
            "expected_duration_s": float(event.expected_duration_s),
            "deficit_pct": float(event.deficit_pct),
            "cause": event.cause,
        } if isinstance(event, StoppageEvent) else {
            "stop_type": event.stop_type,
            "timestamp": event.timestamp,
            "spindle_rpm_before": float(event.spindle_rpm_before),
            "feed_rate_before": float(event.feed_rate_before),
            "feed_override_before": float(event.feed_override_before),
            "time_in_stopped_s": float(event.time_in_stopped_s),
        }),
    }


def build_json_report(
    data_dir: Path,
    events: List[StoppageEvent],
    operator_events: List[OperatorStopEvent],
    baselines: Dict[float, ToolBaseline],
) -> Dict[str, Any]:
    weak_labels = [
        {
            "label": "unexpected_stop",
            "label_source": event.stop_type,
            "confidence": event.severity,
            "case_dir": event.case_dir or None,
            "operation_id": event.operation_id,
            "tool_number": float(event.tool_number),
            "timestamp": event.timestamp,
            "context": dict(event.context),
        }
        for event in operator_events
    ]
    weak_labels.extend(
        {
            "label": "unexpected_stop_candidate",
            "label_source": event.event_type,
            "confidence": event.severity,
            "case_dir": event.case_dir or None,
            "operation_id": event.operation_id,
            "tool_number": float(event.tool_number),
            "timestamp": event.timestamp,
            "context": dict(event.context),
        }
        for event in events
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "operator_stop_events": [_serialize_event(event) for event in operator_events],
        "baseline_anomalies": [_serialize_event(event) for event in events],
        "weak_labels": weak_labels,
        "tool_baselines": {
            str(int(tool_number)): {
                "n_operations": baseline.n_operations,
                "median_total_s": float(baseline.median_total_s),
                "mad_total_s": float(baseline.mad_total_s),
                "median_engagement_s": float(baseline.median_engagement_s),
                "mad_engagement_s": float(baseline.mad_engagement_s),
                "total_durations": dict(baseline.total_durations),
            }
            for tool_number, baseline in baselines.items()
        },
    }


def write_json_report(
    output_path: Path,
    data_dir: Path,
    events: List[StoppageEvent],
    operator_events: List[OperatorStopEvent],
    baselines: Dict[float, ToolBaseline],
) -> Path:
    payload = build_json_report(data_dir, events, operator_events, baselines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect premature stoppage (baseline-aware) in CNC casedata"
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/casedata"),
        help="Path to casedata directory (default: data/casedata)",
    )
    parser.add_argument(
        "--deficit", type=float, default=60.0,
        help="Minimum deficit %% to flag an engagement (default: 60)",
    )
    parser.add_argument(
        "--z-threshold", type=float, default=2.5,
        help="Modified z-score threshold for anomaly (default: 2.5)",
    )
    parser.add_argument(
        "--case", default=None,
        help="Optional case directory name to analyse within the data root",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Optional path to write a machine-readable weak-label report",
    )
    parser.add_argument(
        "--skip-archives", action="store_true",
        help="Ignore archived OF*.tar.gz operations and analyse only unpacked directories",
    )
    parser.add_argument(
        "--fail-on-critical", action="store_true",
        help="Exit with code 1 when critical stops are detected",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Premature Stoppage Detection (baseline-aware)")
    print(f"Data: {args.data_dir.resolve()}")
    if args.case:
        print(f"Case filter: {args.case}")
    print(f"Thresholds: deficit ≥{args.deficit}%  or  z ≤ -{args.z_threshold}")

    events, operator_events, baselines = analyse(
        args.data_dir,
        deficit_threshold=args.deficit,
        z_threshold=args.z_threshold,
        case=args.case,
        include_archives=not args.skip_archives,
    )
    print_report(events, operator_events, baselines)

    if args.output_json is not None:
        written = write_json_report(
            args.output_json,
            args.data_dir,
            events,
            operator_events,
            baselines,
        )
        print(f"\nWrote weak-label report: {written}")

    critical = sum(1 for e in events if e.severity == "critical")
    critical += sum(1 for e in operator_events if e.severity == "critical")
    sys.exit(1 if args.fail_on_critical and critical > 0 else 0)


if __name__ == "__main__":
    main()

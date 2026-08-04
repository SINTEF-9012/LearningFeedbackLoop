"""Experiment data router — raw time-series and per-sample annotations.

Business logic is extracted from the old monolithic app.py into clean,
testable functions.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
from fastapi import APIRouter, Query
from backend.json_utils import finite_float as _safe_float

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiment", tags=["experiment"])


# ── Service helpers (extracted from old inline route handlers) ────────────────

def _load_timeseries(sample_idx: int, want_channels: Set[str]):
    """Load a single sample window from the raw-series NPZ file.

    Tries ``stoppage_raw_series.npz`` first (the current experiment type),
    falling back to ``breakage_raw_series.npz`` for legacy compatibility.
    """
    npz_path = Path("data/breakage_patterns/stoppage_raw_series.npz")
    if not npz_path.exists():
        npz_path = Path("data/breakage_patterns/breakage_raw_series.npz")
    if not npz_path.exists():
        return {
            "error": "No raw_series.npz found. Run feature extraction first.",
            "channels": [],
            "sample_id": None,
            "total_samples": 0,
        }

    data = np.load(npz_path, allow_pickle=True)
    series = data["data"]  # (n_samples, n_channels, n_timesteps)
    channel_names = list(data["channel_names"])
    sample_ids = (
        list(data["sample_ids"])
        if "sample_ids" in data
        else [f"sample_{i}" for i in range(series.shape[0])]
    )
    labels = (
        list(data["labels"]) if "labels" in data else ["unknown"] * series.shape[0]
    )

    n_samples = series.shape[0]
    idx = max(0, min(sample_idx, n_samples - 1))

    if not want_channels:
        want_channels = {
            "Feed_Rate_Actual",
            "Power_Active",
            "Spindle_Speed_Actual",
            "Power_Spindle",
            "Vibration_Severity_X",
            "Vibration_Severity_Y",
        }

    result_channels = []
    for ci, cname in enumerate(channel_names):
        if want_channels and cname not in want_channels:
            continue
        vals = series[idx, ci, :].tolist()
        result_channels.append({
            "name": cname,
            "values": [float(v) if np.isfinite(v) else 0.0 for v in vals],
        })

    return {
        "sample_idx": idx,
        "sample_id": str(sample_ids[idx]),
        "label": str(labels[idx]) if idx < len(labels) else "unknown",
        "channels": result_channels,
        "all_channel_names": channel_names,
        "n_timesteps": int(series.shape[2]),
        "total_samples": n_samples,
    }


def _load_annotations(run_id: str):
    """Load per-sample annotations for an experiment run."""
    import pandas as pd

    # Locate experiment run directory
    exp_root = Path("data/breakage_patterns/stoppage_experiment")
    if not exp_root.exists():
        exp_root = Path("data/breakage_patterns/experiment_runs")

    run_dir = None
    if run_id:
        candidate = exp_root / run_id
        if candidate.exists():
            run_dir = candidate
    if not run_dir:
        candidates = sorted(exp_root.glob("*/experiment_results.json"), reverse=True)
        if candidates:
            run_dir = candidates[0].parent
    if not run_dir:
        return {"error": "No experiment runs found", "samples": [], "run_id": None}

    results_path = run_dir / "experiment_results.json"
    if not results_path.exists():
        return {"error": "results not found", "samples": [], "run_id": str(run_dir.name)}

    with open(results_path) as f:
        results = json.load(f)

    # NPZ index map (prefer stoppage, fall back to breakage)
    npz_path = Path("data/breakage_patterns/stoppage_raw_series.npz")
    if not npz_path.exists():
        npz_path = Path("data/breakage_patterns/breakage_raw_series.npz")
    npz_ids: List[str] = []
    if npz_path.exists():
        npz = np.load(npz_path, allow_pickle=True)
        npz_ids = [str(s) for s in npz.get("sample_ids", [])]
    sid_to_idx = {sid: i for i, sid in enumerate(npz_ids)}

    # Config → phase map
    cfg = results.get("config", {})
    train_ops = set(cfg.get("train_ops", []))
    test_op = cfg.get("test_op", "")
    eval_op = cfg.get("eval_op", "")

    def op_to_phase(op: str) -> str:
        if op in train_ops:
            return "train"
        if op == test_op:
            return "test"
        if op == eval_op:
            return "eval"
        return "unknown"

    # Features CSV for label + operation mapping (prefer stoppage, fall back to breakage)
    feat_path = Path("data/breakage_patterns/stoppage_features.csv")
    if not feat_path.exists():
        feat_path = Path("data/breakage_patterns/breakage_features.csv")
    feat_map: Dict[str, dict] = {}
    if feat_path.exists():
        df = pd.read_csv(
            feat_path,
            usecols=lambda c: c in {
                "sample_id", "label", "operation_id",
                "severity", "stop_type", "gap_seconds",
            },
        )
        for _, row in df.iterrows():
            feat_map[str(row["sample_id"])] = {
                "label": row.get("label", ""),
                "operation_id": row.get("operation_id", ""),
                "severity": row.get("severity"),
                "stop_type": row.get("stop_type", ""),
                "gap_seconds": row.get("gap_seconds"),
            }

    # Per-sample results from phases
    scored: Dict[str, dict] = {}
    for phase_key in (
        "train_phase", "eval_phase", "test_phase",
        "baseline_phase", "train", "eval", "test", "baseline",
    ):
        phase_data = results.get(phase_key)
        if not phase_data or not isinstance(phase_data, dict):
            continue
        phase_name = phase_key.replace("_phase", "")
        for sr in phase_data.get("sample_results", []):
            sid = str(sr.get("sample_id", ""))
            if not sid:
                continue
            scored[sid] = {
                "phase": phase_name,
                "predicted": sr.get("predicted", ""),
                "combined_score": _safe_float(sr.get("combined_score")),
                "pattern_score": _safe_float(sr.get("pattern_score") or sr.get("pattern_rule_score")),
                "model_score": _safe_float(sr.get("model_score") or sr.get("supervised_score")),
                "event_triggered": sr.get("event_triggered", False),
                "patterns_detected": sr.get("patterns_detected", []),
                "correct": sr.get("correct"),
            }

    # Assemble one annotation per NPZ sample
    comparison = results.get("comparison", {})
    threshold = comparison.get("threshold") or cfg.get("threshold", 0.5)

    samples: List[Dict[str, Any]] = []
    for idx, sid in enumerate(npz_ids):
        feat = feat_map.get(sid, {})
        sc = scored.get(sid, {})
        op_id = feat.get("operation_id", "")
        phase = sc.get("phase") or op_to_phase(op_id)
        true_label = feat.get("label", "")

        samples.append({
            "sample_idx": idx,
            "sample_id": sid,
            "phase": phase,
            "true_label": true_label,
            "predicted": sc.get("predicted", ""),
            "combined_score": _safe_float(sc.get("combined_score")),
            "pattern_score": _safe_float(sc.get("pattern_score")),
            "model_score": _safe_float(sc.get("model_score")),
            "event_triggered": sc.get("event_triggered", False),
            "patterns_detected": sc.get("patterns_detected", []),
            "correct": sc.get("correct"),
            "severity": feat.get("severity"),
            "stop_type": feat.get("stop_type", ""),
            "gap_seconds": feat.get("gap_seconds"),
        })

    # Fallback: features CSV order when NPZ unavailable
    if not npz_ids and feat_map:
        for i, (sid, feat) in enumerate(feat_map.items()):
            sc = scored.get(sid, {})
            op_id = feat.get("operation_id", "")
            samples.append({
                "sample_idx": i,
                "sample_id": sid,
                "phase": sc.get("phase") or op_to_phase(op_id),
                "true_label": feat.get("label", ""),
                "predicted": sc.get("predicted", ""),
                "combined_score": _safe_float(sc.get("combined_score")),
                "pattern_score": _safe_float(sc.get("pattern_score")),
                "model_score": _safe_float(sc.get("model_score")),
                "event_triggered": sc.get("event_triggered", False),
                "patterns_detected": sc.get("patterns_detected", []),
                "correct": sc.get("correct"),
                "severity": feat.get("severity"),
                "stop_type": feat.get("stop_type", ""),
                "gap_seconds": feat.get("gap_seconds"),
            })

    return {
        "run_id": str(run_dir.name),
        "threshold": threshold,
        "total_samples": len(samples),
        "samples": samples,
        "phases": sorted({s["phase"] for s in samples}),
        "config": {
            "train_ops": list(train_ops),
            "test_op": test_op,
            "eval_op": eval_op,
        },
    }


# ── Route handlers ───────────────────────────────────────────────────────────

@router.get("/timeseries")
async def experiment_timeseries(
    sample_idx: int = 0,
    channels: str = "",
):
    """Return raw multi-channel time-series data for one sample window."""
    want = set(c.strip() for c in channels.split(",")) if channels else set()
    return _load_timeseries(sample_idx, want)


@router.get("/annotations")
async def experiment_annotations(run_id: str = ""):
    """Per-sample scoring / event annotations for the time-series overlay."""
    return _load_annotations(run_id)


# ── Full-operation waveform endpoint ──────────────────────────────────────────

_CASEDATA_ROOT = Path(os.environ.get("EXPERIMENT_CASEDATA_ROOT", "data/casedata"))
_DEFAULT_OPERATION_ID = os.environ.get("EXPERIMENT_DEFAULT_OPERATION_ID", "")

# Sensor groups per CSV suffix → columns likely wanted for waveform display
_SENSOR_FILES = {
    "TYZBPS": ["Feed_Rate_Actual", "Spindle_Speed_Actual", "Feed_Override",
               "Spindle_Speed_Override", "Tool_Number"],
    "BXCZ3M": ["Power_Spindle", "Power_X1", "Power_Y", "Power_Z"],
    "7DTZHE": ["Vibration_Severity_X", "Vibration_Severity_Y",
               "Chatter_Detection_Amplitude_X", "Chatter_Detection_Amplitude_Y"],
    "92SQBY": ["Power_Active", "Power_Reactive", "Power_Factor"],
}

_DEFAULT_WAVEFORM_CHANNELS = [
    "Feed_Rate_Actual", "Power_Active", "Spindle_Speed_Actual",
    "Power_Spindle", "Vibration_Severity_X", "Vibration_Severity_Y",
]


def _resolve_operation_dir(operation_id: str) -> Path:
    """Resolve an operation directory under the configured casedata root.

    Supports both layouts:
    1) <root>/<operation_id>
    2) <root>/<case_dir>/<operation_id>
    """
    direct = _CASEDATA_ROOT / operation_id
    if direct.exists() and direct.is_dir():
        return direct

    if _CASEDATA_ROOT.exists():
        for case_dir in _CASEDATA_ROOT.iterdir():
            if not case_dir.is_dir():
                continue
            candidate = case_dir / operation_id
            if candidate.exists() and candidate.is_dir():
                return candidate

    return direct


def _list_operation_dirs() -> List[Path]:
    """List operation directories from direct and case-nested layouts."""
    dirs: Dict[str, Path] = {}
    if not _CASEDATA_ROOT.exists():
        return []

    for d in _CASEDATA_ROOT.iterdir():
        if d.is_dir() and d.name.startswith("OF"):
            dirs[d.name] = d

    for case_dir in _CASEDATA_ROOT.iterdir():
        if not case_dir.is_dir():
            continue
        for d in case_dir.iterdir():
            if d.is_dir() and d.name.startswith("OF") and d.name not in dirs:
                dirs[d.name] = d

    return [dirs[k] for k in sorted(dirs)]


def _lttb_downsample(x: np.ndarray, y: np.ndarray, target: int) -> tuple:
    """Largest-Triangle-Three-Buckets downsampling."""
    n = len(x)
    if n <= target:
        return x, y
    bucket_size = (n - 2) / (target - 2)
    out_x = [x[0]]
    out_y = [y[0]]
    a_idx = 0
    for i in range(1, target - 1):
        b_start = int((i) * bucket_size) + 1
        b_end = int((i + 1) * bucket_size) + 1
        b_end = min(b_end, n)
        c_start = int((i + 1) * bucket_size) + 1
        c_end = int((i + 2) * bucket_size) + 1
        c_end = min(c_end, n)
        avg_x = np.mean(x[c_start:c_end]) if c_start < c_end else x[-1]
        avg_y = np.mean(y[c_start:c_end]) if c_start < c_end else y[-1]
        max_area = -1.0
        best = b_start
        for j in range(b_start, b_end):
            area = abs(
                (x[a_idx] - avg_x) * (y[j] - out_y[-1])
                - (x[a_idx] - x[j]) * (avg_y - out_y[-1])
            )
            if area > max_area:
                max_area = area
                best = j
        out_x.append(x[best])
        out_y.append(y[best])
        a_idx = best
    out_x.append(x[-1])
    out_y.append(y[-1])
    return np.array(out_x), np.array(out_y)


def _enrich_regions_with_feedback(
    regions: List[Dict[str, Any]], run_id: str
) -> None:
    """Annotate regions in-place with feedback data from experiment results.

    Matches regions to the experiment's per-sample results by ``sample_id``,
    adding ``feedback_action``, ``feedback_given``, and ``predicted_positive``
    to each region dict.
    """
    exp_root = Path("data/breakage_patterns/stoppage_experiment")
    results_path = None
    # The run_dir may contain sub-directories with experiment_results.json
    candidate = exp_root / run_id
    if candidate.exists():
        for p in candidate.rglob("experiment_results.json"):
            results_path = p
            break
    if not results_path:
        return
    try:
        with open(results_path) as f:
            data = json.load(f)
    except Exception:
        return

    # Build sample_id → feedback map from all phases
    fb_map: Dict[str, Dict[str, Any]] = {}
    for phase_key in ("test_phase", "eval_phase", "test", "eval"):
        phase = data.get(phase_key)
        if not phase or not isinstance(phase, dict):
            continue
        for sr in phase.get("sample_results", []):
            sid = sr.get("sample_id", "")
            if sid:
                fb_map[sid] = {
                    "feedback_given": sr.get("feedback_given", False),
                    "feedback_action": sr.get("feedback_action", ""),
                    "predicted_positive": sr.get("predicted_positive", False),
                    "significance_score": sr.get("significance_score"),
                    "detected_patterns": sr.get("detected_patterns", []),
                }

    for r in regions:
        fb = fb_map.get(r.get("sample_id", ""))
        if fb:
            r["feedback_given"] = fb["feedback_given"]
            r["feedback_action"] = fb["feedback_action"]
            r["predicted_positive"] = fb["predicted_positive"]
            r["significance_score"] = fb.get("significance_score")
            r["detected_patterns"] = fb.get("detected_patterns", [])


def _load_operation_waveform(
    operation_id: str,
    want_channels: Set[str],
    max_points: int = 2000,
    run_id: str = "",
):
    """Load full-operation waveform from raw CSVs with downsampling."""
    import pandas as pd

    op_dir = _resolve_operation_dir(operation_id)
    if not op_dir.exists():
        return {"error": f"Operation directory not found: {operation_id}"}

    # Determine which CSV files to read based on requested channels
    all_available_channels: List[str] = []
    frames = []
    for suffix, cols in _SENSOR_FILES.items():
        csv_files = list(op_dir.glob(f"*_{suffix}.csv"))
        if not csv_files:
            continue
        needed = [c for c in cols if (not want_channels or c in want_channels)]
        if not needed and want_channels:
            # Still read for channel discovery
            all_available_channels.extend(cols)
            continue
        try:
            usecols = ["timestamp"] + [c for c in cols]
            df = pd.read_csv(csv_files[0], usecols=lambda c: c in set(usecols))
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp").sort_index()
            frames.append(df)
            all_available_channels.extend([c for c in df.columns if c != "timestamp"])
        except Exception as e:
            logger.warning("Failed to read %s: %s", csv_files[0], e)

    if not frames:
        return {"error": f"No CSV data found for {operation_id}"}

    # Join all frames on timestamp (outer join, forward-fill gaps)
    combined = frames[0]
    for f in frames[1:]:
        combined = combined.join(f, how="outer", rsuffix="_dup")
    combined = combined.ffill().bfill()

    # Drop duplicate columns
    combined = combined.loc[:, ~combined.columns.str.endswith("_dup")]

    # Filter to requested channels
    if want_channels:
        keep = [c for c in combined.columns if c in want_channels]
    else:
        keep = [c for c in _DEFAULT_WAVEFORM_CHANNELS if c in combined.columns]
    if not keep:
        keep = list(combined.columns[:6])

    total_points = len(combined)

    # Numeric x-axis (seconds from start)
    t0 = combined.index[0]
    x_seconds = np.array([(t - t0).total_seconds() for t in combined.index])

    # Downsample each channel with LTTB
    channels_out = []
    for col in keep:
        raw_vals = combined[col].values.astype(float)
        # Drop NaN entries instead of replacing with 0 (avoid scatter spikes)
        finite_mask = np.isfinite(raw_vals)
        col_x = x_seconds[finite_mask]
        col_y = raw_vals[finite_mask]
        if len(col_x) == 0:
            continue
        # Sort by timestamp to avoid scatter from unsorted outer-join rows
        sort_idx = np.argsort(col_x)
        col_x = col_x[sort_idx]
        col_y = col_y[sort_idx]
        ds_x, ds_y = _lttb_downsample(col_x, col_y, max_points)
        channels_out.append({
            "name": col,
            "timestamps": [round(float(v), 1) for v in ds_x],
            "values": [_safe_float(round(float(v), 4)) for v in ds_y],
        })

    # Load event regions from features CSVs (both breakage and stoppage)
    regions = []
    _region_seen: set = set()  # deduplicate across CSVs

    def _load_regions_from_csv(csv_path: Path, source: str) -> None:
        """Parse event regions from a features CSV and append to ``regions``."""
        if not csv_path.exists():
            return
        try:
            feat_df = pd.read_csv(csv_path)
            op_feats = feat_df[feat_df["operation_id"] == operation_id]
            for _, row in op_feats.iterrows():
                evt_ts = row.get("event_timestamp")
                gap_s = row.get("gap_seconds", 60)
                label = row.get("label", "unknown")
                sample_id = row.get("sample_id", "")
                severity = row.get("severity", "")
                if pd.isna(evt_ts):
                    continue
                # Deduplicate by sample_id
                sid = str(sample_id)
                if sid in _region_seen:
                    continue
                _region_seen.add(sid)
                evt_time = pd.to_datetime(evt_ts, utc=True)
                # Window is [evt_time - gap_s, evt_time]
                window_start = (evt_time - pd.Timedelta(seconds=float(gap_s)) - t0).total_seconds()
                window_end = (evt_time - t0).total_seconds()
                regions.append({
                    "start_s": round(max(0, window_start), 1),
                    "end_s": round(min(float(x_seconds[-1]), window_end), 1),
                    "label": label,
                    "sample_id": sid,
                    "severity": str(severity) if not pd.isna(severity) else "",
                    "event_timestamp": str(evt_time),
                    "source": source,
                })
        except Exception as e:
            logger.warning("Failed to load event regions from %s: %s", csv_path, e)

    _load_regions_from_csv(Path("data/breakage_patterns/breakage_features.csv"), "breakage")
    # Stoppage features (try all gap variants so we capture the run's gap)
    for sp in sorted(Path("data/breakage_patterns").glob("stoppage_features*.csv")):
        _load_regions_from_csv(sp, "stoppage")

    # ── Enrich regions with feedback data from experiment results ──
    if run_id:
        _enrich_regions_with_feedback(regions, run_id)

    # Duration in human-readable form
    duration_s = float(x_seconds[-1])
    duration_h = duration_s / 3600

    return {
        "operation_id": operation_id,
        "channels": channels_out,
        "all_channel_names": sorted(set(all_available_channels)),
        "regions": regions,
        "total_points": total_points,
        "displayed_points": max_points if total_points > max_points else total_points,
        "duration_seconds": round(duration_s, 1),
        "duration_hours": round(duration_h, 2),
        "start_time": str(combined.index[0]),
        "end_time": str(combined.index[-1]),
    }


@router.get("/operation-waveform")
async def operation_waveform(
    operation_id: str = Query(_DEFAULT_OPERATION_ID or "", description="Operation folder name"),
    channels: str = Query("", description="Comma-separated channel names (empty=defaults)"),
    max_points: int = Query(2000, ge=200, le=10000, description="Max points per channel after downsampling"),
    run_id: str = Query("", description="Experiment run ID for feedback enrichment"),
):
    """Full-operation continuous waveform with highlighted event regions."""
    if not operation_id:
        ops = _list_operation_dirs()
        if not ops:
            return {"error": "No operation directories found"}
        operation_id = ops[0].name

    want = set(c.strip() for c in channels.split(",")) if channels else set()
    return _load_operation_waveform(operation_id, want, max_points, run_id=run_id)


@router.get("/operations")
async def list_operations():
    """List available operations with basic metadata."""
    ops = []
    for d in _list_operation_dirs():
        n_csvs = len(list(d.glob("*.csv")))
        ops.append({"id": d.name, "n_csv_files": n_csvs})
    return {"operations": ops}

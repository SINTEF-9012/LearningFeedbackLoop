"""Session management router — CRUD, upload, playback control."""

from __future__ import annotations

import asyncio
import csv
from copy import deepcopy
from functools import lru_cache
import io
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Body, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from ..agents.memory.cycle_tracker import get_cycle_tracker
from ..agents.memory.init import is_initialized as is_memory_initialized
from ..agents.memory.orchestrator import get_orchestrator
from ..agents.processing.harmonic_config import casedata_stoppage_preset, pair_casedata_preset
from ..agents.processing.harmonic_features import select_harmonic_columns
from ..agents.processing.harmonic_peak_pairs import discover_peak_pair_columns
from ..agents.processing.dataset_loader import DatasetLoader, OperationInfo, pd
from ..events import publish_feature
from ..fft_streamer import fft_stream_task
from ..ingestion.registry import create_source, registered_sources
from ..inference_streamer import inference_stream_task
from ..metadata_utils import get_sample_frequency
from ..mqtt_transport import ensure_mqtt_transport_available
from ..session_active_context import build_active_session_context
from ..ingestion.simulated_casedata import SimulatedCasedataSource

from .dependencies import (
    DEFAULT_SAMPLES_PER_TICK,
    DEFAULT_SPEED,
    PlaybackConfigUpdate,
    ReplayRequest,
    SessionConfig,
    get_session_or_404,
    get_sessions_dict,
    json_default,
)

logger = logging.getLogger(__name__)

router = APIRouter()
supplemental_router = APIRouter()

# ── Upload size limit ────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
_SITE_C_CASE_DIR_PREFIX = "site_c - machine_c1 - case_c1"
_CASEDATA_SIGNAL_PREVIEW_COLUMNS = frozenset(
    {
        "Feed_Rate_Actual",
        "Feed_Rate_Commanded",
        "Spindle_Speed_Actual",
        "Spindle_Speed_Commanded",
        "Tool_Number",
        "tool_number",
        "tool",
        "tool_id",
        "Cnc_Tool_Number_RT",
        "Cnc_Tool_Number",
        "CNC_Tool_Number",
        "num_teeth",
        "NumberOfTeeth",
        "CNC_parameters_teeth_num",
        "teeth",
    }
)


# ── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_payload(payload: dict, config: Optional[dict] = None):
    """Normalize uploaded JSON into ``(data, metadata)``."""
    data: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

    # Case 1: Already normalized
    if "channels" in payload:
        for ch, chdata in payload["channels"].items():
            if isinstance(chdata, dict) and "signal" in chdata:
                data[ch] = chdata["signal"]
            elif isinstance(chdata, list):
                data[ch] = chdata
        metadata = payload.get("metadata", {})

    # Case 2: MATLAB-style export
    elif any(k.startswith("Channel_") for k in payload.keys()):
        logger.info("Detected MATLAB-style export")
        for k, v in payload.items():
            if k.startswith("Channel_") and isinstance(v, dict) and "Signal" in v:
                name = v.get("SignalName", k)
                data[name] = v["Signal"]
        if "File_Header" in payload:
            metadata["sample_frequency"] = payload["File_Header"].get("SampleFrequency")
            metadata["file_header"] = payload["File_Header"]
        machining_keys = ["d", "z", "ap", "ae", "vc", "n", "f", "vf", "type", "break", "fg", "fp"]
        metadata["machining"] = {k: payload[k] for k in machining_keys if k in payload}

    # Case 3: Config-driven
    elif config:
        channel_keys = config.get("channel_keys", [])
        signal_field = config.get("signal_field", "Signal")
        name_field = config.get("name_field", "SignalName")
        for ck in channel_keys:
            ch = payload[ck]
            name = ch.get(name_field, ck)
            data[name] = ch[signal_field]
        metadata["sample_frequency"] = config.get("sample_frequency")

    else:
        raise ValueError("Unsupported JSON structure")

    if "playback_speed" in payload:
        metadata["playback_speed"] = payload["playback_speed"]

    return data, metadata


def _extract_sample_labels(payload: Dict[str, Any], data: Dict[str, Any]) -> Optional[List[str]]:
    raw_labels = payload.get("labels")
    if raw_labels is None:
        return None
    if not isinstance(raw_labels, list):
        raise ValueError("labels must be an array when provided")
    if not data:
        raise ValueError("labels provided without any channel data")

    sample_count = min(len(series) for series in data.values())
    if len(raw_labels) != sample_count:
        raise ValueError(
            f"labels length {len(raw_labels)} does not match sample count {sample_count}"
        )

    labels: List[str] = []
    for value in raw_labels:
        if value is None:
            labels.append("unknown")
            continue
        label = str(value).strip()
        labels.append(label or "unknown")
    return labels


def _apply_harmonic_runtime_settings(
    cfg: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Mirror harmonic scorer selection from session config into metadata."""
    meta = metadata if isinstance(metadata, dict) else {}

    def _preferred_pair_dataset() -> str:
        requested = str(
            cfg.get("harmonic_dataset")
            or meta.get("harmonic_dataset")
            or meta.get("harmonic_dataset_name")
            or ""
        ).strip().lower()
        if requested in {"pair_raw", "pair_casedata", "pair_lfl"}:
            return requested
        if requested == "casedata":
            return "pair_lfl"

        casedata_meta = meta.get("casedata") if isinstance(meta.get("casedata"), dict) else {}
        source_hints = " ".join(
            part
            for part in (
                str(meta.get("source") or "").strip().lower(),
                str(casedata_meta.get("root") or "").strip().lower(),
                str(casedata_meta.get("case_dir") or "").strip().lower(),
                str(meta.get("machine_id") or "").strip().lower(),
            )
            if part
        )
        if casedata_meta or "casedata" in source_hints or "site_b" in source_hints or "site_c" in source_hints:
            return "pair_lfl"
        return "pair_raw"

    explicit_pair_dataset = str(
        cfg.get("harmonic_dataset")
        or meta.get("harmonic_dataset")
        or meta.get("harmonic_dataset_name")
        or ""
    ).strip().lower()

    if _is_site_c_casedata_metadata(meta):
        previous_kind = str(cfg.get("harmonic_scorer_kind") or "").strip().lower()
        previous_dataset = str(cfg.get("harmonic_dataset") or "").strip().lower()
        if explicit_pair_dataset == "pair_casedata":
            if previous_kind != "pair" or previous_dataset != "pair_casedata":
                logger.info("SITE_C casedata session: respecting explicit harmonic_dataset=pair_casedata override")
            cfg["harmonic_scorer_kind"] = "pair"
            cfg["harmonic_dataset"] = "pair_casedata"
        else:
            if previous_kind != "pair" or previous_dataset != "pair_lfl":
                logger.info("SITE_C casedata session: pinning harmonic_scorer_kind=pair harmonic_dataset=pair_lfl")
            cfg["harmonic_scorer_kind"] = "pair"
            cfg["harmonic_dataset"] = "pair_lfl"

    kind = str(cfg.get("harmonic_scorer_kind") or "context").strip().lower()
    if kind not in {"context", "pair"}:
        kind = "context"
    cfg["harmonic_scorer_kind"] = kind
    meta["harmonic_scorer_kind"] = kind

    pause_on_alert = bool(cfg.get("pause_on_alert", False))
    cfg["pause_on_alert"] = pause_on_alert
    meta["pause_on_alert"] = pause_on_alert

    dataset = cfg.get("harmonic_dataset")
    dataset_str = str(dataset).strip().lower() if dataset is not None else ""
    context_datasets = {"casedata", "stoppage_1hz", "site_a_line2", "raw_accelerometer"}
    if kind == "pair":
        pair_dataset = _preferred_pair_dataset()
        cfg["harmonic_dataset"] = pair_dataset
        meta["harmonic_dataset"] = pair_dataset
    elif dataset_str in {"", "auto", "default", "none"}:
        cfg.pop("harmonic_dataset", None)
        meta.pop("harmonic_dataset", None)
    elif dataset_str in context_datasets:
        cfg["harmonic_dataset"] = dataset_str
        meta["harmonic_dataset"] = dataset_str
    else:
        cfg.pop("harmonic_dataset", None)
        meta.pop("harmonic_dataset", None)

    return meta


# ── Playback task ────────────────────────────────────────────────────────────

async def playback_task(session_id: str, sessions_dict: Dict[str, Dict[str, Any]]):
    """Stream time-domain frames to subscribers at real-time (or scaled) pace."""
    s = sessions_dict[session_id]
    s.pop("last_error", None)
    cfg = s["config"]
    data = s["data"]
    metadata = s["metadata"]

    fs = get_sample_frequency(metadata)
    Ts = 1.0 / fs
    channels = cfg.get("channels") or list(data.keys())
    speed = float(cfg.get("speed", DEFAULT_SPEED))
    samples_per_tick = max(1, int(cfg.get("samples_per_tick", DEFAULT_SAMPLES_PER_TICK)))

    pos = int(s.get("position", 0))
    if not channels:
        raise ValueError("No channels available to process")
    n_max = min(len(data[ch]) for ch in channels)

    start_wall = time.perf_counter()
    start_pos = pos
    last_speed = speed
    last_samples_per_tick = samples_per_tick
    next_emit_time = start_wall

    try:
        while s.get("running", False) and pos < n_max:
            if s.get("paused", False):
                await asyncio.sleep(0.1)
                start_wall = time.perf_counter()
                start_pos = pos
                next_emit_time = start_wall
                continue

            cfg_now = s.get("config", {})
            speed = float(cfg_now.get("speed", DEFAULT_SPEED))
            samples_per_tick = max(1, int(cfg_now.get("samples_per_tick", DEFAULT_SAMPLES_PER_TICK)))
            if speed != last_speed or samples_per_tick != last_samples_per_tick:
                last_speed = speed
                last_samples_per_tick = samples_per_tick
                start_wall = time.perf_counter()
                start_pos = pos
                next_emit_time = start_wall

            i0 = pos
            i1 = min(pos + samples_per_tick, n_max)
            if i0 >= i1:
                break

            if samples_per_tick == 1:
                t = i0 / fs
                frame: Dict[str, Any] = {"t": t, "i": i0, "fs": fs}
                for ch in channels:
                    arr = data[ch]
                    frame[ch] = arr[i0] if i0 < len(arr) else None
            else:
                t0 = i0 / fs
                t1 = (i1 - 1) / fs
                frame = {"t0": t0, "t1": t1, "i0": i0, "i1": i1, "fs": fs}
                for ch in channels:
                    arr = data[ch]
                    frame[ch] = arr[i0:i1]

            pos = i1
            s["position"] = pos

            for q in list(s.get("subscribers", [])):
                await q.put(frame)

            try:
                payload = {
                    "type": "time",
                    "session_id": session_id,
                    "position": pos,
                    "frame": frame,
                    "metadata": deepcopy(metadata),
                }
                source_name = metadata.get("source")
                if isinstance(source_name, str) and source_name:
                    payload["source"] = source_name
                await publish_feature(session_id, payload)
            except Exception:
                logger.warning("Feature publish failed (time); continuing stream", exc_info=True)

            samples_emitted = pos - start_pos
            target_elapsed = (samples_emitted * Ts) / max(speed, 1e-9)
            next_emit_time = start_wall + target_elapsed
            now = time.perf_counter()
            delay = next_emit_time - now
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)

        s["running"] = False
        eos = {"eos": True, "fs": fs, "final_i": pos}
        for q in list(s.get("subscribers", [])):
            try:
                await q.put(eos)
            except Exception:
                pass

    except Exception as e:
        s["last_error"] = str(e)
        s["running"] = False
        logger.exception("playback_task crashed for session %s", session_id)
    finally:
        s["task"] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _start_fft_task(s: Dict[str, Any]) -> None:
    """Start the FFT streaming task if not already running."""
    fft_task = s.get("fft_task")
    if fft_task is None or (hasattr(fft_task, "done") and fft_task.done()):
        s["running_fft"] = True
        s["fft_task"] = asyncio.create_task(fft_stream_task(s))
    else:
        logger.debug("FFT task already exists and appears running")


def _start_inference_task(s: Dict[str, Any]) -> None:
    """Start the inference streaming task if not already running."""
    inf_task = s.get("inference_task")
    if inf_task is None or (hasattr(inf_task, "done") and inf_task.done()):
        s["running_inference"] = True
        s["inference_task"] = asyncio.create_task(inference_stream_task(s))
    else:
        logger.debug("Inference task already exists and appears running")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_site_c_casedata_metadata(metadata: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(metadata, dict):
        return False

    source = str(metadata.get("source") or "").strip().lower()
    if source and source != "simulated_casedata":
        return False

    casedata_meta = metadata.get("casedata")
    if not isinstance(casedata_meta, dict):
        return False

    case_dir = _text(casedata_meta.get("case_dir"))
    if case_dir is None:
        return False

    return Path(case_dir).name.strip().lower().startswith(_SITE_C_CASE_DIR_PREFIX)


def _coerce_int(value: Any) -> int | None:
    text = _text(value)
    if text is None:
        return None
    if text.upper().startswith("T"):
        text = text[1:]
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _session_state_label(session: Dict[str, Any]) -> str:
    return _session_status_fields(session)["status"]


def _session_total_samples(session: Dict[str, Any]) -> int:
    cached_total = session.get("total_samples")
    if isinstance(cached_total, int) and cached_total >= 0:
        return cached_total

    data = session.get("data")
    if not isinstance(data, dict) or not data:
        return 0

    try:
        return min(len(series) for series in data.values())
    except Exception:
        return 0


def _session_source_fields(session: Dict[str, Any]) -> Dict[str, Any]:
    metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
    source_config = session.get("source_config") if isinstance(session.get("source_config"), dict) else {}
    casedata_meta = metadata.get("casedata") if isinstance(metadata.get("casedata"), dict) else {}

    source = str(
        session.get("source_name")
        or metadata.get("source")
        or source_config.get("source")
        or "simulated_file"
    )
    case_dir = _text(source_config.get("case_dir")) or _text(casedata_meta.get("case_dir"))
    operation_id = _text(source_config.get("operation_id")) or _text(casedata_meta.get("operation_id"))
    tool_id = _text(casedata_meta.get("tool_id"))
    topic = _text(source_config.get("topic"))
    resolved_start_position = source_config.get("start_position")
    requested_start_position = source_config.get("requested_start_position")
    start_at_first_cutting_row = bool(source_config.get("start_at_first_cutting_row", False))
    start_at_label = _text(source_config.get("start_at_label"))
    resolved_start_label = _text(source_config.get("resolved_start_label"))
    resolved_start_label_index = source_config.get("resolved_start_label_index")
    rationale = _text(source_config.get("rationale"))

    if source == SimulatedCasedataSource.name:
        label_parts = [part for part in (case_dir, operation_id) if part]
        source_label = " / ".join(label_parts) or "Casedata stream"
    elif source == "mqtt":
        source_label = topic or "MQTT live"
    else:
        source_label = source

    return {
        "source": source,
        "source_label": source_label,
        "case_dir": case_dir,
        "operation_id": operation_id,
        "tool_id": tool_id,
        "resolved_start_position": int(resolved_start_position) if resolved_start_position is not None else None,
        "requested_start_position": int(requested_start_position) if requested_start_position is not None else None,
        "start_at_first_cutting_row": start_at_first_cutting_row,
        "start_at_label": start_at_label,
        "resolved_start_label": resolved_start_label,
        "resolved_start_label_index": int(resolved_start_label_index) if resolved_start_label_index is not None else None,
        "rationale": rationale,
    }


def _session_status_fields(session: Dict[str, Any]) -> Dict[str, Any]:
    running = bool(session.get("running"))
    paused = bool(session.get("paused"))
    loading = bool(session.get("loading"))
    last_error = _text(session.get("last_error"))
    position = int(session.get("position", 0) or 0)
    total_samples = _session_total_samples(session)
    progress = None
    if total_samples > 0:
        progress = max(0.0, min(1.0, position / max(total_samples, 1)))

    if last_error:
        status = "error"
        status_label = "Error"
    elif loading:
        status = "loading"
        status_label = "Loading"
    elif running:
        status = "paused" if paused else "live"
        status_label = "Paused" if paused else "Live"
    elif total_samples > 0 and position >= total_samples:
        status = "completed"
        status_label = "Completed"
    elif position > 0:
        status = "stopped"
        status_label = "Stopped"
    else:
        status = "idle"
        status_label = "Ready"

    return {
        "status": status,
        "status_label": status_label,
        "running": running,
        "paused": paused,
        "position": position,
        "total_samples": total_samples,
        "progress": progress,
        "last_error": last_error,
        "loading": loading,
    }


def _activate_pending_live_source(session_id: str, session: Dict[str, Any], request: Request) -> None:
    if session.get("loading") or not session.get("pending_live_start"):
        return
    if session.get("running"):
        session["pending_live_start"] = False
        return

    source = session.get("_stream_source")
    if source is None:
        session["pending_live_start"] = False
        return

    loop = getattr(request.app.state, "main_loop", None)
    if loop is None or loop.is_closed():
        return

    async def _activate() -> None:
        if session.get("loading") or not session.get("pending_live_start"):
            return
        session["pending_live_start"] = False
        session["running"] = True
        source.start(session_id, startup_delay=0.15)
        try:
            _start_inference_task(session)
        except Exception:
            logger.exception("Demo: inference task start failed")

    try:
        asyncio.run_coroutine_threadsafe(_activate(), loop).result(timeout=0.5)
    except Exception:
        logger.exception("Demo: failed to activate pending live source for %s", session_id)


def _build_session_summary(session_id: str, session: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        **_session_status_fields(session),
        **_session_source_fields(session),
    }


def _default_casedata_roots() -> List[Path]:
    configured = os.environ.get("SIMULATED_CASEDATA_ROOT", "").strip()
    if configured:
        return [Path(configured)]
    return [Path("data/casedata"), Path("data/site_a")]


def _resolve_casedata_root(requested_root: Any = None, case_dir: str | None = None) -> str:
    configured = _text(requested_root)
    if configured is not None:
        return str(Path(configured))

    roots = _default_casedata_roots()
    if case_dir:
        for root in roots:
            if (Path(root) / case_dir).is_dir():
                return str(root)
            try:
                loader = DatasetLoader(root)
            except FileNotFoundError:
                continue
            if case_dir in loader.list_cases():
                return str(root)

    for root in roots:
        if root.exists():
            return str(root)
    return str(roots[0])


def _casedata_preview_cache_key(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
    except OSError:
        return str(path), 0, 0
    return str(path), int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=2048)
def _preview_casedata_signals_cached(
    machine_state_path: str,
    _mtime_ns: int,
    _size: int,
) -> Dict[str, Any]:
    if pd is None:
        return {}

    path = Path(machine_state_path)
    try:
        frame = pd.read_csv(
            path,
            usecols=lambda column: str(column) in _CASEDATA_SIGNAL_PREVIEW_COLUMNS,
            low_memory=False,
        )
    except Exception:
        logger.debug("Could not preview machine-state row for %s", path, exc_info=True)
        return {}

    if frame.empty:
        return {}

    preview = frame
    if "Feed_Rate_Actual" in preview.columns:
        preview = preview[pd.to_numeric(preview["Feed_Rate_Actual"], errors="coerce").fillna(0.0) > 0.0]
    if "Spindle_Speed_Actual" in preview.columns:
        cutting = preview[pd.to_numeric(preview["Spindle_Speed_Actual"], errors="coerce").fillna(0.0) > 0.0]
        if not cutting.empty:
            preview = cutting

    if not preview.empty and "Tool_Number" in preview.columns:
        tool_numbers = preview["Tool_Number"].map(_coerce_int)
        dominant_tools = tool_numbers.dropna()
        if not dominant_tools.empty:
            dominant_tool = int(dominant_tools.value_counts().index[0])
            dominant_rows = preview[tool_numbers == dominant_tool]
            if not dominant_rows.empty:
                preview = dominant_rows

    row = preview.iloc[0] if not preview.empty else frame.iloc[0]
    return {
        str(column): value
        for column, value in row.items()
        if column != "timestamp"
    }


def _preview_casedata_signals(operation: OperationInfo) -> Dict[str, Any]:
    if pd is None:
        return {}

    machine_state_path = operation.channel_files.get("machine_state")
    if machine_state_path is None or not machine_state_path.exists():
        return {}

    return _preview_casedata_signals_cached(*_casedata_preview_cache_key(machine_state_path))


@lru_cache(maxsize=2048)
def _preview_casedata_harmonics_cached(
    vibration_path: str,
    _mtime_ns: int,
    _size: int,
) -> Dict[str, Any]:
    if pd is None:
        return {}

    path = Path(vibration_path)
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        logger.debug("Could not preview vibration header for %s", path, exc_info=True)
        return {}

    harmonic_columns = select_harmonic_columns(
        list(header.columns),
        list(casedata_stoppage_preset().harmonic_column_patterns or []),
    )
    pair_cfg = pair_casedata_preset()
    pair_specs = discover_peak_pair_columns(
        list(header.columns),
        frequency_patterns=list(pair_cfg.pair_frequency_column_patterns or []),
        amplitude_patterns=list(pair_cfg.pair_amplitude_column_patterns or []),
        k_peaks=int(pair_cfg.k_peaks or 5),
    )
    pair_frequency_columns = [spec.frequency_col for spec in pair_specs if spec.frequency_col]
    pair_amplitude_columns = [spec.amplitude_col for spec in pair_specs if spec.amplitude_col]
    preview_columns = list(dict.fromkeys([*harmonic_columns, *pair_frequency_columns, *pair_amplitude_columns]))
    signal_columns = list(dict.fromkeys([*harmonic_columns, *pair_amplitude_columns]))
    if not preview_columns:
        return {}

    selected_columns = set(preview_columns)
    try:
        frame = pd.read_csv(
            path,
            usecols=lambda column: str(column) in selected_columns,
            low_memory=False,
        )
    except Exception:
        logger.debug("Could not preview vibration row for %s", path, exc_info=True)
        return {}

    if frame.empty:
        return {}

    preview = frame
    try:
        preview_values = frame[signal_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        nonzero_rows = preview_values.abs().sum(axis=1) > 0.0
        if bool(nonzero_rows.any()):
            preview = frame.loc[nonzero_rows]
    except Exception:
        logger.debug("Could not filter harmonic preview rows for %s", path, exc_info=True)

    row = preview.iloc[0] if not preview.empty else frame.iloc[0]
    return {
        str(column): value
        for column, value in row.items()
        if column != "timestamp"
    }


def _preview_casedata_harmonics(operation: OperationInfo) -> Dict[str, Any]:
    if pd is None:
        return {}

    vibration_path = operation.channel_files.get("vibration")
    if vibration_path is None or not vibration_path.exists():
        return {}

    return _preview_casedata_harmonics_cached(*_casedata_preview_cache_key(vibration_path))


def _first_positive_signal_index(series: Any, *, absolute: bool = True) -> Optional[int]:
    if not isinstance(series, list) or not series:
        return None
    for index, raw_value in enumerate(series):
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            continue
        value = abs(numeric) if absolute else numeric
        if value > 0.0:
            return index
    return None


def _first_casedata_cutting_index(data: Dict[str, Any]) -> Optional[int]:
    if not isinstance(data, dict) or not data:
        return None

    spindle_actual = data.get("Spindle_Speed_Actual")
    feed_actual = data.get("Feed_Rate_Actual")
    power_spindle = data.get("Power_Spindle")

    spindle_idx = _first_positive_signal_index(spindle_actual)
    feed_idx = _first_positive_signal_index(feed_actual)
    power_idx = _first_positive_signal_index(power_spindle)

    lengths = [len(series) for series in (spindle_actual, feed_actual, power_spindle) if isinstance(series, list) and series]
    if not lengths:
        return None
    total_samples = min(lengths)

    def _value(series: Any, index: int) -> float:
        if not isinstance(series, list) or index >= len(series):
            return 0.0
        try:
            return abs(float(series[index]))
        except (TypeError, ValueError):
            return 0.0

    for index in range(total_samples):
        if _value(spindle_actual, index) > 0.0 and _value(feed_actual, index) > 0.0:
            return index
    for index in range(total_samples):
        if _value(spindle_actual, index) > 0.0 and _value(power_spindle, index) > 0.0:
            return index
    for index in range(total_samples):
        if _value(feed_actual, index) > 0.0 and _value(power_spindle, index) > 0.0:
            return index

    for candidate in (spindle_idx, feed_idx, power_idx):
        if candidate is not None:
            return candidate
    return None


def _casedata_sindit_asset_iri(case_dir: Optional[str]) -> Optional[str]:
    """Map a casedata case dir to its SINDIT machine-asset IRI.

    e.g. ``"Site_a - MACHINE_A1 - CASE_A1"`` →
    ``"urn:lfl:asset:site_a---machine_a1---case_a1"`` — the deterministic
    convention the SINDIT sync used. Returns ``None`` for an empty case dir; the
    enrichment path degrades gracefully if the asset does not exist.
    """
    text = (case_dir or "").strip()
    if not text:
        return None
    slug = text.lower().replace(" - ", "---").replace(" ", "-")
    return f"urn:lfl:asset:{slug}"


def _resolve_casedata_start_position(
    data: Dict[str, Any],
    *,
    requested_start_position: int,
    start_at_first_cutting_row: bool,
) -> int:
    manual_offset = max(0, int(requested_start_position or 0))
    if not start_at_first_cutting_row:
        return manual_offset

    base_index = _first_casedata_cutting_index(data)
    if base_index is None:
        return manual_offset
    return max(0, base_index + manual_offset)


def _resolve_sample_label_start_position(
    sample_labels: List[Any],
    *,
    requested_start_position: int,
    start_at_label: str | None,
    start_label_lead_in_samples: int,
) -> tuple[int, int | None]:
    resolved_start = max(0, int(requested_start_position or 0))
    label = _text(start_at_label)
    if label is None:
        return resolved_start, None

    normalized_target = label.strip().lower()
    for index, sample_label in enumerate(sample_labels or []):
        if not isinstance(sample_label, str):
            continue
        if sample_label.strip().lower() != normalized_target:
            continue
        lead_in = max(0, int(start_label_lead_in_samples or 0))
        return max(0, index - lead_in), index

    return resolved_start, None


def _summarize_casedata_operation(casedata_root: str, operation: OperationInfo) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "sample_frequency": 1.0,
        "source": SimulatedCasedataSource.name,
        "casedata": {
            "root": casedata_root,
            "case_dir": operation.case_dir,
            "operation_id": operation.operation_id,
            "tool_id": operation.tool_id,
        },
    }
    signals = {
        **_preview_casedata_harmonics(operation),
        **_preview_casedata_signals(operation),
    }
    harmonic_cfg = casedata_stoppage_preset()
    harmonic_columns = select_harmonic_columns(
        list(signals.keys()),
        list(harmonic_cfg.harmonic_column_patterns or []),
    )
    pair_cfg = pair_casedata_preset()
    pair_specs = discover_peak_pair_columns(
        list(signals.keys()),
        frequency_patterns=list(pair_cfg.pair_frequency_column_patterns or []),
        amplitude_patterns=list(pair_cfg.pair_amplitude_column_patterns or []),
        k_peaks=int(pair_cfg.k_peaks or 5),
    )
    pair_columns = [
        column_name
        for spec in pair_specs
        for column_name in (spec.frequency_col, spec.amplitude_col)
        if column_name
    ]

    def _has_nonzero_preview_value(column: str) -> bool:
        try:
            return abs(float(signals.get(column))) > 0.0
        except (TypeError, ValueError):
            return False

    harmonic_preview_available = any(
        _has_nonzero_preview_value(column) for column in harmonic_columns
    )
    pair_preview_available = any(
        _has_nonzero_preview_value(spec.amplitude_col)
        for spec in pair_specs
        if spec.amplitude_col
    )
    active_context = build_active_session_context(
        {
            "data": {key: [value] for key, value in signals.items()},
            "metadata": metadata,
            "position": 1,
            "source_config": {
                "case_dir": operation.case_dir,
                "operation_id": operation.operation_id,
            },
        }
    ) or {}
    missing_fields = list(active_context.get("missing_fields") or [])
    if not harmonic_columns and not pair_columns:
        missing_fields.append("harmonic columns")
    elif not harmonic_preview_available and not pair_preview_available:
        missing_fields.append("harmonic signal")
    harmonic_ready = bool(active_context.get("tool_ready")) and (
        harmonic_preview_available or pair_preview_available
    )

    return {
        "operation_id": operation.operation_id,
        "tool_id": operation.tool_id,
        "tool_label": active_context.get("tool_label") or operation.tool_id,
        "tool_number": active_context.get("tool_number"),
        "n_channels": len(operation.channel_files),
        "harmonic_ready": harmonic_ready,
        "missing_fields": missing_fields,
        "harmonic_column_count": len(harmonic_columns),
        "harmonic_preview_available": harmonic_preview_available,
        "pair_column_count": len(pair_columns),
        "pair_preview_available": pair_preview_available,
    }


def _resolve_casedata_demo_operation(
    loader: DatasetLoader,
    casedata_root: str,
    *,
    operation_id: Any,
    case_dir: str | None,
    valid_tools_only: bool,
) -> str:
    requested_operation = _text(operation_id)
    if requested_operation is not None:
        operation = loader.get_operation(requested_operation, case=case_dir)
        if not valid_tools_only:
            return operation.operation_id

        summary = _summarize_casedata_operation(casedata_root, operation)
        if summary["harmonic_ready"]:
            return operation.operation_id
        raise ValueError(
            f"Operation {operation.operation_id} is not harmonic-ready: missing {', '.join(summary['missing_fields'])}"
        )

    operations = loader.list_operations(case=case_dir)
    if not operations:
        if case_dir:
            raise ValueError(f"No casedata operations found under {casedata_root} for case {case_dir}")
        raise ValueError(f"No casedata operations found under {casedata_root}")

    if not valid_tools_only:
        return operations[0].operation_id

    for operation in operations:
        summary = _summarize_casedata_operation(casedata_root, operation)
        if summary["harmonic_ready"]:
            return operation.operation_id

    if case_dir:
        raise ValueError(f"No harmonic-ready casedata operations found for case {case_dir}")
    raise ValueError(f"No harmonic-ready casedata operations found under {casedata_root}")


def _prepare_casedata_session(
    session_id: str,
    sessions_dict: Dict[str, Dict[str, Any]],
    *,
    casedata_root: str,
    operation_id: str,
    case_dir: str | None,
):
    source = create_source(
        "simulated_casedata",
        sessions_dict,
        casedata_root=casedata_root,
        operation_id=operation_id,
        case_dir=case_dir,
    )
    data, metadata = source.session_data()
    time_axis_unix = source.session_time_axis_unix()
    return source, data, metadata, time_axis_unix


def _prepare_mqtt_session(
    session_id: str,
    sessions_dict: Dict[str, Dict[str, Any]],
    *,
    topic: str,
    broker_host: str,
    broker_port: int,
    sample_frequency: float,
    username: str | None,
    password: str | None,
    qos: int = 0,
):
    source = create_source(
        "mqtt",
        sessions_dict,
        topic=topic,
        broker_host=broker_host,
        broker_port=broker_port,
        sample_frequency=sample_frequency,
        username=username,
        password=password,
        qos=qos,
    )
    metadata: Dict[str, Any] = {
        "sample_frequency": float(sample_frequency),
        "source": "mqtt",
        "mqtt": {
            "topic": topic,
            "broker_host": broker_host,
            "broker_port": int(broker_port),
            "qos": int(qos),
        },
    }
    return source, metadata


def _configure_live_source_session(
    session: Dict[str, Any],
    *,
    fs: float,
    speed: float,
    samples_per_tick: int = 1,
    inference_window_s: float = 10.0,
    inference_stride_samples: int = 1,
    enable_fft: bool = False,
) -> None:
    session["config"]["speed"] = speed
    session["config"]["samples_per_tick"] = samples_per_tick

    session["running_fft"] = bool(enable_fft)
    session["fft_task"] = None
    session["fft_subscribers"] = []
    if enable_fft:
        target_fft_samples = max(32, int(fs * 4))
        nfft = 1 << max(5, int(math.log2(target_fft_samples)))
        if fs <= 10:
            nfft = max(256, nfft)
        session["fft_config"] = {
            "nfft": nfft,
            "overlap": 0.75,
            "window_type": "hann",
            "detrend": True,
            "output": "amplitude",
            "db": False,
            "bin_stride": 1,
            "max_freq_hz": None,
            "inherit_speed": True,
        }

    window_samples = max(1, int(round(fs * max(inference_window_s, 0.001))))
    session["running_inference"] = True
    session["inference_task"] = None
    session["inference_subscribers"] = []
    session["inference_config"] = {
        "window_samples": window_samples,
        "window_seconds": round(window_samples / max(fs, 0.001), 4),
        "sample_rate_hz": fs,
        "stride_samples": max(1, int(inference_stride_samples)),
        "inherit_speed": True,
    }


def _get_source_status(session_id: str, sessions_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    session = get_session_or_404(session_id, sessions_dict)
    source = session.get("_stream_source")
    if source is not None and hasattr(source, "status"):
        return source.status(session_id)

    source_name = str(session.get("source_name") or dict(session.get("metadata") or {}).get("source") or "simulated_file")
    status: Dict[str, Any] = {"kind": source_name}
    source_config = dict(session.get("source_config") or {})
    status.update({k: v for k, v in source_config.items() if k in {"topic", "broker_host", "broker_port", "username", "case_dir", "operation_id"}})
    if "password" in source_config:
        status["password_configured"] = bool(source_config.get("password"))
    return status


# ── Endpoints ────────────────────────────────────────────────────────────────


@supplemental_router.post("/sessions/start-demo")
@router.post("/sessions/start-demo")
async def start_demo_session(request: Request, body: Dict[str, Any] = Body(default={})):
    """Create a session, load a demo dataset, resolve a seek anchor, and start playback.

    Returns immediately with ``{session_id, ws_url, mode, n_events}``.

    Body parameters
    ---------------
    mode : str
        ``"default"`` | ``"labeled"`` | ``"casedata"`` | ``"site_c"`` | ``"site_a"`` | ``"site_b"`` | ``"site_a_line2"``
    speed : float
        Playback speed multiplier (default: ``0.02`` = ~50× slower than real-time)
    reset_priors : bool
        Reset pattern priors before playback starts (default: ``true``)
    start_paused : bool
        Start playback paused so user can click Resume (default: ``false``)
    """
    from .demo import _get_demo_config

    requested_source_name = _text(body.get("source"))
    requested_mode = str(body.get("mode") or "").strip().lower()
    session_file_override = _text(body.get("session_file") or body.get("session_path"))

    if requested_mode:
        mode = requested_mode
    elif requested_source_name == "simulated_casedata":
        mode = "casedata"
    elif requested_source_name == "mqtt":
        mode = "live"
    else:
        mode = "labeled"

    demo_config: Dict[str, Any] = {}
    if mode != "live":
        try:
            demo_config = _get_demo_config(mode, session_file_override=session_file_override)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    source_name = str(requested_source_name or demo_config.get("source") or "simulated_file").strip().lower()
    if source_name not in registered_sources():
        raise HTTPException(status_code=400, detail=f"Unknown source: {source_name}")

    # Default speed is per-mode: low-freq Site_a_line2 plays at sim-time; dense
    # high-freq casedata/labeled demos use a 0.02 default so the 50 s CNC
    # session doesn't flash past in one second. "casedata" overrides to 1.0
    # because the operator expects the chart to move at a visible rate.
    if "speed" in body:
        speed = float(body["speed"])
    elif source_name == "simulated_casedata":
        speed = 1.0
    elif mode == "site_a_line2":
        speed = 1.0
    else:
        speed = 0.02
    reset_priors = bool(body.get("reset_priors", True))
    start_paused = bool(body.get("start_paused", False))
    samples_per_tick = int(body.get("samples_per_tick", 0))  # 0 = auto-scale
    inference_window_s = body.get("inference_window_s", None)  # user-specified inference window
    # Site_a_line2 defaults to a 30 s inference window — long enough for the
    # seed model to see meaningful variance at 1 Hz, short enough that the
    # first inference frame appears within seconds of playback start.
    if inference_window_s is None and mode == "site_a_line2":
        inference_window_s = 30.0

    sessions = get_sessions_dict(request)

    session_path = demo_config.get("session_path")
    casedata_root = None
    case_dir_text = _text(body.get("case_dir") or body.get("casedata_case_dir"))
    if case_dir_text is None and (requested_mode or not requested_source_name):
        case_dir_text = _text(demo_config.get("case_dir"))
    operation_id = None
    valid_tools_only = bool(body.get("valid_tools_only", False))
    mqtt_config: Dict[str, Any] = {}
    demo_rationale = _text(body.get("rationale") or demo_config.get("rationale"))

    if reset_priors:
        try:
            from backend.agents.memory.orchestrator import get_orchestrator

            get_orchestrator().scorer.reset_priors()
            logger.info("Demo: priors reset")
        except Exception:
            logger.debug("Demo: prior reset failed (continuing)", exc_info=True)

    if source_name == "simulated_file":
        if not isinstance(session_path, Path):
            raise HTTPException(status_code=400, detail=f"Demo mode {mode!r} does not resolve to a session file")
        if not session_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Demo data file not found: {session_path}. Generate it first (see scripts/README).",
            )
    elif source_name == "simulated_casedata":
        requested_casedata_root = body.get("casedata_root")
        if requested_casedata_root is None and case_dir_text is None:
            requested_casedata_root = demo_config.get("casedata_root")
        casedata_root = _resolve_casedata_root(requested_casedata_root, case_dir_text)
        explicit_operation_id = body.get("operation_id") or body.get("casedata_operation_id")
        requested_operation_id = explicit_operation_id
        if requested_operation_id is None and (requested_mode or not requested_source_name):
            requested_operation_id = demo_config.get("operation_id")
        try:
            loader = DatasetLoader(casedata_root)
            operation_id = _resolve_casedata_demo_operation(
                loader,
                casedata_root,
                operation_id=requested_operation_id,
                case_dir=case_dir_text,
                valid_tools_only=valid_tools_only,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to prepare casedata source: {e}")
    elif source_name == "mqtt":
        try:
            ensure_mqtt_transport_available()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        topic = _text(body.get("topic"))
        if topic is None:
            raise HTTPException(status_code=400, detail="MQTT topic is required")
        mqtt_config = {
            "topic": topic,
            "broker_host": _text(body.get("broker_host")) or "localhost",
            "broker_port": max(1, int(body.get("broker_port") or 1883)),
            "sample_frequency": max(0.001, float(body.get("sample_frequency") or 1.0)),
            "username": _text(body.get("username")),
            "password": _text(body.get("password")),
            "qos": max(0, int(body.get("qos") or 0)),
        }

    pause_on_alert_raw = body.get("pause_on_alert")
    if isinstance(pause_on_alert_raw, str):
        pause_on_alert = pause_on_alert_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        pause_on_alert = bool(pause_on_alert_raw)

    harmonic_scorer_kind = str(body.get("harmonic_scorer_kind") or "context").strip().lower()
    if harmonic_scorer_kind not in {"context", "pair"}:
        raise HTTPException(status_code=400, detail="harmonic_scorer_kind must be 'context' or 'pair'")

    harmonic_dataset_raw = body.get("harmonic_dataset")
    harmonic_dataset: Optional[str] = None
    if harmonic_dataset_raw is not None:
        dataset_value = str(harmonic_dataset_raw).strip().lower()
        allowed_datasets = {
            "",
            "auto",
            "default",
            "none",
            "casedata",
            "stoppage_1hz",
            "site_a_line2",
            "raw_accelerometer",
            "pair_casedata",
            "pair_lfl",
            "pair_raw",
        }
        if dataset_value not in allowed_datasets:
            raise HTTPException(
                status_code=400,
                detail="harmonic_dataset must be one of auto, casedata, stoppage_1hz, site_a_line2, raw_accelerometer, pair_casedata, pair_lfl, or pair_raw",
            )
        if dataset_value not in {"", "auto", "default", "none"}:
            harmonic_dataset = dataset_value

    # 2. Create session
    session_id = str(int(time.time() * 1000))
    cfg = SessionConfig(
        interval_ms=100,
        channels=None,
        mode="time",
        speed=speed,
        samples_per_tick=samples_per_tick,
        start_paused=start_paused,
        pause_on_alert=pause_on_alert,
        harmonic_scorer_kind=harmonic_scorer_kind,
        harmonic_dataset=harmonic_dataset,
    )
    cfg_dict = cfg.model_dump()
    sessions[session_id] = {
        "session_id": session_id,
        "config": cfg_dict,
        "data": {},
        "metadata": _apply_harmonic_runtime_settings(cfg_dict, {}),
        "raw_file": None,
        "running": False,
        "paused": bool(start_paused),
        "subscribers": [],
        "task": None,
    }
    logger.info("Demo: created session %s (mode=%s)", session_id, mode)

    s = sessions[session_id]
    s["source_name"] = source_name

    if source_name == "simulated_casedata":
        requested_start_position = max(
            0,
            int(body.get("start_position", demo_config.get("requested_start_position") or 0) or 0),
        )
        start_at_first_cutting_row = bool(
            body.get("start_at_first_cutting_row", demo_config.get("start_at_first_cutting_row", False))
        )
        _base_meta: Dict[str, Any] = {
            "sample_frequency": 1.0,
            "source": source_name,
            "casedata": {
                "root": str(casedata_root),
                "case_dir": case_dir_text,
                "operation_id": str(operation_id),
            },
        }
        # Link the session to its SINDIT digital-twin asset so the memory bridge
        # enriches each scored event with twin context (machine state, etc.).
        # The asset IRI is deterministic from the case dir; enrichment degrades
        # gracefully (cached empty) if the asset or SINDIT is absent.
        _asset_iri = _casedata_sindit_asset_iri(case_dir_text)
        if _asset_iri:
            _base_meta["sindit_asset_iri"] = _asset_iri
        s["metadata"] = _apply_harmonic_runtime_settings(s.get("config", {}), _base_meta)
        s["position"] = requested_start_position
        s["loading"] = True
        s["source_config"] = {
            "source": source_name,
            "casedata_root": str(casedata_root),
            "case_dir": case_dir_text,
            "operation_id": str(operation_id),
            "valid_tools_only": valid_tools_only,
            "requested_start_position": requested_start_position,
            "start_position": requested_start_position,
            "start_at_first_cutting_row": start_at_first_cutting_row,
            "rationale": demo_rationale,
        }
        source = create_source(
            "simulated_casedata",
            sessions,
            casedata_root=str(casedata_root),
            operation_id=str(operation_id),
            case_dir=case_dir_text,
        )
        s["_stream_source"] = source

        def _complete_casedata_startup(
            data: Dict[str, Any],
            metadata: Dict[str, Any],
            time_axis_unix: List[float],
        ) -> None:
            session = sessions.get(session_id)
            if session is None:
                return

            resolved_start_position = _resolve_casedata_start_position(
                data,
                requested_start_position=requested_start_position,
                start_at_first_cutting_row=start_at_first_cutting_row,
            )
            clamped_position = resolved_start_position
            if data:
                total_samples = min(len(series) for series in data.values())
                session["total_samples"] = total_samples
                if total_samples > 0:
                    clamped_position = min(resolved_start_position, total_samples - 1)

            session["data"] = data
            session["time_axis_unix"] = list(time_axis_unix or [])
            session["metadata"] = _apply_harmonic_runtime_settings(session.get("config", {}), metadata)
            session["raw_file"] = None
            session["position"] = clamped_position
            session["source_config"] = {
                **dict(session.get("source_config") or {}),
                "start_position": clamped_position,
            }
            _configure_live_source_session(
                session,
                fs=get_sample_frequency(metadata),
                speed=speed,
                samples_per_tick=1,
                inference_window_s=10.0,
                inference_stride_samples=1,
                enable_fft=False,
            )
            session["loading"] = False
            session["pending_live_start"] = True
            session["startup_task"] = None

        def _fail_casedata_startup(error_message: str) -> None:
            session = sessions.get(session_id)
            if session is None:
                return
            session["loading"] = False
            session["pending_live_start"] = False
            session["running"] = False
            session["last_error"] = error_message
            session["startup_task"] = None

        def _finish_casedata_startup() -> None:
            try:
                data, metadata = source.session_data()
                time_axis_unix = source.session_time_axis_unix()
            except Exception as exc:
                logger.exception("Demo: casedata startup failed for %s", session_id)
                _fail_casedata_startup(f"Failed to start casedata stream: {exc}")
                return

            _complete_casedata_startup(data, metadata, time_axis_unix)

        startup_thread = threading.Thread(
            target=_finish_casedata_startup,
            name=f"casedata-startup-{session_id}",
            daemon=True,
        )
        s["startup_task"] = startup_thread
        startup_thread.start()
        return {
            "session_id": session_id,
            "ws_url": f"/streams/{session_id}",
            "mode": mode,
            "source": source_name,
            "n_events": 0,
            "seek": {
                "requested_start_position": requested_start_position,
                "start_at_first_cutting_row": start_at_first_cutting_row,
                "rationale": demo_rationale,
            },
            "status": "loading",
        }

    if source_name == "mqtt":
        try:
            source, metadata = _prepare_mqtt_session(
                session_id,
                sessions,
                **mqtt_config,
            )
        except Exception as e:
            del sessions[session_id]
            raise HTTPException(status_code=400, detail=f"Failed to prepare mqtt source: {e}")

        s["data"] = {}
        s["metadata"] = _apply_harmonic_runtime_settings(s.get("config", {}), metadata)
        s["raw_file"] = None
        s["position"] = 0
        s["source_config"] = {"source": source_name, **mqtt_config}
        _configure_live_source_session(
            s,
            fs=float(mqtt_config["sample_frequency"]),
            speed=speed,
            samples_per_tick=1,
            inference_window_s=10.0,
            inference_stride_samples=1,
            enable_fft=False,
        )
        s["running"] = True
        source.start(session_id, startup_delay=0.15)
        try:
            _start_inference_task(s)
        except Exception:
            logger.exception("Demo: inference task start failed")
        return {
            "session_id": session_id,
            "ws_url": f"/streams/{session_id}",
            "mode": mode,
            "source": source_name,
            "n_events": 0,
            "status": "started",
        }

    # 3. Load & preprocess the demo data file
    try:
        payload = json.loads(session_path.read_text(encoding="utf-8"))
        data, metadata = preprocess_payload(payload)
        sample_labels = _extract_sample_labels(payload, data)
    except Exception as e:
        del sessions[session_id]
        raise HTTPException(status_code=500, detail=f"Failed to load demo data: {e}")

    s["data"] = data
    s["metadata"] = _apply_harmonic_runtime_settings(s.get("config", {}), metadata)
    s["sample_labels"] = sample_labels
    s["raw_file"] = payload
    requested_start_position = max(
        0,
        int(body.get("start_position", demo_config.get("requested_start_position") or 0) or 0),
    )
    start_at_label = _text(body.get("start_at_label") or demo_config.get("start_at_label"))
    start_label_lead_in_samples = max(
        0,
        int(body.get("start_label_lead_in_samples", demo_config.get("start_label_lead_in_samples") or 0) or 0),
    )
    start_position, resolved_start_label_index = _resolve_sample_label_start_position(
        sample_labels,
        requested_start_position=requested_start_position,
        start_at_label=start_at_label,
        start_label_lead_in_samples=start_label_lead_in_samples,
    )
    resolved_start_label = None
    if data:
        total_samples = min(len(series) for series in data.values())
        if total_samples > 0:
            start_position = min(start_position, total_samples - 1)
        s["total_samples"] = total_samples
    if resolved_start_label_index is not None and 0 <= resolved_start_label_index < len(sample_labels):
        resolved_start_label = sample_labels[resolved_start_label_index]
    s["position"] = start_position
    s["source_config"] = {
        "source": source_name,
        "requested_start_position": requested_start_position,
        "start_position": start_position,
        "start_at_label": start_at_label,
        "start_label_lead_in_samples": start_label_lead_in_samples,
        "resolved_start_label_index": resolved_start_label_index,
        "resolved_start_label": resolved_start_label,
        "rationale": demo_rationale,
    }

    _fs = get_sample_frequency(metadata)

    # Agent N (2026-04-24): persist raw signal so memories can be rebound
    # after a restart. Best-effort — failure here must not break upload.
    try:
        from backend.agents.storage.session_signals import save_session_signal
        save_session_signal(session_id, data, metadata, fs=_fs)
    except Exception as _persist_err:  # pragma: no cover - defensive
        logger.warning("persist demo signal failed for %s: %s", session_id, _persist_err)

    # Auto-scale samples_per_tick to produce ~8-12 frames/second.
    # For fs=100 Hz @ speed=2 → spt=25 → tick every 0.125 s (~8 fps).
    # For fs=1 Hz @ speed=8 → spt=1  → tick every 0.125 s (~8 fps).
    if samples_per_tick <= 0:
        _target_fps = 8.0
        samples_per_tick = max(1, int(_fs * speed / _target_fps))
        logger.info(
            "Auto samples_per_tick=%d  (fs=%.1f, speed=%.2f, target_fps=%.0f)",
            samples_per_tick, _fs, speed, _target_fps,
        )
    s["config"]["samples_per_tick"] = samples_per_tick

    _target_fft_samples = max(32, int(_fs * 4))
    _nfft = 1 << max(5, int(math.log2(_target_fft_samples)))
    if inference_window_s is not None:
        # User-specified inference window overrides defaults and floors
        _inf_s = max(5.0, float(inference_window_s))
        _inf_window = max(8, int(_fs * _inf_s))
        _inf_stride = max(4, _inf_window // 2)
        logger.info(
            "User-specified inference_window_s=%.1f → window=%d stride=%d (fs=%.1f)",
            _inf_s, _inf_window, _inf_stride, _fs,
        )
    else:
        _default_inf_s = float(os.environ.get("DEFAULT_INFERENCE_WINDOW_S", "2.0"))
        _inf_window = max(32, int(_fs * _default_inf_s))
        _inf_stride = max(16, _inf_window // 2)

    # Floor for low-frequency data (e.g., Site_a_line2 1 Hz):
    # FFT needs at least 256 samples to be meaningful,
    # inference window needs at least 120 samples (~2 min at 1 Hz)
    # Speed floor — ensure the stream produces at least ~4 fps regardless
    # of sample rate so the demo feels responsive.
    _min_fps = 4.0
    _min_speed = max(0.5, samples_per_tick * _min_fps / max(_fs, 0.01))
    if speed < _min_speed:
        logger.info(
            "Speed floor: overriding speed %.4f → %.2f  "
            "(fs=%.1f, spt=%d, min_fps=%.0f)",
            speed, _min_speed, _fs, samples_per_tick, _min_fps,
        )
        speed = _min_speed
        s["config"]["speed"] = speed

    if _fs <= 10:
        _nfft = max(256, _nfft)
        if inference_window_s is None:
            _inf_window = max(120, _inf_window)
            _inf_stride = max(60, _inf_stride)

        logger.info(
            "Low-freq data (fs=%.1f Hz): floors applied → nfft=%d, inf_window=%d, speed=%.2f",
            _fs, _nfft, _inf_window, speed,
        )

    s.update({
        "running_fft": True,
        "fft_task": None,
        "fft_subscribers": [],
        "fft_config": {
            "nfft": _nfft,
            "overlap": 0.75,
            "window_type": "hann",
            "detrend": True,
            "output": "amplitude",
            "db": False,
            "bin_stride": 1,
            "max_freq_hz": None,
            "inherit_speed": True,
        },
        "running_inference": True,
        "inference_task": None,
        "inference_subscribers": [],
        "inference_config": {
            "window_samples": _inf_window,
            "window_seconds": round(_inf_window / _fs, 4),
            "sample_rate_hz": _fs,
            "stride_samples": _inf_stride,
            "inherit_speed": True,
        },
    })

    # 4. Start playback + FFT + inference
    #    Wrap in a short startup delay so the HTTP response reaches the UI
    #    and WebSocket subscribers connect before the first frame is emitted.
    s["running"] = True

    async def _delayed_playback():
        await asyncio.sleep(0.15)          # just enough for HTTP response + WS handshake
        await playback_task(session_id, sessions)

    s["task"] = asyncio.create_task(_delayed_playback())
    try:
        _start_fft_task(s)
    except Exception:
        logger.exception("Demo: FFT task start failed")
    try:
        _start_inference_task(s)
    except Exception:
        logger.exception("Demo: inference task start failed")

    logger.info("Demo: playback started for %s (%d channels, fs=%.0f Hz)",
                session_id, len(data), _fs)

    return {
        "session_id": session_id,
        "ws_url": f"/streams/{session_id}",
        "mode": mode,
        "source": source_name,
        "n_events": 0,
        "seek": {
            "requested_start_position": requested_start_position,
            "resolved_start_position": start_position,
            "start_at_label": start_at_label,
            "resolved_start_label": resolved_start_label,
            "resolved_start_label_index": resolved_start_label_index,
            "rationale": demo_rationale,
        },
        "status": "started",
    }


@router.get("/sessions/casedata/catalog")
async def casedata_catalog() -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    roots: List[str] = []

    for root_path in _default_casedata_roots():
        try:
            loader = DatasetLoader(root_path)
        except FileNotFoundError:
            continue
        roots.append(str(root_path))

        for case_dir in loader.list_cases():
            operations = loader.list_operations(case=case_dir)
            operation_entries = [
                _summarize_casedata_operation(str(root_path), operation)
                for operation in operations
            ]
            cases.append(
                {
                    "case_dir": case_dir,
                    "label": case_dir,
                    "casedata_root": str(root_path),
                    "default_operation_id": operations[0].operation_id if operations else None,
                    "default_valid_operation_id": next(
                        (entry["operation_id"] for entry in operation_entries if entry["harmonic_ready"]),
                        None,
                    ),
                    "operations": operation_entries,
                }
            )

    cases.sort(key=lambda item: item["label"])
    return {"root": roots[0] if roots else _resolve_casedata_root(), "roots": roots, "cases": cases}


@router.post("/sessions")
def create_session(cfg: SessionConfig, request: Request):
    sessions = get_sessions_dict(request)
    session_id = str(int(time.time() * 1000))
    cfg_dict = cfg.model_dump()
    sessions[session_id] = {
        "session_id": session_id,
        "config": cfg_dict,
        "data": {},
        "metadata": _apply_harmonic_runtime_settings(cfg_dict, {}),
        "raw_file": None,
        "running": False,
        "paused": bool(cfg.start_paused),
        "subscribers": [],
        "task": None,
    }
    return {"session_id": session_id, "ws": f"/streams/{session_id}"}


@router.post("/sessions/{session_id}/upload")
async def upload(session_id: str, request: Request, file: UploadFile = File(...)):
    sessions = get_sessions_dict(request)
    s = get_session_or_404(session_id, sessions)

    # Enforce upload size limit
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload too large ({len(raw)} bytes). Max is {MAX_UPLOAD_BYTES} bytes.",
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    try:
        data, metadata = preprocess_payload(payload)
        sample_labels = _extract_sample_labels(payload, data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

    s["data"] = data
    s["metadata"] = _apply_harmonic_runtime_settings(s.get("config", {}), metadata)
    s["sample_labels"] = sample_labels
    s["raw_file"] = payload

    _fs = get_sample_frequency(metadata)

    # Agent N (2026-04-24): persist raw signal so memories can be rebound
    # after a restart. Best-effort — failure here must not break upload.
    try:
        from backend.agents.storage.session_signals import save_session_signal
        save_session_signal(session_id, data, metadata, fs=_fs)
    except Exception as _persist_err:  # pragma: no cover - defensive
        logger.warning("persist uploaded signal failed for %s: %s", session_id, _persist_err)

    # Adaptive FFT window (~4s, power-of-two)
    _target_fft_samples = max(32, int(_fs * 4))
    _nfft = 1 << max(5, int(math.log2(_target_fft_samples)))

    # Adaptive inference window (~2s)
    _default_inf_s = float(os.environ.get("DEFAULT_INFERENCE_WINDOW_S", "2.0"))
    _inf_window = max(32, int(_fs * _default_inf_s))
    _inf_stride = max(16, _inf_window // 2)

    logger.info(
        "Adaptive config: fs=%.1f Hz → nfft=%d, inf_window=%d, inf_stride=%d",
        _fs, _nfft, _inf_window, _inf_stride,
    )

    # Floor for low-frequency data (e.g., Site_a_line2 1 Hz)
    if _fs <= 10:
        _nfft = max(256, _nfft)
        _inf_window = max(120, _inf_window)
        _inf_stride = max(60, _inf_stride)
        logger.info(
            "Low-freq data (fs=%.1f Hz): floors applied → nfft=%d, inf_window=%d",
            _fs, _nfft, _inf_window,
        )

    s.update({
        "running_fft": True,
        "fft_task": None,
        "fft_subscribers": [],
        "fft_config": {
            "nfft": _nfft,
            "overlap": 0.75,
            "window_type": "hann",
            "detrend": True,
            "output": "amplitude",
            "db": False,
            "bin_stride": 1,
            "max_freq_hz": None,
            "inherit_speed": True,
        },
        "running_inference": True,
        "inference_task": None,
        "inference_subscribers": [],
        "inference_config": {
            "window_samples": _inf_window,
            "window_seconds": round(_inf_window / _fs, 4),
            "sample_rate_hz": _fs,
            "stride_samples": _inf_stride,
            "inherit_speed": True,
        },
    })
    return {"ok": True, "channels": list(data.keys()), "metadata": metadata}


@router.post("/sessions/{session_id}/start")
async def start(session_id: str, request: Request):
    sessions = get_sessions_dict(request)
    s = get_session_or_404(session_id, sessions)
    if not s.get("data"):
        raise HTTPException(status_code=400, detail="No data uploaded for this session")
    s["running"] = True
    s["task"] = asyncio.create_task(playback_task(session_id, sessions))
    logger.info("Started session %s playback task", session_id)

    try:
        _start_fft_task(s)
    except Exception as e:
        logger.exception("Error while starting FFT task: %s", e)

    try:
        _start_inference_task(s)
    except Exception as e:
        logger.exception("Error while starting inference task: %s", e)

    return {"ok": True}


@router.get("/sessions")
def list_sessions(request: Request):
    """Return all active session IDs."""
    sessions = get_sessions_dict(request)
    session_ids = list(sessions.keys())
    for session_id in session_ids:
        _activate_pending_live_source(session_id, sessions[session_id], request)
    return {
        "sessions": session_ids,
        "session_summaries": [
            _build_session_summary(session_id, sessions[session_id])
            for session_id in session_ids
        ],
    }


@router.get("/sessions/{session_id}")
def get_session_info(session_id: str, request: Request):
    """Return metadata, config, channels, status, and raw file."""
    sessions = get_sessions_dict(request)
    s = get_session_or_404(session_id, sessions)
    _activate_pending_live_source(session_id, s, request)
    summary = _build_session_summary(session_id, s)
    return {
        **summary,
        "config": s["config"],
        "channels": list(s["data"].keys()) if s["data"] else [],
        "metadata": s.get("metadata", {}),
        "raw_file": s.get("raw_file"),
        "active_context": build_active_session_context(s),
        "source_status": _get_source_status(session_id, sessions),
    }


@router.get("/sessions/{session_id}/source")
def get_session_source(session_id: str, request: Request):
    sessions = get_sessions_dict(request)
    session = get_session_or_404(session_id, sessions)
    _activate_pending_live_source(session_id, session, request)
    # Prefer a status snapshot already on the session; else derive from source config.
    status = session.get("source_status")
    if not isinstance(status, dict) or not status:
        status = _get_source_status(session_id, sessions)
    return {
        "session_id": session_id,
        "running": session.get("running"),
        "paused": session.get("paused"),
        "position": session.get("position"),
        **status,
    }


@router.post("/sessions/{session_id}/pause")
def pause(session_id: str, request: Request):
    sessions = get_sessions_dict(request)
    s = get_session_or_404(session_id, sessions)
    s["paused"] = True
    return {"ok": True, "paused": True}


@router.post("/sessions/{session_id}/resume")
def resume(session_id: str, request: Request):
    sessions = get_sessions_dict(request)
    s = get_session_or_404(session_id, sessions)
    s["paused"] = False
    return {"ok": True, "paused": False}


# PATCH /sessions/{id}/config is the canonical name; POST /sessions/{id}/playback
# kept for backward compatibility.
@supplemental_router.patch("/sessions/{session_id}/config")
@supplemental_router.post("/sessions/{session_id}/playback")
@router.patch("/sessions/{session_id}/config")
@router.post("/sessions/{session_id}/playback")
def update_config(session_id: str, req: PlaybackConfigUpdate, request: Request):
    """Live-update playback parameters for an existing session."""
    sessions = get_sessions_dict(request)
    s = get_session_or_404(session_id, sessions)
    cfg = s.get("config", {})

    if req.speed is not None:
        spd = float(req.speed)
        if spd <= 0:
            raise HTTPException(status_code=400, detail="speed must be > 0")
        cfg["speed"] = spd

    if req.samples_per_tick is not None:
        spt = int(req.samples_per_tick)
        if spt < 1:
            raise HTTPException(status_code=400, detail="samples_per_tick must be >= 1")
        cfg["samples_per_tick"] = spt

    if req.pause_on_alert is not None:
        cfg["pause_on_alert"] = bool(req.pause_on_alert)

    if req.harmonic_scorer_kind is not None:
        kind = str(req.harmonic_scorer_kind).strip().lower()
        if kind not in {"context", "pair"}:
            raise HTTPException(status_code=400, detail="harmonic_scorer_kind must be 'context' or 'pair'")
        cfg["harmonic_scorer_kind"] = kind

    if req.harmonic_dataset is not None:
        dataset = str(req.harmonic_dataset).strip().lower()
        allowed_datasets = {
            "",
            "auto",
            "default",
            "none",
            "casedata",
            "stoppage_1hz",
            "site_a_line2",
            "raw_accelerometer",
            "pair_casedata",
            "pair_lfl",
            "pair_raw",
        }
        if dataset not in allowed_datasets:
            raise HTTPException(
                status_code=400,
                detail="harmonic_dataset must be one of auto, casedata, stoppage_1hz, site_a_line2, raw_accelerometer, pair_casedata, pair_lfl, or pair_raw",
            )
        if dataset in {"", "auto", "default", "none"}:
            cfg.pop("harmonic_dataset", None)
        else:
            cfg["harmonic_dataset"] = dataset

    metadata = s.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        s["metadata"] = metadata
    _apply_harmonic_runtime_settings(cfg, metadata)

    s["config"] = cfg
    return {"ok": True, "config": cfg}


@router.post("/sessions/{session_id}/replay")
async def replay_session(session_id: str, req: ReplayRequest, request: Request):
    """Restart a session run from the beginning."""
    sessions = get_sessions_dict(request)
    session = get_session_or_404(session_id, sessions)

    session["position"] = 0
    session["metadata"]["playback_speed"] = req.speed
    session["config"]["speed"] = req.speed

    if session.get("task"):
        session["task"].cancel()
    session["running"] = True
    loop = asyncio.get_running_loop()
    session["task"] = loop.create_task(playback_task(session_id, sessions))

    # Restart FFT task
    try:
        old_fft = session.get("fft_task")
        if old_fft is not None:
            try:
                old_fft.cancel()
                try:
                    await old_fft
                except (asyncio.CancelledError, Exception):
                    pass
            except Exception:
                pass
            session["fft_task"] = None
        session["running_fft"] = True
        session["fft_task"] = loop.create_task(fft_stream_task(session))
    except Exception:
        logger.exception("Replay: could not start FFT task")

    return {"status": "restarted", "session_id": session_id, "speed": req.speed}


@supplemental_router.delete("/sessions/{session_id}")
@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Stop all tasks and remove a session."""
    sessions = get_sessions_dict(request)
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    s = sessions[session_id]

    # Stop flags
    s["running"] = False
    s["running_fft"] = False
    s["running_inference"] = False

    source = s.get("_stream_source")
    if source is not None and hasattr(source, "stop"):
        try:
            await source.stop(session_id)
        except Exception:
            logger.exception("Could not stop source for session %s", session_id)

    # Cancel tasks
    for key in ("startup_task", "task", "fft_task", "inference_task"):
        t = s.get(key)
        if isinstance(t, asyncio.Task) and not t.done():
            t.cancel()

    passive_feedback_count = 0
    flushed_cycle = get_cycle_tracker().flush_session(session_id)
    if flushed_cycle is not None and is_memory_initialized():
        try:
            passive_feedback_count = int(
                await get_orchestrator().attach_passive_cycle_outcome(flushed_cycle)
            )
        except Exception:
            logger.exception("Could not attach passive cycle outcome during session shutdown for %s", session_id)

    # Drain queues
    for q_list_key in ("subscribers", "fft_subscribers", "inference_subscribers"):
        for q in list(s.get(q_list_key, [])):
            try:
                q.put_nowait({"eos": True})
            except Exception:
                pass

    del sessions[session_id]
    return {
        "ok": True,
        "deleted": session_id,
        "flushed_cycle": flushed_cycle is not None,
        "passive_feedback_count": passive_feedback_count,
    }


@router.get("/sessions/{session_id}/metadata")
def get_session_metadata(session_id: str, request: Request):
    """Return metadata and number of timesteps played for a given session."""
    sessions = get_sessions_dict(request)
    s = get_session_or_404(session_id, sessions)
    return {
        "metadata": s.get("metadata", {}),
        "timesteps_played": s.get("position", 0),
    }


@router.get("/sessions/{session_id}/download")
def download_played(session_id: str, request: Request, format: str = "json"):
    """Download the portion of the session data that has been played so far."""
    sessions = get_sessions_dict(request)
    s = get_session_or_404(session_id, sessions)
    pos = s.get("position", 0)
    data = s.get("data", {})
    channels = s["config"].get("channels") or list(data.keys())

    played = {ch: data[ch][:pos] for ch in channels if ch in data}

    if format == "json":
        return JSONResponse(
            content={
                "session_id": session_id,
                "position": pos,
                "played": played,
                "metadata": s.get("metadata", {}),
            },
            headers={
                "Content-Disposition": f'attachment; filename="session_{session_id}_played.json"'
            },
        )
    elif format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["t"] + channels)
        for i in range(pos):
            row = [i] + [played[ch][i] if i < len(played[ch]) else "" for ch in channels]
            writer.writerow(row)
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="session_{session_id}_played.csv"'
            },
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")

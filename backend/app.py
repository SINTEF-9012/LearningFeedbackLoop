import asyncio
import json
import importlib
import logging
import os as _os
import time
from contextlib import asynccontextmanager
from dataclasses import replace as _dc_replace
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Path,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute, APIWebSocketRoute
from pydantic import BaseModel

from .computation import compute_fg_fp_for_window_session_multi_ref
from .events import publish_feature
from .fft_streamer import fft_stream_task

from .agents.memory.feature_stream_bridge import start_memory_processor, stop_memory_processor
from .agents.memory.init import initialize_memory_system, shutdown_memory_system

logger = logging.getLogger(__name__)


def _memory_startup_config():
    """Return a memory config suitable for non-blocking app startup."""
    from .agents.config import get_config as _get_memory_config

    config = _get_memory_config()
    if (
        "LAZY_SEED_TRAINING" not in _os.environ
        and bool(getattr(config, "use_classical_models", False))
        and not bool(getattr(config, "lazy_seed_training", False))
    ):
        config = _dc_replace(config, lazy_seed_training=True)
        logger.info(
            "App startup enabling lazy_seed_training for the memory system; set LAZY_SEED_TRAINING explicitly to override this behavior"
        )
    return config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: initialize and run the memory learning loop."""
    app.state.main_loop = asyncio.get_running_loop()
    try:
        initialize_memory_system(config=_memory_startup_config())
        await start_memory_processor()
    except Exception:
        # Best-effort: keep the API up even if memory is unavailable.
        logger.exception("Memory system startup failed; memory-first learning disabled")

    # Phase 0 live twin: push live machine state / cutting params onto the SINDIT
    # asset graph (throttled, gated by SINDIT_LIVE_BRIDGE; no-op when disabled).
    try:
        from .agents.sindit.live_bridge import start_live_bridge

        await start_live_bridge()
    except Exception:
        logger.exception("SINDIT live bridge startup failed; live twin disabled")

    # Agent M (2026-04-24): wire async feedback pipeline — outbox, operator
    # history, and WS broadcast — onto the existing MemoryFeedbackHandler via
    # its register_callback hook. Best-effort; HTTP surface still serves if
    # orchestrator isn't available yet.
    try:
        from .agents.memory.feedback_async import build_default_pipeline
        from .agents.memory.orchestrator import get_orchestrator

        pipeline = build_default_pipeline()
        app.state.feedback_pipeline = pipeline
        try:
            orch = get_orchestrator()
            handler = getattr(orch, "feedback_handler", None)
            if handler is not None and hasattr(handler, "register_callback"):
                handler.register_callback(pipeline.callback)
        except Exception:
            logger.exception(
                "Feedback pipeline registered but orchestrator callback wiring failed"
            )
    except Exception:
        logger.exception("Feedback pipeline initialisation failed")

    yield

    try:
        from .agents.sindit.live_bridge import stop_live_bridge

        await stop_live_bridge()
    except Exception:
        logger.exception("Failed to stop SINDIT live bridge")

    try:
        await stop_memory_processor()
    except Exception:
        logger.exception("Failed to stop memory processor")

    try:
        shutdown_memory_system()
    except Exception:
        logger.exception("Memory system shutdown failed")
    finally:
        app.state.main_loop = None


app = FastAPI(lifespan=lifespan)

# CORS — permit the Vite dev server (5173) and Playwright preview (4173) to
# call the API directly. Override via CORS_ORIGINS env var (comma-separated)
# or set to "*" to allow any origin (development only).
_cors_env = _os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
)
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session state - stored on app.state for proper access from other modules
# Access via get_sessions() dependency or app.state.sessions
app.state.sessions = {}

# Agent L (2026-04-24): SessionManager wraps app.state.sessions so new routes
# can use Depends(get_session_manager) instead of reaching into app.state.
# The dict itself remains the canonical store — existing readers unchanged.
from .session_manager import SessionManager as _SessionManager  # noqa: E402
app.state.session_manager = _SessionManager(app.state.sessions)


def _pause_session_if_configured(session_id: str, alert: Optional[Dict[str, Any]] = None) -> bool:
    session = app.state.sessions.get(session_id)
    if not isinstance(session, dict):
        return False

    cfg = session.get("config") if isinstance(session.get("config"), dict) else {}
    if not bool(cfg.get("pause_on_alert", False)):
        return False

    metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
    source_name = str(
        metadata.get("source")
        or session.get("source_name")
        or ""
    ).strip().lower()
    if source_name == "mqtt":
        return False

    session["paused"] = True
    session["paused_reason"] = "alert"
    if isinstance(alert, dict):
        event_id = alert.get("event_id")
        if isinstance(event_id, str) and event_id:
            session["paused_alert_id"] = event_id
    logging.getLogger(__name__).info("Paused session %s from backend alert dispatch", session_id)
    return True


try:
    from .agents.memory.dispatcher import get_dispatcher as _get_alert_dispatcher  # noqa: E402

    _get_alert_dispatcher().set_session_pause_handler(_pause_session_if_configured)
except Exception as _dispatcher_err:  # pragma: no cover - startup safeguard
    logging.getLogger(__name__).warning(
        "Alert pause handler wiring skipped: %s",
        _dispatcher_err,
    )

def get_sessions() -> Dict[str, Dict[str, Any]]:
    """Dependency to access the sessions dictionary."""
    return app.state.sessions


def get_session_or_404(session_id: str, *, detail: str = "Session not found") -> Dict[str, Any]:
    """Fetch a session dict or raise a 404."""
    sessions_dict = app.state.sessions
    if session_id not in sessions_dict:
        raise HTTPException(status_code=404, detail=detail)
    return sessions_dict[session_id]

# Backwards compatibility alias (will be removed)
sessions = app.state.sessions


@app.get("/health")
async def health() -> Dict[str, str]:
    """Lightweight liveness probe used by experiment API mode."""
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready() -> Dict[str, Any]:
    """Warm-up / readiness probe (distinct from liveness).

    `/health` returns 200 as soon as the server binds, but the classical seed
    model can still be training in the background (LAZY_SEED_TRAINING), during
    which live scoring is slower. This reports whether that background warm-up
    has finished so the UI can show a subtle "backend ready" vs "warming up"
    indicator. Never raises — degrades to ready=False with a reason.
    """
    detail: Dict[str, Any] = {"live": True}
    ready = True
    try:
        from backend.agents.memory.orchestrator import get_orchestrator
        orch = get_orchestrator()
        detail["store"] = type(orch.store).__name__ if getattr(orch, "store", None) else None
        ad = getattr(orch, "anomaly_detector", None)
        if ad is None:
            # Classical models disabled → nothing to warm up.
            detail["classical_models"] = "disabled"
        else:
            seed = getattr(ad, "seed_model", None)
            trained = bool(getattr(seed, "is_trained", False)) if seed is not None else False
            detail["classical_models"] = "ready" if trained else "warming"
            ready = ready and trained
    except Exception as exc:  # pragma: no cover - defensive
        ready = False
        detail["error"] = str(exc)
    return {"ready": ready, **detail}


default_samples_per_tick = 32
default_speed = 1.0  # real-time

class SessionConfig(BaseModel):
    interval_ms: int
    channels: Optional[List[str]] = None
    mode: str = "time"  # or "frequency"

# -----------------------------
# Preprocessing logic
# -----------------------------
def preprocess_payload(payload: dict, config: Optional[dict] = None):
    """
    Normalize uploaded JSON into a consistent structure:
      - data: dict of channel_name -> list of samples
      - metadata: dict of acquisition/tool info
    """
    data = {}
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
        # Extract sample frequency if present
        if "File_Header" in payload:
            metadata["sample_frequency"] = payload["File_Header"].get("SampleFrequency")
            metadata["file_header"] = payload["File_Header"]
        # Extract machining parameters if present
        machining_keys = ["d","z","ap","ae","vc","n","f","vf","type","break","fg","fp"]
        metadata["machining"] = {k: payload[k] for k in machining_keys if k in payload}

    # Case 3: Use config if provided
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

    # Add playback_speed to metadata if present in payload
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

async def playback_task(session_id: str):
    logger.debug("Starting playback task")
    s = sessions[session_id]
    cfg = s["config"]
    data = s["data"]
    metadata = s["metadata"]

    # --- Required: sampling frequency from metadata ---
    fs = float(metadata["file_header"]["SampleFrequency"])  # e.g., 4096.0
    Ts = 1.0 / fs
    # --- Config knobs (user-provided at upload or defaults) ---
    channels = cfg.get("channels") or list(data.keys())
    speed = float(cfg.get("speed", default_speed))           # 1.0 = real-time; <1 slower; >1 faster
    samples_per_tick = int(cfg.get("samples_per_tick", default_samples_per_tick))  # 1 = per-sample; >1 = chunk mode
    samples_per_tick = max(1, samples_per_tick)
    # --- Resume position & limits ---
    pos = int(s.get("position", 0))
    if not channels:
        raise ValueError("No channels available to process")
    n_max = min(len(data[ch]) for ch in channels)  # ensure all channels valid length
    # Absolute scheduling to avoid drift
    start_wall = time.perf_counter()
    start_pos = pos
    next_emit_time = start_wall  # emit immediately first time
    try:
        logger.debug("playback_task starting loop; running=%s", s.get("running", False))
        while s.get("running", False) and pos < n_max:
            # Pause handling
            if s.get("paused", False):
                await asyncio.sleep(0.1)
                # Realign schedule after pause so we don’t “catch up” aggressively
                start_wall = time.perf_counter()
                start_pos = pos
                next_emit_time = start_wall
                continue
            # Determine slice for this tick
            i0 = pos
            i1 = min(pos + samples_per_tick, n_max)   # exclusive
            if i0 >= i1:
                break

            # Build frame
            if samples_per_tick == 1:
                # Per-sample frame: one scalar per channel
                t = i0 / fs
                frame: Dict[str, Any] = {"t": t, "i": i0, "fs": fs}
                for ch in channels:
                    arr = data[ch]
                    frame[ch] = arr[i0] if i0 < len(arr) else None
            else:
                # Chunk frame: arrays per channel (prefer numpy arrays for zero-copy slicing)
                t0 = i0 / fs
                t1 = (i1 - 1) / fs
                frame = {"t0": t0, "t1": t1, "i0": i0, "i1": i1, "fs": fs}
                for ch in channels:
                    arr = data[ch]
                    frame[ch] = arr[i0:i1]  # np.ndarray slice is a view; list slice copies (still ok)

            # Advance position and persist
            pos = i1
            s["position"] = pos

            # Broadcast to subscribers
            for q in list(s.get("subscribers", [])):
                # Consider wrapping in try/except if queues can be closed mid-send
                await q.put(frame)

            # Publish feature event for online agents
            try:
                # publish a lightweight payload; include session id
                payload = {"type": "time", "session_id": session_id, "position": pos, "frame": frame}
                await publish_feature(session_id, payload)
            except Exception:
                # keep streaming even if event publish fails
                logger.warning("Feature publish failed (time); continuing stream", exc_info=True)

            # Schedule next tick based on absolute time and speed
            samples_emitted = pos - start_pos
            target_elapsed = (samples_emitted * Ts) / max(speed, 1e-9)
            next_emit_time = start_wall + target_elapsed
            # Sleep until target time; if behind, just yield
            now = time.perf_counter()
            delay = next_emit_time - now
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)

        # End-of-data
        s["running"] = False

        # (Optional) Send an EOS sentinel to subscribers
        eos = {"eos": True, "fs": fs, "final_i": pos}
        for q in list(s.get("subscribers", [])):
            try:
                await q.put(eos)
            except Exception:
                pass

    finally:
        s["task"] = None


# -----------------------------
# API endpoints
# -----------------------------
@app.post("/sessions")
def create_session(cfg: SessionConfig):
    session_id = str(int(time.time() * 1000))
    sessions[session_id] = {
        "session_id": session_id,
        "config": cfg.dict(),
        "data": {},
        "metadata": {},
        "raw_file": None,
        "running": False,
        "subscribers": [],
        "task": None,
    }
    return {"session_id": session_id, "ws": f"/streams/{session_id}"}

@app.post("/sessions/{session_id}/upload")
async def upload(session_id: str, file: UploadFile = File(...)):
    s = get_session_or_404(session_id)
    payload = json.loads(await file.read())
    try:
        data, metadata = preprocess_payload(payload)
        sample_labels = _extract_sample_labels(payload, data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")
    s["data"] = data
    s["metadata"] = metadata
    s["sample_labels"] = sample_labels
    s["raw_file"] = payload

    s.update({
    # FFT streaming state
    "running_fft": True,
    "fft_task": None,
    "fft_subscribers": [],  # list[asyncio.Queue]
    # FFT configuration (defaults; can be set by user)
    "fft_config": {
        "nfft": 4096,               # FFT window size (power of two recommended)
        "overlap": 0.75,            # e.g., 75% → hop = nfft * (1 - overlap)
        "window_type": "hann",      # "hann" or "rect"
        "detrend": True,
        "output": "amplitude",      # "amplitude" | "power" | "psd"
        "db": False,                # True to send 20*log10 or 10*log10 depending on output
        "bin_stride": 1,            # decimate bins: take every k-th bin to reduce payload
        "max_freq_hz": None,        # band-limit, e.g., 1000.0 (None = Nyquist)
        "inherit_speed": True       # follow s['config']['speed'] of time playback
    }
})
    return {
        "ok": True,
        "channels": list(data.keys()),
        "metadata": metadata
    }

@app.post("/sessions/{session_id}/start")
async def start(session_id: str):
    s = get_session_or_404(session_id)
    s["running"] = True
    s["task"] = asyncio.create_task(playback_task(session_id))
    logger.info("Started session %s playback task", session_id)

    # in /sessions/{sid}/start playback handler:
    try:
        fft_task = s.get("fft_task")
        if fft_task is None or (hasattr(fft_task, "done") and fft_task.done()):
            logger.info("Starting FFT stream task")
            s["running_fft"] = True
            s["fft_task"] = asyncio.create_task(fft_stream_task(s))
        else:
            logger.debug("FFT task already exists and appears running")
    except Exception as e:
        logger.exception("Error while starting FFT task: %s", e)
    return {"ok": True}

@app.websocket("/streams/{session_id}")
async def ws_stream(websocket: WebSocket, session_id: str):
    if session_id not in sessions:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    s = sessions[session_id]
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    s["subscribers"].append(queue)
    try:
        while True:
            frame = await queue.get()
            try:
                await websocket.send_text(json.dumps(frame))
            except Exception:
                # client disconnected, break out
                break
    finally:
        s["subscribers"].remove(queue)


@app.get("/sessions")
def list_sessions():
    """Return all active session IDs."""
    return {"sessions": list(sessions.keys())}

@app.get("/sessions/{session_id}")
def get_session_info(session_id: str):
    """Return metadata, config, channels, status, and raw file."""
    s = get_session_or_404(session_id)
    return {
        "session_id": session_id,
        "config": s["config"],
        "channels": list(s["data"].keys()) if s["data"] else [],
        "metadata": s.get("metadata", {}),
        "running": s["running"],
        "raw_file": s.get("raw_file"),
    }


@app.post("/sessions/{session_id}/pause")
def pause(session_id: str):
    s = get_session_or_404(session_id)
    s["paused"] = True
    return {"ok": True, "paused": True}

@app.post("/sessions/{session_id}/resume")
def resume(session_id: str):
    s = get_session_or_404(session_id)
    s["paused"] = False
    return {"ok": True, "paused": False}


import numpy as np

@app.post("/sessions/{session_id}/analyze")
def analyze(session_id: str, start: int, end: int, channel: str):
    s = get_session_or_404(session_id)
    data = s["data"].get(channel)
    if data is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    segment = np.array(data[start:end])
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / s["metadata"].get("sample_frequency", 1.0))
    spectrum = np.abs(np.fft.rfft(segment)).tolist()
    return {"freqs": freqs.tolist(), "spectrum": spectrum}


from fastapi import Query

@app.get("/sessions/{session_id}/analyze")
def analyze_get(
    session_id: str,
    channel: str,
    start: int = Query(..., ge=0),
    end: int = Query(..., gt=0)
):
    s = get_session_or_404(session_id)
    data = s["data"].get(channel)
    if data is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    if end > len(data):
        end = len(data)
    if start >= end:
        raise HTTPException(status_code=400, detail="Invalid range")

    segment = np.array(data[start:end])
    fs = s.get("metadata", {}).get("sample_frequency", 1.0)
    freqs = np.fft.rfftfreq(len(segment), d=1.0/fs)
    spectrum = np.abs(np.fft.rfft(segment)).tolist()
    return {
        "channel": channel,
        "start": start,
        "end": end,
        "freqs": freqs.tolist(),
        "spectrum": spectrum
    }


from fastapi.responses import JSONResponse, StreamingResponse
import io, csv

@app.get("/sessions/{session_id}/download")
def download_played(session_id: str, format: str = "json"):
    """
    Download the portion of the session data that has been played so far.
    Supports JSON (default) or CSV.
    """
    s = get_session_or_404(session_id)
    pos = s.get("position", 0)
    data = s.get("data", {})
    channels = s["config"].get("channels") or list(data.keys())

    # Slice each channel up to current position
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
        # Build CSV in memory
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


@app.get("/sessions/{session_id}/metadata")
def get_session_metadata(session_id: str):
    """
    Return metadata and number of timesteps played for a given session.
    """
    s = get_session_or_404(session_id)
    position = s.get("position", 0)  # default to 0 if not started
    return {
        "metadata": s.get("metadata", {}),
        "timesteps_played": position
    }

class FFTRequest(BaseModel):
    window_size: int  # number of timesteps to include in the FFT

@app.post("/sessions/{session_id}/fft2")
def compute_fft(session_id: str, req: FFTRequest):
    """
    Compute FFT for the most recent window of data in the session.
    Returns frequency and magnitude spectrum for each channel.
    """
    session = get_session_or_404(session_id)
    data = session.get("data", {})
    position = session.get("position", 0)
    window_size = req.window_size

    if not data:
        raise HTTPException(status_code=400, detail="No data available in session")

    fft_results = {}
    for channel, samples in data.items():
        end_idx = min(position, len(samples))
        start_idx = max(0, end_idx - window_size)
        window = samples[start_idx:end_idx]

        if len(window) < 2:
            fft_results[channel] = {"frequencies": [], "magnitudes": []}
            continue

        signal = np.array(window)
        fft_vals = np.fft.rfft(signal)
        fft_freqs = np.fft.rfftfreq(len(signal), d=1.0)  # assuming unit sampling interval

        fft_results[channel] = {
            "frequencies": fft_freqs.tolist(),
            "magnitudes": np.abs(fft_vals).tolist()
        }

    return {"fft": fft_results}


class FFTTimeRangeRequest(BaseModel):
    min_time: float
    max_time: float
    variables: List[str]

@app.post("/sessions/{session_id}/fft")
def compute_fft_time_range(session_id: str, req: FFTTimeRangeRequest):
    """
    Compute FFT for a selected time window and multiple variables in the session.
    Returns frequency and magnitude spectrum per variable.
    """
    session = get_session_or_404(session_id)

    logger.debug("FFT request received: session_id=%s req=%s", session_id, req)
    data = session.get("data", {})

    if not data:
        raise HTTPException(status_code=400, detail="No data available in session")

    start_idx = int(req.min_time)
    end_idx = int(req.max_time)

    if start_idx >= end_idx:
        raise HTTPException(status_code=400, detail="Invalid time window")

    fft_results: Dict[str, Dict[str, Any]] = {}

    for variable in req.variables:
        if variable not in data:
            fft_results[variable] = {
                "frequencies": [],
                "magnitudes": [],
                "error": f"Variable '{variable}' not found"
            }
            continue

        samples = data[variable]
        if end_idx > len(samples):
            fft_results[variable] = {
                "frequencies": [],
                "magnitudes": [],
                "error": "Selected time window exceeds data length"
            }
            continue

        window = samples[start_idx:end_idx]
        if len(window) < 2:
            fft_results[variable] = {
                "frequencies": [],
                "magnitudes": [],
                "error": "Not enough data points"
            }
            continue

        signal = np.array(window)
        fft_vals = np.fft.rfft(signal)
        fft_freqs = np.fft.rfftfreq(len(signal), d=1.0)  # Adjust d=... if sampling rate is known

        fft_results[variable] = {
            "frequencies": fft_freqs.tolist(),
            "magnitudes": np.abs(fft_vals).tolist()
        }

    return {"fft": fft_results}



# Request model
class WindowModel(BaseModel):
    t_min: Optional[float] = 0.0
    t_max: Optional[float] = 0.0

class OptionsModel(BaseModel):
    method: Optional[str] = "goertzel"
    return_peak: Optional[bool] = False
    detrend: Optional[bool] = True
    window_type: Optional[str] = "hann"

class ComputeRequest(BaseModel):
    window: Optional[WindowModel] = WindowModel()
    channels: Optional[List[str]] = None
    options: Optional[OptionsModel] = OptionsModel()
    variables: Optional[Dict[str, Any]] = {}

@app.post("/sessions/{session_id}/amplitudes/fg-fp")
def amplitudes_endpoint(session_id: str, req: Optional[ComputeRequest] = None):
    get_session_or_404(session_id, detail=f"Session {session_id} not found")
    try:
        logger.debug("Amplitude request: session_id=%s req=%s", session_id, req)
        result = compute_fg_fp_for_window_session_multi_ref(
            session=sessions[session_id],
            request=req.dict()
        )
        return {"ok": True, "result": result}
    except Exception as e:
        logger.exception("Amplitude computation failed")
        return {"ok": False, "error": str(e)}

class ReplayRequest(BaseModel):
    speed: float = 1.0  # playback speed multiplier

@app.post("/sessions/{session_id}/replay")
async def replay_session(session_id: str, req: ReplayRequest):
    """
    Restart a session run from the beginning, with a given playback speed.
    """
    session = get_session_or_404(session_id)

    # Reset position to beginning
    session["position"] = 0

    # Store playback speed (your streaming loop should respect this)
    session["metadata"]["playback_speed"] = req.speed
    session["config"]["speed"] = req.speed
    logger.debug("Replay requested for session %s", session_id)
    if session.get("task"):
        session["task"].cancel()
    session["running"] = True
    # Start a new playback task
    loop = asyncio.get_running_loop()
    session["task"] = loop.create_task(playback_task(session_id))
    # Ensure FFT streamer restarts on replay as well: cancel any old task and start new
    try:
        old_fft = session.get("fft_task")
        # If an old fft task exists, cancel it and wait for it to finish so its
        # finally-block does not overwrite the new task state (race condition).
        if old_fft is not None:
            try:
                old_fft.cancel()
                # await the old task to let its finally block run; ignore
                # CancelledError which is expected when canceling a task.
                try:
                    await old_fft
                except asyncio.CancelledError:
                    # expected when cancelling the task
                    pass
                except Exception:
                    pass
            except Exception:
                pass
            session["fft_task"] = None

        # start fresh fft task
        session["running_fft"] = True
        session["fft_task"] = loop.create_task(fft_stream_task(session))
    except Exception:
        # If FFT task cannot be started for some reason, keep replaying time-domain
        logger.exception("Replay: could not start FFT task")
        pass
    return {"status": "restarted", "session_id": session_id, "speed": req.speed}

# fastapi_ws_fft.py

# fastapi_fft_control.py

router = APIRouter()

@router.post("/sessions/{session_id}/fft/start")
async def start_fft(session_id: str):
    s = get_session_or_404(session_id)

    if s.get("fft_task") is None or s.get("running_fft") is False:
        s["running_fft"] = True
        # fft_stream_task expects the session object (dict), pass it so
        # the task can read/update session state directly.
        s["fft_task"] = asyncio.create_task(fft_stream_task(s))
        return {"ok": True, "msg": "FFT task started"}
    else:
        return {"ok": True, "msg": "FFT task already running"}

@router.post("/sessions/{session_id}/fft/stop")
async def stop_fft(session_id: str):
    s = get_session_or_404(session_id)
    s["running_fft"] = False
    return {"ok": True, "msg": "FFT task stopping"}

app.include_router(router)

# ws_fft.py
ws_router = APIRouter()

@ws_router.websocket("/sessions/{session_id}/fft")
async def ws_fft(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in sessions:
        await websocket.close(code=4404)
        return

    s = sessions[session_id]
    q: asyncio.Queue = asyncio.Queue(maxsize=8)
    subs = s.setdefault("fft_subscribers", [])
    subs.append(q)
    logger.info(
        "[ws_fft] client connected for session %s; total_fft_subscribers=%s",
        session_id,
        len(subs),
    )

    try:
        while True:
            msg = await q.get()
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        logger.info("[ws_fft] client disconnected for session %s", session_id)
    finally:
        subs = s.get("fft_subscribers", [])
        if q in subs:
            subs.remove(q)
            logger.debug("[ws_fft] removed subscriber; remaining=%s", len(subs))

app.include_router(ws_router)

# Graph outbox status for durable Neo4j best-effort writes.
# ---------------------------------------------------------------------------
# Router mounting
# ---------------------------------------------------------------------------
# Optional routers are mounted best-effort: a missing optional dependency
# disables one feature area, it does not stop the app booting. The failure mode
# used to be invisible though — a silent 404 whose only trace was one log line
# at start-up. Every attempt is now recorded and served from
# ``GET /health/routers`` so "why is this endpoint 404ing?" is answerable
# without digging through logs. See DEBT-13.

ROUTER_LOAD_STATUS: List[Dict[str, Any]] = []


def _mount_router(
    module_path: str,
    *,
    label: str,
    disabled: str,
    **include_kwargs: Any,
) -> None:
    """Import ``module_path`` and mount its ``router``, recording the outcome.

    ``disabled`` states, in plain words, what stops working if this fails —
    it is surfaced to whoever is looking at /health/routers.
    """
    entry: Dict[str, Any] = {"router": label, "module": module_path, "loaded": False}
    try:
        module = importlib.import_module(module_path)
        app.include_router(getattr(module, "router"), **include_kwargs)
        entry["loaded"] = True
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["disabled"] = disabled
        logger.warning(
            "%s router failed to import; %s: %s", label, disabled, exc
        )
    ROUTER_LOAD_STATUS.append(entry)


# Mounted before the broad /agent router so /agent/memory/{memory_id} cannot
# shadow /agent/memory/graph-outbox — order matters here.
_mount_router(
    "backend.routers.graph_outbox",
    label="graph_outbox",
    disabled="graph outbox status disabled",
)
_mount_router(
    "backend.agents.router",
    label="agents",
    disabled="agent dispatch endpoints disabled",
    prefix="/agent",
)
for _module, _label, _disabled in (
    ("backend.routers.config", "config", "config endpoints disabled"),
    ("backend.routers.fleet_knowledge", "fleet_knowledge", "fleet endpoints disabled"),
    ("backend.routers.reconfig", "reconfig", "reconfig endpoints disabled"),
    ("backend.routers.demo_director", "demo_director", "scripted demo events disabled"),
    ("backend.routers.domain", "domain", "domain endpoints disabled"),
    ("backend.routers.feedback", "feedback", "async feedback disabled"),
    ("backend.routers.sindit", "sindit", "sindit endpoints disabled"),
    ("backend.routers.sessions", "sessions", "session endpoints disabled"),
    ("backend.routers.harmonic", "harmonic", "harmonic endpoints disabled"),
    ("backend.routers.analysis", "analysis", "analysis endpoints disabled"),
    ("backend.routers.experiment", "experiment", "experiment endpoints disabled"),
    ("backend.routers.dataset", "dataset", "dataset endpoints disabled"),
    ("backend.routers.streams", "streams", "stream endpoints disabled"),
):
    _mount_router(_module, label=_label, disabled=_disabled)


@app.get("/health/routers")
async def health_routers() -> Dict[str, Any]:
    """Report which optional routers loaded, and why any did not.

    Answers "this endpoint 404s — is it broken, or did its router fail to
    import?" without needing access to the start-up logs.
    """
    failed = [r for r in ROUTER_LOAD_STATUS if not r["loaded"]]
    return {
        "total": len(ROUTER_LOAD_STATUS),
        "loaded": len(ROUTER_LOAD_STATUS) - len(failed),
        "failed": len(failed),
        "routers": ROUTER_LOAD_STATUS,
    }


_LEGACY_SESSION_ROUTE_PATHS = {
    "/sessions",
    "/sessions/{session_id}",
    "/sessions/{session_id}/upload",
    "/sessions/{session_id}/start",
    "/sessions/{session_id}/pause",
    "/sessions/{session_id}/resume",
    "/sessions/{session_id}/analyze",
    "/sessions/{session_id}/download",
    "/sessions/{session_id}/metadata",
    "/sessions/{session_id}/replay",
}

_LEGACY_SESSION_WS_ROUTE_PATHS = {
    "/streams/{session_id}",
    "/sessions/{session_id}/fft",
}


def _prune_legacy_http_session_routes() -> None:
    app.router.routes = [
        route
        for route in app.router.routes
        if not (
            isinstance(route, APIRoute)
            and route.endpoint.__module__ == __name__
            and route.path in _LEGACY_SESSION_ROUTE_PATHS
        )
    ]


def _prune_legacy_websocket_session_routes() -> None:
    app.router.routes = [
        route
        for route in app.router.routes
        if not (
            isinstance(route, APIWebSocketRoute)
            and route.endpoint.__module__ == __name__
            and route.path in _LEGACY_SESSION_WS_ROUTE_PATHS
        )
    ]


_prune_legacy_http_session_routes()
_prune_legacy_websocket_session_routes()


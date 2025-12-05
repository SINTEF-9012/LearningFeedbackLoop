import asyncio, json, time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, Path, WebSocketDisconnect, APIRouter
from pydantic import BaseModel
from requests import session

from computation import compute_fg_fp_for_window_session_multi_ref
from fft_streamer import fft_stream_task

app = FastAPI()
sessions: Dict[str, Dict[str, Any]] = {}


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
        print("Detected MATLAB-style export")
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

# -----------------------------
# Playback task
# -----------------------------
async def playback_task2(session_id: str):
    s = sessions[session_id]
    cfg = s["config"]
    interval = cfg["interval_ms"] / 1000.0
    pos = s.get("position", 0)  # resume from last known position
    data = s["data"]
    channels = cfg["channels"] or list(data.keys())

    try:
        while s["running"]:
            if s.get("paused", False):
                # If paused, just wait a bit and continue loop
                await asyncio.sleep(0.1)
                continue

            frame = {"t": pos}
            for ch in channels:
                frame[ch] = data[ch][pos] if pos < len(data[ch]) else None
            pos += 1
            s["position"] = pos  # update current position

            # broadcast to subscribers
            for q in list(s["subscribers"]):
                await q.put(frame)

            await asyncio.sleep(interval)

            # stop at end of data
            if pos >= len(data[channels[0]]):
                s["running"] = False
    finally:
        s["task"] = None


async def playback_task(session_id: str):
    print("Starting playback task111", flush=True)
    s = sessions[session_id]
    cfg = s["config"]
    data = s["data"]
    metadata = s["metadata"]
    print("0", flush=True)

    # --- Required: sampling frequency from metadata ---
    print("metadata",metadata["file_header"], flush=True)
    fs = float(metadata["file_header"]["SampleFrequency"])  # e.g., 4096.0
    print("fs", fs, flush=True)
    Ts = 1.0 / fs
    print("1", flush=True)
    # --- Config knobs (user-provided at upload or defaults) ---
    channels = cfg.get("channels") or list(data.keys())
    speed = float(cfg.get("speed", default_speed))           # 1.0 = real-time; <1 slower; >1 faster
    samples_per_tick = int(cfg.get("samples_per_tick", default_samples_per_tick))  # 1 = per-sample; >1 = chunk mode
    samples_per_tick = max(1, samples_per_tick)
    print("2", flush=True)
    # --- Resume position & limits ---
    pos = int(s.get("position", 0))
    if not channels:
        raise ValueError("No channels available to process")
    n_max = min(len(data[ch]) for ch in channels)  # ensure all channels valid length
    print("3", flush=True)
    # Absolute scheduling to avoid drift
    start_wall = time.perf_counter()
    start_pos = pos
    next_emit_time = start_wall  # emit immediately first time
    print("In this", flush=True)
    try:
        print(s.get("running", False), flush=True)
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
                frame = {"t": t, "i": i0, "fs": fs}
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
                print(frame)
                await q.put(frame)

            # Publish feature event for online agents
            try:
                from events import publish_feature
                # publish a lightweight payload; include session id
                payload = {"type": "time", "session_id": session_id, "position": pos, "frame": frame}
                await publish_feature(session_id, payload)
            except Exception:
                # keep streaming even if event publish fails
                pass

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
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    payload = json.loads(await file.read())
    try:
        data, metadata = preprocess_payload(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")
    sessions[session_id]["data"] = data
    sessions[session_id]["metadata"] = metadata
    sessions[session_id]["raw_file"] = payload

    sessions[session_id].update({
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
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    s["running"] = True
    s["task"] = asyncio.create_task(playback_task(session_id))
    print(f"Started session {session_id} playback task")

    # in /sessions/{sid}/start playback handler:
    try:
        fft_task = sessions[session_id].get("fft_task")
        if fft_task is None or (hasattr(fft_task, "done") and fft_task.done()):
            print("Starting FFT stream task", flush=True)
            sessions[session_id]["running_fft"] = True
            sessions[session_id]["fft_task"] = asyncio.create_task(fft_stream_task(s))
        else:
            print("FFT task already exists and appears running", flush=True)
    except Exception as e:
        print(f"Error while starting FFT task: {e}", flush=True)
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
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
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
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id]["paused"] = True
    return {"ok": True, "paused": True}

@app.post("/sessions/{session_id}/resume")
def resume(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id]["paused"] = False
    return {"ok": True, "paused": False}


import numpy as np

@app.post("/sessions/{session_id}/analyze")
def analyze(session_id: str, start: int, end: int, channel: str):
    s = sessions[session_id]
    data = s["data"].get(channel)
    if data is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    segment = np.array(data[start:end])
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / s["metadata"].get("sample_frequency", 1.0))
    spectrum = np.abs(np.fft.rfft(segment)).tolist()
    return {"freqs": freqs.tolist(), "spectrum": spectrum}


import numpy as np
from fastapi import Query

@app.get("/sessions/{session_id}/analyze")
def analyze(
    session_id: str,
    channel: str,
    start: int = Query(..., ge=0),
    end: int = Query(..., gt=0)
):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
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
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    s = sessions[session_id]
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
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    s = sessions[session_id]
    position = s.get("position", 0)  # default to 0 if not started
    return {
        "metadata": s.get("metadata", {}),
        "timesteps_played": position
    }

    from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from pydantic import BaseModel
import numpy as np
class FFTRequest(BaseModel):
    window_size: int  # number of timesteps to include in the FFT

@app.post("/sessions/{session_id}/fft2")
def compute_fft(session_id: str, req: FFTRequest):
    """
    Compute FFT for the most recent window of data in the session.
    Returns frequency and magnitude spectrum for each channel.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
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


class FFTRequest(BaseModel):
    min_time: float
    max_time: float
    variables: List[str]

@app.post("/sessions/{session_id}/fft")
def compute_fft(session_id: str, req: FFTRequest):
    """
    Compute FFT for a selected time window and multiple variables in the session.
    Returns frequency and magnitude spectrum per variable.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    print("FFT request received:", session_id, req)
    session = sessions[session_id]
    data = session.get("data", {})

    if not data:
        raise HTTPException(status_code=400, detail="No data available in session")

    start_idx = int(req.min_time)
    end_idx = int(req.max_time)

    if start_idx >= end_idx:
        raise HTTPException(status_code=400, detail="Invalid time window")

    fft_results: Dict[str, Dict[str, List[float]]] = {}

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
def amplitudes_endpoint(session_id: str, req: ComputeRequest = None):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    try:
        print("Amplitude request:", session_id, req)
        result = compute_fg_fp_for_window_session_multi_ref(
            session=sessions[session_id],
            request=req.dict()
        )
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}

class ReplayRequest(BaseModel):
    speed: float = 1.0  # playback speed multiplier

@app.post("/sessions/{session_id}/replay")
async def replay_session(session_id: str, req: ReplayRequest):
    """
    Restart a session run from the beginning, with a given playback speed.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # Reset position to beginning
    session["position"] = 0

    # Store playback speed (your streaming loop should respect this)
    session["metadata"]["playback_speed"] = req.speed
    session["config"]["speed"] = req.speed
    print("here?")
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
        print("cannot start fft", flush=True)
        pass
    return {"status": "restarted", "session_id": session_id, "speed": req.speed}


    print("Replaying session", session_id, "at speed", req.speed)

    return {"status": "restarted", "session_id": session_id, "speed": req.speed}

# fastapi_ws_fft.py

# fastapi_fft_control.py

router = APIRouter()

@router.post("/sessions/{session_id}/fft/start")
async def start_fft(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]

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
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
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
    print(f"[ws_fft] client connected for session {session_id}; total_fft_subscribers={len(subs)}", flush=True)

    try:
        while True:
            msg = await q.get()
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        print(f"[ws_fft] client disconnected for session {session_id}", flush=True)
    finally:
        subs = s.get("fft_subscribers", [])
        if q in subs:
            subs.remove(q)
            print(f"[ws_fft] removed subscriber; remaining={len(subs)}", flush=True)

app.include_router(ws_router)

# Agents router (agent dispatch endpoints)
try:
    from agents.router import router as agents_router
    app.include_router(agents_router, prefix="/agent")
except Exception as e:
    # best-effort: if agents package missing optional deps, keep server running
    print(f"Agents router failed to import; agent endpoints disabled: {e}")


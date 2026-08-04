"""Analysis router — FFT computation, amplitude analysis, spectral analysis.

The old ``/fft`` (time-range) and ``/fft2`` (last-N) endpoints are merged
into a single ``POST /sessions/{session_id}/fft``. The spectral analysis
surface now lives here too, with ``GET /sessions/{session_id}/analyze`` and
the backward-compatible ``POST /sessions/{session_id}/analyze`` sharing the
same implementation. The ``/fft/start`` and ``/fft/stop`` manual controls are
dropped because FFT tasks auto-start on upload/start.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request

from ..computation import compute_fg_fp_for_window_session_multi_ref
from ..metadata_utils import get_sample_frequency

from .dependencies import (
    ComputeRequest,
    FFTRequest,
    get_session_or_404,
    get_sessions_dict,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _analyze_channel(
    session: Dict[str, Any],
    *,
    channel: str,
    start: int,
    end: int,
) -> Dict[str, Any]:
    data = session["data"].get(channel)
    if data is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    if end > len(data):
        end = len(data)
    if start >= end:
        raise HTTPException(status_code=400, detail="Invalid range")

    segment = np.array(data[start:end])
    fs = get_sample_frequency(session.get("metadata", {}), default=1.0)
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(segment)).tolist()

    return {
        "channel": channel,
        "start": start,
        "end": end,
        "freqs": freqs.tolist(),
        "spectrum": spectrum,
    }


# ── Unified FFT endpoint ─────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/fft")
def compute_fft(session_id: str, req: FFTRequest, request: Request):
    """Compute FFT for a session's data.

    Two modes, selected automatically:

    * **Window mode** — supply ``window_size``; returns FFT of the last
      *window_size* samples relative to the current playback position.
    * **Range mode** — supply ``min_index`` + ``max_index`` (and optionally
      ``variables``); returns FFT over a specific sample range.

    If both are supplied, range mode takes precedence.
    """
    sessions = get_sessions_dict(request)
    session = get_session_or_404(session_id, sessions)

    data = session.get("data", {})
    if not data:
        raise HTTPException(status_code=400, detail="No data available in session")

    metadata = session.get("metadata", {})

    # ── Resolve backward-compat aliases ──────────────────────────────
    min_idx = req.min_index
    max_idx = req.max_index
    if min_idx is None and req.min_time is not None:
        min_idx = int(req.min_time)
    if max_idx is None and req.max_time is not None:
        max_idx = int(req.max_time)

    # ── Range mode ───────────────────────────────────────────────────
    if min_idx is not None and max_idx is not None:
        if min_idx >= max_idx:
            raise HTTPException(status_code=400, detail="min_index must be < max_index")

        _fs = get_sample_frequency(metadata, default=1.0)
        variables = req.variables or list(data.keys())
        fft_results: Dict[str, Dict[str, Any]] = {}

        for variable in variables:
            if variable not in data:
                fft_results[variable] = {
                    "frequencies": [],
                    "magnitudes": [],
                    "error": f"Variable '{variable}' not found",
                }
                continue

            samples = data[variable]
            if max_idx > len(samples):
                fft_results[variable] = {
                    "frequencies": [],
                    "magnitudes": [],
                    "error": "Selected range exceeds data length",
                }
                continue

            window = samples[min_idx:max_idx]
            if len(window) < 2:
                fft_results[variable] = {
                    "frequencies": [],
                    "magnitudes": [],
                    "error": "Not enough data points",
                }
                continue

            signal = np.array(window)
            fft_vals = np.fft.rfft(signal)
            fft_freqs = np.fft.rfftfreq(len(signal), d=1.0 / _fs)
            fft_results[variable] = {
                "frequencies": fft_freqs.tolist(),
                "magnitudes": np.abs(fft_vals).tolist(),
            }

        return {"fft": fft_results}

    # ── Window mode ──────────────────────────────────────────────────
    if req.window_size is not None and req.window_size > 0:
        position = session.get("position", 0)
        _fs = get_sample_frequency(metadata, default=1.0)
        fft_results = {}

        for channel, samples in data.items():
            end_idx = min(position, len(samples))
            start_idx = max(0, end_idx - req.window_size)
            window = samples[start_idx:end_idx]

            if len(window) < 2:
                fft_results[channel] = {"frequencies": [], "magnitudes": []}
                continue

            signal = np.array(window)
            fft_vals = np.fft.rfft(signal)
            fft_freqs = np.fft.rfftfreq(len(signal), d=1.0 / _fs)
            fft_results[channel] = {
                "frequencies": fft_freqs.tolist(),
                "magnitudes": np.abs(fft_vals).tolist(),
            }

        return {"fft": fft_results}

    raise HTTPException(
        status_code=400,
        detail="Provide either window_size (window mode) or min_index+max_index (range mode).",
    )


# ── Analyze ─────────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/analyze")
def analyze(
    session_id: str,
    request: Request,
    channel: str = Query(...),
    start: int = Query(..., ge=0),
    end: int = Query(..., gt=0),
):
    """Compute basic FFT spectrum for a channel over ``[start, end)``."""
    sessions = get_sessions_dict(request)
    s = get_session_or_404(session_id, sessions)

    return _analyze_channel(s, channel=channel, start=start, end=end)



@router.post("/sessions/{session_id}/analyze")
def analyze_post(
    session_id: str,
    request: Request,
    channel: str = Query(...),
    start: int = Query(..., ge=0),
    end: int = Query(..., gt=0),
):
    sessions = get_sessions_dict(request)
    s = get_session_or_404(session_id, sessions)

    return _analyze_channel(s, channel=channel, start=start, end=end)


# ── Amplitude computation ────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/amplitudes/fg-fp")
def amplitudes_endpoint(
    session_id: str,
    request: Request,
    req: Optional[ComputeRequest] = None,
):
    sessions = get_sessions_dict(request)
    s = get_session_or_404(session_id, sessions, detail=f"Session {session_id} not found")
    try:
        logger.debug("Amplitude request: session_id=%s req=%s", session_id, req)
        result = compute_fg_fp_for_window_session_multi_ref(
            session=s,
            request=req.dict() if req else {},
        )
        return {"ok": True, "result": result}
    except Exception as e:
        logger.exception("Amplitude computation failed")
        raise HTTPException(status_code=500, detail=f"Amplitude computation failed: {e}")

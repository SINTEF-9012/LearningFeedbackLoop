"""Shared dependencies and helpers used by multiple routers.

This module avoids circular imports: routers import from here,
never from ``backend.app``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Session defaults ─────────────────────────────────────────────────────────

DEFAULT_SAMPLES_PER_TICK = 32
DEFAULT_SPEED = 1.0  # real-time


# ── Pydantic models shared across routers ────────────────────────────────────

class SessionConfig(BaseModel):
    interval_ms: int
    channels: Optional[List[str]] = None
    mode: str = "time"  # or "frequency"
    speed: float = DEFAULT_SPEED
    samples_per_tick: int = DEFAULT_SAMPLES_PER_TICK
    start_paused: bool = False
    pause_on_alert: bool = False
    harmonic_scorer_kind: str = "context"  # "context" or "pair"
    harmonic_dataset: Optional[str] = None


class PlaybackConfigUpdate(BaseModel):
    """Live-update playback parameters (renamed from PlaybackUpdate)."""
    speed: Optional[float] = None
    samples_per_tick: Optional[int] = None
    pause_on_alert: Optional[bool] = None
    harmonic_scorer_kind: Optional[str] = None
    harmonic_dataset: Optional[str] = None


class ReplayRequest(BaseModel):
    speed: float = 1.0


class FFTRequest(BaseModel):
    """Unified FFT request — supports both window-size and index-range modes.

    * Supply ``window_size`` to get the last N samples (old ``/fft2``).
    * Supply ``min_index`` + ``max_index`` (+ optionally ``variables``) to get
      a specific range (old ``/fft``).
    * Both may be present; ``min_index``/``max_index`` takes precedence.
    """
    window_size: Optional[int] = None
    min_index: Optional[int] = None
    max_index: Optional[int] = None
    variables: Optional[List[str]] = None
    # Backward-compat aliases (deprecated — prefer min_index/max_index)
    min_time: Optional[float] = None
    max_time: Optional[float] = None


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


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_sessions_dict(request: Request) -> Dict[str, Dict[str, Any]]:
    """FastAPI dependency — returns the sessions dict from app.state."""
    return request.app.state.sessions


def get_session_or_404(
    session_id: str,
    sessions_dict: Dict[str, Dict[str, Any]],
    *,
    detail: str = "Session not found",
) -> Dict[str, Any]:
    """Fetch a session from the dict or raise HTTP 404."""
    if session_id not in sessions_dict:
        raise HTTPException(status_code=404, detail=detail)
    return sessions_dict[session_id]


def json_default(obj: Any):
    """Best-effort JSON serializer for numpy arrays / scalars sent over WS."""
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(obj, "item", None)
    if callable(item):
        return item()
    return str(obj)

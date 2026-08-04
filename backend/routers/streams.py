"""WebSocket stream endpoints — time-domain, FFT, inference.

Validates session existence *before* accepting the connection.

Agent Q (Round 20, 2026-04-24): optional server-side LTTB
downsampling on chunk / FFT frames via ``?downsample=<int>`` query
parameter. When set, frames with per-channel arrays at least that
long are downsampled to exactly ``<int>`` points before being
JSON-serialised and sent. Per-sample frames and small frames pass
through unchanged. Invalid values fall back silently to no
downsampling (stream must not break).
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ._stream_downsample import maybe_downsample_frame
from .dependencies import get_sessions_dict, json_default

logger = logging.getLogger(__name__)

router = APIRouter()


def _parse_downsample(raw: str | None) -> int:
    """Parse the ``?downsample=`` query param. Returns 0 when disabled."""
    if raw is None:
        return 0
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(0, val)


# ── Time-domain stream ───────────────────────────────────────────────────────

@router.websocket("/streams/{session_id}")
async def ws_stream(websocket: WebSocket, session_id: str):
    sessions = websocket.app.state.sessions
    if session_id not in sessions:
        await websocket.close(code=4404)
        return

    downsample = _parse_downsample(websocket.query_params.get("downsample"))

    await websocket.accept()
    s = sessions[session_id]
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    s["subscribers"].append(queue)
    try:
        while True:
            frame = await queue.get()
            if downsample > 2:
                frame = maybe_downsample_frame(frame, downsample)
            try:
                await websocket.send_text(json.dumps(frame, default=json_default))
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        subs = s.get("subscribers", [])
        if queue in subs:
            subs.remove(queue)


# ── FFT stream ───────────────────────────────────────────────────────────────

@router.websocket("/sessions/{session_id}/fft")
async def ws_fft(websocket: WebSocket, session_id: str):
    sessions = websocket.app.state.sessions
    if session_id not in sessions:
        await websocket.close(code=4404)
        return

    downsample = _parse_downsample(websocket.query_params.get("downsample"))

    await websocket.accept()
    s = sessions[session_id]
    q: asyncio.Queue = asyncio.Queue(maxsize=8)
    subs = s.setdefault("fft_subscribers", [])
    subs.append(q)
    logger.info(
        "[ws_fft] client connected for session %s; total_fft_subscribers=%s",
        session_id, len(subs),
    )

    try:
        while True:
            msg = await q.get()
            if downsample > 2:
                msg = maybe_downsample_frame(msg, downsample)
            try:
                await websocket.send_text(json.dumps(msg, default=json_default))
            except Exception:
                break
    except WebSocketDisconnect:
        logger.info("[ws_fft] client disconnected for session %s", session_id)
    finally:
        subs = s.get("fft_subscribers", [])
        if q in subs:
            subs.remove(q)
            logger.debug("[ws_fft] removed subscriber; remaining=%s", len(subs))


# ── Inference stream ─────────────────────────────────────────────────────────

@router.websocket("/sessions/{session_id}/inference")
async def ws_inference(websocket: WebSocket, session_id: str):
    sessions = websocket.app.state.sessions
    if session_id not in sessions:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    s = sessions[session_id]
    q: asyncio.Queue = asyncio.Queue(maxsize=16)
    subs = s.setdefault("inference_subscribers", [])
    subs.append(q)
    logger.info(
        "[ws_inference] client connected for session %s; total_inference_subscribers=%s",
        session_id, len(subs),
    )

    try:
        while True:
            msg = await q.get()
            try:
                await websocket.send_text(json.dumps(msg, default=json_default))
            except Exception:
                break
    except WebSocketDisconnect:
        logger.info("[ws_inference] client disconnected for session %s", session_id)
    finally:
        subs = s.get("inference_subscribers", [])
        if q in subs:
            subs.remove(q)
            logger.debug("[ws_inference] removed subscriber; remaining=%s", len(subs))

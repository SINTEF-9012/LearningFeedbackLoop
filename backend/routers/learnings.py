"""Live learnings router.

Exposes the in-memory learnings bus so the UI can mirror what is also being
published to MQTT.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..events import bus, subscribe_learnings
from ..ingestion.schema import envelope_to_dict


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/learnings", tags=["learnings"])


async def _stream_learnings(websocket: WebSocket, session_id: str | None = None) -> None:
    await websocket.accept()
    queue = subscribe_learnings(session_id)
    channel = f"learnings.{session_id}" if session_id else "learnings"
    try:
        while True:
            payload = await queue.get()
            try:
                await websocket.send_json(envelope_to_dict(payload))
            except TypeError:
                await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("learnings websocket failed")
    finally:
        bus.unsubscribe(channel, queue)


@router.websocket("/ws")
async def learnings_ws(websocket: WebSocket) -> None:
    await _stream_learnings(websocket)


@router.websocket("/ws/{session_id}")
async def learnings_ws_session(websocket: WebSocket, session_id: str) -> None:
    await _stream_learnings(websocket, session_id=session_id)
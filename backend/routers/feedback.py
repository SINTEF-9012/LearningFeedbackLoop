"""Async feedback router — Agent M (2026-04-24).

Exposes the operator-feedback outbox, history, and live broadcast
created by :mod:`backend.agents.memory.feedback_async`.

Endpoints:
    GET  /agent/memory/feedback/operators               — summary per operator
    GET  /agent/memory/feedback/operators/{operator_id} — events for one operator
    GET  /agent/memory/feedback/outbox                  — pending count + head
    WS   /agent/memory/feedback/ws                      — live broadcast

The pipeline is constructed once at app startup (see :mod:`backend.app`
lifespan) and stashed on ``app.state.feedback_pipeline``. On failure to
attach to the orchestrator, all HTTP endpoints still work against the
outbox + in-memory history replay — just no live hookup to feedback
processing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/memory/feedback", tags=["feedback"])


class OperatorSummary(BaseModel):
    operator_id: str
    total: int
    actions: Dict[str, int] = Field(default_factory=dict)


class OperatorSummaryResponse(BaseModel):
    operators: List[OperatorSummary]


class OperatorEvent(BaseModel):
    memory_id: str
    action: str
    operator_id: str
    created_at: str
    sequence: int
    data: Dict[str, Any] = Field(default_factory=dict)


class OperatorEventsResponse(BaseModel):
    operator_id: str
    count: int
    events: List[OperatorEvent]


class OutboxStatusResponse(BaseModel):
    pending: int
    head: List[OperatorEvent] = Field(default_factory=list)


class ExplanationFeedbackRequest(BaseModel):
    memory_id: str
    helpful: bool
    operator_id: str = "ui"
    session_id: Optional[str] = None
    signature: Optional[str] = None
    summary_source: Optional[str] = None
    explanation_source: Optional[str] = None


class ExplanationFeedbackResponse(BaseModel):
    success: bool
    memory_id: str
    action: str


def _get_pipeline(request: Request):
    """Return the app-wide :class:`FeedbackPipeline` or 503."""
    pipeline = getattr(request.app.state, "feedback_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="feedback pipeline not initialised",
        )
    return pipeline


@router.get("/operators", response_model=OperatorSummaryResponse)
def list_operators(request: Request) -> OperatorSummaryResponse:
    """Per-operator action counts.

    Cheap: reads the in-memory ``OperatorFeedbackHistory``.
    """
    pipeline = _get_pipeline(request)
    summary = pipeline.history.summary()
    ops = [
        OperatorSummary(
            operator_id=op_id,
            total=sum(counts.values()),
            actions=dict(counts),
        )
        for op_id, counts in sorted(summary.items())
    ]
    return OperatorSummaryResponse(operators=ops)


@router.get("/operators/{operator_id}", response_model=OperatorEventsResponse)
def get_operator_events(
    operator_id: str,
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
) -> OperatorEventsResponse:
    pipeline = _get_pipeline(request)
    events = pipeline.history.for_operator(operator_id)
    if not events:
        # Return empty rather than 404 — operator may just have no activity yet.
        return OperatorEventsResponse(operator_id=operator_id, count=0, events=[])
    # Most-recent first.
    tail = list(reversed(events))[:limit]
    return OperatorEventsResponse(
        operator_id=operator_id,
        count=len(events),
        events=[OperatorEvent(**e.to_dict()) for e in tail],
    )


@router.get("/outbox", response_model=OutboxStatusResponse)
def outbox_status(
    request: Request,
    head: int = Query(0, ge=0, le=100, description="Return first N pending events"),
) -> OutboxStatusResponse:
    pipeline = _get_pipeline(request)
    pending = pipeline.outbox.pending_count()
    head_events: List[OperatorEvent] = []
    if head:
        for i, ev in enumerate(pipeline.outbox.iter_pending()):
            if i >= head:
                break
            head_events.append(OperatorEvent(**ev.to_dict()))
    return OutboxStatusResponse(pending=pending, head=head_events)


@router.post("/explanation", response_model=ExplanationFeedbackResponse)
async def explanation_feedback(
    body: ExplanationFeedbackRequest,
    request: Request,
) -> ExplanationFeedbackResponse:
    """Log whether the operator found an alert explanation helpful."""
    pipeline = _get_pipeline(request)
    action = "explanation_helpful" if body.helpful else "explanation_unhelpful"
    await pipeline.callback(
        body.memory_id,
        action,
        {
            "operator_id": body.operator_id,
            "session_id": body.session_id,
            "signature": body.signature,
            "summary_source": body.summary_source,
            "explanation_source": body.explanation_source,
            "helpful": bool(body.helpful),
        },
    )
    return ExplanationFeedbackResponse(
        success=True,
        memory_id=body.memory_id,
        action=action,
    )


@router.websocket("/ws")
async def feedback_ws(websocket: WebSocket):
    """Live broadcast of feedback events to connected operators."""
    pipeline = getattr(websocket.app.state, "feedback_pipeline", None)
    if pipeline is None:
        await websocket.close(code=1011)  # internal error
        return

    await websocket.accept()

    async def _send(payload: Dict[str, Any]) -> None:
        await websocket.send_json(payload)

    await pipeline.broadcaster.subscribe(_send)
    try:
        # Keep the socket open; we don't expect client messages, but
        # ``receive_text`` will raise ``WebSocketDisconnect`` on close.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("feedback_ws: unexpected error, closing subscriber")
    finally:
        await pipeline.broadcaster.unsubscribe(_send)

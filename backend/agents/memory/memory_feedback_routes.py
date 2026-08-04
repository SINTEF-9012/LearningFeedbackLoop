from __future__ import annotations

import asyncio

from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .dispatcher import get_dispatcher
from .feedback import FeedbackAction, MemoryFeedbackRequest, MemoryFeedbackResponse
from .orchestrator import get_orchestrator


router = APIRouter()


class FeedbackHistoryResponse(BaseModel):
    memory_id: str
    events: List[Dict[str, Any]]
    stats: Dict[str, Any]


class SignatureMuteRequest(BaseModel):
    session_id: str
    signature: Optional[str] = None
    pattern_keys: List[str] = Field(default_factory=list)
    muted: bool = True
    source: str = "ui"
    reason: Optional[str] = None


class SignatureMuteResponse(BaseModel):
    success: bool
    session_id: str
    signature: Optional[str] = None
    muted: bool
    state: Optional[Dict[str, Any]] = None


class DocLinkFeedbackValue(str, Enum):
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"


class DocLinkFeedbackRequest(BaseModel):
    feedback: DocLinkFeedbackValue
    user_id: str = "operator"
    reason: Optional[str] = None


class DocLinkFeedbackResponse(BaseModel):
    success: bool
    memory_id: str
    doc_id: str
    feedback: DocLinkFeedbackValue
    doc_link: Dict[str, Any] = Field(default_factory=dict)


@router.get("/{memory_id}/feedback", response_model=FeedbackHistoryResponse)
async def get_memory_feedback(memory_id: str, limit: int = Query(default=200, ge=1, le=1000)):
    """Return append-only feedback event history for a memory."""
    orchestrator = get_orchestrator()
    memory = orchestrator.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    events: List[Dict[str, Any]] = []
    if hasattr(orchestrator.store, "list_feedback_events"):
        try:
            events = list(orchestrator.store.list_feedback_events(memory_id, limit=int(limit)))
        except Exception:
            events = []

    if not events:
        records = orchestrator.feedback_handler.get_feedback_history(memory_id)
        events = [record.model_dump() for record in records]

    return FeedbackHistoryResponse(
        memory_id=memory_id,
        events=events,
        stats=orchestrator.feedback_handler.get_feedback_stats(memory_id),
    )


@router.patch("/{memory_id}/feedback", response_model=MemoryFeedbackResponse)
async def add_memory_feedback(memory_id: str, request: MemoryFeedbackRequest):
    """Add feedback to a memory."""
    orchestrator = get_orchestrator()

    if request.action == FeedbackAction.DISMISS:
        try:
            dispatcher = get_dispatcher()
            memory = orchestrator.get_memory(memory_id)
            if memory:
                dispatcher.set_cooldown(memory.session_id)
        except Exception:
            pass

    response = await orchestrator.feedback_handler.process_feedback(
        memory_id=memory_id,
        request=request,
    )

    if not response.success:
        raise HTTPException(status_code=404, detail=response.message)

    return response


@router.patch("/{memory_id}/doc_links/{doc_id}/feedback", response_model=DocLinkFeedbackResponse)
async def set_doc_link_feedback(memory_id: str, doc_id: str, request: DocLinkFeedbackRequest):
    """Record operator feedback for a memory-scoped documentation citation."""
    orchestrator = get_orchestrator()
    memory = orchestrator.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    store = getattr(orchestrator, "store", None)
    if store is None or not hasattr(store, "set_doc_link_feedback"):
        raise HTTPException(status_code=501, detail="Doc-link feedback is unavailable for this store")

    try:
        updated_link = await asyncio.to_thread(
            store.set_doc_link_feedback,
            memory_id=memory_id,
            doc_id=doc_id,
            feedback=request.feedback.value,
            user_id=request.user_id,
            reason=request.reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if updated_link is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} is not linked to memory {memory_id}")

    if hasattr(store, "add_feedback_event"):
        pattern_key = str(updated_link.get("pattern_key") or "").strip()
        event_data = {
            "source": "doc_link_feedback",
            "doc_id": doc_id,
            "citation": updated_link.get("citation"),
            "query_used": updated_link.get("query_used"),
            "pattern_key": pattern_key or None,
            "feedback": request.feedback.value,
            "reason": request.reason,
        }
        try:
            store.add_feedback_event(
                memory_id=memory_id,
                action=f"doc_link_{request.feedback.value}",
                user_id=request.user_id,
                pattern_keys=[pattern_key] if pattern_key else None,
                context_key="doc_link_feedback",
                data=event_data,
                weight=1.0,
            )
        except Exception:
            pass

    return DocLinkFeedbackResponse(
        success=True,
        memory_id=memory_id,
        doc_id=doc_id,
        feedback=request.feedback,
        doc_link=dict(updated_link),
    )


@router.post("/{memory_id}/confirm")
async def confirm_memory(
    memory_id: str,
    user_id: str = "operator",
    reason: Optional[str] = None,
    episode_id: Optional[str] = None,
):
    """Shortcut to confirm a memory as significant.

    ``episode_id`` (plan 1.4) dedupes the learning update per episode — pass the
    alert's ``recurrence.episode_id`` so a multi-window episode nudges the priors
    once, not once per window.
    """
    request = MemoryFeedbackRequest(
        action=FeedbackAction.CONFIRM,
        user_id=user_id,
        reason=reason,
        episode_id=episode_id,
    )
    return await add_memory_feedback(memory_id, request)


@router.post("/{memory_id}/dismiss")
async def dismiss_memory(
    memory_id: str,
    user_id: str = "operator",
    reason: Optional[str] = None,
    episode_id: Optional[str] = None,
):
    """Shortcut to dismiss a memory. ``episode_id`` dedupes learning (plan 1.4)."""
    request = MemoryFeedbackRequest(
        action=FeedbackAction.DISMISS,
        user_id=user_id,
        reason=reason,
        episode_id=episode_id,
    )
    return await add_memory_feedback(memory_id, request)


@router.post("/alerts/signature-mute", response_model=SignatureMuteResponse)
async def set_signature_mute(request: SignatureMuteRequest):
    """Mute or unmute a recurring alert signature for a session."""
    dispatcher = get_dispatcher()

    signature = request.signature
    if request.muted:
        state = dispatcher.set_signature_muted(
            request.session_id,
            pattern_keys=list(request.pattern_keys or []),
            signature=signature,
            source=request.source,
            reason=request.reason,
        )
        if state is None:
            raise HTTPException(status_code=400, detail="signature or pattern_keys required")
        return SignatureMuteResponse(
            success=True,
            session_id=request.session_id,
            signature=state.get("signature"),
            muted=True,
            state=state,
        )

    removed = dispatcher.clear_signature_muted(
        request.session_id,
        pattern_keys=list(request.pattern_keys or []),
        signature=signature,
    )
    resolved = signature or dispatcher._suppression_signature(list(request.pattern_keys or []))
    return SignatureMuteResponse(
        success=bool(removed),
        session_id=request.session_id,
        signature=resolved,
        muted=False,
        state=None,
    )
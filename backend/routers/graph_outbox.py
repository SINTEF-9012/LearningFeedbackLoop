"""Read-only status surface for the Neo4j graph write outbox."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.agents.memory.init import get_store

router = APIRouter(prefix="/agent/memory/graph-outbox", tags=["graph-outbox"])


class GraphOutboxIntent(BaseModel):
    kind: str
    created_at: str
    sequence: int
    payload: Dict[str, Any] = Field(default_factory=dict)


class GraphOutboxStatusResponse(BaseModel):
    backend: str
    enabled: bool
    pending: int
    head: List[GraphOutboxIntent] = Field(default_factory=list)


class GraphOutboxReplayResponse(BaseModel):
    backend: str
    enabled: bool
    processed: int
    pending_before: int
    pending_after: int


def _backend_name(store: Any) -> str:
    name = getattr(store.__class__, "__name__", "unknown")
    return str(name).lower()


@router.get("", response_model=GraphOutboxStatusResponse)
def graph_outbox_status(
    head: int = Query(0, ge=0, le=100, description="Return first N pending graph write intents"),
) -> GraphOutboxStatusResponse:
    store = get_store()
    if store is None:
        raise HTTPException(status_code=503, detail="memory store not initialised")

    enabled = bool(getattr(store, "graph_outbox_enabled", lambda: False)())
    pending = int(getattr(store, "graph_outbox_pending_count", lambda: 0)()) if enabled else 0
    head_rows = []
    if enabled and head > 0 and hasattr(store, "list_pending_graph_writes"):
        head_rows = list(store.list_pending_graph_writes(limit=head))

    return GraphOutboxStatusResponse(
        backend=_backend_name(store),
        enabled=enabled,
        pending=pending,
        head=[GraphOutboxIntent(**row) for row in head_rows],
    )


@router.post("/replay", response_model=GraphOutboxReplayResponse)
def replay_graph_outbox(
    limit: int = Query(0, ge=0, le=1000, description="Replay at most N pending graph write intents; 0 means all"),
) -> GraphOutboxReplayResponse:
    store = get_store()
    if store is None:
        raise HTTPException(status_code=503, detail="memory store not initialised")

    backend = _backend_name(store)
    enabled = bool(getattr(store, "graph_outbox_enabled", lambda: False)())
    if not enabled:
        return GraphOutboxReplayResponse(
            backend=backend,
            enabled=False,
            processed=0,
            pending_before=0,
            pending_after=0,
        )

    pending_before = int(getattr(store, "graph_outbox_pending_count", lambda: 0)())
    processed = int(getattr(store, "replay_pending_graph_writes", lambda limit=0: 0)(limit=limit))
    pending_after = int(getattr(store, "graph_outbox_pending_count", lambda: 0)())
    return GraphOutboxReplayResponse(
        backend=backend,
        enabled=True,
        processed=processed,
        pending_before=pending_before,
        pending_after=pending_after,
    )
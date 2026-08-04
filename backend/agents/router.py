from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, HTTPException, Request, Depends

from .config import AGENT_REGISTRY
from .processing.compute import ComputeAgent
from .llm.rag import LLMAgent
from .llm.integrated import IntegratedAgent
from .processing.online import OnlineAgent
from .llm.retriever import RetrieverAgent
from .processing.monitoring import MonitoringAgent
from .processing.analytics import AnalyticsAgent
from .processing.stoppage_predictor import StoppagePredictor, get_predictor

# [PROTOTYPE_LLM_MEMORY_V1] - Import memory router
from .memory.router import router as memory_router


def get_sessions() -> Dict[str, Dict[str, Any]]:
    """Get sessions from the main app.
    
    This uses a lazy import to avoid circular dependencies.
    The sessions dict is stored on app.state.sessions.
    """
    try:
        from ..app import app
        return app.state.sessions
    except Exception:
        return {}


router = APIRouter()

# [PROTOTYPE_LLM_MEMORY_V1] - Include memory router
router.include_router(memory_router)

# Simple in-process registry (name -> agent instance)
_AGENTS: Dict[str, Any] = {}
_AGENTS_STARTED: bool = False
_AGENTS_START_LOCK = asyncio.Lock()


# Keywords used for auto-routing to the new agents.
# Deliberately avoids overly generic words ("search", "find", "state")
# that would misclassify general queries.
_RETRIEVER_KEYWORDS = {"document", "knowledge", "manual", "retrieve", "lookup", "reference", "guide", "instruction", "datasheet"}
_MONITORING_KEYWORDS = {"status", "machine", "asset", "sensor", "spindle", "temperature", "connection", "health", "live", "reading", "vibration", "rpm", "feed"}
_ANALYTICS_KEYWORDS = {"trend", "chart", "history", "stats", "statistics", "frequency", "count", "summary", "cycle", "analyse", "analyze", "analytics", "report"}
_STOPPAGE_KEYWORDS = {"stoppage", "breakage", "stop", "predict", "prediction", "tool_break", "fault"}

# Generic terms that only count if combined with another keyword
_WEAK_RETRIEVER = {"search", "find"}
_WEAK_MONITORING = {"state"}
_WEAK_ANALYTICS = {"pattern"}


def _classify_query(action: Optional[str], args: Dict[str, Any]) -> Optional[str]:
    """Return a best-guess agent name from action/args, or None."""
    text = " ".join([
        action or "",
        str(args.get("query", "")),
        str(args.get("q", "")),
    ]).lower().split()
    tokens = set(text)

    # Strong keyword matches (single token sufficient)
    retriever_hits = tokens & _RETRIEVER_KEYWORDS
    monitoring_hits = tokens & _MONITORING_KEYWORDS
    analytics_hits = tokens & _ANALYTICS_KEYWORDS

    # Strong keywords match immediately; weak keywords are a softer signal
    if retriever_hits:
        return "retriever"
    if monitoring_hits:
        return "monitoring"
    if analytics_hits:
        return "analytics"
    # Weak keywords — lower-confidence classification fallback
    if tokens & _WEAK_RETRIEVER:
        return "retriever"
    if tokens & _WEAK_MONITORING:
        return "monitoring"
    if tokens & _WEAK_ANALYTICS:
        return "analytics"
    if tokens & _STOPPAGE_KEYWORDS:
        return "stoppage"
    return None


def register_default_agents():
    # lazy import/construct agents
    _AGENTS.setdefault("compute", ComputeAgent())
    _AGENTS.setdefault("llm.rag", LLMAgent())
    _AGENTS.setdefault("online", OnlineAgent())
    _AGENTS.setdefault("retriever", RetrieverAgent())
    _AGENTS.setdefault("monitoring", MonitoringAgent())
    _AGENTS.setdefault("analytics", AnalyticsAgent())
    # Stoppage predictor (lazy-loaded — only fails if no trained model)
    try:
        _AGENTS.setdefault("stoppage", get_predictor(gap_s=10.0))
    except Exception:
        import logging
        logging.getLogger(__name__).info("Stoppage predictor not available (no trained model)")

    # Agent I (2026-04-24): fan-out + synthesize across registered sub-agents.
    # ``get_agent`` is a late-bound lookup so sub-agents registered after this
    # call are still resolvable.
    _AGENTS.setdefault(
        "integrated",
        IntegratedAgent(get_agent=lambda name: _AGENTS.get(name)),
    )


async def _ensure_agents_started() -> None:
    """Initialize agents lazily.

    FastAPI's `router.on_event` is deprecated; this keeps equivalent behavior
    without relying on startup hooks.
    """
    global _AGENTS_STARTED
    if _AGENTS_STARTED:
        return

    async with _AGENTS_START_LOCK:
        if _AGENTS_STARTED:
            return

        register_default_agents()

        # Start any agents that expose an async start() method.
        # Await them so the agent is fully initialised before handling
        # requests (fixes B5: race condition).
        for name, agent in _AGENTS.items():
            start = getattr(agent, "start", None)
            if callable(start):
                try:
                    res = start()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    # Log but continue — agent degrades gracefully.
                    import logging
                    logging.getLogger(__name__).warning(
                        "Agent '%s' start() failed", name, exc_info=True,
                    )

        _AGENTS_STARTED = True


@router.post("/dispatch/{session_id}")
async def dispatch(session_id: str, request: Request):
    """Dispatch an agent request for a session.

    Body example:
      { "agent": "compute", "action": "amplitudes", "args": { ... }, "stream": false }

    Agent L phase 2 (2026-04-24): prefer the explicit per-action endpoints
    ``POST /agent/{agent_name}/{action}/{session_id}`` below. ``/dispatch`` is
    retained as a legacy alias for the UI and existing scripts.
    """
    body = await request.json()

    agent_name = body.get("agent", "auto")
    action = body.get("action")
    args = body.get("args", {}) or {}

    return await _run_agent(
        session_id=session_id,
        agent_name=agent_name,
        action=action,
        args=args,
    )


# ── Agent L phase 2: explicit per-action endpoint ────────────────────────────
# Each agent can be reached via POST /agent/{agent_name}/{action}/{session_id}
# with a JSON body of ``{"args": {...}}``. This replaces the string-matching
# in /dispatch (``action.startswith("fft")``) with an explicit, discoverable,
# OpenAPI-documented surface.

@router.post("/{agent_name}/{action}/{session_id}")
async def run_agent_explicit(
    agent_name: str,
    action: str,
    session_id: str,
    request: Request,
):
    """Explicit agent endpoint — no auto-classification.

    Body:
        { "args": { ... }, "stream": false }

    ``agent_name`` must be one of the registered agent keys (see
    ``register_default_agents``). Use the legacy ``/dispatch`` endpoint with
    ``{"agent": "auto"}`` to keep keyword-based routing.
    """
    # Guard reserved sub-paths that belong to /agent/memory/* so a stray
    # POST /agent/memory/events doesn't collide with this catch-all.
    _RESERVED_AGENTS = {"memory", "dispatch"}
    if agent_name in _RESERVED_AGENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Path conflicts with reserved agent namespace '{agent_name}'",
        )

    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        body = {}

    args = body.get("args", body) if isinstance(body, dict) else {}
    if not isinstance(args, dict):
        args = {}

    return await _run_agent(
        session_id=session_id,
        agent_name=agent_name,
        action=action,
        args=args,
    )


async def _run_agent(
    *,
    session_id: str,
    agent_name: str,
    action: Optional[str],
    args: Dict[str, Any],
) -> Dict[str, Any]:
    """Shared dispatch core used by both /dispatch and the explicit endpoint."""
    await _ensure_agents_started()

    if agent_name == "auto":
        # 1) Compute agent for FFT / signal-level actions
        if action and (action.startswith("fft") or action in ("amplitudes", "analyze")):
            agent_name = "compute"
        else:
            # 2) Try keyword-based routing to new agents
            classified = _classify_query(action, args)
            if classified:
                agent_name = classified
            else:
                # 3) Default fallback: LLM RAG
                agent_name = "llm.rag"

    agent = _AGENTS.get(agent_name)
    if agent is None:
        raise HTTPException(status_code=400, detail=f"Unknown agent '{agent_name}'")

    dispatch_id = str(uuid.uuid4())
    ctx = {"dispatch_id": dispatch_id, "session_id": session_id}

    # Inject live session object (if available) into args/context so agents can operate on it
    sessions = get_sessions()
    if sessions and session_id in sessions:
        ctx["session"] = sessions[session_id]
        # also prefer to put session in args for backward compatibility
        args.setdefault("session", sessions[session_id])

    try:
        result = agent.handle_request(
            session_id=session_id, action=action, args=args, context=ctx,
        )
        if asyncio.iscoroutine(result):
            result = await result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "ok": True,
        "dispatch_id": dispatch_id,
        "agent": agent_name,
        "result": result,
    }

import asyncio
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request

from .config import AGENT_REGISTRY
from .compute_agent import ComputeAgent
from .llm_rag import LLMAgent
from .online_agent import OnlineAgent

# Import the global sessions mapping from the app so we can inject session references
_SESSIONS = None

def _get_sessions():
    """Lazy import to avoid circular dependencies."""
    global _SESSIONS
    if _SESSIONS is None:
        try:
            # Try importing from app module in same package level
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            import app as _app_module
            _SESSIONS = getattr(_app_module, "sessions", {})
        except Exception:
            _SESSIONS = {}
    return _SESSIONS

router = APIRouter()

# Simple in-process registry (name -> agent instance)
_AGENTS: Dict[str, Any] = {}


def register_default_agents():
    # lazy import/construct agents
    _AGENTS.setdefault("compute", ComputeAgent())
    _AGENTS.setdefault("llm.rag", LLMAgent())
    _AGENTS.setdefault("online", OnlineAgent())


@router.on_event("startup")
async def _startup_agents():
    register_default_agents()
    # Start any agents that expose an async start() method
    for name, agent in _AGENTS.items():
        start = getattr(agent, "start", None)
        if callable(start):
            try:
                res = start()
                if asyncio.iscoroutine(res):
                    # schedule start but don't await here
                    asyncio.create_task(res)
            except Exception:
                # swallow errors to keep startup resilient
                pass


@router.post("/dispatch/{session_id}")
async def dispatch(session_id: str, request: Request):
    """Dispatch an agent request for a session.

    Body example:
      { "agent": "compute", "action": "amplitudes", "args": { ... }, "stream": false }
    """
    body = await request.json()
    agent_name = body.get("agent", "auto")
    action = body.get("action")
    args = body.get("args", {}) or {}
    stream = bool(body.get("stream", False))

    if agent_name == "auto":
        # simple fallback: prefer compute if action matches, else llm.rag
        if action and action.startswith("fft") or action in ("amplitudes", "analyze"):
            agent_name = "compute"
        else:
            agent_name = "llm.rag"

    agent = _AGENTS.get(agent_name)
    if agent is None:
        raise HTTPException(status_code=400, detail=f"Unknown agent '{agent_name}'")

    dispatch_id = str(uuid.uuid4())

    # create a small request context
    ctx = {"dispatch_id": dispatch_id, "session_id": session_id}

    # Inject live session object (if available) into args/context so agents can operate on it
    sessions = _get_sessions()
    if sessions and session_id in sessions:
        ctx["session"] = sessions[session_id]
        # also prefer to put session in args for backward compatibility
        args.setdefault("session", sessions[session_id])

    # For now we support only sync/async single-call handlers that return JSON-serializable
    try:
        result = agent.handle_request(session_id=session_id, action=action, args=args, context=ctx)
        if asyncio.iscoroutine(result):
            result = await result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"ok": True, "dispatch_id": dispatch_id, "agent": agent_name, "result": result}

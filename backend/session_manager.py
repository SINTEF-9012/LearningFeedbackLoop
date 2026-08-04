"""SessionManager — single mutation surface for per-session playback state.

Agent L (API hardening, 2026-04-24).

This class wraps the per-session dict (``app.state.sessions``) so that all
creation/mutation/teardown flows through one object. Today it is a thin
facade over the dict — preserving backward compatibility with the many
call sites that read ``session["key"]`` directly — but new code should use
the manager's methods so that future changes (persistence, metrics,
locking) have a single point of contact.

Design notes:
- ``app.state.sessions`` remains the canonical dict (read paths keep
  working unchanged). ``SessionManager`` keeps a reference to it.
- A module-level ``get_session_manager`` FastAPI dependency is provided
  for new routes; existing routes that use ``get_sessions_dict`` are
  unaffected.
- Auth / multi-tenant gates are deliberately out of scope (deferred per
  user direction 2026-04-23). A ``TODO(auth)`` marker is left for the
  eventual hook.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


# Agent Q (2026-04-24): LRU cap on resident sessions. Default is 0
# (unbounded, matching historical behaviour). Set via the
# ``LFL_MAX_SESSIONS`` environment variable or by passing
# ``max_sessions`` to ``SessionManager``. Evicted sessions are
# passed to an optional ``on_evict`` callback so the caller can
# tear down tasks/queues/ws subscribers; if no callback is supplied
# the session dict is simply dropped.
def _default_max_sessions() -> int:
    raw = os.environ.get("LFL_MAX_SESSIONS", "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        logger.warning("LFL_MAX_SESSIONS=%r is not an integer; ignoring", raw)
        return 0
    return max(0, value)


class SessionManager:
    """Thin façade over the per-session state dict.

    Backed directly by the dict on ``app.state.sessions`` so existing
    readers continue to work. New mutation sites should go through this
    class exclusively.

    Agent Q (2026-04-24): supports an optional LRU cap. When
    ``max_sessions > 0`` and a new session would exceed the cap, the
    least-recently-touched session is evicted first. Callers that
    need to tear down background tasks / ws queues on eviction should
    pass an ``on_evict`` callback.
    """

    def __init__(
        self,
        sessions: Dict[str, Dict[str, Any]],
        *,
        max_sessions: Optional[int] = None,
        on_evict: Optional[Any] = None,  # Callable[[str, Dict], None]
    ):
        self._sessions = sessions
        # Lock guarding insert/delete; individual session dict mutation is
        # still the caller's responsibility for now (existing code is not
        # thread-safe and adding a per-session lock would be a bigger
        # refactor). Agent Q may revisit.
        self._lock = asyncio.Lock()
        self._max_sessions = (
            int(max_sessions) if max_sessions is not None else _default_max_sessions()
        )
        self._on_evict = on_evict
        # Track recency independently of the dict so eviction is O(1).
        # The dict remains the canonical store; ``_lru`` only mirrors key order.
        self._lru: "OrderedDict[str, None]" = OrderedDict(
            (sid, None) for sid in self._sessions.keys()
        )

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    @property
    def raw(self) -> Dict[str, Dict[str, Any]]:
        """Return the underlying dict (backward-compat escape hatch)."""
        return self._sessions

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self._sessions.get(session_id)
        if session is not None:
            self._touch(session_id)
        return session

    def get_or_404(self, session_id: str, *, detail: str = "Session not found") -> Dict[str, Any]:
        s = self._sessions.get(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail=detail)
        self._touch(session_id)
        return s

    def ids(self) -> Iterator[str]:
        return iter(list(self._sessions.keys()))

    def __len__(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------------
    # LRU bookkeeping (Agent Q, 2026-04-24)
    # ------------------------------------------------------------------

    @property
    def max_sessions(self) -> int:
        """LRU cap. 0 means unbounded (historical default)."""
        return self._max_sessions

    def _touch(self, session_id: str) -> None:
        """Mark a session as most-recently-used."""
        if session_id in self._lru:
            self._lru.move_to_end(session_id)

    def _evict_one(self) -> Optional[str]:
        """Remove the least-recently-used session. Returns its id or None."""
        if not self._lru:
            return None
        oldest_id, _ = self._lru.popitem(last=False)
        state = self._sessions.pop(oldest_id, None)
        if state is not None and self._on_evict is not None:
            try:
                self._on_evict(oldest_id, state)
            except Exception:
                logger.exception(
                    "SessionManager: on_evict callback raised for %s", oldest_id
                )
        return oldest_id

    def evict_to_cap(self) -> List[str]:
        """Evict least-recently-used sessions until under the cap."""
        evicted: List[str] = []
        if self._max_sessions <= 0:
            return evicted
        while len(self._sessions) > self._max_sessions:
            ident = self._evict_one()
            if ident is None:
                break
            evicted.append(ident)
        return evicted

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, session_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a new session; overwrites any existing entry.

        Returns the stored dict. When an LRU cap is active, the
        least-recently-used session is evicted if inserting would
        exceed ``max_sessions``.
        """
        # TODO(auth): gate on tenant/role here once auth lands.
        already_present = session_id in self._sessions
        self._sessions[session_id] = state
        self._lru[session_id] = None
        self._lru.move_to_end(session_id)
        if not already_present:
            self.evict_to_cap()
        return state

    def remove(self, session_id: str) -> Optional[Dict[str, Any]]:
        self._lru.pop(session_id, None)
        return self._sessions.pop(session_id, None)

    def clear(self) -> None:
        self._lru.clear()
        self._sessions.clear()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_session_manager(request: Request) -> SessionManager:
    """FastAPI dependency returning the process-wide SessionManager.

    The manager is created once at app startup (see ``backend.app``) and
    stored on ``app.state.session_manager``. New routes should take this
    as a ``Depends(get_session_manager)`` parameter instead of reaching
    into ``request.app.state.sessions`` directly.
    """
    mgr = getattr(request.app.state, "session_manager", None)
    if mgr is None:
        # Fallback: construct one lazily if app.py hasn't wired it yet.
        # This keeps tests that patch ``app.state.sessions`` directly
        # working without modification.
        sessions = getattr(request.app.state, "sessions", None)
        if sessions is None:
            sessions = {}
            request.app.state.sessions = sessions
        mgr = SessionManager(sessions)
        request.app.state.session_manager = mgr
    return mgr

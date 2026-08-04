"""
MemoryStoreProtocol — Abstract interface for memory persistence backends.

All memory stores (SQLite, Neo4j, etc.) must conform to this protocol so that
the rest of the system (orchestrator, scorer, feedback handler, retriever) can
work with any backend transparently.

Key design decisions:
- ``update()`` accepts a **dict** (not **kwargs) so the signature is uniform
  across implementations and easy to forward through async boundaries.
- Methods that only read never require a transaction; implementations may
  use read-replicas or caching at their discretion.
- Optional methods (feedback, traces) raise ``NotImplementedError`` by default
  so lightweight test doubles can omit them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, runtime_checkable, Protocol

from ..core.schemas import Memory, MemoryQuery


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    """Structural (duck-typed) interface for memory persistence backends."""

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def store(self, memory: Memory) -> str:
        """Persist a memory (insert or upsert). Return the memory ID."""
        ...

    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID, or ``None``."""
        ...

    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Apply *updates* dict to an existing memory. Return success flag.

        Implementations must handle nested metadata merging (e.g. the key
        ``"metadata"`` containing a sub-dict should be merged, not replaced).
        """
        ...

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID. Return ``True`` if it existed."""
        ...

    # ------------------------------------------------------------------
    # Queries / listing
    # ------------------------------------------------------------------

    def list_all(self, limit: int = 1000, visibility: str = "active") -> List[Memory]:
        """Return recent memories, newest first."""
        ...

    def list_by_session(self, session_id: str, limit: int = 100) -> List[Memory]:
        """Return memories for a session, newest first."""
        ...

    def search(
        self,
        text_query: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        session_id: Optional[str] = None,
    ) -> List[Memory]:
        """Simple text / time-range / session search."""
        ...

    def count(self) -> int:
        """Total number of stored memories."""
        ...

    def stats(self) -> Dict[str, Any]:
        """Backend-specific statistics dict."""
        ...

    def clear(self) -> None:
        """Delete **all** memories. Use with caution."""
        ...

    # ------------------------------------------------------------------
    # Feedback events (append-only audit log)
    # ------------------------------------------------------------------

    def add_feedback_event(
        self,
        *,
        memory_id: Optional[str],
        action: str,
        user_id: str,
        timestamp: Optional[str] = None,
        pattern_keys: Optional[List[str]] = None,
        context_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
    ) -> str:
        """Append a feedback event. Return the event ID."""
        ...

    def list_feedback_events(self, memory_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """List feedback events for a memory."""
        ...

    def list_pattern_keys_with_feedback(self) -> List[str]:
        """Return distinct pattern keys that have received feedback."""
        ...

    def get_feedback_counts(
        self,
        *,
        pattern_key: str,
        context_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Return ``(confirm_count, dismiss_count)`` for a pattern/context."""
        ...

    # ------------------------------------------------------------------
    # Traces (scoring / retrieval audit)
    # ------------------------------------------------------------------

    def add_trace(
        self,
        *,
        session_id: Optional[str],
        memory_id: Optional[str],
        trace_type: str,
        payload: Dict[str, Any],
        created_at: Optional[str] = None,
    ) -> str:
        """Append a trace record. Return the trace ID."""
        ...

    def list_traces(
        self,
        *,
        memory_id: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_type: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """List trace records matching the given filters."""
        ...

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release any resources (connections, file handles)."""
        ...

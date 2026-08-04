"""In-memory implementation of the memory store.

Extracted from ``memory/orchestrator.py`` (Phase 2), where a 400-line storage
implementation had no business living: storage belongs beside the SQLite
(:mod:`.store`) and Neo4j (:mod:`.neo4j_store`) backends.

It adapts the orchestrator's plain ``{memory_id: Memory}`` dict so the retriever
and feedback handler can query memories when no external store is configured —
useful for tests and for running without a database. It is not durable: nothing
here survives a restart.

**This is a partial store.** It covers what the retriever and feedback handler
actually call, but does not implement the whole of
:class:`~backend.agents.storage.protocol.MemoryStoreProtocol` — ``close``,
``list_by_session``, ``search`` and ``stats`` are absent. That predates the
move and is why it is an *adapter* rather than a store. Prefer a real store in
any deployment.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..core.schemas import Memory


class InMemoryStoreAdapter:
    """
    Adapter to make the orchestrator's in-memory dict look like a MemoryStore.
    
    This allows the retriever and feedback handler to access memories
    stored in the orchestrator's `_memories` dict when no external store
    is provided.
    
    [PROTOTYPE_LLM_MEMORY_V1] - This is a temporary solution.
    Production should always use a real MemoryStore.
    """
    
    def __init__(self, memories_dict: Dict[str, Memory]):
        """
        Args:
            memories_dict: Reference to orchestrator's _memories dict
        """
        self._memories = memories_dict

        # Minimal in-memory feedback + trace persistence for tests.
        self._feedback_events: List[Dict[str, Any]] = []
        self._traces: List[Dict[str, Any]] = []
        self._doc_link_feedback_lock = threading.Lock()
    
    def get(self, memory_id: str) -> Optional[Memory]:
        """Get memory by ID."""
        return self._memories.get(memory_id)
    
    def store(self, memory: Memory) -> str:
        """Store a memory and return its ID."""
        self._memories[memory.id] = memory
        return memory.id
    
    def save(self, memory: Memory) -> str:
        """Alias for store()."""
        return self.store(memory)
    
    def list_all(self, limit: Optional[int] = None) -> List[Memory]:
        """List all memories."""
        memories = list(self._memories.values())
        if limit:
            memories = memories[:limit]
        return memories
    
    def query(
        self,
        pattern_keys: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Memory]:
        """Query memories with optional filters."""
        memories = list(self._memories.values())
        
        if session_id:
            memories = [m for m in memories if m.session_id == session_id]
        
        if pattern_keys:
            pattern_set = set(pattern_keys)
            memories = [
                m for m in memories 
                if any(pk.key in pattern_set for pk in m.pattern_keys)
            ]
        
        return memories[:limit]
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update memory fields."""
        memory = self._memories.get(memory_id)
        if not memory:
            return False
        
        # Handle nested updates like "metadata.user_confirmed"
        for key, value in updates.items():
            if "." in key:
                # Nested update (e.g., "metadata.user_confirmed")
                parts = key.split(".")
                if parts[0] == "metadata" and memory.metadata is not None:
                    memory.metadata[parts[1]] = value
            else:
                # Direct attribute update
                if hasattr(memory, key):
                    setattr(memory, key, value)
        
        return True

    # ----------------------------
    # Feedback event compatibility
    # ----------------------------

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
        weight: Optional[float] = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        try:
            normalized_weight = float(weight) if weight is not None else 1.0
        except (TypeError, ValueError):
            normalized_weight = 1.0
        self._feedback_events.append(
            {
                "id": event_id,
                "memory_id": memory_id,
                "action": str(action),
                "user_id": str(user_id),
                "timestamp": ts,
                "context_key": context_key,
                "context": dict(context or {}),
                "data": dict(data or {}),
                "pattern_keys": list(pattern_keys or []),
                "weight": normalized_weight,
            }
        )
        return event_id

    def list_feedback_events(self, memory_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        events = [e for e in self._feedback_events if e.get("memory_id") == memory_id]
        events.sort(key=lambda e: str(e.get("timestamp") or ""))
        return events[: int(limit)]

    def list_pattern_keys_with_feedback(self, user_id: Optional[str] = None) -> List[str]:
        keys = set()
        for e in self._feedback_events:
            if user_id is not None and e.get("user_id") != str(user_id):
                continue
            for pk in e.get("pattern_keys") or []:
                if pk:
                    keys.add(str(pk))
        return sorted(keys)

    def get_feedback_counts(
        self,
        *,
        pattern_key: str,
        context_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> tuple[int, int]:
        pk = str(pattern_key).strip()
        confirm = 0
        dismiss = 0
        for e in self._feedback_events:
            if user_id is not None and e.get("user_id") != str(user_id):
                continue
            if context_key is None:
                pass
            else:
                if e.get("context_key") != context_key:
                    continue
            if pk not in (e.get("pattern_keys") or []):
                continue
            action = str(e.get("action") or "").lower()
            try:
                weight = float(e.get("weight", 1.0) or 1.0)
            except (TypeError, ValueError):
                weight = 1.0
            if action in {"confirm", "missed"}:
                confirm += weight
            elif action == "dismiss":
                dismiss += weight
        return (confirm, dismiss)

    # ----------------------------
    # Trace compatibility
    # ----------------------------

    def add_trace(
        self,
        *,
        session_id: Optional[str],
        memory_id: Optional[str],
        trace_type: str,
        payload: Dict[str, Any],
        created_at: Optional[str] = None,
    ) -> str:
        trace_id = str(uuid.uuid4())
        ts = created_at or datetime.now(timezone.utc).isoformat()
        self._traces.append(
            {
                "id": trace_id,
                "session_id": session_id,
                "memory_id": memory_id,
                "trace_type": str(trace_type),
                "created_at": ts,
                "payload": dict(payload or {}),
            }
        )
        return trace_id

    def list_traces(
        self,
        *,
        memory_id: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_type: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        traces = list(self._traces)
        if memory_id is not None:
            traces = [t for t in traces if t.get("memory_id") == memory_id]
        if session_id is not None:
            traces = [t for t in traces if t.get("session_id") == session_id]
        if trace_type is not None:
            traces = [t for t in traces if t.get("trace_type") == trace_type]
        traces.sort(key=lambda t: str(t.get("created_at") or ""))
        return traces[: int(limit)]
    
    # ----------------------------
    # Co-occurrence compatibility
    # ----------------------------

    def upsert_co_occurrence(
        self,
        pattern_key_a: str,
        pattern_key_b: str,
        session_id: str,
    ) -> None:
        """Track co-occurrence in memory (mirror of store method)."""
        if pattern_key_a == pattern_key_b:
            return
        a, b = sorted([pattern_key_a, pattern_key_b])
        key = f"{a}|{b}"
        if not hasattr(self, "_co_occurrence"):
            self._co_occurrence: Dict[str, int] = {}
        self._co_occurrence[key] = self._co_occurrence.get(key, 0) + 1

    def propagate_prior_update(
        self,
        pattern_key: str,
        delta: float,
        decay: float = 0.3,
        max_hops: int = 1,
    ) -> List[Any]:
        """No-op for in-memory adapter (no persistent priors)."""
        return []

    def get_co_occurrence_edges(self) -> List[Dict[str, Any]]:
        """Return in-memory co-occurrence edges."""
        if not hasattr(self, "_co_occurrence"):
            return []
        edges = []
        for key, weight in self._co_occurrence.items():
            a, b = key.split("|", 1)
            edges.append({"source": a, "target": b, "weight": weight})
        return edges

    def get_co_occurring_patterns(
        self,
        pattern_key: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the top-k co-occurring patterns for *pattern_key*."""
        if not hasattr(self, "_co_occurrence"):
            return []
        pk = str(pattern_key).strip()
        matches: List[Dict[str, Any]] = []
        for key, weight in self._co_occurrence.items():
            a, b = key.split("|", 1)
            if a == pk or b == pk:
                matches.append({"source": a, "target": b, "weight": weight})
        matches.sort(key=lambda x: x["weight"], reverse=True)
        return matches[:top_k]

    def persist_doc_links(
        self,
        *,
        memory_id: str,
        pattern_keys: List[str],
        doc_links: List[Dict[str, Any]],
    ) -> int:
        memory = self._memories.get(memory_id)
        if memory is None or not doc_links:
            return 0
        memory.metadata = dict(memory.metadata or {})
        memory.metadata["doc_links"] = list(doc_links)
        return len(doc_links)

    @staticmethod
    def _normalize_doc_link(link: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(link or {})
        item["query_used"] = str(item.get("query_used") or "")
        item["pattern_key"] = str(item.get("pattern_key") or "")
        item["doc_feedback"] = str(item.get("doc_feedback") or "").strip() or None
        item["helpful_count"] = int(item.get("helpful_count") or 0)
        item["not_helpful_count"] = int(item.get("not_helpful_count") or 0)
        item["feedback_score"] = float(item.get("feedback_score") or 0.0)
        evidence_entities = item.get("evidence_entities") or []
        item["evidence_entities"] = list(evidence_entities) if isinstance(evidence_entities, list) else []
        return item

    @classmethod
    def _sort_doc_links(cls, doc_links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            [cls._normalize_doc_link(link) for link in doc_links if isinstance(link, dict)],
            key=lambda item: (
                float(item.get("feedback_score") or 0.0),
                float(item.get("score") or 0.0),
            ),
            reverse=True,
        )

    def get_doc_links(
        self,
        memory_id: str,
        *,
        score_floor: float = 0.0,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        memory = self._memories.get(memory_id)
        if memory is None:
            return []
        doc_links = self._sort_doc_links(
            [
                dict(link)
                for link in (dict(memory.metadata or {}).get("doc_links") or [])
                if isinstance(link, dict)
            ]
        )
        filtered: List[Dict[str, Any]] = []
        for link in doc_links:
            try:
                score = float(link.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if score >= float(score_floor):
                filtered.append(link)
        filtered.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        if limit > 0:
            filtered = filtered[: int(limit)]
        return filtered

    def set_doc_link_feedback(
        self,
        *,
        memory_id: str,
        doc_id: str,
        feedback: str,
        user_id: str,
        reason: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        feedback_value = str(feedback or "").strip().lower()
        if feedback_value not in {"helpful", "not_helpful"}:
            raise ValueError("feedback must be 'helpful' or 'not_helpful'")

        with self._doc_link_feedback_lock:
            memory = self._memories.get(memory_id)
            if memory is None:
                return None
            metadata = dict(memory.metadata or {})
            doc_links = [
                dict(link)
                for link in (metadata.get("doc_links") or [])
                if isinstance(link, dict)
            ]
            updated_link: Optional[Dict[str, Any]] = None
            for index, link in enumerate(doc_links):
                if str(link.get("id") or "") != str(doc_id):
                    continue
                current = self._normalize_doc_link(link)
                if feedback_value == "helpful":
                    current["helpful_count"] = int(current.get("helpful_count") or 0) + 1
                else:
                    current["not_helpful_count"] = int(current.get("not_helpful_count") or 0) + 1
                current["feedback_score"] = float(current.get("helpful_count") or 0) - float(current.get("not_helpful_count") or 0)
                current["doc_feedback"] = feedback_value
                current["doc_feedback_user_id"] = str(user_id or "")
                current["doc_feedback_reason"] = reason
                current["doc_feedback_updated_at"] = datetime.now(timezone.utc).isoformat()
                doc_links[index] = current
                updated_link = current
                break
            if updated_link is None:
                return None
            metadata["doc_links"] = self._sort_doc_links(doc_links)
            memory.metadata = metadata
            return dict(updated_link)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all memories."""
        self._memories.clear()
    
    def count(self) -> int:
        """Count total memories."""
        return len(self._memories)

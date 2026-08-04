"""
Memory Store - SQLite-based persistent storage for LLM-RAG agent memories.

This module provides the core storage layer for the memory system, handling
CRUD operations, pattern key indexing, and optional ANN/embedding indices.
"""

from __future__ import annotations

import sqlite3
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

import numpy as np

from ..core.schemas import (
    Memory,
    PatternKey,
    TimeRange,
    NumericMetrics,
    MetricsSummary,  # Alias for NumericMetrics
    MemoryProvenance,
)
from ..patterns.signatures import infer_pattern_kind, normalize_signature_key
from .pattern_index import PatternIndex
from .ann_index import ANNIndex


class MemoryStore:
    """
    Hybrid memory store with SQLite backend and optional index layers.
    
    Architecture:
    - SQLite: Main storage for Memory records (JSON-serialized)
    - PatternIndex: Inverted index for fast pattern key lookups
    - ANNIndex: FAISS-based approximate nearest neighbor for numeric metrics
    
    Thread Safety:
    - SQLite operations are serialized via check_same_thread=False
    - Indices are rebuilt on startup from persisted state
    """
    
    DB_VERSION = 5  # v5: adds weighted feedback events for fractional learning signals

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
    
    def __init__(
        self,
        db_path: str = "memories.db",
        enable_ann: bool = True,
        enable_embeddings: bool = False,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the memory store.
        
        Args:
            db_path: Path to SQLite database file
            enable_ann: Whether to enable FAISS ANN index for numeric metrics
            enable_embeddings: Whether to enable text embedding index
            embedding_model: Sentence-transformer model name (if enable_embeddings)
        """
        self.db_path = Path(db_path)
        self.enable_ann = enable_ann
        self.enable_embeddings = enable_embeddings
        self.embedding_model = embedding_model or "all-MiniLM-L6-v2"
        self._doc_link_feedback_lock = threading.Lock()

        # ':memory:' SQLite databases are per-connection. Tests expect a working
        # in-memory database with tables available throughout the store lifetime.
        self._in_memory = str(self.db_path) == ":memory:"
        self._conn: Optional[sqlite3.Connection] = None
        if self._in_memory:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        
        # Initialize indices
        self.pattern_index = PatternIndex()
        self.ann_index: Optional[ANNIndex] = None
        
        if self.enable_ann:
            self.ann_index = ANNIndex()
        
        # Initialize database
        self._init_db()
        self._rebuild_indices()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        if self._in_memory:
            assert self._conn is not None
            yield self._conn
            return

        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def close(self) -> None:
        """Close any long-lived resources held by this store."""
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def _serialize_time_range(self, time_range: Any) -> Optional[str]:
        if time_range is None:
            return None
        if isinstance(time_range, tuple) and len(time_range) == 2:
            return json.dumps([float(time_range[0]), float(time_range[1])])
        if isinstance(time_range, list) and len(time_range) == 2:
            return json.dumps([float(time_range[0]), float(time_range[1])])
        if isinstance(time_range, TimeRange):
            return time_range.model_dump_json()

    @staticmethod
    def _serialize_pattern_key(pk: PatternKey) -> Dict[str, Any]:
        data = pk.model_dump()
        additional = dict(data.get("additional") or {})
        pattern_type = pk.pattern_type.value if getattr(pk, "pattern_type", None) else None
        additional.setdefault("kind", infer_pattern_kind(pk.key, pattern_type))
        data["additional"] = additional
        return data
        # Best-effort: try pydantic-like dict
        if isinstance(time_range, dict):
            return json.dumps(time_range)
        return None

    def _deserialize_time_range(self, raw: Optional[str]) -> Any:
        if not raw:
            return (0.0, 0.0)
        try:
            parsed = json.loads(raw)
        except Exception:
            return (0.0, 0.0)
        if isinstance(parsed, list) and len(parsed) == 2:
            return (float(parsed[0]), float(parsed[1]))
        if isinstance(parsed, dict):
            # If it looks like a full TimeRange, keep it as TimeRange.
            if {"i0", "i1", "t0", "t1", "fs"}.issubset(parsed.keys()):
                return TimeRange(**parsed)
            # Otherwise treat as simple time span.
            if "t0" in parsed and "t1" in parsed:
                return (float(parsed["t0"]), float(parsed["t1"]))
        return (0.0, 0.0)

    @staticmethod
    def _pattern_match_key(key: str) -> str:
        raw = str(key or "").strip()
        canonical = normalize_signature_key(raw)
        if canonical.startswith("signature:"):
            return canonical
        return raw.lower()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Main memories table - schema v2 with pattern_keys list
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    annotation_text TEXT NOT NULL,
                    pattern_keys_json TEXT NOT NULL,
                    metrics_json TEXT,
                    time_range_json TEXT,
                    channels_json TEXT,
                    tags_json TEXT,
                    label TEXT,
                    provenance_json TEXT,
                    metadata_json TEXT,
                    visibility TEXT DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    created_by TEXT DEFAULT 'operator',
                    embedding_vector BLOB
                )
            """)
            
            # Migration: check for old schema and migrate
            cursor.execute("PRAGMA table_info(memories)")
            columns = {row[1] for row in cursor.fetchall()}
            
            # If old schema (has 'content' column), migrate to new schema
            if 'content' in columns and 'annotation_text' not in columns:
                self._migrate_v1_to_v2(conn)
            
            # Indices for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_session 
                ON memories(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_visibility 
                ON memories(visibility)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_created 
                ON memories(created_at)
            """)
            
            # Pattern key components table for fast lookups
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_key_components (
                    memory_id TEXT NOT NULL,
                    component_type TEXT NOT NULL,
                    component_value TEXT NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pattern_components
                ON pattern_key_components(component_type, component_value)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_pattern_edges (
                    memory_id TEXT NOT NULL,
                    pattern_key TEXT NOT NULL,
                    strength REAL NOT NULL DEFAULT 1.0,
                    source_metric TEXT,
                    kind TEXT,
                    PRIMARY KEY (memory_id, pattern_key),
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_pattern_edges_memory
                ON memory_pattern_edges(memory_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_pattern_edges_pattern
                ON memory_pattern_edges(pattern_key)
            """)
            
            # Version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            cursor.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (self.DB_VERSION,)
            )

            # -----------------------------------------------------------------
            # Feedback event log (append-only) for learning/auditability
            # -----------------------------------------------------------------
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT,
                    action TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    context_key TEXT,
                    context_json TEXT,
                    data_json TEXT
                )
            """)

            cursor.execute("PRAGMA table_info(feedback_events)")
            feedback_columns = {str(row[1]) for row in cursor.fetchall()}
            if "weight" not in feedback_columns:
                cursor.execute(
                    "ALTER TABLE feedback_events ADD COLUMN weight REAL NOT NULL DEFAULT 1.0"
                )

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_event_patterns (
                    event_id TEXT NOT NULL,
                    pattern_key TEXT NOT NULL,
                    FOREIGN KEY (event_id) REFERENCES feedback_events(id) ON DELETE CASCADE
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_events_memory
                ON feedback_events(memory_id, timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_patterns
                ON feedback_event_patterns(pattern_key)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_events_context_action
                ON feedback_events(context_key, action)
            """)

            # -----------------------------------------------------------------
            # Trace persistence (scoring/retrieval)
            # -----------------------------------------------------------------
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_traces (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    memory_id TEXT,
                    trace_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_traces_memory
                ON memory_traces(memory_id, created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_traces_session
                ON memory_traces(session_id, created_at)
            """)

            # -----------------------------------------------------------------
            # Co-occurrence edges (pattern-pair weights)
            # -----------------------------------------------------------------
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS co_occurrence (
                    pattern_a TEXT NOT NULL,
                    pattern_b TEXT NOT NULL,
                    weight INTEGER NOT NULL DEFAULT 1,
                    first_session TEXT,
                    last_session TEXT,
                    PRIMARY KEY (pattern_a, pattern_b)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_co_occurrence_a
                ON co_occurrence(pattern_a)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_co_occurrence_b
                ON co_occurrence(pattern_b)
            """)

            conn.commit()

    # ------------------------------------------------------------------
    # Feedback events (learning signal)
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
        """Append a feedback event and return its ID.

        This is intentionally append-only to preserve audit history.
        """
        event_id = str(uuid.uuid4())
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        pats = [str(p).strip() for p in (pattern_keys or []) if str(p).strip()]
        try:
            feedback_weight = max(0.0, float(weight))
        except (TypeError, ValueError):
            feedback_weight = 1.0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO feedback_events
                (id, memory_id, action, user_id, timestamp, weight, context_key, context_json, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    memory_id,
                    str(action),
                    str(user_id),
                    ts,
                    feedback_weight,
                    context_key,
                    json.dumps(context or {}) if context is not None else None,
                    json.dumps(data or {}) if data is not None else None,
                ),
            )
            for pk in pats:
                cursor.execute(
                    "INSERT INTO feedback_event_patterns (event_id, pattern_key) VALUES (?, ?)",
                    (event_id, pk),
                )
            conn.commit()
        return event_id

    def list_feedback_events(self, memory_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, memory_id, action, user_id, timestamp, weight, context_key, context_json, data_json
                FROM feedback_events
                WHERE memory_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (memory_id, int(limit)),
            )
            rows = [dict(r) for r in cursor.fetchall()]

            # Attach patterns
            for row in rows:
                cursor.execute(
                    "SELECT pattern_key FROM feedback_event_patterns WHERE event_id = ?",
                    (row["id"],),
                )
                row["pattern_keys"] = [r[0] for r in cursor.fetchall()]
                row["context"] = json.loads(row["context_json"]) if row.get("context_json") else {}
                row["data"] = json.loads(row["data_json"]) if row.get("data_json") else {}
                row["weight"] = float(row.get("weight") or 1.0)
                if isinstance(row["data"], dict) and "comment" in row["data"]:
                    row["comment"] = row["data"].get("comment")
                row.pop("context_json", None)
                row.pop("data_json", None)
        return rows

    def get_similar_with_resolution(
        self,
        memory_id: str,
        k: int = 5,
        *,
        include_dismissed: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return similar memories plus shared-pattern and feedback context.

        SQLite mirror of the Neo4j history surface. Similarity is currently
        based on shared pattern keys/signatures rather than an explicit
        relationship edge.
        """
        query_memory = self.get(memory_id)
        if query_memory is None:
            return []

        query_edge_map = self._get_memory_pattern_edges(memory_id)
        query_patterns_by_key: Dict[str, PatternKey] = {}
        for pk in query_memory.pattern_keys:
            query_patterns_by_key.setdefault(self._pattern_match_key(pk.key), pk)

        if not query_patterns_by_key:
            return []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories WHERE id != ? ORDER BY created_at DESC", (memory_id,))
            rows = cursor.fetchall()

        enriched: List[Dict[str, Any]] = []
        for row in rows:
            candidate = self._deserialize_memory(row)
            candidate_edge_map = self._get_memory_pattern_edges(candidate.id or "")
            candidate_patterns_by_key: Dict[str, PatternKey] = {}
            for pk in candidate.pattern_keys:
                candidate_patterns_by_key.setdefault(self._pattern_match_key(pk.key), pk)

            shared_match_keys = [
                key for key in query_patterns_by_key.keys()
                if key in candidate_patterns_by_key
            ]
            if not shared_match_keys:
                continue

            feedbacks = self.list_feedback_events(candidate.id or "", limit=200)
            feedbacks = sorted(
                feedbacks,
                key=lambda fb: str(fb.get("timestamp") or ""),
            )
            confirm_count = 0.0
            dismiss_count = 0.0
            last_action: Optional[str] = None
            last_comment: Optional[str] = None
            last_action_ts: Optional[str] = None
            for fb in feedbacks:
                action = str(fb.get("action") or "")
                feedback_weight = float(fb.get("weight") or 1.0)
                if action == "confirm":
                    confirm_count += feedback_weight
                elif action == "dismiss":
                    dismiss_count += feedback_weight
                if action:
                    last_action = action
                comment = fb.get("comment") or (fb.get("data") or {}).get("comment")
                if isinstance(comment, str) and comment.strip():
                    last_comment = comment.strip()
                timestamp = fb.get("timestamp")
                if isinstance(timestamp, str) and timestamp.strip():
                    last_action_ts = timestamp

            if not include_dismissed and last_action == "dismiss" and confirm_count == 0:
                continue

            shared_pattern_keys: List[str] = []
            shared_pattern_details: List[Dict[str, Any]] = []
            for match_key in shared_match_keys:
                query_pk = query_patterns_by_key[match_key]
                candidate_pk = candidate_patterns_by_key[match_key]
                rendered_key = match_key if match_key.startswith("signature:") else query_pk.key
                query_edge = query_edge_map.get(match_key, {})
                candidate_edge = candidate_edge_map.get(match_key, {})
                shared_pattern_keys.append(rendered_key)
                shared_pattern_details.append({
                    "key": rendered_key,
                    "query_strength": float(query_edge.get("strength", getattr(query_pk, "confidence", 1.0) or 1.0)),
                    "query_source_metric": query_edge.get("source_metric", getattr(query_pk, "source_metric", None)),
                    "candidate_strength": float(candidate_edge.get("strength", getattr(candidate_pk, "confidence", 1.0) or 1.0)),
                    "candidate_source_metric": candidate_edge.get("source_metric", getattr(candidate_pk, "source_metric", None)),
                })

            enriched.append({
                "id": candidate.id,
                "session_id": candidate.session_id,
                "created_at": candidate.created_at.isoformat() if candidate.created_at else None,
                "annotation_text": candidate.annotation_text,
                "label": candidate.label,
                "shared_pattern_keys": shared_pattern_keys,
                "shared_pattern_details": shared_pattern_details,
                "feedback": {
                    "confirm_count": confirm_count,
                    "dismiss_count": dismiss_count,
                    "last_action": last_action,
                    "last_comment": last_comment,
                    "last_action_ts": last_action_ts,
                },
            })

        enriched.sort(
            key=lambda entry: (
                float(entry["feedback"]["confirm_count"]),
                len(entry["shared_pattern_keys"]),
                entry.get("created_at") or "",
            ),
            reverse=True,
        )
        return enriched[: int(k)]

    def _get_memory_pattern_edges(self, memory_id: str) -> Dict[str, Dict[str, Any]]:
        if not memory_id:
            return {}

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT pattern_key, strength, source_metric, kind
                FROM memory_pattern_edges
                WHERE memory_id = ?
                """,
                (memory_id,),
            )
            rows = cursor.fetchall()

        edge_map: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            match_key = self._pattern_match_key(row[0])
            edge_map[match_key] = {
                "strength": float(row[1]),
                "source_metric": row[2],
                "kind": row[3],
            }
        return edge_map

    def list_pattern_keys_with_feedback(self, user_id: Optional[str] = None) -> List[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if user_id is None:
                cursor.execute("SELECT DISTINCT pattern_key FROM feedback_event_patterns")
            else:
                cursor.execute(
                    """
                    SELECT DISTINCT p.pattern_key
                    FROM feedback_event_patterns p
                    JOIN feedback_events e ON e.id = p.event_id
                    WHERE e.user_id = ?
                    """,
                    (str(user_id),),
                )
            return [r[0] for r in cursor.fetchall()]

    def get_feedback_counts(
        self,
        *,
        pattern_key: str,
        context_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Return (confirm_count, dismiss_count) for a pattern in a context."""
        pattern_key = str(pattern_key).strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if context_key is None and user_id is None:
                cursor.execute(
                    """
                    SELECT e.action, SUM(COALESCE(e.weight, 1.0))
                    FROM feedback_events e
                    JOIN feedback_event_patterns p ON p.event_id = e.id
                    WHERE p.pattern_key = ?
                    GROUP BY e.action
                    """,
                    (pattern_key,),
                )
            elif context_key is None:
                cursor.execute(
                    """
                    SELECT e.action, SUM(COALESCE(e.weight, 1.0))
                    FROM feedback_events e
                    JOIN feedback_event_patterns p ON p.event_id = e.id
                    WHERE p.pattern_key = ? AND e.user_id = ?
                    GROUP BY e.action
                    """,
                    (pattern_key, str(user_id)),
                )
            elif user_id is None:
                cursor.execute(
                    """
                    SELECT e.action, SUM(COALESCE(e.weight, 1.0))
                    FROM feedback_events e
                    JOIN feedback_event_patterns p ON p.event_id = e.id
                    WHERE p.pattern_key = ? AND e.context_key = ?
                    GROUP BY e.action
                    """,
                    (pattern_key, str(context_key)),
                )
            else:
                cursor.execute(
                    """
                    SELECT e.action, SUM(COALESCE(e.weight, 1.0))
                    FROM feedback_events e
                    JOIN feedback_event_patterns p ON p.event_id = e.id
                    WHERE p.pattern_key = ? AND e.context_key = ? AND e.user_id = ?
                    GROUP BY e.action
                    """,
                    (pattern_key, str(context_key), str(user_id)),
                )
            counts = {str(row[0]).lower(): float(row[1]) for row in cursor.fetchall()}
        confirm = counts.get("confirm", 0) + counts.get("missed", 0)
        return (confirm, counts.get("dismiss", 0))

    # ------------------------------------------------------------------
    # Trace persistence (auditability)
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
        trace_id = str(uuid.uuid4())
        ts = created_at or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memory_traces (id, session_id, memory_id, trace_type, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    session_id,
                    memory_id,
                    str(trace_type),
                    ts,
                    json.dumps(payload or {}, sort_keys=True),
                ),
            )
            conn.commit()
        return trace_id

    def list_traces(
        self,
        *,
        memory_id: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_type: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT id, session_id, memory_id, trace_type, created_at, payload_json FROM memory_traces WHERE 1=1"
        params: List[Any] = []
        if memory_id is not None:
            sql += " AND memory_id = ?"
            params.append(memory_id)
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if trace_type is not None:
            sql += " AND trace_type = ?"
            params.append(trace_type)
        sql += " ORDER BY created_at ASC LIMIT ?"
        params.append(int(limit))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = [dict(r) for r in cursor.fetchall()]

        for row in rows:
            row["payload"] = json.loads(row.get("payload_json") or "{}")
            row.pop("payload_json", None)
        return rows

    # ------------------------------------------------------------------
    # Co-occurrence edges (Gap #4 — parity with Neo4jMemoryStore)
    # ------------------------------------------------------------------

    def upsert_co_occurrence(
        self,
        pattern_key_a: str,
        pattern_key_b: str,
        session_id: str,
    ) -> None:
        """Increment the co-occurrence weight between two patterns.

        Matches the ``Neo4jMemoryStore.upsert_co_occurrence`` signature so
        the orchestrator can call this uniformly regardless of backend.
        """
        if pattern_key_a == pattern_key_b:
            return
        # Canonical ordering to avoid duplicate edges
        a, b = sorted([pattern_key_a, pattern_key_b])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO co_occurrence (pattern_a, pattern_b, weight, first_session, last_session) "
                "VALUES (?, ?, 1, ?, ?) "
                "ON CONFLICT(pattern_a, pattern_b) DO UPDATE SET "
                "weight = weight + 1, last_session = excluded.last_session",
                (a, b, session_id, session_id),
            )
            conn.commit()

    def propagate_prior_update(
        self,
        pattern_key: str,
        delta: float,
        decay: float = 0.3,
        max_hops: int = 1,
    ) -> List[Tuple[str, float]]:
        """Propagate a prior delta to co-occurring patterns.

        Reads co-occurrence edges from the ``co_occurrence`` table and
        applies a decayed delta to each neighbour's prior via the
        in-memory ``PatternIndex``.

        Returns list of ``(neighbor_key, applied_delta)`` for audit.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT pattern_a, pattern_b, weight FROM co_occurrence "
                "WHERE pattern_a = ? OR pattern_b = ? "
                "ORDER BY weight DESC LIMIT 20",
                (pattern_key, pattern_key),
            )
            rows = cursor.fetchall()

        updates: List[Tuple[str, float]] = []
        for row in rows:
            neighbor = row[1] if row[0] == pattern_key else row[0]
            weight = float(row[2])
            applied = delta * decay * min(weight / 10.0, 1.0)
            if abs(applied) < 1e-6:
                continue
            # Update prior in the in-memory PatternIndex
            if self.pattern_index and hasattr(self.pattern_index, "update_prior"):
                try:
                    self.pattern_index.update_prior(neighbor, applied)
                except Exception:
                    pass
            updates.append((neighbor, applied))
        return updates

    def get_co_occurrence_edges(self) -> List[Dict[str, Any]]:
        """Return all co-occurrence edges for graph visualisation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT pattern_a, pattern_b, weight, first_session, last_session "
                "FROM co_occurrence ORDER BY weight DESC"
            )
            return [
                {
                    "source": row[0],
                    "target": row[1],
                    "weight": row[2],
                    "first_session": row[3],
                    "last_session": row[4],
                }
                for row in cursor.fetchall()
            ]

    def get_co_occurring_patterns(
        self,
        pattern_key: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the top-k co-occurring patterns for *pattern_key*.

        Each result dict has keys: ``source``, ``target``, ``weight``.
        """
        pk = str(pattern_key).strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT pattern_a, pattern_b, weight FROM co_occurrence "
                "WHERE pattern_a = ? OR pattern_b = ? "
                "ORDER BY weight DESC LIMIT ?",
                (pk, pk, int(top_k)),
            )
            results: List[Dict[str, Any]] = []
            for row in cursor.fetchall():
                results.append({
                    "source": row[0],
                    "target": row[1],
                    "weight": row[2],
                })
        return results

    def _migrate_v1_to_v2(self, conn: sqlite3.Connection) -> None:
        """Migrate from schema v1 (single pattern_key) to v2 (pattern_keys list)."""
        logger = logging.getLogger(__name__)
        logger.info("Migrating database from schema v1 to v2...")
        
        cursor = conn.cursor()
        
        # Create new table with v2 schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories_v2 (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                annotation_text TEXT NOT NULL,
                pattern_keys_json TEXT NOT NULL,
                metrics_json TEXT,
                time_range_json TEXT,
                channels_json TEXT,
                tags_json TEXT,
                label TEXT,
                provenance_json TEXT,
                metadata_json TEXT,
                visibility TEXT DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                created_by TEXT DEFAULT 'operator',
                embedding_vector BLOB
            )
        """)
        
        # Copy data with transformation
        cursor.execute("SELECT * FROM memories")
        for row in cursor.fetchall():
            row_dict = dict(row)
            # Transform single pattern_key to pattern_keys list
            pattern_key_json = row_dict.get('pattern_key_json', '{}')
            pattern_keys_json = json.dumps([json.loads(pattern_key_json)] if pattern_key_json else [])
            
            cursor.execute("""
                INSERT INTO memories_v2 
                (id, session_id, annotation_text, pattern_keys_json, metrics_json, 
                 time_range_json, channels_json, tags_json, label, provenance_json,
                 metadata_json, visibility, created_at, updated_at, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row_dict.get('id'),
                row_dict.get('session_id', 'migrated'),
                row_dict.get('content', ''),  # content -> annotation_text
                pattern_keys_json,
                row_dict.get('metrics_json'),
                row_dict.get('time_range_json'),
                json.dumps([]),  # channels
                row_dict.get('tags_json'),
                None,  # label
                json.dumps({"compute_version": "1.0"}),  # provenance
                json.dumps({"migrated_from_v1": True}),  # metadata
                'active',
                row_dict.get('created_at'),
                row_dict.get('updated_at'),
                'system',
            ))
        
        # Drop old table and rename new one
        cursor.execute("DROP TABLE memories")
        cursor.execute("ALTER TABLE memories_v2 RENAME TO memories")
        conn.commit()
        
        logger.info("Migration to schema v2 complete")
    
    def _rebuild_indices(self) -> None:
        """Rebuild in-memory indices from database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, pattern_keys_json, metrics_json FROM memories")
            
            for row in cursor.fetchall():
                memory_id = row["id"]
                pattern_keys_json = row["pattern_keys_json"]
                
                # Parse pattern_keys list
                if pattern_keys_json:
                    pattern_keys = json.loads(pattern_keys_json)
                    for pk_data in pattern_keys:
                        pattern_key = PatternKey(**pk_data)
                        # Add to pattern index
                        self.pattern_index.add(memory_id, pattern_key)
                
                # Add to ANN index if enabled and metrics exist
                if self.ann_index and row["metrics_json"]:
                    metrics = MetricsSummary(**json.loads(row["metrics_json"]))
                    # Convert metrics to a numeric vector for ANN indexing
                    vector = self._metrics_to_vector(metrics)
                    if vector is not None:
                        self.ann_index.insert(vector, memory_id)
    
    def _metrics_to_vector(self, metrics: MetricsSummary) -> Optional[np.ndarray]:
        """Convert a MetricsSummary to a numeric vector for ANN indexing."""
        values = []
        # Extract numeric values from metrics in a consistent order
        for key in sorted(metrics.means.keys()):
            values.append(metrics.means.get(key, 0.0))
        for key in sorted(metrics.stds.keys()):
            values.append(metrics.stds.get(key, 0.0))

        # Back-compat: metrics.rms may be a scalar.
        if isinstance(metrics.rms, dict):
            for key in sorted(metrics.rms.keys()):
                values.append(metrics.rms.get(key, 0.0))
        elif isinstance(metrics.rms, (int, float)):
            values.append(float(metrics.rms))

        for key in sorted(metrics.peaks.keys()):
            values.append(metrics.peaks.get(key, 0.0))
        for key in sorted(metrics.dominant_freqs.keys()):
            values.append(metrics.dominant_freqs.get(key, 0.0))
        for key in sorted(metrics.spectral_centroids.keys()):
            values.append(metrics.spectral_centroids.get(key, 0.0))
        
        if not values:
            return None
        return np.array(values, dtype=np.float32)
    
    def _serialize_memory(self, memory: Memory) -> Dict[str, Any]:
        """Serialize a Memory to database row format (schema v2)."""
        # Serialize pattern_keys list
        pattern_keys_json = json.dumps([
            self._serialize_pattern_key(pk) for pk in memory.pattern_keys
        ]) if memory.pattern_keys else "[]"
        
        metadata = dict(memory.metadata or {})
        # Persist vectors for unit tests / lightweight clients.
        if memory.numeric_vector is not None:
            metadata["_numeric_vector"] = memory.numeric_vector
        if memory.text_embedding is not None:
            metadata["_text_embedding"] = memory.text_embedding

        return {
            "id": memory.id,
            "session_id": memory.session_id,
            "annotation_text": memory.annotation_text,
            "pattern_keys_json": pattern_keys_json,
            "metrics_json": memory.metrics.model_dump_json() if memory.metrics else None,
            "time_range_json": self._serialize_time_range(memory.time_range),
            "channels_json": json.dumps(memory.channels) if memory.channels else "[]",
            "tags_json": json.dumps(memory.tags) if memory.tags else "[]",
            "label": memory.label,
            "provenance_json": memory.provenance.model_dump_json() if memory.provenance else None,
            "metadata_json": json.dumps(metadata) if metadata else "{}",
            "visibility": memory.visibility,
            "created_at": memory.created_at.isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "created_by": memory.created_by,
        }
    
    def _deserialize_memory(self, row: sqlite3.Row) -> Memory:
        """Deserialize a database row to Memory object (schema v2)."""
        # Parse pattern_keys list
        pattern_keys_json = row["pattern_keys_json"]
        pattern_keys = []
        if pattern_keys_json:
            for pk_data in json.loads(pattern_keys_json):
                pk = PatternKey(**pk_data)
                additional = dict(pk.additional or {})
                pattern_type = pk.pattern_type.value if pk.pattern_type else None
                additional.setdefault("kind", infer_pattern_kind(pk.key, pattern_type))
                pk.additional = additional
                pattern_keys.append(pk)
        
        # Parse provenance
        provenance = MemoryProvenance()
        if row["provenance_json"]:
            provenance = MemoryProvenance(**json.loads(row["provenance_json"]))
        
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        numeric_vector = metadata.get("_numeric_vector")
        text_embedding = metadata.get("_text_embedding")

        created_at = datetime.fromisoformat(row["created_at"])
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        return Memory(
            id=row["id"],
            session_id=row["session_id"],
            annotation_text=row["annotation_text"],
            pattern_keys=pattern_keys,
            metrics=NumericMetrics(**json.loads(row["metrics_json"])) if row["metrics_json"] else NumericMetrics(),
            time_range=self._deserialize_time_range(row["time_range_json"]),
            channels=json.loads(row["channels_json"]) if row["channels_json"] else [],
            tags=json.loads(row["tags_json"]) if row["tags_json"] else [],
            label=row["label"],
            provenance=provenance,
            metadata=metadata,
            numeric_vector=numeric_vector,
            text_embedding=text_embedding,
            visibility=row["visibility"] or "active",
            created_at=created_at,
            created_by=row["created_by"] or "operator",
        )

    # ------------------------------------------------------------------
    # Test-friendly CRUD surface (thin wrappers over existing API)
    # ------------------------------------------------------------------

    def create(self, memory: Memory) -> str:
        return self.store(memory)

    def list(self, session_id: Optional[str] = None, offset: int = 0, limit: int = 100) -> List[Memory]:
        sql = "SELECT * FROM memories"
        params: List[Any] = []
        if session_id:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " ORDER BY id ASC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return [self._deserialize_memory(row) for row in cursor.fetchall()]

    def update(self, memory_id: str, updates: Optional[Dict[str, Any]] = None, **fields: Any) -> bool:
        """Update a memory by ID.

        Accepts updates as a dict (preferred) or as keyword arguments
        for backward compatibility.  When both are given the dict wins
        and kwargs are merged in.
        """
        merged: Dict[str, Any] = dict(updates) if updates else {}
        merged.update(fields)
        if not merged:
            return False

        memory = self.get(memory_id)
        if memory is None:
            return False

        for key, value in merged.items():
            setattr(memory, key, value)

        self.store(memory)
        return True

    def search(
        self,
        text_query: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        session_id: Optional[str] = None,
    ) -> List[Memory]:
        sql = "SELECT * FROM memories WHERE 1=1"
        params: List[Any] = []
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if text_query:
            sql += " AND annotation_text LIKE ?"
            params.append(f"%{text_query}%")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = [self._deserialize_memory(row) for row in cursor.fetchall()]

        if time_range is None:
            return rows

        q0, q1 = float(time_range[0]), float(time_range[1])
        results: List[Memory] = []
        for mem in rows:
            tr = mem.time_range
            if isinstance(tr, tuple) and len(tr) == 2:
                m0, m1 = float(tr[0]), float(tr[1])
            elif isinstance(tr, TimeRange):
                m0, m1 = float(tr.t0), float(tr.t1)
            else:
                continue
            # overlap
            if m0 <= q1 and m1 >= q0:
                results.append(mem)
        return results
    
    def store(self, memory: Memory) -> str:
        """
        Store a new memory or update an existing one.
        
        Args:
            memory: Memory object to store
            
        Returns:
            The memory ID (generated if not provided)
        """
        # Assign ID if not present
        if not memory.id:
            memory.id = str(uuid.uuid4())
        
        # Set timestamps
        now = datetime.now(timezone.utc)
        if not memory.created_at:
            memory.created_at = now
        
        data = self._serialize_memory(memory)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Upsert memory (schema v2)
            cursor.execute("""
                INSERT OR REPLACE INTO memories 
                (id, session_id, annotation_text, pattern_keys_json, metrics_json,
                 time_range_json, channels_json, tags_json, label, provenance_json,
                 metadata_json, visibility, created_at, updated_at, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["id"], data["session_id"], data["annotation_text"],
                data["pattern_keys_json"], data["metrics_json"], data["time_range_json"],
                data["channels_json"], data["tags_json"], data["label"],
                data["provenance_json"], data["metadata_json"], data["visibility"],
                data["created_at"], data["updated_at"], data["created_by"]
            ))
            
            # Update pattern key components
            cursor.execute(
                "DELETE FROM pattern_key_components WHERE memory_id = ?",
                (memory.id,)
            )
            cursor.execute(
                "DELETE FROM memory_pattern_edges WHERE memory_id = ?",
                (memory.id,)
            )
            
            # Insert pattern key components for fast lookups (all pattern_keys)
            for pk in memory.pattern_keys:
                additional = dict(pk.additional or {})
                pattern_type_value = pk.pattern_type.value if pk.pattern_type else None
                kind = str(additional.get("kind") or infer_pattern_kind(pk.key, pattern_type_value))
                source_metric = pk.source_metric or additional.get("source_metric")
                strength = float(pk.confidence if pk.confidence is not None else additional.get("confidence", 1.0) or 1.0)

                components = [
                    ("pattern_type", pattern_type_value),
                    ("key", pk.key),
                    ("condition", pk.condition),
                    ("machine_type", pk.machine_type),
                    ("fault_type", pk.fault_type),
                    ("channel", pk.channel),
                ]
                if additional:
                    for key, value in additional.items():
                        components.append((f"additional.{key}", value))

                cursor.execute(
                    """
                    INSERT INTO memory_pattern_edges
                    (memory_id, pattern_key, strength, source_metric, kind)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (memory.id, pk.key, strength, source_metric, kind),
                )
                
                for comp_type, comp_value in components:
                    if comp_value:
                        cursor.execute("""
                            INSERT INTO pattern_key_components 
                            (memory_id, component_type, component_value)
                            VALUES (?, ?, ?)
                        """, (memory.id, comp_type, comp_value))
            
            conn.commit()
        
        # Update in-memory indices (add all pattern_keys)
        for pk in memory.pattern_keys:
            self.pattern_index.add(memory.id, pk)
        
        if self.ann_index and memory.metrics:
            self.ann_index.add(memory.id, memory.metrics)
        
        return memory.id

    def persist_doc_links(
        self,
        *,
        memory_id: str,
        pattern_keys: List[str],
        doc_links: List[Dict[str, Any]],
    ) -> int:
        memory = self.get(memory_id)
        if memory is None or not doc_links:
            return 0
        memory.metadata = dict(memory.metadata or {})
        memory.metadata["doc_links"] = self._sort_doc_links(
            [dict(link) for link in doc_links if isinstance(link, dict)]
        )
        self.store(memory)
        return len(memory.metadata["doc_links"])

    def get_doc_links(
        self,
        memory_id: str,
        *,
        score_floor: float = 0.0,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        memory = self.get(memory_id)
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
            memory = self.get(memory_id)
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
            self.store(memory)
            return dict(updated_link)
    
    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            
            if row:
                return self._deserialize_memory(row)
        return None
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
        
        if deleted:
            self.pattern_index.remove(memory_id)
            if self.ann_index:
                self.ann_index.remove(memory_id)
        
        return deleted
    
    def query(self, query: MemoryQuery) -> List[Memory]:
        """
        Query memories using multiple retrieval strategies.
        
        Strategy priority:
        1. Pattern key match (exact/partial)
        2. ANN search by numeric metrics (if query.similar_to_metrics)
        3. Tag/source/session filtering
        4. Time range filtering
        5. Limit and ranking
        """
        candidate_ids: Optional[set] = None
        
        # Pattern-based retrieval
        if query.pattern:
            pattern_matches = self.pattern_index.query(
                query.pattern,
                partial_match=query.partial_pattern_match
            )
            candidate_ids = set(pattern_matches)
        
        # ANN-based retrieval for similar metrics
        if query.similar_to_metrics and self.ann_index:
            ann_matches = self.ann_index.query(
                query.similar_to_metrics,
                k=query.limit * 3  # Over-fetch for post-filtering
            )
            if candidate_ids is None:
                candidate_ids = set(ann_matches)
            else:
                # Intersection if we already have pattern matches
                candidate_ids &= set(ann_matches)
        
        # Build SQL query for remaining filters
        sql_parts = ["SELECT * FROM memories WHERE 1=1"]
        params: List[Any] = []
        
        if candidate_ids is not None:
            if not candidate_ids:
                return []  # No matches from index queries
            placeholders = ",".join("?" * len(candidate_ids))
            sql_parts.append(f"AND id IN ({placeholders})")
            params.extend(candidate_ids)
        
        if query.tags:
            # Tag filtering (any match)
            for tag in query.tags:
                sql_parts.append("AND tags_json LIKE ?")
                params.append(f'%"{tag}"%')
        
        if query.source:
            sql_parts.append("AND source = ?")
            params.append(query.source)
        
        if query.session_id:
            sql_parts.append("AND session_id = ?")
            params.append(query.session_id)
        
        if query.min_confidence is not None:
            sql_parts.append("AND confidence >= ?")
            params.append(query.min_confidence)
        
        # Time range filter (overlapping)
        if query.time_range:
            sql_parts.append("""
                AND time_range_json IS NOT NULL
                AND json_extract(time_range_json, '$.start') <= ?
                AND json_extract(time_range_json, '$.end') >= ?
            """)
            params.extend([query.time_range.end, query.time_range.start])
        
        # Order and limit
        sql_parts.append("ORDER BY confidence DESC, updated_at DESC")
        sql_parts.append(f"LIMIT {query.limit}")
        
        sql = " ".join(sql_parts)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            return [self._deserialize_memory(row) for row in rows]
    
    def list_by_session(self, session_id: str, limit: int = 100) -> List[Memory]:
        """List all memories for a given session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM memories WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit)
            )
            return [self._deserialize_memory(row) for row in cursor.fetchall()]

    def list_all(self, limit: int = 1000, visibility: str = "active") -> List[Memory]:
        """List recent memories.

        This is intentionally simple and is used by prototype components like the
        memory retriever, which need a stable way to fetch candidates.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if visibility:
                cursor.execute(
                    "SELECT * FROM memories WHERE visibility = ? ORDER BY created_at DESC LIMIT ?",
                    (visibility, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            return [self._deserialize_memory(row) for row in cursor.fetchall()]
    
    def count(self) -> int:
        """Get total count of memories."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            return cursor.fetchone()[0]
    
    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM memories")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT session_id) FROM memories WHERE session_id IS NOT NULL")
            sessions = cursor.fetchone()[0]
            
            cursor.execute("SELECT source, COUNT(*) FROM memories GROUP BY source")
            by_source = dict(cursor.fetchall())
            
        return {
            "total_memories": total,
            "unique_sessions": sessions,
            "by_source": by_source,
            "pattern_index_size": self.pattern_index.size(),
            "ann_index_size": self.ann_index.size if self.ann_index else 0,
        }
    
    def clear(self) -> None:
        """Clear all memories (use with caution)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories")
            cursor.execute("DELETE FROM pattern_key_components")
            conn.commit()
        
        self.pattern_index.clear()
        if self.ann_index:
            self.ann_index.clear()

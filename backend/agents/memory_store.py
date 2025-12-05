"""
Memory Store - SQLite-based persistent storage for LLM-RAG agent memories.

This module provides the core storage layer for the memory system, handling
CRUD operations, pattern key indexing, and optional ANN/embedding indices.
"""

from __future__ import annotations

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

import numpy as np

from .memory_schema import (
    Memory,
    MemoryQuery,
    PatternKey,
    TimeRange,
    MetricsSummary,
)
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
    
    DB_VERSION = 1
    
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
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Main memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    pattern_key_json TEXT NOT NULL,
                    metrics_json TEXT,
                    time_range_json TEXT,
                    tags_json TEXT,
                    source TEXT,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    session_id TEXT,
                    embedding_vector BLOB
                )
            """)
            
            # Indices for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_session 
                ON memories(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_source 
                ON memories(source)
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
            
            # Version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            cursor.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (self.DB_VERSION,)
            )
            
            conn.commit()
    
    def _rebuild_indices(self) -> None:
        """Rebuild in-memory indices from database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, pattern_key_json, metrics_json FROM memories")
            
            for row in cursor.fetchall():
                memory_id = row["id"]
                pattern_key = PatternKey(**json.loads(row["pattern_key_json"]))
                
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
        for key in sorted(metrics.rms.keys()):
            values.append(metrics.rms.get(key, 0.0))
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
        """Serialize a Memory to database row format."""
        return {
            "id": memory.id,
            "content": memory.content,
            "pattern_key_json": memory.pattern_key.model_dump_json(),
            "metrics_json": memory.metrics.model_dump_json() if memory.metrics else None,
            "time_range_json": memory.time_range.model_dump_json() if memory.time_range else None,
            "tags_json": json.dumps(memory.tags) if memory.tags else None,
            "source": memory.source,
            "confidence": memory.confidence,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "session_id": memory.session_id,
        }
    
    def _deserialize_memory(self, row: sqlite3.Row) -> Memory:
        """Deserialize a database row to Memory object."""
        return Memory(
            id=row["id"],
            content=row["content"],
            pattern_key=PatternKey(**json.loads(row["pattern_key_json"])),
            metrics=MetricsSummary(**json.loads(row["metrics_json"])) if row["metrics_json"] else None,
            time_range=TimeRange(**json.loads(row["time_range_json"])) if row["time_range_json"] else None,
            tags=json.loads(row["tags_json"]) if row["tags_json"] else None,
            source=row["source"],
            confidence=row["confidence"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            session_id=row["session_id"],
        )
    
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
        now = datetime.utcnow()
        if not memory.created_at:
            memory.created_at = now
        memory.updated_at = now
        
        data = self._serialize_memory(memory)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Upsert memory
            cursor.execute("""
                INSERT OR REPLACE INTO memories 
                (id, content, pattern_key_json, metrics_json, time_range_json,
                 tags_json, source, confidence, created_at, updated_at, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["id"], data["content"], data["pattern_key_json"],
                data["metrics_json"], data["time_range_json"], data["tags_json"],
                data["source"], data["confidence"], data["created_at"],
                data["updated_at"], data["session_id"]
            ))
            
            # Update pattern key components
            cursor.execute(
                "DELETE FROM pattern_key_components WHERE memory_id = ?",
                (memory.id,)
            )
            
            # Insert pattern key components for fast lookups
            pk = memory.pattern_key
            components = [
                ("condition", pk.condition),
                ("machine_type", pk.machine_type),
                ("fault_type", pk.fault_type),
                ("channel", pk.channel),
            ]
            if pk.additional:
                for key, value in pk.additional.items():
                    components.append((f"additional.{key}", value))
            
            for comp_type, comp_value in components:
                if comp_value:
                    cursor.execute("""
                        INSERT INTO pattern_key_components 
                        (memory_id, component_type, component_value)
                        VALUES (?, ?, ?)
                    """, (memory.id, comp_type, comp_value))
            
            conn.commit()
        
        # Update in-memory indices
        self.pattern_index.add(memory.id, memory.pattern_key)
        
        if self.ann_index and memory.metrics:
            self.ann_index.add(memory.id, memory.metrics)
        
        return memory.id
    
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
            "ann_index_size": self.ann_index.size() if self.ann_index else 0,
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

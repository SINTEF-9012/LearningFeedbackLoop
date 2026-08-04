"""
Storage module - Persistence layer for memories and indexes.

This module contains:
- store: SQLite-based MemoryStore
- in_memory_store: non-durable in-memory adapter (tests / no-database runs)
- neo4j_store: graph-backed store
- ann_index: FAISS-based approximate nearest neighbor index
- pattern_index: Pattern key indexing

All stores satisfy the duck-typed ``protocol.MemoryStoreProtocol``.
``neo4j_store`` is deliberately not re-exported here: it imports the optional
Neo4j driver, and this package must stay importable without it.
"""

from .store import MemoryStore
from .in_memory_store import InMemoryStoreAdapter
from .ann_index import ANNIndex
from .pattern_index import PatternIndex

__all__ = [
    "MemoryStore",
    "InMemoryStoreAdapter",
    "ANNIndex",
    "PatternIndex",
]

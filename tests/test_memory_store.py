"""
Unit tests for the memory store module.

Tests cover:
- Memory record creation
- Memory retrieval by ID
- Memory listing with filters
- Memory updates
- Memory deletion
- Search functionality
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from agents.memory_store import MemoryStore
from agents.memory_schema import Memory, PatternKey, NumericMetrics


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def memory_store(temp_db):
    """Create a memory store with temporary database."""
    return MemoryStore(db_path=temp_db)


@pytest.fixture
def sample_memory():
    """Create a sample memory object."""
    return Memory(
        id="test-mem-001",
        session_id="session-123",
        time_range=(0.0, 10.0),
        annotation_text="Test annotation",
        pattern_keys=[
            PatternKey(category="freq", value="high"),
            PatternKey(category="amp", value="loud"),
        ],
        numeric_metrics=NumericMetrics(
            mean=1.5,
            std=0.3,
            rms=1.52,
            peak=2.1,
            crest_factor=1.38,
            snr_db=25.0,
        ),
        numeric_vector=[0.1, 0.2, 0.3, 0.4],
        text_embedding=[0.5, 0.6, 0.7, 0.8],
        provenance="test",
        created_at=datetime.utcnow().isoformat(),
    )


class TestMemoryStoreCreation:
    """Tests for memory store initialization."""
    
    def test_create_store(self, temp_db):
        """Should create a new store with database."""
        store = MemoryStore(db_path=temp_db)
        assert store is not None
        assert os.path.exists(temp_db)
    
    def test_create_in_memory_store(self):
        """Should support in-memory database."""
        store = MemoryStore(db_path=":memory:")
        assert store is not None
    
    def test_table_created(self, memory_store):
        """Database should have memories table."""
        import sqlite3
        conn = sqlite3.connect(memory_store.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        assert cursor.fetchone() is not None
        conn.close()


class TestMemoryCreate:
    """Tests for creating memory records."""
    
    def test_create_memory(self, memory_store, sample_memory):
        """Should create a memory record."""
        mem_id = memory_store.create(sample_memory)
        assert mem_id == sample_memory.id
    
    def test_create_with_generated_id(self, memory_store):
        """Should generate ID if not provided."""
        mem = Memory(
            session_id="session-456",
            time_range=(5.0, 15.0),
            annotation_text="Auto-ID test",
        )
        mem_id = memory_store.create(mem)
        assert mem_id is not None
        assert len(mem_id) > 0
    
    def test_create_minimal_memory(self, memory_store):
        """Should create memory with minimal fields."""
        mem = Memory(
            session_id="session-789",
            time_range=(0.0, 5.0),
        )
        mem_id = memory_store.create(mem)
        assert mem_id is not None


class TestMemoryGet:
    """Tests for retrieving memory records."""
    
    def test_get_existing_memory(self, memory_store, sample_memory):
        """Should retrieve existing memory by ID."""
        memory_store.create(sample_memory)
        
        retrieved = memory_store.get(sample_memory.id)
        
        assert retrieved is not None
        assert retrieved.id == sample_memory.id
        assert retrieved.session_id == sample_memory.session_id
        assert retrieved.annotation_text == sample_memory.annotation_text
    
    def test_get_nonexistent_memory(self, memory_store):
        """Should return None for non-existent ID."""
        retrieved = memory_store.get("nonexistent-id")
        assert retrieved is None
    
    def test_get_preserves_time_range(self, memory_store, sample_memory):
        """Should preserve time_range tuple."""
        memory_store.create(sample_memory)
        retrieved = memory_store.get(sample_memory.id)
        
        assert retrieved.time_range == sample_memory.time_range
    
    def test_get_preserves_pattern_keys(self, memory_store, sample_memory):
        """Should preserve pattern keys."""
        memory_store.create(sample_memory)
        retrieved = memory_store.get(sample_memory.id)
        
        assert len(retrieved.pattern_keys) == len(sample_memory.pattern_keys)
        assert retrieved.pattern_keys[0].category == "freq"
        assert retrieved.pattern_keys[0].value == "high"


class TestMemoryList:
    """Tests for listing memory records."""
    
    def test_list_empty(self, memory_store):
        """Should return empty list for empty store."""
        memories = memory_store.list()
        assert memories == []
    
    def test_list_all(self, memory_store, sample_memory):
        """Should list all memories."""
        memory_store.create(sample_memory)
        
        mem2 = Memory(
            id="test-mem-002",
            session_id="session-123",
            time_range=(10.0, 20.0),
        )
        memory_store.create(mem2)
        
        memories = memory_store.list()
        assert len(memories) == 2
    
    def test_list_filter_by_session(self, memory_store):
        """Should filter by session_id."""
        mem1 = Memory(id="m1", session_id="session-A", time_range=(0, 5))
        mem2 = Memory(id="m2", session_id="session-B", time_range=(0, 5))
        mem3 = Memory(id="m3", session_id="session-A", time_range=(5, 10))
        
        memory_store.create(mem1)
        memory_store.create(mem2)
        memory_store.create(mem3)
        
        session_a = memory_store.list(session_id="session-A")
        assert len(session_a) == 2
        
        session_b = memory_store.list(session_id="session-B")
        assert len(session_b) == 1
    
    def test_list_with_limit(self, memory_store):
        """Should respect limit parameter."""
        for i in range(10):
            mem = Memory(id=f"m{i}", session_id="s", time_range=(i, i+1))
            memory_store.create(mem)
        
        memories = memory_store.list(limit=5)
        assert len(memories) == 5
    
    def test_list_with_offset(self, memory_store):
        """Should respect offset parameter."""
        for i in range(10):
            mem = Memory(id=f"m{i:02d}", session_id="s", time_range=(i, i+1))
            memory_store.create(mem)
        
        memories = memory_store.list(offset=5, limit=100)
        assert len(memories) == 5


class TestMemoryUpdate:
    """Tests for updating memory records."""
    
    def test_update_annotation(self, memory_store, sample_memory):
        """Should update annotation text."""
        memory_store.create(sample_memory)
        
        success = memory_store.update(
            sample_memory.id,
            annotation_text="Updated annotation"
        )
        
        assert success is True
        
        retrieved = memory_store.get(sample_memory.id)
        assert retrieved.annotation_text == "Updated annotation"
    
    def test_update_nonexistent(self, memory_store):
        """Should return False for non-existent ID."""
        success = memory_store.update("nonexistent", annotation_text="test")
        assert success is False
    
    def test_update_multiple_fields(self, memory_store, sample_memory):
        """Should update multiple fields at once."""
        memory_store.create(sample_memory)
        
        new_patterns = [PatternKey(category="new", value="pattern")]
        success = memory_store.update(
            sample_memory.id,
            annotation_text="New text",
            pattern_keys=new_patterns,
        )
        
        assert success is True
        
        retrieved = memory_store.get(sample_memory.id)
        assert retrieved.annotation_text == "New text"
        assert len(retrieved.pattern_keys) == 1
        assert retrieved.pattern_keys[0].category == "new"


class TestMemoryDelete:
    """Tests for deleting memory records."""
    
    def test_delete_existing(self, memory_store, sample_memory):
        """Should delete existing memory."""
        memory_store.create(sample_memory)
        
        success = memory_store.delete(sample_memory.id)
        assert success is True
        
        retrieved = memory_store.get(sample_memory.id)
        assert retrieved is None
    
    def test_delete_nonexistent(self, memory_store):
        """Should return False for non-existent ID."""
        success = memory_store.delete("nonexistent")
        assert success is False


class TestMemorySearch:
    """Tests for searching memory records."""
    
    def test_search_by_text(self, memory_store):
        """Should search by annotation text."""
        mem1 = Memory(
            id="m1",
            session_id="s",
            time_range=(0, 5),
            annotation_text="This has a specific keyword"
        )
        mem2 = Memory(
            id="m2",
            session_id="s",
            time_range=(5, 10),
            annotation_text="This does not"
        )
        
        memory_store.create(mem1)
        memory_store.create(mem2)
        
        results = memory_store.search(text_query="keyword")
        assert len(results) == 1
        assert results[0].id == "m1"
    
    def test_search_by_time_range(self, memory_store):
        """Should search by time range overlap."""
        mem1 = Memory(id="m1", session_id="s", time_range=(0.0, 10.0))
        mem2 = Memory(id="m2", session_id="s", time_range=(15.0, 25.0))
        mem3 = Memory(id="m3", session_id="s", time_range=(8.0, 18.0))
        
        memory_store.create(mem1)
        memory_store.create(mem2)
        memory_store.create(mem3)
        
        # Search for overlap with 5-12
        results = memory_store.search(time_range=(5.0, 12.0))
        
        ids = {m.id for m in results}
        assert "m1" in ids  # Overlaps at 5-10
        assert "m3" in ids  # Overlaps at 8-12
        assert "m2" not in ids  # No overlap


class TestMemoryStoreEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_vectors(self, memory_store):
        """Should handle empty vectors."""
        mem = Memory(
            id="m1",
            session_id="s",
            time_range=(0, 5),
            numeric_vector=[],
            text_embedding=[],
        )
        
        mem_id = memory_store.create(mem)
        retrieved = memory_store.get(mem_id)
        
        assert retrieved.numeric_vector == []
        assert retrieved.text_embedding == []
    
    def test_special_characters_in_annotation(self, memory_store):
        """Should handle special characters in text."""
        mem = Memory(
            id="m1",
            session_id="s",
            time_range=(0, 5),
            annotation_text="Test with 'quotes' and \"double quotes\" and emoji 🚀",
        )
        
        mem_id = memory_store.create(mem)
        retrieved = memory_store.get(mem_id)
        
        assert "emoji 🚀" in retrieved.annotation_text
    
    def test_large_vectors(self, memory_store):
        """Should handle large vectors."""
        large_vec = list(range(1024))
        mem = Memory(
            id="m1",
            session_id="s",
            time_range=(0, 5),
            numeric_vector=large_vec,
        )
        
        mem_id = memory_store.create(mem)
        retrieved = memory_store.get(mem_id)
        
        assert len(retrieved.numeric_vector) == 1024
        assert retrieved.numeric_vector[100] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

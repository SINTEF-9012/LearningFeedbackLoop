"""
Unit tests for the pattern index module.

Tests cover:
- Adding patterns for memory IDs
- Removing patterns
- Searching by single pattern
- Searching with multiple patterns (AND)
- Persistence (save/load)
"""

import pytest
import tempfile
import os
import json

from backend.agents.storage.pattern_index import PatternIndex


@pytest.fixture
def temp_index_file():
    """Create a temporary file for index persistence."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def pattern_index():
    """Create an in-memory pattern index."""
    return PatternIndex()


class TestPatternIndexBasic:
    """Basic pattern index operations."""
    
    def test_create_index(self):
        """Should create an empty index."""
        idx = PatternIndex()
        assert idx is not None
    
    def test_add_single_pattern(self, pattern_index):
        """Should add a single pattern."""
        pattern_index.add("mem1", ["freq:high"])
        
        results = pattern_index.search(["freq:high"])
        assert "mem1" in results
    
    def test_add_multiple_patterns(self, pattern_index):
        """Should add multiple patterns for one memory."""
        patterns = ["freq:high", "amp:loud", "snr:good"]
        pattern_index.add("mem1", patterns)
        
        for p in patterns:
            results = pattern_index.search([p])
            assert "mem1" in results
    
    def test_add_same_pattern_multiple_memories(self, pattern_index):
        """Should index multiple memories under same pattern."""
        pattern_index.add("mem1", ["freq:high"])
        pattern_index.add("mem2", ["freq:high"])
        pattern_index.add("mem3", ["freq:high"])
        
        results = pattern_index.search(["freq:high"])
        assert len(results) == 3
        assert "mem1" in results
        assert "mem2" in results
        assert "mem3" in results


class TestPatternIndexSearch:
    """Tests for pattern search functionality."""
    
    def test_search_nonexistent_pattern(self, pattern_index):
        """Should return empty set for non-existent pattern."""
        results = pattern_index.search(["nonexistent:pattern"])
        assert len(results) == 0
    
    def test_search_single_pattern(self, pattern_index):
        """Should find memories with single pattern."""
        pattern_index.add("mem1", ["freq:high", "amp:loud"])
        pattern_index.add("mem2", ["freq:low", "amp:loud"])
        pattern_index.add("mem3", ["freq:high", "amp:quiet"])
        
        results = pattern_index.search(["freq:high"])
        assert len(results) == 2
        assert "mem1" in results
        assert "mem3" in results
    
    def test_search_multiple_patterns_and(self, pattern_index):
        """Should intersect multiple patterns (AND logic)."""
        pattern_index.add("mem1", ["freq:high", "amp:loud"])
        pattern_index.add("mem2", ["freq:high", "amp:quiet"])
        pattern_index.add("mem3", ["freq:low", "amp:loud"])
        
        # Only mem1 has both
        results = pattern_index.search(["freq:high", "amp:loud"])
        assert len(results) == 1
        assert "mem1" in results
    
    def test_search_partial_match(self, pattern_index):
        """Should only return memories matching ALL patterns."""
        pattern_index.add("mem1", ["freq:high"])
        pattern_index.add("mem2", ["freq:high", "amp:loud"])
        
        # mem1 doesn't have amp:loud
        results = pattern_index.search(["freq:high", "amp:loud"])
        assert len(results) == 1
        assert "mem2" in results
    
    def test_search_or_mode(self, pattern_index):
        """Should support OR mode searching."""
        pattern_index.add("mem1", ["freq:high"])
        pattern_index.add("mem2", ["amp:loud"])
        pattern_index.add("mem3", ["freq:high", "amp:loud"])
        
        results = pattern_index.search_any(["freq:high", "amp:loud"])
        assert len(results) == 3


class TestPatternIndexRemove:
    """Tests for removing patterns."""
    
    def test_remove_memory_from_index(self, pattern_index):
        """Should remove all patterns for a memory ID."""
        pattern_index.add("mem1", ["freq:high", "amp:loud"])
        pattern_index.add("mem2", ["freq:high"])
        
        pattern_index.remove("mem1")
        
        # mem1 should no longer appear
        results = pattern_index.search(["freq:high"])
        assert "mem1" not in results
        assert "mem2" in results
        
        results = pattern_index.search(["amp:loud"])
        assert "mem1" not in results
    
    def test_remove_nonexistent_memory(self, pattern_index):
        """Should handle removing non-existent memory gracefully."""
        # Should not raise
        pattern_index.remove("nonexistent")
    
    def test_remove_specific_patterns(self, pattern_index):
        """Should remove specific patterns for a memory."""
        pattern_index.add("mem1", ["freq:high", "amp:loud", "snr:good"])
        
        pattern_index.remove("mem1", patterns=["amp:loud"])
        
        # freq:high should still be indexed
        results = pattern_index.search(["freq:high"])
        assert "mem1" in results
        
        # amp:loud should be removed
        results = pattern_index.search(["amp:loud"])
        assert "mem1" not in results


class TestPatternIndexPersistence:
    """Tests for index persistence."""
    
    def test_save_and_load(self, temp_index_file):
        """Should save and load index."""
        idx1 = PatternIndex()
        idx1.add("mem1", ["freq:high", "amp:loud"])
        idx1.add("mem2", ["freq:low"])
        
        idx1.save(temp_index_file)
        
        idx2 = PatternIndex()
        idx2.load(temp_index_file)
        
        results = idx2.search(["freq:high"])
        assert "mem1" in results
        
        results = idx2.search(["freq:low"])
        assert "mem2" in results
    
    def test_load_from_constructor(self, temp_index_file):
        """Should load from path in constructor."""
        idx1 = PatternIndex()
        idx1.add("mem1", ["freq:high"])
        idx1.save(temp_index_file)
        
        idx2 = PatternIndex(index_path=temp_index_file)
        
        results = idx2.search(["freq:high"])
        assert "mem1" in results
    
    def test_load_nonexistent_file(self):
        """Should handle non-existent file gracefully."""
        idx = PatternIndex(index_path="/nonexistent/path.json")
        # Should start with empty index
        results = idx.search(["anything"])
        assert len(results) == 0


class TestPatternIndexEdgeCases:
    """Edge case tests."""
    
    def test_empty_patterns_list(self, pattern_index):
        """Should handle empty patterns list."""
        pattern_index.add("mem1", [])
        
        # Memory added but with no patterns
        results = pattern_index.search([])
        # Empty search should return nothing or everything depending on implementation
        # Here we expect empty result for empty query
        assert len(results) == 0
    
    def test_duplicate_patterns(self, pattern_index):
        """Should handle duplicate patterns in add."""
        pattern_index.add("mem1", ["freq:high", "freq:high", "freq:high"])
        
        results = pattern_index.search(["freq:high"])
        assert len(results) == 1
    
    def test_special_characters_in_pattern(self, pattern_index):
        """Should handle special characters in patterns."""
        pattern_index.add("mem1", ["category:value-with-dash", "cat:with_underscore"])
        
        results = pattern_index.search(["category:value-with-dash"])
        assert "mem1" in results
        
        results = pattern_index.search(["cat:with_underscore"])
        assert "mem1" in results
    
    def test_get_all_patterns(self, pattern_index):
        """Should list all unique patterns."""
        pattern_index.add("mem1", ["freq:high", "amp:loud"])
        pattern_index.add("mem2", ["freq:low", "amp:loud"])
        
        all_patterns = pattern_index.get_all_patterns()
        
        assert "freq:high" in all_patterns
        assert "freq:low" in all_patterns
        assert "amp:loud" in all_patterns
        assert len(all_patterns) == 3
    
    def test_get_patterns_for_memory(self, pattern_index):
        """Should retrieve patterns for a specific memory."""
        pattern_index.add("mem1", ["freq:high", "amp:loud"])
        
        patterns = pattern_index.get_patterns("mem1")
        
        assert "freq:high" in patterns
        assert "amp:loud" in patterns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

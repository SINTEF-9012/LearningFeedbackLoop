"""
Pattern Index - Inverted index for fast pattern key lookups.

This module provides an in-memory inverted index that enables efficient
retrieval of memories by their pattern key components.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Set, Dict, Optional
from dataclasses import dataclass, field

from .memory_schema import PatternKey


@dataclass
class PatternIndex:
    """
    In-memory inverted index for PatternKey lookups.
    
    Structure:
    - Each component (condition, machine_type, fault_type, channel) has its own posting list
    - Memory IDs are stored in sets for fast intersection
    - Supports both exact match and partial match queries
    
    Performance:
    - Add: O(k) where k = number of non-null pattern key components
    - Query: O(m) where m = number of matching memories
    - Remove: O(k)
    """
    
    # Posting lists: component_type -> value -> set of memory_ids
    _condition_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _machine_type_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _fault_type_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _channel_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _additional_index: Dict[str, Dict[str, Set[str]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))
    
    # Reverse index: memory_id -> PatternKey (for removal)
    _memory_patterns: Dict[str, PatternKey] = field(default_factory=dict)
    
    def add(self, memory_id: str, pattern_key: PatternKey) -> None:
        """
        Add a memory to the index.
        
        Args:
            memory_id: Unique memory identifier
            pattern_key: The pattern key to index
        """
        # Remove existing entry if present (update case)
        if memory_id in self._memory_patterns:
            self.remove(memory_id)
        
        # Store pattern for reverse lookup
        self._memory_patterns[memory_id] = pattern_key
        
        # Index each component
        if pattern_key.condition:
            self._condition_index[pattern_key.condition.lower()].add(memory_id)
        
        if pattern_key.machine_type:
            self._machine_type_index[pattern_key.machine_type.lower()].add(memory_id)
        
        if pattern_key.fault_type:
            self._fault_type_index[pattern_key.fault_type.lower()].add(memory_id)
        
        if pattern_key.channel:
            self._channel_index[pattern_key.channel.lower()].add(memory_id)
        
        # Index additional fields
        if pattern_key.additional:
            for key, value in pattern_key.additional.items():
                self._additional_index[key.lower()][str(value).lower()].add(memory_id)
    
    def remove(self, memory_id: str) -> bool:
        """
        Remove a memory from the index.
        
        Returns:
            True if removed, False if not found
        """
        pattern_key = self._memory_patterns.pop(memory_id, None)
        if pattern_key is None:
            return False
        
        # Remove from each index
        if pattern_key.condition:
            self._condition_index[pattern_key.condition.lower()].discard(memory_id)
        
        if pattern_key.machine_type:
            self._machine_type_index[pattern_key.machine_type.lower()].discard(memory_id)
        
        if pattern_key.fault_type:
            self._fault_type_index[pattern_key.fault_type.lower()].discard(memory_id)
        
        if pattern_key.channel:
            self._channel_index[pattern_key.channel.lower()].discard(memory_id)
        
        if pattern_key.additional:
            for key, value in pattern_key.additional.items():
                self._additional_index[key.lower()][str(value).lower()].discard(memory_id)
        
        return True
    
    def query(
        self,
        pattern: PatternKey,
        partial_match: bool = False
    ) -> List[str]:
        """
        Query for memories matching a pattern.
        
        Args:
            pattern: Pattern key to match against
            partial_match: If True, match any component. If False, match all specified components.
            
        Returns:
            List of matching memory IDs
        """
        result_sets: List[Set[str]] = []
        
        # Collect matching sets for each specified component
        if pattern.condition:
            matches = self._condition_index.get(pattern.condition.lower(), set())
            result_sets.append(matches)
        
        if pattern.machine_type:
            matches = self._machine_type_index.get(pattern.machine_type.lower(), set())
            result_sets.append(matches)
        
        if pattern.fault_type:
            matches = self._fault_type_index.get(pattern.fault_type.lower(), set())
            result_sets.append(matches)
        
        if pattern.channel:
            matches = self._channel_index.get(pattern.channel.lower(), set())
            result_sets.append(matches)
        
        if pattern.additional:
            for key, value in pattern.additional.items():
                matches = self._additional_index.get(key.lower(), {}).get(str(value).lower(), set())
                result_sets.append(matches)
        
        if not result_sets:
            # No constraints specified - return all
            return list(self._memory_patterns.keys())
        
        if partial_match:
            # Union: match any component
            result = set().union(*result_sets)
        else:
            # Intersection: match all components
            result = result_sets[0].copy()
            for s in result_sets[1:]:
                result &= s
        
        return list(result)
    
    def query_by_component(
        self,
        component: str,
        value: str
    ) -> List[str]:
        """
        Query by a single component.
        
        Args:
            component: One of 'condition', 'machine_type', 'fault_type', 'channel'
            value: The value to match
            
        Returns:
            List of matching memory IDs
        """
        value_lower = value.lower()
        
        if component == "condition":
            return list(self._condition_index.get(value_lower, set()))
        elif component == "machine_type":
            return list(self._machine_type_index.get(value_lower, set()))
        elif component == "fault_type":
            return list(self._fault_type_index.get(value_lower, set()))
        elif component == "channel":
            return list(self._channel_index.get(value_lower, set()))
        else:
            # Check additional index
            return list(self._additional_index.get(component.lower(), {}).get(value_lower, set()))
    
    def get_all_values(self, component: str) -> List[str]:
        """
        Get all unique values for a component.
        
        Args:
            component: One of 'condition', 'machine_type', 'fault_type', 'channel'
            
        Returns:
            List of unique values
        """
        if component == "condition":
            return list(self._condition_index.keys())
        elif component == "machine_type":
            return list(self._machine_type_index.keys())
        elif component == "fault_type":
            return list(self._fault_type_index.keys())
        elif component == "channel":
            return list(self._channel_index.keys())
        else:
            return list(self._additional_index.get(component.lower(), {}).keys())
    
    def size(self) -> int:
        """Get total number of indexed memories."""
        return len(self._memory_patterns)
    
    def clear(self) -> None:
        """Clear the entire index."""
        self._condition_index.clear()
        self._machine_type_index.clear()
        self._fault_type_index.clear()
        self._channel_index.clear()
        self._additional_index.clear()
        self._memory_patterns.clear()
    
    def stats(self) -> Dict[str, int]:
        """Get index statistics."""
        return {
            "total_memories": len(self._memory_patterns),
            "unique_conditions": len(self._condition_index),
            "unique_machine_types": len(self._machine_type_index),
            "unique_fault_types": len(self._fault_type_index),
            "unique_channels": len(self._channel_index),
            "additional_keys": len(self._additional_index),
        }

"""Pattern Index.

This module is intentionally small and test-friendly.

Primary API (used by unit tests):
- add(memory_id, patterns: list[str])
- search(patterns: list[str]) -> set[str]          # AND semantics
- search_any(patterns: list[str]) -> set[str]      # OR semantics
- remove(memory_id, patterns: list[str] | None)
- save(path) / load(path)
- get_all_patterns() / get_patterns(memory_id)

Compatibility:
- MemoryStore and other runtime code may call add(memory_id, PatternKey) or
    query(pattern: PatternKey, partial_match=...). Those calls are supported by
    indexing/matching on PatternKey.key.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Union

from ..core.schemas import PatternKey


def _normalize_pattern(pattern: str) -> str:
    return str(pattern).strip()


def _coerce_patterns(patterns: Union[PatternKey, str, Iterable[str]]) -> List[str]:
    if isinstance(patterns, PatternKey):
        if patterns.key:
            return [_normalize_pattern(patterns.key)]
        return []
    if isinstance(patterns, str):
        return [_normalize_pattern(patterns)] if patterns.strip() else []
    return [_normalize_pattern(p) for p in patterns if str(p).strip()]


@dataclass
class PatternIndex:
    """String-pattern inverted index."""

    index_path: Optional[str] = None
    _pattern_to_memories: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _memory_to_patterns: Dict[str, Set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index_path:
            self.load(self.index_path)

    def add(self, memory_id: str, patterns: Union[PatternKey, str, List[str]]) -> None:
        pats = _coerce_patterns(patterns)
        if not pats:
            # Keep behavior simple for tests: don't index empty pattern lists.
            self._memory_to_patterns.setdefault(memory_id, set())
            return

        existing = self._memory_to_patterns.get(memory_id)
        if existing is None:
            existing = set()
            self._memory_to_patterns[memory_id] = existing

        for p in pats:
            if p in existing:
                continue
            existing.add(p)
            self._pattern_to_memories[p].add(memory_id)

    def remove(self, memory_id: str, patterns: Optional[List[str]] = None) -> None:
        if memory_id not in self._memory_to_patterns:
            return

        if patterns is None:
            pats = list(self._memory_to_patterns.get(memory_id, set()))
        else:
            pats = _coerce_patterns(patterns)

        for p in pats:
            self._pattern_to_memories.get(p, set()).discard(memory_id)
            if p in self._pattern_to_memories and not self._pattern_to_memories[p]:
                self._pattern_to_memories.pop(p, None)
            self._memory_to_patterns.get(memory_id, set()).discard(p)

        if not self._memory_to_patterns.get(memory_id):
            self._memory_to_patterns.pop(memory_id, None)

    def search(self, patterns: List[str]) -> Set[str]:
        pats = _coerce_patterns(patterns)
        if not pats:
            return set()

        result: Optional[Set[str]] = None
        for p in pats:
            mems = self._pattern_to_memories.get(p, set())
            if result is None:
                result = set(mems)
            else:
                result &= mems
            if not result:
                return set()
        return result or set()

    def search_any(self, patterns: List[str]) -> Set[str]:
        pats = _coerce_patterns(patterns)
        if not pats:
            return set()

        result: Set[str] = set()
        for p in pats:
            result |= self._pattern_to_memories.get(p, set())
        return result

    def get_all_patterns(self) -> Set[str]:
        return set(self._pattern_to_memories.keys())

    def get_patterns(self, memory_id: str) -> Set[str]:
        return set(self._memory_to_patterns.get(memory_id, set()))

    def clear(self) -> None:
        """Remove all indexed patterns.

        Runtime code (e.g., MemoryStore.clear()) expects this method.
        """
        self._pattern_to_memories.clear()
        self._memory_to_patterns.clear()

    def size(self) -> int:
        """Return number of distinct patterns in the index."""
        return len(self._pattern_to_memories)

    def save(self, path: str) -> None:
        payload = {
            "pattern_to_memories": {k: sorted(v) for k, v in self._pattern_to_memories.items()},
            "memory_to_patterns": {k: sorted(v) for k, v in self._memory_to_patterns.items()},
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def load(self, path: str) -> None:
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return

        self._pattern_to_memories.clear()
        self._memory_to_patterns.clear()

        for pattern, mem_ids in (payload.get("pattern_to_memories") or {}).items():
            self._pattern_to_memories[_normalize_pattern(pattern)].update(mem_ids or [])

        for mem_id, patterns in (payload.get("memory_to_patterns") or {}).items():
            self._memory_to_patterns[mem_id] = set(_coerce_patterns(patterns or []))

    # ---------------------------------------------------------------------
    # Compatibility helpers (legacy callers)
    # ---------------------------------------------------------------------

    def query(self, pattern: PatternKey, partial_match: bool = False) -> List[str]:
        # Current runtime code uses PatternKey-based matching. Treat it as a key
        # string lookup (exact). partial_match is ignored.
        _ = partial_match
        return sorted(self.search([pattern.key] if pattern and pattern.key else []))


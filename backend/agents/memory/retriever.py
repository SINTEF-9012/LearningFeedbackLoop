"""
Memory Retriever - Context-aware retrieval of similar memories.

# ===========================================================================
# DRAFT/PROTOTYPE - Tag: [PROTOTYPE_LLM_MEMORY_V1]
# This module implements the three-layer matching strategy for memory retrieval.
# Layer 1: Context filter (cutting conditions)
# Layer 2: Pattern key matching (symbolic)
# Layer 3: Numeric similarity (feature vectors)
# ===========================================================================

Retrieves memories that are relevant to a query based on:
1. Similar cutting conditions (context)
2. Matching or related pattern keys
3. Numeric feature vector similarity
4. User feedback weighting
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np

from ..core.schemas import (
    Memory,
    PatternKey,
    PatternType,
    NumericMetrics,
    MemoryQueryResult,
    TimeRange,
)
from ..core.context import CuttingContext, ContextTolerance
from ..core.metrics import WindowMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level sentence-transformer cache (loaded once, used by both
# MemoryRetriever and RetrieverAgent).
# ---------------------------------------------------------------------------
_EMBEDDING_MODEL: Any = None
_EMBEDDING_MODEL_LOADED: bool = False


def _get_embedding_model() -> Any:
    """Return the cached sentence-transformer model, loading it on first call."""
    global _EMBEDDING_MODEL, _EMBEDDING_MODEL_LOADED
    if _EMBEDDING_MODEL_LOADED:
        return _EMBEDDING_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        logger.info("Loaded embedding model: paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as exc:
        logger.warning("Sentence-transformer unavailable: %s", exc)
        _EMBEDDING_MODEL = None
    _EMBEDDING_MODEL_LOADED = True
    return _EMBEDDING_MODEL


# [PROTOTYPE_LLM_MEMORY_V1] - Match result
@dataclass
class MemoryMatch:
    """A matched memory with scoring breakdown."""
    memory: Memory
    
    # Overall score (0-1)
    relevance_score: float
    
    # Score components
    context_score: float = 0.0  # Layer 1
    pattern_score: float = 0.0  # Layer 2
    vector_score: float = 0.0  # Layer 3
    feedback_boost: float = 0.0  # User feedback adjustment
    
    # Match details
    matched_patterns: List[str] = field(default_factory=list)
    match_reasons: List[str] = field(default_factory=list)
    
    def to_query_result(self) -> MemoryQueryResult:
        """Convert to MemoryQueryResult schema."""
        return MemoryQueryResult(
            memory=self.memory,
            relevance_score=self.relevance_score,
            match_reasons=self.match_reasons,
            pattern_matches=[
                PatternKey(pattern_type=PatternType.CUSTOM, key=p)
                for p in self.matched_patterns
            ],
        )


# [PROTOTYPE_LLM_MEMORY_V1] - Retrieval configuration
@dataclass
class RetrievalConfig:
    """Configuration for memory retrieval."""
    # Weight for each layer
    context_weight: float = 0.3
    pattern_weight: float = 0.4
    vector_weight: float = 0.3
    
    # Context matching
    context_tolerance: ContextTolerance = field(default_factory=ContextTolerance)
    require_context_match: bool = False  # If True, filter out non-matching contexts
    
    # Pattern matching
    exact_pattern_boost: float = 1.0
    partial_pattern_boost: float = 0.5
    
    # Feedback weighting
    confirmed_boost: float = 1.3  # Multiplier for confirmed memories
    dismissed_penalty: float = 0.5  # Multiplier for dismissed memories
    
    # Retrieval limits
    max_candidates: int = 100  # Max candidates after context filter
    

# [PROTOTYPE_LLM_MEMORY_V1] - Pattern matching utilities
class PatternMatcher:
    """
    Utilities for pattern key matching.
    
    Supports:
    - Exact matching
    - Partial/hierarchical matching (RATIO_Fx_Fy:>5 ~ RATIO_Fx_Fy:2-5)
    - Pattern family matching (all RATIO patterns)
    """
    
    @staticmethod
    def exact_match(p1: str, p2: str) -> bool:
        """Check exact pattern match."""
        return p1.lower() == p2.lower()
    
    @staticmethod
    def family_match(p1: str, p2: str) -> bool:
        """
        Check if patterns belong to same family.
        
        Examples:
        - "RATIO_Fx_Fy:>5" and "RATIO_Fx_Fy:2-5" -> True (same variables)
        - "RATIO_Fx_Fy:>5" and "RATIO_Fz_Fy:>5" -> False (different variables)
        """
        # Extract family (prefix before last colon)
        def get_family(p: str) -> str:
            if ":" in p:
                return p.rsplit(":", 1)[0].lower()
            return p.lower()
        
        return get_family(p1) == get_family(p2)
    
    @staticmethod
    def type_match(p1: str, p2: str) -> bool:
        """
        Check if patterns are same type (first part of key).
        
        Examples:
        - "RATIO_Fx_Fy:>5" and "RATIO_Fz_My:>3" -> True (both RATIO)
        - "RATIO_Fx_Fy:>5" and "SPECTRAL_PEAK_512Hz" -> False
        """
        def get_type(p: str) -> str:
            return p.split("_")[0].lower()
        
        return get_type(p1) == get_type(p2)
    
    @staticmethod
    def score_pattern_similarity(query_patterns: List[str], memory_patterns: List[str]) -> Tuple[float, List[str]]:
        """
        Score pattern similarity between query and memory.

        Returns:
            (score, list of matched patterns)

        Note (Agent Q Round 19, 2026-04-24): this method now delegates to
        :class:`QueryPatternIndex`. For hot paths that score a single query
        against many candidates, build a ``QueryPatternIndex`` once and
        call :meth:`QueryPatternIndex.score_against` per candidate — that
        avoids re-tokenising the query on every invocation. The static
        method stays for backward compatibility with external callers.
        """
        return QueryPatternIndex(query_patterns).score_against(memory_patterns)


# Agent Q (Round 19, 2026-04-24): pre-index to accelerate the pattern
# scoring hot path. ``_score_and_rank`` calls ``score_pattern_similarity``
# once per candidate memory (up to ``max_candidates``); precomputing the
# query-side lower/family/type tokens here moves work out of the
# per-candidate inner loop.
class QueryPatternIndex:
    """Precomputed query-side tokens for pattern similarity scoring.

    Build once per retrieve() call with the query's pattern keys, then
    call :meth:`score_against` per candidate memory. Preserves the
    legacy semantics of ``PatternMatcher.score_pattern_similarity``
    exactly: for each query pattern, scan memory patterns in order and
    take the first match, with exact > family > type precedence applied
    only when that specific memory pattern matches.
    """

    __slots__ = ("_patterns", "_tokens")

    def __init__(self, query_patterns: List[str]):
        self._patterns: List[str] = list(query_patterns)
        # Tuple of (lower, family, type) per query pattern, in order.
        self._tokens: List[Tuple[str, str, str]] = [
            self._tokenise(qp) for qp in self._patterns
        ]

    @staticmethod
    def _tokenise(pattern: str) -> Tuple[str, str, str]:
        low = pattern.lower()
        family = low.rsplit(":", 1)[0] if ":" in low else low
        ptype = low.split("_", 1)[0]
        return (low, family, ptype)

    def score_against(self, memory_patterns: List[str]) -> Tuple[float, List[str]]:
        """Score this query against a candidate memory's pattern list.

        Returns ``(score, matched_patterns)`` with identical semantics to
        :meth:`PatternMatcher.score_pattern_similarity`.
        """
        if not self._patterns or not memory_patterns:
            return (0.0, [])

        # Tokenise memory patterns once per candidate. ``mem_tokens`` is
        # an ordered list so the legacy "first match wins" behaviour is
        # preserved.
        mem_tokens = [(mp,) + self._tokenise(mp) for mp in memory_patterns]

        matched: List[str] = []
        score_sum = 0.0
        for q_low, q_family, q_type in self._tokens:
            for mp, m_low, m_family, m_type in mem_tokens:
                if q_low == m_low:
                    matched.append(mp)
                    score_sum += 1.0
                    break
                if q_family == m_family:
                    matched.append(mp)
                    score_sum += 0.6
                    break
                if q_type == m_type:
                    matched.append(mp)
                    score_sum += 0.3
                    break

        score = score_sum / len(self._patterns)
        return (min(1.0, score), matched)


# [PROTOTYPE_LLM_MEMORY_V1] - Main retriever class
class MemoryRetriever:
    """
    Retrieves similar memories using three-layer matching.
    
    [INTEGRATION_POINT] Requires MemoryStore for data access.
    """
    
    def __init__(
        self,
        memory_store: Any,  # MemoryStore instance
        config: Optional[RetrievalConfig] = None
    ):
        self.store = memory_store
        self.config = config or RetrievalConfig()
        
        # [PROTOTYPE_LLM_MEMORY_V1] - In-memory feedback cache
        # Production should use database
        self._feedback_cache: Dict[str, str] = {}  # memory_id -> "confirmed" | "dismissed"
    
    def retrieve(
        self,
        query_patterns: List[PatternKey],
        query_metrics: Optional[WindowMetrics] = None,
        query_context: Optional[CuttingContext] = None,
        session_id: Optional[str] = None,
        top_k: int = 10,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[MemoryMatch]:
        """
        Retrieve memories similar to query.
        
        Args:
            query_patterns: Pattern keys from current event
            query_metrics: Computed metrics for current window
            query_context: Cutting conditions for current operation
            session_id: Current session (can exclude or boost same-session)
            top_k: Number of results to return
            exclude_ids: Memory IDs to exclude (e.g., current event)
        
        Returns:
            List of MemoryMatch sorted by relevance
        """
        exclude_ids = exclude_ids or set()

        # --- Fast path: Neo4j vector search ---
        # When the store is Neo4jMemoryStore and we have metrics, use the
        # native vector index for Layer 3, then refine with Layers 1 & 2.
        if (
            query_metrics is not None
            and hasattr(self.store, 'vector_search')
        ):
            return self._retrieve_via_vector_index(
                query_patterns=query_patterns,
                query_metrics=query_metrics,
                query_context=query_context,
                top_k=top_k,
                exclude_ids=exclude_ids,
            )

        # --- Standard path: in-memory retrieval ---
        # Layer 1: Context filter
        candidates = self._filter_by_context(query_context)
        logger.debug("Layer 1 (context): %d candidates", len(candidates))
        
        # Limit candidates for efficiency
        if len(candidates) > self.config.max_candidates:
            # [PROTOTYPE_LLM_MEMORY_V1] - Simple random sample
            # Should use smarter selection in production
            candidates = random.sample(candidates, self.config.max_candidates)
        
        return self._score_and_rank(
            candidates, query_patterns, query_metrics, query_context,
            top_k, exclude_ids,
        )
    
    def update_feedback(self, memory_id: str, feedback_type: str):
        """
        Update feedback for a memory.
        
        [INTEGRATION_POINT] Should persist to database in production.
        """
        self._feedback_cache[memory_id] = feedback_type
    
    def _filter_by_context(
        self, 
        query_context: Optional[CuttingContext]
    ) -> List[Memory]:
        """
        Layer 1: Filter memories by cutting context.
        
        [PROTOTYPE_LLM_MEMORY_V1] - Currently loads all and filters in-memory.
        Production should use SQL filtering.
        """
        # Get all active memories
        # [INTEGRATION_POINT] MemoryStore should expose a list/filter method
        all_memories = self._get_all_memories()
        
        if query_context is None or not self.config.require_context_match:
            return all_memories
        
        filtered = []
        for memory in all_memories:
            # Try to get stored context from metadata
            if hasattr(memory, 'metadata') and memory.metadata:
                stored_context = memory.metadata.get('cutting_context')
                if stored_context:
                    try:
                        mem_ctx = CuttingContext(**stored_context)
                        is_match, _ = query_context.matches(mem_ctx, self.config.context_tolerance)
                        if is_match:
                            filtered.append(memory)
                            continue
                    except Exception:
                        pass
            
            # Include memories without context info
            if not self.config.require_context_match:
                filtered.append(memory)
        
        return filtered
    
    def _get_all_memories(self) -> List[Memory]:
        """
        Get all memories from store.
        
        [PROTOTYPE_LLM_MEMORY_V1] - Placeholder implementation.
        Should use proper MemoryStore interface.
        """
        # Prefer a stable list API if available.
        if hasattr(self.store, "list_all"):
            return self.store.list_all(limit=self.config.max_candidates)

        # Back-compat for older store implementations.
        if hasattr(self.store, "query"):
            try:
                return self.store.query(limit=self.config.max_candidates)
            except TypeError:
                logger.exception("MemoryStore.query signature is incompatible")
                return []

        logger.warning("MemoryStore does not expose list interface")
        return []
    
    def _vector_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity between vectors.
        
        [PROTOTYPE_LLM_MEMORY_V1] - Simple cosine similarity.
        Should use domain-specific normalization per cutting regime.
        """
        # Handle dimension mismatch
        if v1.shape != v2.shape:
            min_len = min(len(v1), len(v2))
            v1 = v1[:min_len]
            v2 = v2[:min_len]
        
        # Cosine similarity
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        similarity = np.dot(v1, v2) / (norm1 * norm2)
        # Convert from [-1, 1] to [0, 1]
        return float((similarity + 1.0) / 2.0)
    
    def _get_feedback_boost(self, memory_id: str) -> float:
        """Get feedback-based score multiplier."""
        feedback = self._feedback_cache.get(memory_id)
        if feedback == "confirmed":
            return self.config.confirmed_boost
        elif feedback == "dismissed":
            return self.config.dismissed_penalty
        return 1.0  # Neutral


    # ------------------------------------------------------------------
    # Neo4j vector-index retrieval (Phase 5)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_query_text(patterns: List[PatternKey], context: Optional[CuttingContext] = None) -> str:
        """Build a text string from query patterns + context for embedding.

        The Neo4j vector index stores **text embeddings** (384-dim,
        sentence-transformer).  We must embed the query in the *same*
        space — not use the numeric metrics vector which lives in a
        completely different vector space.
        """
        parts: List[str] = []
        for p in patterns:
            parts.append(p.key)
            if p.fault_type:
                parts.append(p.fault_type)
        if context:
            if context.tool_type:
                parts.append(f"tool:{context.tool_type}")
            if context.workpiece_material:
                parts.append(f"material:{context.workpiece_material}")
            if context.machine_type:
                parts.append(f"machine:{context.machine_type}")
        return " ".join(parts) if parts else "vibration event"

    @staticmethod
    def _compute_text_embedding(text: str) -> Optional[List[float]]:
        """Compute a 384-dim text embedding using the project's sentence-transformer.

        The model is cached at module level so it is loaded at most once.
        """
        try:
            model = _get_embedding_model()
            if model is None:
                return None
            return model.encode(text).tolist()
        except Exception:
            return None

    def _retrieve_via_vector_index(
        self,
        query_patterns: List[PatternKey],
        query_metrics: WindowMetrics,
        query_context: Optional[CuttingContext],
        top_k: int,
        exclude_ids: Set[str],
    ) -> List[MemoryMatch]:
        """Use Neo4j's native vector index for efficient ANN search.

        Builds a **text embedding** from the query patterns and context
        (same vector space as the stored embeddings) and queries the
        Neo4j vector index.  The returned candidates are then re-ranked
        using the full three-layer scoring (context + pattern + vector).

        Falls back to the standard in-memory path on error.
        """
        try:
            query_text = self._build_query_text(query_patterns, query_context)
            query_embedding = self._compute_text_embedding(query_text)
            if query_embedding is None:
                raise RuntimeError("Text embedding model unavailable")

            # Fetch more candidates than top_k so re-ranking has room
            raw_results = self.store.vector_search(
                query_embedding=query_embedding,
                top_k=min(top_k * 3, self.config.max_candidates),
            )
        except Exception as exc:
            logger.warning("Neo4j vector search failed, falling back: %s", exc)
            candidates = self._filter_by_context(query_context)
            return self._score_and_rank(
                candidates, query_patterns, query_metrics, query_context,
                top_k, exclude_ids,
            )

        # Use the candidates from vector search but re-score with all 3 layers
        candidate_memories = [mem for mem, _score in raw_results]
        return self._score_and_rank(
            candidate_memories, query_patterns, query_metrics, query_context,
            top_k, exclude_ids,
        )

    def _score_and_rank(
        self,
        candidates: List[Memory],
        query_patterns: List[PatternKey],
        query_metrics: Optional[WindowMetrics],
        query_context: Optional[CuttingContext],
        top_k: int,
        exclude_ids: Set[str],
    ) -> List[MemoryMatch]:
        """Re-usable scoring loop (extracted from retrieve())."""
        query_pattern_keys = [p.key for p in query_patterns]
        query_vector = query_metrics.to_vector() if query_metrics else None

        # Agent Q (Round 19, 2026-04-24): build the query-side pattern
        # index once and reuse for every candidate. Previously each
        # candidate went through ``PatternMatcher.score_pattern_similarity``
        # which re-tokenised the query on every call.
        query_index = QueryPatternIndex(query_pattern_keys)

        matches: List[MemoryMatch] = []
        for memory in candidates:
            if memory.id in exclude_ids:
                continue

            mem_pattern_keys = [p.key for p in memory.pattern_keys]
            pattern_score, matched_patterns = query_index.score_against(
                mem_pattern_keys,
            )
            vector_score = 0.0
            if query_vector is not None and memory.numeric_vector:
                mem_vec = np.array(memory.numeric_vector, dtype=np.float32)
                vector_score = self._vector_similarity(query_vector, mem_vec)

            context_score = 0.5
            if query_context and hasattr(memory, 'metadata') and memory.metadata:
                stored = memory.metadata.get('cutting_context')
                if stored:
                    try:
                        mem_ctx = CuttingContext(**stored)
                        _, context_score = query_context.matches(
                            mem_ctx, self.config.context_tolerance,
                        )
                    except Exception:
                        pass

            raw_score = (
                self.config.context_weight * context_score
                + self.config.pattern_weight * pattern_score
                + self.config.vector_weight * vector_score
            )
            feedback_boost = self._get_feedback_boost(memory.id)
            final_score = raw_score * feedback_boost

            match_reasons: List[str] = []
            if matched_patterns:
                match_reasons.append(f"Patterns: {', '.join(matched_patterns[:3])}")
            if vector_score > 0.7:
                match_reasons.append(f"High feature similarity ({vector_score:.2f})")
            if context_score > 0.8:
                match_reasons.append("Similar cutting conditions")

            matches.append(MemoryMatch(
                memory=memory,
                relevance_score=final_score,
                context_score=context_score,
                pattern_score=pattern_score,
                vector_score=vector_score,
                feedback_boost=feedback_boost,
                matched_patterns=matched_patterns,
                match_reasons=match_reasons,
            ))

        matches.sort(key=lambda m: m.relevance_score, reverse=True)
        return matches[:top_k]


# [PROTOTYPE_LLM_MEMORY_V1] - Factory function
def create_retriever(memory_store: Any, config: Optional[RetrievalConfig] = None) -> MemoryRetriever:
    """Create a MemoryRetriever instance."""
    return MemoryRetriever(memory_store, config)

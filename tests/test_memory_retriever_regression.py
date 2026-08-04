from __future__ import annotations

from backend.agents.core.schemas import Memory, PatternKey, PatternType
from backend.agents.memory.retriever import MemoryRetriever, RetrievalConfig
from backend.agents.storage.store import MemoryStore


def test_memory_retriever_does_not_call_store_query_with_limit_kwarg() -> None:
    """Regression test for API-mode failures.

    Previously, MemoryRetriever attempted to call MemoryStore.query(limit=...),
    but MemoryStore.query expects a MemoryQuery object.
    """
    store = MemoryStore(db_path=":memory:", enable_ann=False)
    try:
        memory_id = store.create(
            Memory(
                session_id="s1",
                time_range=(0.0, 1.0),
                channels=["Fx"],
                annotation_text="chatter",
                pattern_keys=[
                    PatternKey(pattern_type=PatternType.CUSTOM, key="CHATTER_DETECTED"),
                    PatternKey(pattern_type=PatternType.CUSTOM, key="RATIO_Fx_Fy:>5"),
                ],
            )
        )

        retriever = MemoryRetriever(store, config=RetrievalConfig(max_candidates=25))
        matches = retriever.retrieve(
            query_patterns=[
                PatternKey(pattern_type=PatternType.CUSTOM, key="CHATTER_DETECTED"),
                PatternKey(pattern_type=PatternType.CUSTOM, key="RATIO_Fx_Fy:>5"),
            ],
            top_k=5,
            exclude_ids={""},
        )

        assert matches, "Expected at least one match"
        assert matches[0].memory.id == memory_id
        assert matches[0].relevance_score > 0.5
    finally:
        store.close()

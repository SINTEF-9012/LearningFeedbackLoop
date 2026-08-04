from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import time

from backend.agents.core.schemas import Memory, PatternKey, PatternType
from backend.agents.storage.store import MemoryStore


def test_sqlite_store_doc_link_feedback_reorders_links_by_feedback() -> None:
    store = MemoryStore(db_path=":memory:", enable_ann=False, enable_embeddings=False)
    memory = Memory(
        id="mem-doc-links",
        session_id="session-1",
        time_range=(0.0, 1.0),
        created_at=datetime.now(timezone.utc),
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="fault:chatter")],
    )

    store.store(memory)
    store.persist_doc_links(
        memory_id="mem-doc-links",
        pattern_keys=["fault:chatter"],
        doc_links=[
            {
                "id": "doc-high-score",
                "citation": "high score",
                "score": 0.95,
                "query_used": "high score query",
                "pattern_key": "fault:chatter",
            },
            {
                "id": "doc-helpful",
                "citation": "helpful lower raw score",
                "score": 0.71,
                "query_used": "helpful query",
                "pattern_key": "fault:chatter",
            },
        ],
    )

    updated = store.set_doc_link_feedback(
        memory_id="mem-doc-links",
        doc_id="doc-helpful",
        feedback="helpful",
        user_id="operator",
        reason="Most relevant guidance",
    )
    links = store.get_doc_links("mem-doc-links")

    assert updated is not None
    assert updated["helpful_count"] == 1
    assert updated["feedback_score"] == 1.0
    assert links[0]["id"] == "doc-helpful"
    assert links[0]["doc_feedback"] == "helpful"


def test_sqlite_store_doc_link_feedback_returns_none_for_missing_doc() -> None:
    store = MemoryStore(db_path=":memory:", enable_ann=False, enable_embeddings=False)
    memory = Memory(
        id="mem-doc-links-missing",
        session_id="session-1",
        time_range=(0.0, 1.0),
        created_at=datetime.now(timezone.utc),
    )
    store.store(memory)

    updated = store.set_doc_link_feedback(
        memory_id="mem-doc-links-missing",
        doc_id="doc-missing",
        feedback="helpful",
        user_id="operator",
    )

    assert updated is None


def test_sqlite_store_doc_link_feedback_preserves_counts_under_same_process_concurrency(monkeypatch) -> None:
    store = MemoryStore(db_path=":memory:", enable_ann=False, enable_embeddings=False)
    memory = Memory(
        id="mem-doc-links-concurrent",
        session_id="session-1",
        time_range=(0.0, 1.0),
        created_at=datetime.now(timezone.utc),
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="fault:chatter")],
    )

    store.store(memory)
    store.persist_doc_links(
        memory_id="mem-doc-links-concurrent",
        pattern_keys=["fault:chatter"],
        doc_links=[
            {
                "id": "doc-helpful",
                "citation": "helpful lower raw score",
                "score": 0.71,
                "query_used": "helpful query",
                "pattern_key": "fault:chatter",
            },
        ],
    )

    original_get = store.get

    def slow_get(memory_id: str):
        memory_obj = original_get(memory_id)
        time.sleep(0.02)
        return memory_obj

    monkeypatch.setattr(store, "get", slow_get)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                store.set_doc_link_feedback,
                memory_id="mem-doc-links-concurrent",
                doc_id="doc-helpful",
                feedback="helpful",
                user_id=f"operator-{index}",
            )
            for index in range(2)
        ]
        for future in futures:
            assert future.result() is not None

    links = store.get_doc_links("mem-doc-links-concurrent")
    assert links[0]["helpful_count"] == 2
    assert links[0]["feedback_score"] == 2.0
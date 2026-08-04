"""Agent E — unit tests for get_similar_with_resolution post-processing.

We don't have a live Neo4j. The test constructs a Neo4jMemoryStore via
``__new__`` (skipping __init__) and stubs ``_run`` to return canned rows.
This validates the post-processing: feedback roll-up, shared-pattern
intersection, ranking.
"""
from __future__ import annotations

from typing import Any, Dict, List

from backend.agents.storage.neo4j_store import Neo4jMemoryStore


def _store_with_rows(rows: List[Dict[str, Any]]) -> Neo4jMemoryStore:
    store = Neo4jMemoryStore.__new__(Neo4jMemoryStore)  # bypass __init__
    # Monkeypatch _run to return our canned rows regardless of Cypher.
    store._run = lambda *args, **kwargs: rows  # type: ignore[method-assign]
    store._database = "neo4j"
    return store


def _row(
    mid: str,
    *,
    shared: List[str],
    cand_patterns: List[str],
    feedbacks: List[Dict[str, Any]],
    shared_pattern_details: List[Dict[str, Any]] | None = None,
    created_at: str = "2026-04-20T10:00:00+00:00",
    annotation: str = "",
    label: str | None = None,
) -> Dict[str, Any]:
    return {
        "mem_props": {
            "id": mid,
            "session_id": "s1",
            "created_at": created_at,
            "annotation_text": annotation,
            "label": label,
            "machine_uri": "urn:test:cnc1",
        },
        "shared_patterns": shared,
        "shared_pattern_details": shared_pattern_details or [],
        "feedbacks": feedbacks,
    }


def test_basic_roll_up_confirm_and_dismiss():
    rows = [
        _row(
            "m1",
            shared=["fault:chatter"],
            cand_patterns=["fault:chatter"],
            feedbacks=[
                {"action": "confirm", "timestamp": "2026-04-01T00:00:00+00:00", "comment": "changed tool"},
                {"action": "confirm", "timestamp": "2026-04-05T00:00:00+00:00"},
                {"action": "dismiss", "timestamp": "2026-04-10T00:00:00+00:00", "comment": "false alarm"},
            ],
        )
    ]
    store = _store_with_rows(rows)
    out = store.get_similar_with_resolution("q", k=5)
    assert len(out) == 1
    entry = out[0]
    assert entry["id"] == "m1"
    assert entry["shared_pattern_keys"] == ["fault:chatter"]
    assert entry["feedback"]["confirm_count"] == 2
    assert entry["feedback"]["dismiss_count"] == 1
    # Last action is sorted by timestamp — dismiss is most recent
    assert entry["feedback"]["last_action"] == "dismiss"
    assert entry["feedback"]["last_comment"] == "false alarm"


def test_shared_pattern_details_preserve_strength_metadata():
    rows = [
        _row(
            "m1",
            shared=["signature:spindle_shift_phase_change"],
            cand_patterns=["signature:spindle_shift_phase_change"],
            shared_pattern_details=[
                {
                    "key": "signature:spindle_shift_phase_change",
                    "query_strength": 0.61,
                    "query_source_metric": "phase_differences",
                    "candidate_strength": 0.83,
                    "candidate_source_metric": "phase_differences",
                }
            ],
            feedbacks=[],
        )
    ]

    store = _store_with_rows(rows)
    out = store.get_similar_with_resolution("q", k=5)

    assert out[0]["shared_pattern_keys"] == ["signature:spindle_shift_phase_change"]
    assert out[0]["shared_pattern_details"] == [
        {
            "key": "signature:spindle_shift_phase_change",
            "query_strength": 0.61,
            "query_source_metric": "phase_differences",
            "candidate_strength": 0.83,
            "candidate_source_metric": "phase_differences",
        }
    ]


def test_ranking_prefers_confirmed_and_more_shared_patterns():
    rows = [
        _row("m_old", shared=["p1"], cand_patterns=["p1"],
             feedbacks=[{"action": "dismiss", "timestamp": "2026-01-01"}],
             created_at="2026-04-23T00:00:00+00:00"),
        _row("m_confirmed", shared=["p1"], cand_patterns=["p1"],
             feedbacks=[{"action": "confirm", "timestamp": "2026-02-01"}],
             created_at="2026-04-01T00:00:00+00:00"),
        _row("m_more_shared", shared=["p1", "p2"], cand_patterns=["p1", "p2"],
             feedbacks=[{"action": "confirm", "timestamp": "2026-02-01"}],
             created_at="2026-04-15T00:00:00+00:00"),
    ]
    store = _store_with_rows(rows)
    out = store.get_similar_with_resolution("q", k=3)
    # Confirmed + more shared patterns wins the top slot.
    assert out[0]["id"] == "m_more_shared"
    # Second: confirmed with 1 shared pattern.
    assert out[1]["id"] == "m_confirmed"
    # Last: dismissed-only, even though it's the most recent.
    assert out[2]["id"] == "m_old"


def test_include_dismissed_false_filters_dismiss_only():
    rows = [
        _row("m_dismiss", shared=["p1"], cand_patterns=["p1"],
             feedbacks=[{"action": "dismiss", "timestamp": "2026-04-01"}]),
        _row("m_confirm", shared=["p1"], cand_patterns=["p1"],
             feedbacks=[{"action": "confirm", "timestamp": "2026-04-01"}]),
    ]
    store = _store_with_rows(rows)
    out = store.get_similar_with_resolution("q", k=5, include_dismissed=False)
    assert [e["id"] for e in out] == ["m_confirm"]


def test_empty_feedback_handled():
    rows = [
        _row("m_none", shared=["p1"], cand_patterns=["p1"], feedbacks=[]),
    ]
    store = _store_with_rows(rows)
    out = store.get_similar_with_resolution("q", k=3)
    assert len(out) == 1
    fb = out[0]["feedback"]
    assert fb["confirm_count"] == 0
    assert fb["dismiss_count"] == 0
    assert fb["last_action"] is None
    assert fb["last_comment"] is None


def test_k_truncates():
    rows = [
        _row(f"m{i}", shared=["p1"], cand_patterns=["p1"],
             feedbacks=[{"action": "confirm", "timestamp": f"2026-04-{i:02d}"}])
        for i in range(1, 11)
    ]
    store = _store_with_rows(rows)
    out = store.get_similar_with_resolution("q", k=3)
    assert len(out) == 3

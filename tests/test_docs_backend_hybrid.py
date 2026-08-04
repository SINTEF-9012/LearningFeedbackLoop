from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from backend.agents.llm.docs_backend import Neo4jDocsBackend, _dedupe_entity_matches


def _doc_match(doc_id: str, *, document_type: str = "manual", score: float = 0.5, evidence_entities=None):
    match = {
        "id": doc_id,
        "text": f"text for {doc_id}",
        "source": "site_a",
        "usecase": "SITE_A",
        "file_name": f"{doc_id}.pdf",
        "page": 1,
        "machine": "MACHINE_A1",
        "document_type": document_type,
        "language": "en",
        "score": score,
        "citation": f"SITE_A / {doc_id}.pdf / p.1 / machine=MACHINE_A1",
    }
    if evidence_entities:
        match["evidence_entities"] = evidence_entities
    return match


def test_search_uses_hybrid_path_when_entities_exist():
    backend = object.__new__(Neo4jDocsBackend)
    backend.name = "docs_neo4j"
    backend._driver = object()
    backend._driver_error = None
    backend._ensure_driver = lambda: None
    backend._entities_available = lambda params: True
    backend._hybrid_search = lambda params: [_doc_match("hybrid-doc", score=0.7)]
    backend._text_search = lambda params: []

    result = asyncio.run(
        Neo4jDocsBackend.search(
            backend,
            "chatter",
            top_k=3,
            usecase="SITE_A",
        )
    )

    assert result["matches"][0]["id"] == "hybrid-doc"


def test_hybrid_search_fuses_vector_and_graph_results():
    backend = object.__new__(Neo4jDocsBackend)
    backend._vector_search = lambda params, **kwargs: [
        _doc_match("manual-doc", document_type="manual", score=0.9),
        _doc_match("spreadsheet-doc", document_type="spreadsheet", score=0.85),
    ]
    backend._entity_vector_search = lambda params: []
    backend._entity_alias_search = lambda params: [{"id": "entity-1", "name": "Chatter", "type": "Symptom", "score": 1.0}]
    backend._seed_entities_from_chunks = lambda chunk_ids, top_k: []
    backend._graph_expand_search = lambda params, seed_entity_ids: [
        _doc_match(
            "manual-doc",
            document_type="manual",
            score=2.0,
            evidence_entities=[{"id": "entity-1", "name": "Chatter", "type": "Symptom"}],
        ),
        _doc_match(
            "spreadsheet-doc",
            document_type="spreadsheet",
            score=1.0,
            evidence_entities=[{"id": "entity-1", "name": "Chatter", "type": "Symptom"}],
        ),
    ]

    matches = Neo4jDocsBackend._hybrid_search(
        backend,
        {
            "top_k": 2,
            "search_text": "spindle chatter",
            "usecase": "SITE_A",
            "usecase_aliases": ["site_a"],
            "source_candidates": [],
            "machine": "",
            "machine_tokens": [],
            "document_type": "",
            "usecase_machine_tokens": [],
        },
    )

    assert matches[0]["id"] == "manual-doc"
    assert matches[0]["evidence_entities"][0]["id"] == "entity-1"
    assert matches[1]["id"] == "spreadsheet-doc"
    assert matches[0]["score"] > matches[1]["score"]
    assert matches[0]["score"] > 0.18


def test_rrf_merge_demotes_spreadsheet_scores():
    backend = object.__new__(Neo4jDocsBackend)

    matches = Neo4jDocsBackend._rrf_merge(
        backend,
        [
            [
                _doc_match("manual-doc", document_type="manual", score=0.8),
                _doc_match("spreadsheet-doc", document_type="spreadsheet", score=0.8),
            ]
        ],
        top_k=2,
    )

    assert matches[0]["id"] == "manual-doc"
    assert matches[1]["id"] == "spreadsheet-doc"
    assert matches[0]["score"] > matches[1]["score"]
    assert matches[0]["score"] == 0.8
    assert matches[1]["score"] == 0.48


def test_rrf_merge_promotes_graph_supported_matches_without_vector_score():
    backend = object.__new__(Neo4jDocsBackend)
    backend._doc_feedback_signals = lambda doc_ids: {}

    matches = Neo4jDocsBackend._rrf_merge(
        backend,
        [
            [
                {
                    **_doc_match("graph-doc", document_type="manual", score=3.0),
                    "graph_support": 3,
                    "_score_source": "graph",
                }
            ]
        ],
        top_k=1,
    )

    assert matches[0]["id"] == "graph-doc"
    assert matches[0]["score"] == 0.21


def test_rank_matches_with_feedback_promotes_positive_doc_feedback():
    backend = object.__new__(Neo4jDocsBackend)
    backend._doc_feedback_signals = lambda doc_ids: {
        "feedback-doc": {"helpful_count": 3, "not_helpful_count": 0, "feedback_score": 3.0},
        "raw-doc": {"helpful_count": 0, "not_helpful_count": 0, "feedback_score": 0.0},
    }

    matches = Neo4jDocsBackend._rank_matches_with_feedback(
        backend,
        [
            _doc_match("raw-doc", score=0.82),
            _doc_match("feedback-doc", score=0.74),
        ],
        top_k=2,
    )

    assert [match["id"] for match in matches] == ["feedback-doc", "raw-doc"]
    assert matches[0]["feedback_score"] == 3.0
    assert matches[0]["ranking_score"] > matches[0]["score"]


def test_doc_feedback_signals_reads_memory_side_doc_links():
    backend = object.__new__(Neo4jDocsBackend)
    captured = {}

    def _run(query, **kwargs):
        captured["query"] = query
        captured["kwargs"] = kwargs
        return [
            {
                "metadata_json": json.dumps(
                    {
                        "doc_links": [
                            {"id": "doc-1", "helpful_count": 2, "not_helpful_count": 1},
                            {"id": "doc-2", "helpful_count": 1, "not_helpful_count": 0},
                        ]
                    }
                )
            }
        ]

    backend._driver = object()
    backend._run = _run

    signals = Neo4jDocsBackend._doc_feedback_signals(backend, ["doc-1", "doc-2", "missing"])

    assert "doc_link_ids" in captured["query"]
    assert captured["kwargs"] == {"doc_ids": ["doc-1", "doc-2", "missing"]}
    assert signals["doc-1"] == {"helpful_count": 2, "not_helpful_count": 1, "feedback_score": 1.0}
    assert signals["doc-2"] == {"helpful_count": 1, "not_helpful_count": 0, "feedback_score": 1.0}
    assert signals["missing"] == {"helpful_count": 0, "not_helpful_count": 0, "feedback_score": 0.0}


def test_structured_returns_one_record_per_source_document():
    backend = object.__new__(Neo4jDocsBackend)
    backend.name = "docs_neo4j"
    backend._driver = object()
    backend._driver_error = None
    backend._ensure_driver = lambda: None
    backend._build_filter_params = lambda **kwargs: {}
    backend._where_clause = lambda alias: "true"
    calls = []

    def _run(query, **kwargs):
        calls.append((query, kwargs))
        return [
            {"doc": {"id": "chunk-1", "file_name": "alpha.xlsx", "page": 1, "document_type": "spreadsheet", "usecase": "SITE_B", "machine": "b1001", "text": "alpha p1"}},
            {"doc": {"id": "chunk-2", "file_name": "beta.pdf", "page": 3, "document_type": "manual", "usecase": "SITE_B", "machine": "b1001", "text": "beta p3"}},
        ]

    backend._run = _run

    result = asyncio.run(Neo4jDocsBackend.structured(backend, {"limit": 10}))

    assert len(result["records"]) == 2
    assert result["records"][0]["file_name"] == "alpha.xlsx"
    assert result["records"][1]["file_name"] == "beta.pdf"
    assert "collect(d)[0] AS doc" in calls[0][0]


def test_dedupe_entity_matches_collapses_shared_canonical_ids():
    matches = _dedupe_entity_matches(
        [
            {"id": "entity-1", "canonical_id": "urn:lfl:asset:machine_a1", "name": "MACHINE_A1", "type": "Machine", "score": 0.4},
            {"id": "entity-2", "canonical_id": "urn:lfl:asset:machine_a1", "name": "MACHINE_A1", "type": "Machine", "score": 0.9},
        ]
    )

    assert len(matches) == 1
    assert matches[0]["id"] == "entity-2"
    assert matches[0]["canonical_id"] == "urn:lfl:asset:machine_a1"


def test_status_reports_semantic_coverage_summary():
    backend = object.__new__(Neo4jDocsBackend)
    backend.name = "docs_neo4j"
    backend._driver = object()
    backend._driver_error = None
    backend._ensure_driver = lambda: None

    def _run(query, **kwargs):
        del kwargs
        if "collect(DISTINCT coalesce(d.usecase" in query:
            return [{"sources": ["SITE_A", "SITE_C"], "machines": ["MACHINE_A1", "c1001"]}]
        if "docs_with_mentions" in query and "file_count" in query:
            return [
                {"usecase": "SITE_A", "document_count": 12, "docs_with_mentions": 10, "file_count": 2},
                {"usecase": "SITE_C", "document_count": 4, "docs_with_mentions": 0, "file_count": 1},
            ]
        if "canonical_entity_count" in query:
            return [
                {"usecase": "SITE_A", "entity_count": 8, "canonical_entity_count": 2},
                {"usecase": "SITE_C", "entity_count": 0, "canonical_entity_count": 0},
            ]
        if "count(m) AS mention_count" in query:
            return [{"usecase": "SITE_A", "mention_count": 22}]
        if "count(r) AS relation_count" in query:
            return [{"usecase": "SITE_A", "relation_count": 7}]
        raise AssertionError(query)

    backend._run = _run

    result = asyncio.run(Neo4jDocsBackend.status(backend))

    assert result["ready"] is True
    assert result["document_count"] == 16
    assert result["entity_count"] == 8
    assert result["mention_count"] == 22
    assert result["relation_count"] == 7
    assert result["docs_with_mentions"] == 10
    assert result["docs_without_mentions"] == 6
    assert result["semantic_gap_usecases"] == ["SITE_C"]
    assert result["twin_health"] == {
        "status": "warning",
        "headline": "1/2 ready",
        "summary": "1/2 usecases semantically ready · 2 canonical entities",
        "semantic_ready_usecases": 1,
        "total_usecases": 2,
        "canonical_entity_count": 2,
        "semantic_coverage_ratio": 0.625,
        "semantic_gap_usecases": ["SITE_C"],
    }
    assert result["message"] == "Semantic coverage missing for: SITE_C."
    assert result["usecase_coverage"] == [
        {
            "usecase": "SITE_A",
            "document_count": 12,
            "file_count": 2,
            "entity_count": 8,
            "canonical_entity_count": 2,
            "mention_count": 22,
            "relation_count": 7,
            "docs_with_mentions": 10,
            "docs_without_mentions": 2,
            "semantic_coverage_ratio": 0.8333,
            "semantic_ready": True,
        },
        {
            "usecase": "SITE_C",
            "document_count": 4,
            "file_count": 1,
            "entity_count": 0,
            "canonical_entity_count": 0,
            "mention_count": 0,
            "relation_count": 0,
            "docs_with_mentions": 0,
            "docs_without_mentions": 4,
            "semantic_coverage_ratio": 0.0,
            "semantic_ready": False,
        },
    ]
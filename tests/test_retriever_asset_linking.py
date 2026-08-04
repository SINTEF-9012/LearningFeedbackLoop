from __future__ import annotations

import asyncio

import backend.agents.llm.retriever as retriever_module
from backend.agents.llm.retriever import RetrieverAgent


class _FakeSinditClient:
    async def search_assets(self, query: str):
        return [
            {
                "iri": "urn:lfl:asset:machine_a1",
                "label": "MACHINE_A1",
                "assetType": "Machine",
            }
        ]


class _FakeDocsBackend:
    async def search(self, query: str, **kwargs):
        return {
            "backend": "docs_neo4j",
            "matches": [
                {
                    "id": "doc-1",
                    "file_name": "manual.pdf",
                    "citation": "SITE_A / manual.pdf / p.1 / machine=MACHINE_A1",
                    "machine": "MACHINE_A1",
                    "machine_uri": "urn:lfl:asset:machine_a1",
                    "evidence_entities": [
                        {
                            "id": "entity-1",
                            "canonical_id": "urn:lfl:asset:machine_a1",
                            "name": "MACHINE_A1",
                            "type": "Machine",
                        }
                    ],
                }
            ],
        }


def test_retriever_links_documents_to_shared_asset_identity():
    agent = RetrieverAgent()
    agent._sindit_client = _FakeSinditClient()
    agent._docs_backend = _FakeDocsBackend()

    async def _memory_search(args):
        return {"matches": []}

    agent._memory_search = _memory_search  # type: ignore[assignment]

    result = asyncio.run(agent._search({"query": "MACHINE_A1", "include_docs": True}))

    assert result["sindit_assets"][0]["canonical_id"] == "urn:lfl:asset:machine_a1"
    assert result["sindit_assets"][0]["related_documents"][0]["id"] == "doc-1"
    assert result["documents"][0]["related_assets"][0]["canonical_id"] == "urn:lfl:asset:machine_a1"
    assert len(result["linked_assets"]) == 1
    assert set(result["linked_assets"][0]["sources"]) == {"docs", "sindit"}


class _FakeSqliteStore:
    db_path = "/tmp/memories.db"

    def count(self) -> int:
        return 7


class _FakeNeo4jStatusStore:
    _driver = object()
    _database = "neo4j"

    def count(self) -> int:
        return 5

    def subgraph_integrity(self):
        return {
            "healthy": False,
            "mixed_label_nodes": 1,
            "disallowed_cross_graph_edges": 2,
            "disallowed_relationship_types": ["ON_MACHINE"],
            "memory_labels": ["Memory"],
            "knowledge_labels": ["Document", "Entity"],
            "allowed_cross_relationships": ["CITES", "DOCUMENTED_BY"],
        }


class _FakeStatusDocsBackend:
    async def status(self):
        return {
            "backend": "docs_neo4j",
            "ready": True,
            "document_count": 12,
            "entity_count": 8,
            "mention_count": 22,
            "relation_count": 7,
            "docs_with_mentions": 10,
            "docs_without_mentions": 2,
            "semantic_coverage_ratio": 0.8333,
            "semantic_gap_usecases": [],
            "sources": ["SITE_A"],
            "machines": ["MACHINE_A1"],
            "twin_health": {
                "status": "ok",
                "headline": "1/1 ready",
                "summary": "1/1 usecases semantically ready · 2 canonical ids across 8 entities",
                "semantic_ready_usecases": 1,
                "total_usecases": 1,
                "canonical_entity_count": 2,
                "semantic_coverage_ratio": 0.8333,
                "semantic_gap_usecases": [],
            },
            "usecase_coverage": [
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
                }
            ],
            "message": "ok",
        }


def test_retriever_status_reports_resolved_memory_store(monkeypatch):
    monkeypatch.setattr(retriever_module, "STORAGE_BACKEND", "neo4j")

    agent = RetrieverAgent()
    agent._neo4j_store = _FakeSqliteStore()
    agent._docs_backend = _FakeStatusDocsBackend()

    status = asyncio.run(agent._status())

    assert status["configured_storage_backend"] == "neo4j"
    assert status["storage_backend"] == "sqlite"
    assert status["neo4j_memory_count"] is None
    assert status["memory_graph"] == {
        "configured_backend": "neo4j",
        "resolved_backend": "sqlite",
        "store_class": "_FakeSqliteStore",
        "db_path": "/tmp/memories.db",
        "count": 7,
    }
    assert status["docs"] == {
        "backend": "docs_neo4j",
        "ready": True,
        "document_count": 12,
        "entity_count": 8,
        "mention_count": 22,
        "relation_count": 7,
        "docs_with_mentions": 10,
        "docs_without_mentions": 2,
        "semantic_coverage_ratio": 0.8333,
        "semantic_gap_usecases": [],
        "sources": ["SITE_A"],
        "machines": ["MACHINE_A1"],
        "twin_health": {
            "status": "ok",
            "headline": "1/1 ready",
            "summary": "1/1 usecases semantically ready · 2 canonical ids across 8 entities",
            "semantic_ready_usecases": 1,
            "total_usecases": 1,
            "canonical_entity_count": 2,
            "semantic_coverage_ratio": 0.8333,
            "semantic_gap_usecases": [],
        },
        "usecase_coverage": [
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
            }
        ],
        "message": "ok",
    }
    assert status["knowledge_graph"] == {
        "backend": "docs_neo4j",
        "ready": True,
        "document_count": 12,
        "entity_count": 8,
        "mention_count": 22,
        "relation_count": 7,
        "docs_with_mentions": 10,
        "docs_without_mentions": 2,
        "semantic_coverage_ratio": 0.8333,
        "semantic_gap_usecases": [],
        "sources": ["SITE_A"],
        "machines": ["MACHINE_A1"],
        "twin_health": {
            "status": "ok",
            "headline": "1/1 ready",
            "summary": "1/1 usecases semantically ready · 2 canonical ids across 8 entities",
            "semantic_ready_usecases": 1,
            "total_usecases": 1,
            "canonical_entity_count": 2,
            "semantic_coverage_ratio": 0.8333,
            "semantic_gap_usecases": [],
        },
        "usecase_coverage": [
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
            }
        ],
        "message": "ok",
    }
def test_retriever_status_reports_memory_subgraph_integrity(monkeypatch):
    monkeypatch.setattr(retriever_module, "STORAGE_BACKEND", "neo4j")

    agent = RetrieverAgent()
    agent._neo4j_store = _FakeNeo4jStatusStore()
    agent._docs_backend = _FakeStatusDocsBackend()

    status = asyncio.run(agent._status())

    assert status["storage_backend"] == "neo4j"
    assert status["neo4j_memory_count"] == 5
    assert status["memory_graph"] == {
        "configured_backend": "neo4j",
        "resolved_backend": "neo4j",
        "store_class": "_FakeNeo4jStatusStore",
        "database": "neo4j",
        "count": 5,
        "subgraph_integrity": {
            "healthy": False,
            "mixed_label_nodes": 1,
            "disallowed_cross_graph_edges": 2,
            "disallowed_relationship_types": ["ON_MACHINE"],
            "memory_labels": ["Memory"],
            "knowledge_labels": ["Document", "Entity"],
            "allowed_cross_relationships": ["CITES", "DOCUMENTED_BY"],
        },
    }
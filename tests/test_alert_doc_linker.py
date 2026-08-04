from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.agents.memory.router as memory_router
from backend.agents.core.schemas import Memory, PatternKey, PatternType, TimeRange
from backend.agents.domain_config import reset_active_domain
from backend.agents.llm.alert_doc_linker import build_alert_doc_queries, propose_alert_doc_links


class DummyDocsBackend:
    def __init__(self) -> None:
        self.calls = []

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        usecase: str | None = None,
        source_filter: str | None = None,
        machine: str | None = None,
        document_type: str | None = None,
    ):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "usecase": usecase,
                "machine": machine,
            }
        )
        if query == "SPINDLE_POWER_SURGE":
            return {
                "matches": [
                    {
                        "id": "doc-1",
                        "citation": "SITE_A / spindle.pdf / p.12 / machine=MACHINE_A1",
                        "page": 12,
                        "score": 0.91,
                        "file_name": "spindle.pdf",
                        "machine": "MACHINE_A1",
                        "usecase": "SITE_A",
                        "text": "Check spindle load and power surge conditions.",
                    },
                    {
                        "id": "doc-2",
                        "citation": "SITE_A / weak.pdf / p.9 / machine=MACHINE_A1",
                        "page": 9,
                        "score": 0.4,
                        "file_name": "weak.pdf",
                        "machine": "MACHINE_A1",
                        "usecase": "SITE_A",
                        "text": "Low-confidence weak match.",
                    },
                ]
            }
        return {
            "matches": [
                {
                    "id": "doc-1",
                    "citation": "SITE_A / spindle.pdf / p.12 / machine=MACHINE_A1",
                    "page": 12,
                    "score": 0.72,
                    "file_name": "spindle.pdf",
                    "machine": "MACHINE_A1",
                    "usecase": "SITE_A",
                    "text": "Lower-scoring duplicate for the same page.",
                },
                {
                    "id": "doc-3",
                    "citation": "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1",
                    "page": 330,
                    "score": 0.85,
                    "file_name": "chatter.pdf",
                    "machine": "MACHINE_A1",
                    "usecase": "SITE_A",
                    "text": "Regenerative chatter and harmonic energy guidance.",
                    "evidence_entities": [{"id": "e-1", "name": "Chatter", "type": "Symptom"}],
                },
            ]
        }


def test_build_alert_doc_queries_humanizes_fault_signatures() -> None:
    reset_active_domain()
    queries = build_alert_doc_queries(
        ["fault:tool_breakage"],
        cutting_context={"tool_type": "end mill"},
        channel_names=["Vibration_Severity_X", "Power_Spindle"],
    )

    assert len(queries) == 1
    assert queries[0]["pattern_key"] == "fault:tool_breakage"
    assert "Sudden catastrophic tool failure" in queries[0]["query"]
    assert "HF energy burst" in queries[0]["query"]
    assert "end mill" in queries[0]["query"]


def test_build_alert_doc_queries_preserves_uppercase_domain_rules() -> None:
    queries = build_alert_doc_queries(["SPINDLE_POWER_SURGE"])

    assert queries == [{"pattern_key": "SPINDLE_POWER_SURGE", "query": "SPINDLE_POWER_SURGE"}]


def test_propose_alert_doc_links_dedupes_and_keeps_best_match() -> None:
    backend = DummyDocsBackend()

    result = asyncio.run(
        propose_alert_doc_links(
            backend,
            pattern_keys=["fault:tool_breakage", "SPINDLE_POWER_SURGE"],
            usecase="SITE_A",
            machine="MACHINE_A1",
            cutting_context={"tool_type": "end mill"},
            channel_names=["Vibration_Severity_X", "Power_Spindle"],
            top_k=2,
            score_floor=0.6,
        )
    )

    assert len(result["query_candidates"]) == 2
    assert [link["id"] for link in result["doc_links"]] == ["doc-1", "doc-3"]
    assert result["doc_links"][0]["score"] == 0.91
    assert result["doc_links"][0]["query_used"] == "SPINDLE_POWER_SURGE"
    assert all(link["score"] >= 0.6 for link in result["doc_links"])


def test_propose_alert_doc_links_prefers_feedback_ranked_matches() -> None:
    class FeedbackAwareBackend:
        async def search(self, query: str, **_kwargs):
            return {
                "matches": [
                    {
                        "id": "doc-feedback",
                        "citation": "SITE_A / preferred.pdf / p.4 / machine=MACHINE_A1",
                        "page": 4,
                        "score": 0.74,
                        "ranking_score": 0.86,
                        "feedback_score": 3.0,
                        "file_name": "preferred.pdf",
                        "machine": "MACHINE_A1",
                        "usecase": "SITE_A",
                        "text": "Operator-validated guidance.",
                    },
                    {
                        "id": "doc-raw",
                        "citation": "SITE_A / raw.pdf / p.2 / machine=MACHINE_A1",
                        "page": 2,
                        "score": 0.82,
                        "ranking_score": 0.82,
                        "feedback_score": 0.0,
                        "file_name": "raw.pdf",
                        "machine": "MACHINE_A1",
                        "usecase": "SITE_A",
                        "text": "Higher raw retrieval score but no positive feedback.",
                    },
                ]
            }

    result = asyncio.run(
        propose_alert_doc_links(
            FeedbackAwareBackend(),
            pattern_keys=["fault:chatter"],
            usecase="SITE_A",
            machine="MACHINE_A1",
            score_floor=0.6,
        )
    )

    assert [link["id"] for link in result["doc_links"]] == ["doc-feedback", "doc-raw"]


def test_memory_alert_doc_links_endpoint_returns_grounded_links(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(memory_router.router)

    memory = Memory(
        id="mem-1",
        session_id="session-1",
        time_range=TimeRange(i0=0, i1=1, t0=0.0, t1=1.0, fs=1.0),
        channels=["Vibration_Severity_X", "Power_Spindle"],
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="fault:tool_breakage")],
        metadata={
            "source": "SITE_A",
            "cutting_context": {"machine_id": "MACHINE_A1", "tool_type": "end mill"},
        },
    )
    orchestrator = SimpleNamespace(get_memory=lambda memory_id: memory if memory_id == "mem-1" else None)
    backend = DummyDocsBackend()

    monkeypatch.setattr(memory_router, "get_orchestrator", lambda: orchestrator)
    monkeypatch.setattr(memory_router, "get_docs_backend", lambda: backend)

    client = TestClient(app)
    response = client.get("/memory/alerts/mem-1/doc_links?top_k=2&score_floor=0.6")

    assert response.status_code == 200
    body = response.json()
    assert body["memory_id"] == "mem-1"
    assert body["usecase"] == "SITE_A"
    assert body["machine"] == "MACHINE_A1"
    assert body["query_candidates"]
    assert body["doc_links"][0]["citation"] == "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1"
    assert body["doc_links"][0]["evidence_entities"][0]["name"] == "Chatter"


def test_memory_alert_doc_links_endpoint_prefers_persisted_links_when_available(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(memory_router.router)

    memory = Memory(
        id="mem-2",
        session_id="session-1",
        time_range=TimeRange(i0=0, i1=1, t0=0.0, t1=1.0, fs=1.0),
        channels=["Vibration_Severity_X"],
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="fault:chatter")],
        metadata={
            "source": "SITE_A",
            "cutting_context": {"machine_id": "MACHINE_A1", "tool_type": "end mill"},
        },
    )
    persisted_links = [
        {
            "id": "doc-persisted",
            "citation": "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1",
            "page": 330,
            "score": 0.98,
            "query_used": "regenerative chatter harmonic vibration tooth passing",
            "pattern_key": "fault:chatter",
            "evidence_entities": [{"id": "e-1", "name": "Chatter", "type": "Symptom"}],
        }
    ]
    orchestrator = SimpleNamespace(
        get_memory=lambda memory_id: memory if memory_id == "mem-2" else None,
        store=SimpleNamespace(
            get_doc_links=lambda memory_id, *, score_floor, limit: persisted_links
            if memory_id == "mem-2"
            else []
        ),
    )
    backend = DummyDocsBackend()

    monkeypatch.setattr(memory_router, "get_orchestrator", lambda: orchestrator)
    monkeypatch.setattr(memory_router, "get_docs_backend", lambda: backend)

    client = TestClient(app)
    response = client.get("/memory/alerts/mem-2/doc_links")

    assert response.status_code == 200
    body = response.json()
    assert body["doc_links"][0]["id"] == "doc-persisted"
    assert body["query_candidates"] == [
        {
            "pattern_key": "fault:chatter",
            "query": "regenerative chatter harmonic vibration tooth passing",
        }
    ]
    assert backend.calls == []


def test_memory_detail_endpoint_includes_persisted_doc_links(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(memory_router.router)

    memory = Memory(
        id="mem-3",
        session_id="session-1",
        time_range=TimeRange(i0=0, i1=1, t0=0.0, t1=1.0, fs=1.0),
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="fault:chatter")],
        metadata={"source": "SITE_A"},
    )
    persisted_links = [
        {
            "id": "doc-persisted",
            "citation": "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1",
            "page": 330,
            "score": 0.98,
            "query_used": "regenerative chatter harmonic vibration tooth passing",
            "pattern_key": "fault:chatter",
            "evidence_entities": [{"id": "e-1", "name": "Chatter", "type": "Symptom"}],
        }
    ]
    orchestrator = SimpleNamespace(
        get_memory=lambda memory_id: memory if memory_id == "mem-3" else None,
        feedback_handler=SimpleNamespace(get_feedback_stats=lambda memory_id: {"confirm_count": 0, "dismiss_count": 0}),
        store=SimpleNamespace(
            get_doc_links=lambda memory_id, *, score_floor, limit: persisted_links
            if memory_id == "mem-3"
            else []
        ),
    )

    monkeypatch.setattr(memory_router, "get_orchestrator", lambda: orchestrator)

    client = TestClient(app)
    response = client.get("/memory/mem-3")

    assert response.status_code == 200
    body = response.json()
    assert body["memory"]["id"] == "mem-3"
    assert body["doc_links"][0]["id"] == "doc-persisted"
    assert body["doc_links"][0]["evidence_entities"][0]["name"] == "Chatter"
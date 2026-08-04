from __future__ import annotations

import asyncio
from types import SimpleNamespace

import backend.agents.memory.router as memory_router


class _FakeStore:
    def __init__(self, memory, *, doc_links=None) -> None:
        self._memory = memory
        self._doc_links = list(doc_links or [])

    def get(self, memory_id: str):
        return self._memory if memory_id == self._memory.id else None

    def get_doc_links(self, memory_id: str, *, score_floor: float, limit: int):
        assert memory_id == self._memory.id
        assert score_floor == 0.0
        assert limit > 0
        return list(self._doc_links)


class _FakeExplainer:
    def is_available(self) -> bool:
        return False


class _ForbiddenDocsBackend:
    async def search(self, *args, **kwargs):
        raise AssertionError("docs search should be skipped for unscoped memories")


def test_chat_about_memory_skips_unscoped_docs_search(monkeypatch) -> None:
    memory = SimpleNamespace(
        id="mem-1",
        session_id="session-1",
        pattern_keys=[SimpleNamespace(key="CHATTER_DETECTED")],
        metadata={"significance_score": 0.82},
        machine_uri=None,
    )
    orchestrator = SimpleNamespace(
        store=_FakeStore(memory),
        explainer=_FakeExplainer(),
    )

    monkeypatch.setenv("STRICT_USECASE_GROUNDING", "1")
    monkeypatch.setattr(memory_router, "get_orchestrator", lambda: orchestrator)
    monkeypatch.setattr(memory_router, "get_docs_backend", lambda: _ForbiddenDocsBackend())

    response = asyncio.run(
        memory_router.chat_about_memory(
            "mem-1",
            memory_router.ChatRequest(message="What does this mean?"),
        )
    )

    assert response.memory_context is not None
    assert response.memory_context["usecase"] is None
    assert response.memory_context["docs_scope_reason"] == "no_usecase_scope"
    assert response.memory_context["documents"] == []


def test_chat_about_memory_prefers_persisted_doc_links(monkeypatch) -> None:
    memory = SimpleNamespace(
        id="mem-1",
        session_id="session-1",
        pattern_keys=[SimpleNamespace(key="CHATTER_DETECTED")],
        metadata={"significance_score": 0.82},
        machine_uri=None,
    )
    orchestrator = SimpleNamespace(
        store=_FakeStore(
            memory,
            doc_links=[
                {
                    "id": "doc-1",
                    "citation": "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1",
                    "text": "Check chatter frequency against tooth passing harmonics.",
                    "evidence_entities": [{"name": "Chatter", "type": "Symptom"}],
                }
            ],
        ),
        explainer=_FakeExplainer(),
    )

    monkeypatch.setenv("STRICT_USECASE_GROUNDING", "1")
    monkeypatch.setattr(memory_router, "get_orchestrator", lambda: orchestrator)
    monkeypatch.setattr(memory_router, "get_docs_backend", lambda: _ForbiddenDocsBackend())

    response = asyncio.run(
        memory_router.chat_about_memory(
            "mem-1",
            memory_router.ChatRequest(message="What does this mean?"),
        )
    )

    assert response.memory_context is not None
    assert response.memory_context["docs_scope_reason"] == "persisted_links"
    assert response.memory_context["documents"] == ["SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1"]
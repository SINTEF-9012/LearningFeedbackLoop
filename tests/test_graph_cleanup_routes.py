from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.agents.memory import graph_routes as graph_mod
from backend.agents.memory import router as router_mod


@pytest.mark.asyncio
async def test_preview_memory_graph_cleanup_route_returns_store_preview(monkeypatch):
    orchestrator = SimpleNamespace(
        store=SimpleNamespace(
            preview_memory_graph_cleanup=lambda: {
                "scope": "memory_graph",
                "total_nodes_to_delete": 12,
                "total_relationships_to_delete": 15,
                "legacy_candidate_summary": {
                    "candidate_memories": 9,
                    "candidate_sessions": 4,
                },
            }
        )
    )
    monkeypatch.setattr(graph_mod, "get_orchestrator", lambda: orchestrator)

    payload = await graph_mod.preview_memory_graph_cleanup()

    assert payload["scope"] == "memory_graph"
    assert payload["total_nodes_to_delete"] == 12
    assert payload["legacy_candidate_summary"]["candidate_memories"] == 9


@pytest.mark.asyncio
async def test_clear_memory_graph_route_clears_scoped_graph_and_resets_runtime(monkeypatch):
    calls = []

    class _Store:
        def clear_memory_graph(self):
            calls.append("clear_memory_graph")
            return {"Memory": 3, "Pattern": 2}

    class _Scorer:
        def __init__(self):
            self.reset_called = False

        def reset_feedback_state(self):
            self.reset_called = True

    scorer = _Scorer()
    memory_cache = {"m-1": object()}
    orchestrator = SimpleNamespace(
        store=_Store(),
        scorer=scorer,
        _memories=memory_cache,
    )
    monkeypatch.setattr(graph_mod, "get_orchestrator", lambda: orchestrator)

    payload = await graph_mod.clear_memory_graph_data()

    assert calls == ["clear_memory_graph"]
    assert payload == {
        "deleted": True,
        "scope": "memory_graph",
        "counts": {"Memory": 3, "Pattern": 2},
        "priors_reset": True,
    }
    assert memory_cache == {}
    assert scorer.reset_called is True


@pytest.mark.asyncio
async def test_clear_legacy_candidate_memory_route_refreshes_priors_without_reset(monkeypatch):
    calls = []

    class _Store:
        def clear_legacy_candidate_memories(self):
            calls.append("clear_legacy_candidate_memories")
            return {"Memory": 5, "Pattern": 2}

    class _Scorer:
        def __init__(self):
            self.refresh_called = False
            self.reset_called = False

        def refresh_priors(self):
            self.refresh_called = True

        def reset_feedback_state(self):
            self.reset_called = True

    scorer = _Scorer()
    memory_cache = {"m-1": object()}
    orchestrator = SimpleNamespace(
        store=_Store(),
        scorer=scorer,
        _memories=memory_cache,
    )
    monkeypatch.setattr(graph_mod, "get_orchestrator", lambda: orchestrator)

    payload = await graph_mod.clear_legacy_candidate_memory_data()

    assert calls == ["clear_legacy_candidate_memories"]
    assert payload == {
        "deleted": True,
        "scope": "legacy_candidates",
        "counts": {"Memory": 5, "Pattern": 2},
        "priors_refreshed": True,
    }
    assert memory_cache == {}
    assert scorer.refresh_called is True
    assert scorer.reset_called is False


@pytest.mark.asyncio
async def test_clear_all_graph_route_rejects_when_flag_disabled(monkeypatch):
    monkeypatch.delenv("ALLOW_GRAPH_CLEAR_ALL", raising=False)

    with pytest.raises(HTTPException) as exc:
        await graph_mod.clear_all_graph_data()

    assert exc.value.status_code == 403
    assert "ALLOW_GRAPH_CLEAR_ALL" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_clear_all_graph_route_runs_when_flag_enabled(monkeypatch):
    calls = []

    class _Store:
        def clear_all(self):
            calls.append("clear_all")
            return {"Memory": 4, "Document": 2}

    class _Scorer:
        def __init__(self):
            self.reset_called = False

        def reset_feedback_state(self):
            self.reset_called = True

    scorer = _Scorer()
    memory_cache = {"m-1": object()}
    orchestrator = SimpleNamespace(
        store=_Store(),
        scorer=scorer,
        _memories=memory_cache,
    )
    monkeypatch.setenv("ALLOW_GRAPH_CLEAR_ALL", "1")
    monkeypatch.setattr(graph_mod, "get_orchestrator", lambda: orchestrator)

    payload = await graph_mod.clear_all_graph_data()

    assert calls == ["clear_all"]
    assert payload == {
        "deleted": True,
        "counts": {"Memory": 4, "Document": 2},
        "priors_reset": True,
    }
    assert memory_cache == {}
    assert scorer.reset_called is True


def test_clear_all_graph_route_is_hidden_from_openapi():
    app = FastAPI()
    app.include_router(router_mod.router)

    with TestClient(app) as client:
        paths = client.get('/openapi.json').json()['paths']

    assert '/memory/graph/clear-all' not in paths
    assert '/memory/graph/clear-memory' in paths
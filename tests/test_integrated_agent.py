"""Tests for IntegratedAgent (Agent I, 2026-04-24)."""
from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from backend.agents.llm.integrated import IntegratedAgent


class _StubAgent:
    def __init__(self, *, answer: str = "ok", delay: float = 0.0, raises: bool = False):
        self._answer = answer
        self._delay = delay
        self._raises = raises
        self.calls: list[tuple[str, Dict[str, Any]]] = []

    async def handle_request(self, session_id, action, args, context):
        self.calls.append((action, dict(args)))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raises:
            raise RuntimeError("boom")
        return {"answer": self._answer}


class _StubLLM:
    def __init__(self, answer: str = "synthesized", available: bool = True):
        self._answer = answer
        self._available = available
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return self._available

    async def handle_request(self, session_id, action, args, context):
        self.calls.append(args.get("question", ""))
        return {"answer": self._answer}


def _make_integrated(registry: Dict[str, Any], llm=None, **kw) -> IntegratedAgent:
    return IntegratedAgent(
        get_agent=lambda n: registry.get(n),
        llm_agent=llm,
        **kw,
    )


@pytest.mark.asyncio
async def test_handle_request_happy_path_calls_all_and_synthesizes():
    registry = {
        "retriever": _StubAgent(answer="doc snippet"),
        "monitoring": _StubAgent(answer="spindle OK"),
        "analytics": _StubAgent(answer="uptrend last 24h"),
        "stoppage": _StubAgent(answer="no imminent stoppage"),
    }
    llm = _StubLLM(answer="combined answer")
    agent = _make_integrated(registry, llm=llm)

    out = await agent.handle_request("sess-1", "query", {"question": "what's happening?"}, {})

    assert set(out["sub_results"].keys()) == {"retriever", "monitoring", "analytics", "stoppage"}
    assert out["degraded"] == []
    assert out["synthesis"] == "combined answer"
    assert out["synthesis_source"] == "llm"
    assert all(a.calls for a in registry.values())
    assert llm.calls  # LLM was consulted


@pytest.mark.asyncio
async def test_missing_subagent_is_reported_as_degraded():
    registry = {"retriever": _StubAgent(answer="hit")}  # no monitoring/analytics/stoppage
    agent = _make_integrated(registry)  # no LLM → falls back to concat

    out = await agent.handle_request("s", "query", {"question": "q"}, {})

    assert "monitoring" in out["degraded"]
    assert "analytics" in out["degraded"]
    assert "stoppage" in out["degraded"]
    assert out["sub_results"]["monitoring"]["error"].startswith("sub-agent")
    assert out["synthesis_source"] == "concat"
    assert "retriever" in (out["synthesis"] or "")


@pytest.mark.asyncio
async def test_timeout_marks_subagent_degraded_without_crashing():
    registry = {
        "retriever": _StubAgent(answer="fast"),
        "monitoring": _StubAgent(answer="slow", delay=1.0),
        "analytics": _StubAgent(answer="ok"),
        "stoppage": _StubAgent(answer="ok"),
    }
    agent = _make_integrated(registry, default_timeout_s=0.05)

    out = await agent.handle_request("s", "query", {"question": "q"}, {})

    assert "monitoring" in out["degraded"]
    assert "timeout" in out["sub_results"]["monitoring"]["error"]
    # Others succeeded
    for name in ("retriever", "analytics", "stoppage"):
        assert name not in out["degraded"]


@pytest.mark.asyncio
async def test_subagent_exception_does_not_propagate():
    registry = {
        "retriever": _StubAgent(raises=True),
        "monitoring": _StubAgent(answer="ok"),
        "analytics": _StubAgent(answer="ok"),
        "stoppage": _StubAgent(answer="ok"),
    }
    agent = _make_integrated(registry)

    out = await agent.handle_request("s", "query", {"question": "q"}, {})

    assert "retriever" in out["degraded"]
    assert out["sub_results"]["retriever"]["error"] == "boom"
    assert "monitoring" not in out["degraded"]


@pytest.mark.asyncio
async def test_synthesize_false_skips_llm():
    registry = {"retriever": _StubAgent(answer="x")}
    llm = _StubLLM()
    agent = _make_integrated(registry, llm=llm)

    out = await agent.handle_request("s", "query", {"question": "q", "synthesize": False}, {})

    assert out["synthesis"] is None
    assert out["synthesis_source"] == "skipped"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_llm_unavailable_falls_back_to_concat():
    registry = {"retriever": _StubAgent(answer="x")}
    llm = _StubLLM(available=False)
    agent = _make_integrated(registry, llm=llm)

    out = await agent.handle_request("s", "query", {"question": "q"}, {})

    assert out["synthesis_source"] == "unavailable"
    assert llm.calls == []
    assert "retriever" in (out["synthesis"] or "")


@pytest.mark.asyncio
async def test_missing_question_returns_error():
    agent = _make_integrated({})
    out = await agent.handle_request("s", "query", {}, {})
    assert "error" in out


@pytest.mark.asyncio
async def test_unsupported_action_returns_error():
    agent = _make_integrated({})
    out = await agent.handle_request("s", "frobnicate", {"question": "q"}, {})
    assert "error" in out

from __future__ import annotations

import time

import pytest

from backend.agents.llm.rag import LLMAgent


@pytest.mark.asyncio
async def test_rag_agent_uses_groq_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    agent = LLMAgent()
    agent._available = True
    agent._last_check = time.time()

    called: dict[str, str] = {}

    async def fake_call(prompt: str):
        called["provider"] = "groq"
        called["prompt"] = prompt
        return {"answer": "groq-answer"}

    agent._call_groq_async = fake_call  # type: ignore[method-assign]

    result = await agent.handle_request("session-1", "query", {"question": "What changed?"}, {})

    assert called["provider"] == "groq"
    assert "What changed?" in called["prompt"]
    assert result["answer"] == "groq-answer"
    assert result["provider"] == "groq"
    assert result["llm_available"] is True


@pytest.mark.asyncio
async def test_rag_agent_respects_ollama_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    agent = LLMAgent()
    agent._available = True
    agent._last_check = time.time()

    called: dict[str, str] = {}

    async def fake_call(prompt: str):
        called["provider"] = "ollama"
        called["prompt"] = prompt
        return {"answer": "ollama-answer"}

    agent._call_ollama_async = fake_call  # type: ignore[method-assign]

    result = await agent.handle_request("session-1", "query", {"question": "What changed?"}, {})

    assert called["provider"] == "ollama"
    assert "What changed?" in called["prompt"]
    assert result["answer"] == "ollama-answer"
    assert result["provider"] == "ollama"
    assert result["llm_available"] is True
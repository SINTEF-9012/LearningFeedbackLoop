"""Tests for the LLM comment classifier (Agent I, 2026-04-24)."""
from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from backend.agents.llm.comment_classifier import (
    _classify_heuristic,
    _coerce_llm_result,
    classify_comment,
)


class _FakeLLM:
    def __init__(self, *, answer: str = "", available: bool = True, delay: float = 0.0, raises: bool = False):
        self._answer = answer
        self._available = available
        self._delay = delay
        self._raises = raises
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return self._available

    async def handle_request(self, session_id, action, args, context):
        self.calls.append(args.get("question", ""))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raises:
            raise RuntimeError("llm failure")
        return {"answer": self._answer}


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------


def test_heuristic_detects_tool_change_phrases():
    for text in (
        "Replaced the tool and restarted the job.",
        "Swapped tool due to wear.",
        "Tool change performed.",
        "Changed the tool after chipping.",
    ):
        out = _classify_heuristic(text)
        assert out["tool_change"] is True, text
        assert out["source"] == "heuristic"


def test_heuristic_no_tool_change_for_unrelated_text():
    out = _classify_heuristic("Spindle sounds fine, just some chatter at end.")
    assert out["tool_change"] is False
    assert out["root_cause"] == "chatter"


def test_heuristic_identifies_root_cause_and_action():
    out = _classify_heuristic("Tool was broken from a hard spot — replaced tool.")
    assert out["tool_change"] is True
    assert out["root_cause"] in ("tool_break", "material")
    assert out["action_taken"] == "replacement"


def test_heuristic_handles_empty_text():
    out = _classify_heuristic("")
    assert out == {
        "root_cause": None,
        "action_taken": None,
        "tool_change": False,
        "source": "heuristic",
    }


# ---------------------------------------------------------------------------
# LLM JSON parsing
# ---------------------------------------------------------------------------


def test_coerce_llm_result_parses_clean_json():
    raw = '{"root_cause": "tool_wear", "action_taken": "replaced tool", "tool_change": true}'
    out = _coerce_llm_result(raw)
    assert out == {
        "root_cause": "tool_wear",
        "action_taken": "replaced tool",
        "tool_change": True,
        "source": "llm",
    }


def test_coerce_llm_result_extracts_embedded_json():
    raw = 'Sure — here is the answer: {"root_cause": null, "action_taken": "stopped", "tool_change": false} done.'
    out = _coerce_llm_result(raw)
    assert out is not None
    assert out["action_taken"] == "stopped"
    assert out["tool_change"] is False
    assert out["root_cause"] is None


def test_coerce_llm_result_rejects_non_json():
    assert _coerce_llm_result("I think the tool was fine.") is None
    assert _coerce_llm_result("") is None


def test_coerce_llm_result_handles_stringly_boolean():
    raw = '{"root_cause": "chatter", "action_taken": "adjusted", "tool_change": "yes"}'
    out = _coerce_llm_result(raw)
    assert out is not None
    assert out["tool_change"] is True


# ---------------------------------------------------------------------------
# Async classify_comment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_comment_uses_llm_when_valid_json_returned():
    llm = _FakeLLM(answer='{"root_cause":"tool_break","action_taken":"replaced tool","tool_change":true}')
    out = await classify_comment("something broke", llm_agent=llm)
    assert out["source"] == "llm"
    assert out["tool_change"] is True
    assert out["root_cause"] == "tool_break"
    assert llm.calls


@pytest.mark.asyncio
async def test_classify_comment_falls_back_on_bad_llm_output():
    llm = _FakeLLM(answer="not json at all")
    out = await classify_comment("Replaced tool due to wear.", llm_agent=llm)
    assert out["source"] == "heuristic"
    assert out["tool_change"] is True


@pytest.mark.asyncio
async def test_classify_comment_respects_timeout():
    llm = _FakeLLM(answer='{"root_cause":null,"action_taken":null,"tool_change":false}', delay=0.5)
    out = await classify_comment("tool chang", llm_agent=llm, timeout_s=0.02)
    assert out["source"] == "heuristic"
    assert out["tool_change"] is True  # heuristic catches "tool chang"


@pytest.mark.asyncio
async def test_classify_comment_on_llm_exception_uses_heuristic():
    llm = _FakeLLM(raises=True)
    out = await classify_comment("Swapped tool.", llm_agent=llm)
    assert out["source"] == "heuristic"
    assert out["tool_change"] is True


@pytest.mark.asyncio
async def test_classify_comment_without_llm_uses_heuristic():
    out = await classify_comment("Changed the tool.", llm_agent=None)
    assert out["source"] == "heuristic"
    assert out["tool_change"] is True


@pytest.mark.asyncio
async def test_classify_comment_skips_llm_when_unavailable():
    llm = _FakeLLM(answer='{"tool_change":true}', available=False)
    out = await classify_comment("tool change", llm_agent=llm)
    assert out["source"] == "heuristic"
    assert llm.calls == []

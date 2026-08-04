from __future__ import annotations

import pytest

from backend.agents.llm.explainer import ExplanationContext, ExplainerConfig, LLMExplainer
from backend.agents.core.schemas import Memory, PatternKey
from backend.agents.memory.scorer import SignificanceAction, SignificanceResult


class _StructuredCapturingExplainer(LLMExplainer):
    def __init__(self, *, response_text: str) -> None:
        super().__init__(ExplainerConfig(provider="ollama", model="test-model"))
        self.response_text = response_text
        self.calls: list[dict[str, object]] = []

    def is_available(self) -> bool:
        return True

    async def _call_llm_json_async(self, prompt: str, *, use_system_role: bool = False):
        self.calls.append({"prompt": prompt, "use_system_role": use_system_role})
        return self._parse_json_object(self.response_text)


def _context() -> ExplanationContext:
    return ExplanationContext(
        pattern_keys=["fault:tool_breakage"],
        significance=SignificanceResult(
            is_significant=True,
            score=0.84,
            action=SignificanceAction.ALERT,
            reasons=["power_spindle_delta_max exceeded threshold"],
            triggered_rules=["rules", "history"],
        ),
        feature_evidence={
            "fault:tool_breakage": [
                {
                    "feature": "power_spindle_delta_max",
                    "value": 23.4,
                    "threshold": 15.0,
                    "direction": "above",
                }
            ]
        },
        classical_model={"anomaly_detector_score": 0.82},
        feedback_stats={"fault:tool_breakage": {"confirms": 8, "dismisses": 1, "prior": 0.89}},
    )


@pytest.mark.asyncio
async def test_explain_grounded_async_formats_structured_json_response() -> None:
    explainer = _StructuredCapturingExplainer(
        response_text=(
            '{"indication":"Tool breakage is likely developing",'
            '"evidence":["power_spindle_delta_max reached 23.4 versus a 15.0 threshold",'
            '"anomaly_detector_score remained elevated at 0.82"],'
            '"concern":"Historical confirmation is 8 out of 9 similar cases",'
            '"operator_action":"Inspect the cutting edge and holder before continuing"}'
        )
    )

    text, source = await explainer.explain_grounded_async(_context())

    assert source == "llm"
    assert "Tool breakage is likely developing." in text
    assert "23.4 versus a 15.0 threshold." in text
    assert "anomaly_detector_score remained elevated at 0.82." in text
    assert "Historical confirmation is 8 out of 9 similar cases." in text
    assert "Inspect the cutting edge and holder before continuing." in text
    assert explainer.calls[0]["use_system_role"] is True
    assert "Return only a JSON object" in str(explainer.calls[0]["prompt"])


@pytest.mark.asyncio
async def test_explain_grounded_async_falls_back_on_invalid_structured_output() -> None:
    explainer = _StructuredCapturingExplainer(
        response_text='{"indication":"Tool breakage is likely developing","evidence":[],"concern":"","operator_action":""}'
    )

    text, source = await explainer.explain_grounded_async(_context())

    assert source == "fallback"
    assert "Significance 0.84" in text


@pytest.mark.asyncio
async def test_generate_memory_summary_async_uses_structured_summary_field() -> None:
    explainer = _StructuredCapturingExplainer(
        response_text='{"summary":"Tool breakage signature detected during roughing cut on titanium."}'
    )
    memory = Memory(
        session_id="session-1",
        time_range=(0.0, 1.0),
        pattern_keys=[PatternKey(key="fault:tool_breakage")],
    )

    text = await explainer.generate_memory_summary_async(memory)

    assert text == "Tool breakage signature detected during roughing cut on titanium."
    assert "\"summary\": str" in str(explainer.calls[0]["prompt"])


@pytest.mark.asyncio
async def test_generate_memory_summary_async_falls_back_on_invalid_structured_output() -> None:
    explainer = _StructuredCapturingExplainer(response_text='{"summary":""}')
    memory = Memory(
        session_id="session-1",
        time_range=(0.0, 1.0),
        pattern_keys=[PatternKey(key="fault:tool_breakage")],
    )

    text = await explainer.generate_memory_summary_async(memory)

    assert text is not None
    assert "fault:tool_breakage" in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "expected_key", "expected_value"),
    [
        ("groq", "response_format", {"type": "json_object"}),
        ("ollama", "format", "json"),
    ],
)
async def test_call_llm_async_sets_structured_output_payload(monkeypatch, provider, expected_key, expected_value) -> None:
    captured: list[dict[str, object]] = []

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None, headers=None):
            captured.append({"url": url, "json": json, "headers": headers})
            if provider == "groq":
                return _FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})
            return _FakeResponse({"message": {"content": '{"ok": true}'}})

    monkeypatch.setattr("backend.agents.llm.explainer.httpx.AsyncClient", _FakeAsyncClient)

    config = ExplainerConfig(
        provider=provider,
        model="test-model",
        groq_api_key="test-key",
        groq_api_url="https://groq.example",
        ollama_url="http://ollama.example/api/chat",
    )
    explainer = LLMExplainer(config)

    payload = await explainer._call_llm_json_async("Return JSON", use_system_role=True)

    assert payload == {"ok": True}
    assert captured[0]["json"][expected_key] == expected_value


def test_llm_cache_separates_text_and_json_modes() -> None:
    explainer = LLMExplainer(ExplainerConfig(provider="ollama", model="test-model"))

    explainer._set_cache("same-prompt", "plain-text")
    explainer._set_cache("same-prompt", '{"structured": true}', response_mode="json")

    assert explainer._get_cached("same-prompt") == "plain-text"
    assert explainer._get_cached("same-prompt", response_mode="json") == '{"structured": true}'
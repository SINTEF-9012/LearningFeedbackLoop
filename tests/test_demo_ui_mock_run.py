from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import demo_ui_mock_run as demo_ui_mock_run


def test_llm_preflight_only_when_required_and_explanations_enabled(monkeypatch):
    monkeypatch.delenv("REQUIRE_LLM", raising=False)
    monkeypatch.delenv("GENERATE_EXPLANATIONS", raising=False)

    should_preflight, reason = demo_ui_mock_run._should_preflight_llm(
        {"REQUIRE_LLM": "true", "GENERATE_EXPLANATIONS": "true"}
    )

    assert should_preflight is True
    assert "both true" in reason


def test_llm_preflight_skips_when_explanations_disabled(monkeypatch):
    monkeypatch.delenv("REQUIRE_LLM", raising=False)
    monkeypatch.delenv("GENERATE_EXPLANATIONS", raising=False)

    should_preflight, reason = demo_ui_mock_run._should_preflight_llm(
        {"REQUIRE_LLM": "true", "GENERATE_EXPLANATIONS": "false"}
    )

    assert should_preflight is False
    assert reason == "GENERATE_EXPLANATIONS is false"


def test_disable_llm_flag_overrides_llm_preflight(monkeypatch):
    monkeypatch.delenv("REQUIRE_LLM", raising=False)
    monkeypatch.delenv("GENERATE_EXPLANATIONS", raising=False)

    should_preflight, reason = demo_ui_mock_run._should_preflight_llm(
        {"REQUIRE_LLM": "true", "GENERATE_EXPLANATIONS": "true"},
        disable_llm=True,
    )

    assert should_preflight is False
    assert reason == "disabled via --disable-llm"


def test_disable_llm_mode_patches_backend_config(monkeypatch):
    calls = []

    def fake_patch_json(base_url, path, payload, *, timeout=120.0):
        calls.append((base_url, path, payload, timeout))
        return {"ok": True, "changed": {"generate_explanations": False}}

    monkeypatch.setattr(demo_ui_mock_run, "_patch_json", fake_patch_json)

    demo_ui_mock_run._configure_runtime_llm_mode(
        "http://localhost:8000",
        disable_llm=True,
    )

    assert calls == [
        (
            "http://localhost:8000",
            "/agent/memory/config",
            {"generate_explanations": False},
            15.0,
        )
    ]
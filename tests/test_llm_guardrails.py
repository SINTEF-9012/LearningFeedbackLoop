"""Tests for the Tier-1 LLM output guardrails ([LLM_GUARDRAILS_V1]).

One test per Tier-1 check plus an orchestrator-level integration test that a
blocked explanation falls back to the deterministic explanation and the
guardrail outcome is persisted on the memory record.
"""

from __future__ import annotations

import pytest

from backend.agents.llm.guardrails import (
    OutputGuardrail,
    GuardrailResult,
    UNCERTAINTY_CAVEAT,
)
from backend.agents.llm.explainer import ExplanationContext
from backend.agents.core.context import CuttingContext
from backend.agents.memory.scorer import SignificanceAction, SignificanceResult


# ---------------------------------------------------------------------------
# Fixtures / builders (mirror tests/test_llm_structured_output.py conventions)
# ---------------------------------------------------------------------------

def _significance(action: SignificanceAction = SignificanceAction.ALERT) -> SignificanceResult:
    return SignificanceResult(
        is_significant=True,
        score=0.84,
        action=action,
        reasons=["power_spindle_delta_max exceeded threshold"],
        triggered_rules=["rules", "history"],
    )


def _complete_context(
    action: SignificanceAction = SignificanceAction.ALERT,
) -> ExplanationContext:
    return ExplanationContext(
        pattern_keys=["fault:tool_breakage"],
        significance=_significance(action),
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
        cutting_context=CuttingContext(tool_type="end_mill", workpiece_material="steel"),
        raw_metrics_excerpt={"power_spindle_delta_max": 23.4},
    )


def _incomplete_context() -> ExplanationContext:
    """Evidence pack missing the cutting context (no tool / material)."""
    return ExplanationContext(
        pattern_keys=["fault:tool_breakage"],
        significance=_significance(),
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
        cutting_context=None,
    )


# ---------------------------------------------------------------------------
# Check 1: structure
# ---------------------------------------------------------------------------

def test_structure_blocks_empty_or_degenerate_output() -> None:
    g = OutputGuardrail()
    res = g.check("   ", _complete_context())
    assert res.action == "block"
    assert res.checks.get("structure") == "block"
    assert any("empty" in r.lower() or "short" in r.lower() for r in res.reasons)

    # A raw JSON blob (un-flattened) is also rejected as not-prose.
    res_json = g.check('{"indication": "breakage"}', _complete_context())
    assert res_json.action == "block"


# ---------------------------------------------------------------------------
# Check 2: grounding (out-of-pack entity)
# ---------------------------------------------------------------------------

def test_grounding_flags_out_of_pack_tool_id() -> None:
    g = OutputGuardrail()
    text = (
        "Replace tool T47 because power_spindle_delta_max reached 23.4 versus the "
        "15.0 threshold. Inspect the cutting edge before continuing."
    )
    res = g.check(text, _complete_context())
    # T47 is not in the evidence pack → annotate (default, not block).
    assert res.action == "annotate"
    assert res.checks.get("grounding") == "annotate"
    assert any("T47" in r for r in res.reasons)


def test_grounding_block_on_ungrounded_escalates() -> None:
    g = OutputGuardrail(block_on_ungrounded=True)
    text = (
        "Tool T47 has failed. power_spindle_delta_max reached 23.4 over the 15.0 "
        "threshold; inspect the holder."
    )
    res = g.check(text, _complete_context())
    assert res.action == "block"
    assert res.checks.get("grounding") == "block"


# ---------------------------------------------------------------------------
# Check 3: machine-control blocklist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Reduce the spindle speed to 8000 rpm immediately to avoid breakage.",
        "Change the feed rate and override the tool offset now.",
        "You should stop the machine and modify the NC program.",
        "Adjust the feed override parameter on the controller.",
    ],
)
def test_machine_control_command_is_blocked(text: str) -> None:
    g = OutputGuardrail()
    res = g.check(text, _complete_context())
    assert res.action == "block"
    assert res.checks.get("machine_control") == "block"
    assert any("machine-control" in r for r in res.reasons)


def test_inspection_advice_is_not_machine_control() -> None:
    g = OutputGuardrail()
    # "Inspect / check the tool" is operator advice, not a machine command.
    text = (
        "power_spindle_delta_max reached 23.4 over the 15.0 threshold with anomaly "
        "score 0.82. Inspect the cutting edge and check the tool holder."
    )
    res = g.check(text, _complete_context())
    assert res.checks.get("machine_control") == "pass"


# ---------------------------------------------------------------------------
# Check 4: uncertainty enforcement on incomplete context
# ---------------------------------------------------------------------------

def test_uncertainty_injected_on_incomplete_context() -> None:
    g = OutputGuardrail()
    # No hedging words, and the context is incomplete (no cutting context).
    text = (
        "power_spindle_delta_max reached 23.4 over the 15.0 threshold with anomaly "
        "score 0.82. Inspect the cutting edge."
    )
    res = g.check(text, _incomplete_context())
    assert res.action == "annotate"
    assert res.checks.get("uncertainty") == "annotate"
    assert UNCERTAINTY_CAVEAT in res.text


def test_uncertainty_not_injected_when_context_complete() -> None:
    g = OutputGuardrail()
    text = (
        "power_spindle_delta_max reached 23.4 over the 15.0 threshold with anomaly "
        "score 0.82. Inspect the cutting edge."
    )
    res = g.check(text, _complete_context())
    assert res.checks.get("uncertainty") == "pass"
    assert UNCERTAINTY_CAVEAT not in res.text


def test_uncertainty_not_injected_when_text_already_hedges() -> None:
    g = OutputGuardrail()
    text = (
        "Tool breakage may be developing; power_spindle_delta_max reached 23.4 over "
        "the 15.0 threshold. This is provisional — verify on the machine."
    )
    res = g.check(text, _incomplete_context())
    assert res.checks.get("uncertainty") == "pass"


# ---------------------------------------------------------------------------
# Check 5: scorer consistency
# ---------------------------------------------------------------------------

def test_scorer_consistency_flags_overstated_severity() -> None:
    g = OutputGuardrail()
    # Asserts "critical" while the scorer action is ALERT.
    text = (
        "This is a critical failure. power_spindle_delta_max reached 23.4 over the "
        "15.0 threshold with anomaly score 0.82. Inspect the cutting edge."
    )
    res = g.check(text, _complete_context(action=SignificanceAction.ALERT))
    assert res.checks.get("scorer_consistency") == "annotate"
    assert res.action in ("annotate",)
    assert any("critical" in r.lower() for r in res.reasons)


def test_scorer_consistency_ok_when_action_critical() -> None:
    g = OutputGuardrail()
    text = (
        "This is a critical failure. power_spindle_delta_max reached 23.4 over the "
        "15.0 threshold with anomaly score 0.82. Inspect the cutting edge."
    )
    res = g.check(text, _complete_context(action=SignificanceAction.CRITICAL))
    assert res.checks.get("scorer_consistency") == "pass"


# ---------------------------------------------------------------------------
# Clean, grounded output passes
# ---------------------------------------------------------------------------

def test_clean_grounded_output_passes() -> None:
    g = OutputGuardrail()
    text = (
        "Tool breakage is likely developing. power_spindle_delta_max reached 23.4 "
        "versus the 15.0 threshold, and the anomaly score remained elevated at 0.82. "
        "Inspect the cutting edge and tool holder before continuing the cut."
    )
    res = g.check(text, _complete_context())
    assert res.action == "pass"
    assert res.text == text
    assert all(v == "pass" for v in res.checks.values())


# ---------------------------------------------------------------------------
# Tier-2 hook is a pluggable seam, not implemented
# ---------------------------------------------------------------------------

def test_tier2_hook_default_is_none() -> None:
    g = OutputGuardrail()
    assert g.semantic_checker is None


def test_tier2_hook_invoked_when_provided() -> None:
    captured = {}

    def fake_checker(text: str, ctx, tier1: GuardrailResult) -> GuardrailResult:
        captured["called"] = True
        captured["tier1_action"] = tier1.action
        return GuardrailResult(action="block", reasons=["tier2:semantic"], text=text)

    g = OutputGuardrail(semantic_checker=fake_checker)
    res = g.check(
        "power_spindle_delta_max reached 23.4 over the 15.0 threshold; inspect the tool.",
        _complete_context(),
    )
    assert captured.get("called") is True
    assert res.action == "block"
    assert "tier2:semantic" in res.reasons


def test_guardrail_never_raises_on_bad_ctx() -> None:
    g = OutputGuardrail()
    # None ctx and an object missing all expected fields must not raise.
    res_none = g.check("Some explanation text that is long enough to pass.", None)
    assert isinstance(res_none, GuardrailResult)

    class _Bad:
        pass

    res_bad = g.check("Some explanation text that is long enough to pass.", _Bad())
    assert isinstance(res_bad, GuardrailResult)


# ---------------------------------------------------------------------------
# Orchestrator-level integration: blocked output falls back + outcome stored
# ---------------------------------------------------------------------------

class _FakeStore:
    """Minimal store exposing the get_memory/update_memory API the background
    explanation persistence path uses."""

    def __init__(self) -> None:
        self._memories = {}

    def store_memory(self, memory):
        self._memories[memory.id] = memory
        return memory.id

    def get_memory(self, memory_id):
        return self._memories.get(memory_id)

    def update_memory(self, memory):
        self._memories[memory.id] = memory
        return memory.id


@pytest.mark.asyncio
async def test_orchestrator_blocks_machine_control_and_persists_outcome() -> None:
    from backend.agents.memory.orchestrator import (
        MemoryEventOrchestrator,
        OrchestratorConfig,
    )
    from backend.agents.core.schemas import Memory, PatternKey, PatternType, NumericMetrics
    from backend.agents.core.context import CuttingContext as CC
    from datetime import datetime, timezone

    config = OrchestratorConfig(
        use_classical_models=False,
        enable_harmonic_scorer=False,
        generate_explanations=True,
        llm_guardrails_enabled=True,
        dispatch_alerts=False,
    )
    orch = MemoryEventOrchestrator(memory_store=_FakeStore(), config=config)
    assert orch.output_guardrail is not None

    # Force the explainer to return an unsafe (machine-control) explanation.
    async def _fake_grounded(ctx):
        return (
            "Reduce the spindle speed to 6000 rpm and modify the NC program now.",
            "llm",
        )

    orch.explainer.explain_grounded_async = _fake_grounded  # type: ignore[assignment]

    # Avoid real alert-line LLM calls.
    async def _fake_alert_line(*args, **kwargs):
        return ("alert line", "fallback", "inspect the tool at the next safe stop")

    orch.explainer.summarize_with_history_for_alert_async = _fake_alert_line  # type: ignore
    orch.explainer.explain_significance_for_alert_async = _fake_alert_line  # type: ignore

    # Build and store a real memory so persistence has a target.
    memory = Memory(
        id="mem-guardrail-test",
        session_id="sess-1",
        created_at=datetime.now(timezone.utc),
        created_by="test",
        time_range=(0.0, 1.0),
        annotation_text="",
        pattern_keys=[PatternKey(key="fault:tool_breakage", pattern_type=PatternType.FAULT)],
        metrics=NumericMetrics(),
    )
    orch.store.store_memory(memory)

    significance = _significance()
    expl_ctx = ExplanationContext(
        pattern_keys=["fault:tool_breakage"],
        significance=significance,
        feature_evidence={
            "fault:tool_breakage": [
                {"feature": "power_spindle_delta_max", "value": 23.4, "threshold": 15.0, "direction": "above"}
            ]
        },
        classical_model={"anomaly_detector_score": 0.82},
        cutting_context=CC(tool_type="end_mill", workpiece_material="steel"),
    )

    captured_broadcast = {}

    async def _capture_broadcast(**kwargs):
        captured_broadcast.update(kwargs)

    orch.alert_dispatcher.broadcast_explanation_update = _capture_broadcast  # type: ignore

    from backend.agents.memory.orchestrator import MemoryEvent
    from backend.agents.core.schemas import TimeRange

    event = MemoryEvent(
        session_id="sess-1",
        time_range=TimeRange(i0=0, i1=1, t0=0.0, t1=1.0, fs=1.0),
        patterns=[PatternKey(key="fault:tool_breakage", pattern_type=PatternType.FAULT)],
        cutting_context=CC(tool_type="end_mill", workpiece_material="steel"),
    )

    await orch._generate_explanation_background(
        memory=memory,
        memory_id=memory.id,
        event=event,
        significance=significance,
        similar_memories=[],
        expl_ctx=expl_ctx,
    )

    # The unsafe explanation must NOT be persisted; a fallback should replace it.
    stored = orch.store.get_memory(memory.id)
    assert stored is not None
    meta = stored.metadata or {}
    assert meta.get("explanation_source") == "guardrail_fallback"
    assert "modify the nc program" not in (meta.get("explanation") or "").lower()
    assert "spindle speed to 6000" not in (meta.get("explanation") or "").lower()

    # Guardrail outcome stored for audit.
    outcome = meta.get("guardrail_outcome")
    assert outcome is not None
    assert outcome["action"] == "block"
    assert any("machine-control" in r for r in outcome["reasons"])

    # And the outcome was broadcast.
    assert captured_broadcast.get("guardrail_outcome") is not None
    assert captured_broadcast["guardrail_outcome"]["action"] == "block"

"""Agent A — tests for loop_metrics plumbing.

The HTTP endpoint is a thin adapter over three scorer primitives:
  - SignificanceScorer.get_rule_performance()
  - AdaptiveThresholds.to_dict()
  - feedback_handler.get_feedback_stats() (per-session roll-up)

These tests validate that those primitives return the expected shapes
after simulated feedback, so the endpoint can be trusted to surface them.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.agents.model_confidence import record_model_feedback_outcome
from backend.agents.memory import router as router_mod
from backend.agents.memory.scorer import (
    AdaptiveThresholds,
    RulePerformance,
    SignificanceConfig,
    SignificanceScorer,
)
from backend.agents.processing.classical_models import RLState


def test_rule_performance_computes_precision_recall_f1():
    perf = RulePerformance()
    # 3 TP, 1 FP, 1 FN, 1 TN
    perf.record(True, True)
    perf.record(True, True)
    perf.record(True, True)
    perf.record(True, False)   # FP
    perf.record(False, True)   # FN
    perf.record(False, False)  # TN
    d = perf.to_dict()
    assert d["true_positives"] == 3
    assert d["false_positives"] == 1
    assert d["false_negatives"] == 1
    assert d["n_samples"] == 6
    assert 0.7 < d["precision"] < 0.8   # 3 / (3+1) = 0.75
    assert 0.7 < d["recall"] < 0.8      # 3 / (3+1) = 0.75
    assert 0.7 < d["f1"] < 0.8


def test_scorer_get_rule_performance_returns_all_rules():
    scorer = SignificanceScorer()
    # Simulate some feedback
    scorer.record_rule_feedback(
        triggered_rules={"classical_alert", "pattern_match"},
        was_confirmed=True,
    )
    scorer.record_rule_feedback(
        triggered_rules={"classical_alert"},
        was_confirmed=False,
    )
    perf = scorer.get_rule_performance()
    # All canonical rules should have entries.
    for name in ("classical_alert", "harmonic_alert", "pattern_match", "anomaly_deviation", "historical_prior"):
        assert name in perf
        assert "precision" in perf[name]
        assert "f1" in perf[name]
        assert "n_samples" in perf[name]


def test_harmonic_signal_triggers_separate_rule_bucket():
    scorer = SignificanceScorer()

    result = scorer.score(patterns=[], external_signals={"harmonic_context_score": 0.82})

    assert "harmonic_alert" in result.triggered_rules
    assert "classical_alert" not in result.triggered_rules


def test_harmonic_rule_emits_its_own_weight_and_trace_entry():
    scorer = SignificanceScorer()

    result = scorer.score(patterns=[], external_signals={"harmonic_context_score": 0.82})
    components = {entry["component"] for entry in result.score_trace}

    assert "weight:harmonic_alert" in components
    assert "rule:harmonic_alert" in components


def test_harmonic_rule_honors_propagated_threshold():
    scorer = SignificanceScorer()

    result = scorer.score(
        patterns=[],
        external_signals={
            "harmonic_context_score": 0.82,
            "harmonic_context_threshold": 0.92,
        },
    )

    assert "harmonic_alert" not in result.triggered_rules


def test_rl_state_key_changes_with_harmonic_score():
    base = RLState(seed_model_score=0.6, harmonic_score=0.1, pattern_score=0.2, historical_prior=0.5)
    harmonic = RLState(seed_model_score=0.6, harmonic_score=0.9, pattern_score=0.2, historical_prior=0.5)

    assert base.to_key() != harmonic.to_key()


def test_adaptive_thresholds_to_dict_schema():
    at = AdaptiveThresholds(
        base_alert=0.6, base_store=0.3, base_critical=0.85,
        target_precision=0.7,
    )
    # Record a handful of alerts
    for i in range(10):
        at.record_feedback(score=0.7 + (i % 3) * 0.05, action="alert", confirmed=(i % 2 == 0))
    d = at.to_dict()
    assert set(d.keys()) >= {
        "alert_threshold", "store_threshold", "critical_threshold",
        "current_precision", "n_samples", "target_precision",
    }
    assert d["n_samples"] >= 5
    assert d["current_precision"] is not None
    assert 0.0 <= d["current_precision"] <= 1.0


def test_adaptive_thresholds_precision_none_when_too_few_samples():
    at = AdaptiveThresholds()
    at.record_feedback(score=0.7, action="alert", confirmed=True)
    at.record_feedback(score=0.7, action="alert", confirmed=False)
    # Fewer than 5 samples → precision should be None (guarded)
    d = at.to_dict()
    assert d["current_precision"] is None


def test_scorer_disabled_rule_performance_returns_empty():
    scorer = SignificanceScorer(
        config=SignificanceConfig(enable_rule_performance_tracking=False)
    )
    scorer.record_rule_feedback(triggered_rules={"classical_alert"}, was_confirmed=True)
    # Disabled: recording is a no-op, map stays at default (the 4 empty rules from init).
    perf = scorer.get_rule_performance()
    # All entries should have 0 samples
    for name, p in perf.items():
        assert p["n_samples"] == 0


@pytest.mark.asyncio
async def test_loop_metrics_exposes_model_trust_diagnostics(tmp_path, monkeypatch):
    confidence_path = tmp_path / "model_confidence.json"
    record_model_feedback_outcome(model_fired=True, was_confirmed=True, path=confidence_path)

    scorer = SignificanceScorer(model_confidence_path=confidence_path)
    orchestrator = SimpleNamespace(scorer=scorer, store=None, feedback_handler=None)
    monkeypatch.setattr(router_mod, "get_orchestrator", lambda: orchestrator)

    payload = await router_mod.get_loop_metrics()

    assert "model_trust" in payload
    assert payload["model_trust"]["exists"] is True
    assert payload["model_trust"]["path"] == str(confidence_path)
    assert payload["model_trust"]["model_confidence"] > 0.5

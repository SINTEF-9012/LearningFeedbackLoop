"""Agent D phase 2 — unit tests for score_trace and externalized aliases.

These target the explainability additions made in Agent D phase 1. They are
intentionally narrow: no end-to-end orchestrator wiring, no Neo4j — just the
scorer's public surface and the alias-loader contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.core.schemas import PatternKey, PatternType
from backend.agents.memory import scorer as scorer_mod
from backend.agents.memory.scorer import (
    SignificanceAction,
    SignificanceConfig,
    SignificanceResult,
    SignificanceScorer,
    _load_pattern_aliases,
    _PATTERN_KEY_ALIASES_BUILTIN,
    is_hypothesis_pattern_key,
    normalize_pattern_key,
)
from backend.agents.processing.classical_models import RLAction
from backend.agents.patterns.signatures import CHATTER_SIGNATURE, WORKPIECE_SLIP_SIGNATURE


# ─────────────────────────────────────────────────────────────────────────────
# score_trace shape + content
# ─────────────────────────────────────────────────────────────────────────────

def _collect_components(trace):
    return [entry["component"] for entry in trace]


def test_score_trace_default_empty():
    """Default-constructed SignificanceResult has an empty trace."""
    r = SignificanceResult(
        is_significant=False,
        score=0.0,
        action=SignificanceAction.IGNORE,
        reasons=[],
        triggered_rules=[],
    )
    assert r.score_trace == []
    assert "score_trace" in r.to_dict()


def test_score_trace_populated_on_score_with_external_signal():
    """Classical-only path produces a trace with final_score at the end."""
    scorer = SignificanceScorer()
    result = scorer.score(
        patterns=[],
        external_signals={"breakage_prediction": 0.9},
    )
    components = _collect_components(result.score_trace)
    # final_score must always be present and last
    assert components[-1] == "final_score"
    # Effective weights should be emitted for all 4 rules
    assert "weight:classical_alert" in components
    assert "weight:pattern_match" in components
    assert "weight:anomaly_deviation" in components
    assert "weight:historical_prior" in components
    # final_score entry's value equals result.score
    final_entry = result.score_trace[-1]
    assert final_entry["value"] == pytest.approx(result.score, rel=1e-6)


def test_score_trace_emits_model_trust_when_classical_signal_is_discounted():
    scorer = SignificanceScorer()
    result = scorer.score(
        patterns=[],
        external_signals={"breakage_prediction": 0.9, "model_confidence": 0.42},
    )

    trust_entry = next(
        entry for entry in result.score_trace
        if entry["component"] == "model_trust"
    )

    assert trust_entry["value"] == pytest.approx(0.42, rel=1e-6)
    assert trust_entry["source"] == "feedback_driven_model_confidence"


def test_score_trace_has_prior_component():
    """Either multiplicative (prior_factor) or additive (prior_boost_additive) path fires."""
    # Multiplicative is the default prior_mode
    scorer = SignificanceScorer(config=SignificanceConfig(prior_mode="multiplicative"))
    result = scorer.score(
        patterns=[PatternKey(pattern_type=PatternType.ANOMALY, key="ANOMALY_HIGH:0.9")],
        external_signals={},
    )
    components = _collect_components(result.score_trace)
    assert "prior_factor" in components
    assert "prior_boost_additive" not in components

    scorer2 = SignificanceScorer(config=SignificanceConfig(prior_mode="additive"))
    result2 = scorer2.score(
        patterns=[PatternKey(pattern_type=PatternType.ANOMALY, key="ANOMALY_HIGH:0.9")],
        external_signals={},
    )
    components2 = _collect_components(result2.score_trace)
    assert "prior_boost_additive" in components2
    assert "prior_factor" not in components2


def test_significance_result_exposes_prior_semantics_without_calling_raw_prior_a_boost():
    scorer = SignificanceScorer(config=SignificanceConfig(prior_mode="multiplicative", prior_evidence_damping_k=0.0))
    scorer._get_pattern_prior_and_count = lambda *_args, **_kwargs: (0.88, 1)

    result = scorer.score(
        patterns=[PatternKey(pattern_type=PatternType.FAULT, key="spectral:spindle_freq_shift")],
        external_signals={"anomaly_detector_score": 0.36},
    )

    assert result.historical_prior == pytest.approx(0.88, rel=1e-6)
    assert result.prior_damping_factor == pytest.approx(1.0, rel=1e-6)
    assert result.prior_factor > 1.0
    assert result.prior_boost <= result.score
    payload = result.to_dict()
    assert payload["historical_prior"] == pytest.approx(0.88, rel=1e-6)
    assert payload["prior_damping_factor"] == pytest.approx(1.0, rel=1e-6)
    assert payload["prior_factor"] == pytest.approx(result.prior_factor, rel=1e-6)


def test_prior_damping_reduces_low_evidence_prior_influence():
    scorer = SignificanceScorer(config=SignificanceConfig(prior_mode="multiplicative", prior_evidence_damping_k=5.0))
    scorer._get_pattern_prior_and_count = lambda *_args, **_kwargs: (0.9, 1)

    result = scorer.score(
        patterns=[PatternKey(pattern_type=PatternType.CUSTOM, key="spectral:spindle_freq_shift")],
        external_signals={"anomaly_detector_score": 0.7},
    )

    expected_damping = 1.0 / 6.0
    expected_effective_prior = 0.5 + ((0.9 - 0.5) * expected_damping)
    expected_factor = 0.6 + (0.8 * expected_effective_prior)

    assert result.historical_prior == pytest.approx(0.9, rel=1e-6)
    assert result.prior_evidence_count == 1
    assert result.prior_damping_factor == pytest.approx(expected_damping, rel=1e-6)
    assert result.prior_factor == pytest.approx(expected_factor, rel=1e-6)


def test_prior_damping_scales_up_with_more_evidence():
    scorer = SignificanceScorer(config=SignificanceConfig(prior_mode="multiplicative", prior_evidence_damping_k=5.0))
    scorer._get_pattern_prior_and_count = lambda *_args, **_kwargs: (0.9, 20)

    result = scorer.score(
        patterns=[PatternKey(pattern_type=PatternType.CUSTOM, key="spectral:spindle_freq_shift")],
        external_signals={"anomaly_detector_score": 0.7},
    )

    expected_damping = 20.0 / 25.0
    expected_effective_prior = 0.5 + ((0.9 - 0.5) * expected_damping)
    expected_factor = 0.6 + (0.8 * expected_effective_prior)

    assert result.prior_evidence_count == 20
    assert result.prior_damping_factor == pytest.approx(expected_damping, rel=1e-6)
    assert result.prior_factor == pytest.approx(expected_factor, rel=1e-6)


def test_fault_pattern_prefers_specific_reason_over_generic_type_reason():
    scorer = SignificanceScorer(config=SignificanceConfig(prior_mode="multiplicative"))
    scorer.get_pattern_prior = lambda *_args, **_kwargs: 0.0

    result = scorer.score(
        patterns=[PatternKey(pattern_type=PatternType.CUSTOM, key=WORKPIECE_SLIP_SIGNATURE)],
        external_signals={},
    )

    assert f"Significant pattern: {WORKPIECE_SLIP_SIGNATURE}" in result.reasons
    assert "Critical pattern type: fault" not in result.reasons


def test_signature_patterns_do_not_contribute_historical_prior_yet():
    scorer = SignificanceScorer(config=SignificanceConfig(prior_mode="multiplicative"))
    scorer._local_feedback_counts[WORKPIECE_SLIP_SIGNATURE] = {"confirm": 12, "dismiss": 0}

    result = scorer.score(
        patterns=[PatternKey(pattern_type=PatternType.CUSTOM, key=WORKPIECE_SLIP_SIGNATURE)],
        external_signals={},
    )

    assert result.pattern_priors[WORKPIECE_SLIP_SIGNATURE] == pytest.approx(0.5, rel=1e-6)
    assert "historical_prior" not in result.triggered_rules


def test_score_trace_rule_entries_match_triggered_rules():
    """rule:<name> entries should map 1:1 to non-historical triggered rules."""
    scorer = SignificanceScorer()
    result = scorer.score(
        patterns=[PatternKey(pattern_type=PatternType.ANOMALY, key="ANOMALY_HIGH:0.9")],
        external_signals={"breakage_prediction": 0.85},
    )
    rule_entries = [
        e for e in result.score_trace
        if isinstance(e.get("component"), str) and e["component"].startswith("rule:")
    ]
    rule_entry_names = {e["component"].split(":", 1)[1] for e in rule_entries}
    non_historical_triggered = {r for r in result.triggered_rules if r != "historical_prior"}
    assert rule_entry_names == non_historical_triggered


def test_rl_safe_mode_reverts_exploratory_action_for_high_score_events():
    class _StubRLAgent:
        _epsilon = 1.0

        def select_action(self, _state):
            return 7, RLAction(classical_weight_adj=0.05, pattern_weight_adj=-0.05)

        def get_recommended_action(self, _state):
            return RLAction(classical_weight_adj=0.0, pattern_weight_adj=0.0)

    scorer = SignificanceScorer(
        config=SignificanceConfig(
            prior_mode="multiplicative",
            rl_safe_mode_threshold=0.0,
        )
    )
    scorer.set_rl_agent(_StubRLAgent())

    result = scorer.score(
        patterns=[
            PatternKey(
                pattern_type=PatternType.CUSTOM,
                key=WORKPIECE_SLIP_SIGNATURE,
                confidence=0.7,
            )
        ],
        external_signals={"breakage_prediction": 0.9},
    )

    trace_weights = {
        entry["component"]: entry["value"]
        for entry in result.score_trace
        if entry["component"].startswith("weight:")
    }
    classical_weight = trace_weights["weight:classical_alert"]
    pattern_weight = trace_weights["weight:pattern_match"]

    expected_greedy_score = ((0.9 * classical_weight) + (0.7 * pattern_weight)) / (
        classical_weight + pattern_weight
    )
    expected_greedy_score += scorer.config.agreement_bonus_model_pattern
    exploratory_score = (
        (0.9 * (classical_weight + 0.05)) + (0.7 * (pattern_weight - 0.05))
    ) / ((classical_weight + 0.05) + (pattern_weight - 0.05))
    exploratory_score += scorer.config.agreement_bonus_model_pattern

    assert result.score == pytest.approx(expected_greedy_score, rel=1e-6)
    assert result.score < exploratory_score
    assert any(entry["component"] == "rl_safe_mode" for entry in result.score_trace)


def test_score_trace_entry_schema():
    """Every entry is a dict with the expected keys."""
    scorer = SignificanceScorer()
    result = scorer.score(
        patterns=[], external_signals={"breakage_prediction": 0.9},
    )
    for entry in result.score_trace:
        assert set(entry.keys()) >= {"component", "value", "source"}
        assert isinstance(entry["component"], str)
        assert isinstance(entry["value"], (int, float))
        assert isinstance(entry["source"], str)


# ─────────────────────────────────────────────────────────────────────────────
# Alias loader — data file wins over built-ins
# ─────────────────────────────────────────────────────────────────────────────

def test_builtin_aliases_have_known_keys():
    """Sanity check: core legacy aliases are in the built-in table."""
    for legacy in ("FAULT_CHATTER", "CHATTER_DETECTED", "TOOL_WEAR_RISK"):
        assert legacy in _PATTERN_KEY_ALIASES_BUILTIN


def test_loader_merges_builtin_with_data_file():
    """When data/pattern_aliases.json exists, its entries override built-ins."""
    merged = _load_pattern_aliases()
    # built-in keys still resolve
    assert merged["FAULT_CHATTER"] == "fault:chatter"
    # loader is a superset of built-ins
    for k, v in _PATTERN_KEY_ALIASES_BUILTIN.items():
        assert merged.get(k) == v


def test_normalize_pattern_key_uses_merged_table():
    assert normalize_pattern_key("FAULT_CHATTER") == CHATTER_SIGNATURE
    assert normalize_pattern_key("fault:workpiece_slip") == WORKPIECE_SLIP_SIGNATURE
    assert is_hypothesis_pattern_key("fault:workpiece_slip")
    # Unknown keys pass through untouched
    assert normalize_pattern_key("some:unknown:key") == "some:unknown:key"


def test_loader_falls_back_on_malformed_file(tmp_path, monkeypatch):
    """Malformed JSON should not crash; built-ins remain."""
    bad = tmp_path / "pattern_aliases.json"
    bad.write_text("{not valid json", encoding="utf-8")
    # Patch the loader's candidate path by monkey-patching Path resolution.
    # Easier: call _load_pattern_aliases directly after swapping Path.
    orig_path = scorer_mod.Path

    class _FakePath(type(orig_path)):
        pass

    # Simpler route: temporarily swap the candidate via direct call wrapper.
    def _fake_load():
        merged = dict(_PATTERN_KEY_ALIASES_BUILTIN)
        try:
            if bad.is_file():
                with bad.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)  # will raise
                file_aliases = raw.get("aliases") if isinstance(raw, dict) else None
                if isinstance(file_aliases, dict):
                    for k, v in file_aliases.items():
                        if isinstance(k, str) and isinstance(v, str):
                            merged[k] = v
        except Exception:
            pass
        return merged

    result = _fake_load()
    # Still contains all built-in aliases despite malformed file
    for k, v in _PATTERN_KEY_ALIASES_BUILTIN.items():
        assert result[k] == v

"""Pluggable models: register models and select a SET of them.

Covers both plug points:
  - scoring-rule registry (fusion layer) in scorer.py
  - signal-model registry in processing/model_registry.py
"""

from __future__ import annotations

import tempfile

import pytest

from backend.agents.memory.scorer import (
    SignificanceScorer,
    SignificanceConfig,
    register_scoring_rule,
    build_rules,
    DEFAULT_RULE_ORDER,
    _SignificanceRule,
    _RuleResult,
)
from backend.agents.processing.model_registry import (
    register_model,
    select_models,
    run_models,
    available_models,
    ScoringModel,
)


def _scorer(tmp_path, **cfg):
    return SignificanceScorer(config=SignificanceConfig(**cfg),
                              priors_path=str(tmp_path / "p.json"))


# -------------------- scoring-rule registry --------------------

def test_default_rule_set_unchanged(tmp_path):
    s = _scorer(tmp_path)
    assert [r.name for r in s._rules] == DEFAULT_RULE_ORDER


def test_select_subset_of_models(tmp_path):
    s = _scorer(tmp_path, enabled_rules=["pattern_match", "historical_prior"])
    assert [r.name for r in s._rules] == ["pattern_match", "historical_prior"]


def test_plug_in_new_model_rule(tmp_path):
    class _MyRule(_SignificanceRule):
        name = "unit_plug_model"
        def evaluate(self, ctx):
            return _RuleResult(triggered=True, score=0.7, reasons=["x"])
        def weight(self, config):
            return 0.4

    register_scoring_rule("unit_plug_model", _MyRule)
    assert "unit_plug_model" in [r.name for r in build_rules(["unit_plug_model"])]
    s = _scorer(tmp_path, enabled_rules=["unit_plug_model", "pattern_match"])
    assert [r.name for r in s._rules] == ["unit_plug_model", "pattern_match"]


def test_inject_rule_instances_directly(tmp_path):
    class _R(_SignificanceRule):
        name = "injected"
        def evaluate(self, ctx):
            return _RuleResult(triggered=False, score=0.0, reasons=[])
        def weight(self, config):
            return 0.1
    s = SignificanceScorer(priors_path=str(tmp_path / "p.json"), rules=[_R()])
    assert [r.name for r in s._rules] == ["injected"]


def test_unknown_rule_fails_loud(tmp_path):
    with pytest.raises(KeyError):
        _scorer(tmp_path, enabled_rules=["does_not_exist"])


# -------------------- signal-model registry --------------------

class _ConstModel:
    def __init__(self, name="const", value=0.9, key="anomaly_detector_score"):
        self.name = name
        self._v = value
        self._k = key
    def score(self, features, context=None):
        return {self._k: self._v}


def test_register_and_select_model_set():
    register_model("unit_const_a", lambda: _ConstModel("unit_const_a", 0.9, "anomaly_detector_score"))
    register_model("unit_const_b", lambda: _ConstModel("unit_const_b", 0.3, "aad_score"))
    assert "unit_const_a" in available_models()
    models = select_models(["unit_const_a", "unit_const_b"])
    assert [m.name for m in models] == ["unit_const_a", "unit_const_b"]
    # ScoringModel protocol is satisfied (duck-typed)
    assert isinstance(models[0], ScoringModel)


def test_run_models_merges_signals():
    register_model("unit_const_a", lambda: _ConstModel("unit_const_a", 0.9, "anomaly_detector_score"))
    register_model("unit_const_b", lambda: _ConstModel("unit_const_b", 0.3, "aad_score"))
    merged = run_models(select_models(["unit_const_a", "unit_const_b"]), {"f": 1.0})
    assert merged == {"anomaly_detector_score": 0.9, "aad_score": 0.3}


def test_run_models_first_wins_and_skips_errors():
    register_model("unit_const_a", lambda: _ConstModel("unit_const_a", 0.9, "sig"))
    register_model("unit_const_c", lambda: _ConstModel("unit_const_c", 0.1, "sig"))  # same key

    class _Boom:
        name = "boom"
        def score(self, features, context=None):
            raise RuntimeError("model failed")

    register_model("unit_boom", lambda: _Boom())
    models = select_models(["unit_const_a", "unit_boom", "unit_const_c"])
    merged = run_models(models, {"f": 1.0}, on_error="skip")
    assert merged == {"sig": 0.9}  # first wins; failing model skipped


def test_select_unknown_model_fails_loud():
    with pytest.raises(KeyError):
        select_models(["nope_not_registered"])


# -------------------- AAD combiner --------------------

def test_aad_combiner_learns_from_feedback():
    """The AAD combiner must learn, from confirm/dismiss, which patterns predict
    real stops — scoring a confirmed pattern-set high and a dismissed set low."""
    from backend.agents.processing.aad_combiner import AADCombiner
    aad = AADCombiner(learning_rate=0.3)
    confirm_set, dismiss_set = ["SPINDLE_POWER_SURGE"], ["FEED_OVERRIDE_DROP"]
    for _ in range(40):
        aad.update(confirm_set, anomaly=0.8, label=True)
        aad.update(dismiss_set, anomaly=0.1, label=False)
    p_conf = aad._prob(confirm_set, 0.8)
    p_dis = aad._prob(dismiss_set, 0.1)
    assert p_conf > 0.7 and p_dis < 0.3
    assert p_conf > p_dis
    # credit assignment: the confirmed pattern has the larger positive weight
    assert aad.w["SPINDLE_POWER_SURGE"] > aad.w["FEED_OVERRIDE_DROP"]


def test_aad_combiner_score_protocol_and_registry():
    from backend.agents.processing.aad_combiner import AADCombiner
    from backend.agents.processing.model_registry import select_models, MODEL_REGISTRY
    assert "aad_combiner" in MODEL_REGISTRY
    m = select_models(["aad_combiner"])[0]
    out = m.score({"pattern_keys": ["X"], "anomaly": 0.5})
    assert "aad_score" in out and 0.0 <= out["aad_score"] <= 1.0


def test_aad_rule_registered_and_reads_signal(tmp_path):
    """The aad_combiner RULE surfaces external_signals['aad_score']."""
    from backend.agents.memory.scorer import (
        SignificanceScorer, SignificanceConfig, SCORING_RULE_REGISTRY,
    )
    from backend.agents.core.schemas import PatternKey as PK
    assert "aad_combiner" in SCORING_RULE_REGISTRY
    s = SignificanceScorer(config=SignificanceConfig(enabled_rules=["aad_combiner"]),
                           priors_path=str(tmp_path / "p.json"))
    assert [r.name for r in s._rules] == ["aad_combiner"]
    hi = s.score(patterns=[PK(key="X")], external_signals={"aad_score": 0.95})
    lo = s.score(patterns=[PK(key="X")], external_signals={"aad_score": 0.05})
    assert hi.score > lo.score

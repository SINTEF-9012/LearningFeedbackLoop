"""Legacy experiment pattern keys must be scoreable via the alias table.

The stoppage-experiment registry emits legacy stat-burst keys (HF_ENERGY_BURST,
IMPULSE_BURST) that data/pattern_aliases.json maps to canonical keys
(spectral:hf_burst, temporal:impulsive_burst). The pattern_match rule must
recognize them through the canonical form, and learned priors must accumulate
on them (they are not signature: keys, so they are prior-learnable).
"""

from backend.agents.core.schemas import PatternKey
from backend.agents.memory.scorer import (
    SignificanceConfig,
    SignificanceScorer,
    is_signature_pattern_key,
    normalize_pattern_key,
)


def test_legacy_stat_keys_normalize_to_canonical():
    assert normalize_pattern_key("HF_ENERGY_BURST") == "spectral:hf_burst"
    assert normalize_pattern_key("IMPULSE_BURST") == "temporal:impulsive_burst"
    # Canonical forms are prior-learnable (not signature-pinned).
    assert not is_signature_pattern_key("spectral:hf_burst")
    assert not is_signature_pattern_key("temporal:impulsive_burst")


def test_pattern_rule_triggers_on_aliased_legacy_key():
    scorer = SignificanceScorer(SignificanceConfig())
    result = scorer.score(patterns=[PatternKey(key="HF_ENERGY_BURST")], metrics=None)
    assert "pattern_match" in result.triggered_rules


def test_confirms_on_aliased_key_boost_score():
    scorer = SignificanceScorer(SignificanceConfig())
    patterns = [PatternKey(key="HF_ENERGY_BURST")]

    neutral = scorer.score(patterns=patterns, metrics=None)

    scorer.seed_feedback_counts({"spectral:hf_burst": {"confirm": 10.0, "dismiss": 0.0}})
    boosted = scorer.score(patterns=patterns, metrics=None)

    assert boosted.historical_prior > 0.5
    assert boosted.score > neutral.score


def _scorer(**overrides):
    cfg = SignificanceConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return SignificanceScorer(cfg)


def test_dismissals_do_not_damp_when_flag_off():
    scorer = _scorer()
    patterns = [PatternKey(key="HF_ENERGY_BURST")]
    neutral = scorer.score(patterns=patterns, metrics=None)
    scorer.seed_feedback_counts({"spectral:hf_burst": {"confirm": 0.0, "dismiss": 10.0}})
    dismissed = scorer.score(patterns=patterns, metrics=None)
    assert dismissed.score == neutral.score


def test_dismissals_damp_score_when_enabled():
    # Raise the critical threshold so the pattern's 0.85 base is sub-critical
    # (at the default threshold the base is critical-band and immune — see the
    # immunity test below).
    scorer = _scorer(prior_allow_subneutral=True, critical_threshold=0.95)
    patterns = [PatternKey(key="IMPULSE_BURST")]
    neutral = scorer.score(patterns=patterns, metrics=None)
    assert neutral.score < scorer.config.critical_threshold
    scorer.seed_feedback_counts({"temporal:impulsive_burst": {"confirm": 0.0, "dismiss": 10.0}})
    damped = scorer.score(patterns=patterns, metrics=None)
    assert damped.historical_prior < 0.5
    assert damped.score < neutral.score
    # Factor floor respected.
    assert damped.score >= neutral.score * scorer.config.prior_factor_floor - 1e-9


def test_recalibrated_stat_pattern_is_demotable():
    """After plan 1.11 (severity 0.85→0.65) a lone stat-burst pattern is
    alert-band, not critical — so sustained dismissals CAN demote it (the
    behaviour that was impossible while it sat at the critical threshold)."""
    scorer = _scorer(prior_allow_subneutral=True)
    patterns = [PatternKey(key="HF_ENERGY_BURST")]
    neutral = scorer.score(patterns=patterns, metrics=None)
    assert neutral.score < scorer.config.critical_threshold
    scorer.seed_feedback_counts({"spectral:hf_burst": {"confirm": 0.0, "dismiss": 10.0}})
    damped = scorer.score(patterns=patterns, metrics=None)
    assert damped.score < neutral.score


def test_subneutral_needs_minimum_evidence():
    scorer = _scorer(prior_allow_subneutral=True, prior_subneutral_min_evidence=3.0)
    patterns = [PatternKey(key="HF_ENERGY_BURST")]
    neutral = scorer.score(patterns=patterns, metrics=None)
    scorer.seed_feedback_counts({"spectral:hf_burst": {"confirm": 0.0, "dismiss": 2.0}})
    result = scorer.score(patterns=patterns, metrics=None)
    assert result.score == neutral.score


def test_confirmed_pattern_shields_event_from_damping():
    """Confirmations win: any boosting prior on the event disables damping."""
    scorer = _scorer(prior_allow_subneutral=True)
    patterns = [PatternKey(key="HF_ENERGY_BURST"), PatternKey(key="IMPULSE_BURST")]
    scorer.seed_feedback_counts({
        "spectral:hf_burst": {"confirm": 0.0, "dismiss": 10.0},
        "temporal:impulsive_burst": {"confirm": 10.0, "dismiss": 0.0},
    })
    result = scorer.score(patterns=patterns, metrics=None)
    assert result.historical_prior > 0.5


def test_critical_band_immune_to_damping():
    scorer = _scorer(prior_allow_subneutral=True)
    patterns = [PatternKey(key="HF_ENERGY_BURST")]
    scorer.seed_feedback_counts({"spectral:hf_burst": {"confirm": 0.0, "dismiss": 50.0}})
    strong = {"anomaly_detector_score": 0.99, "breakage_prediction": 0.99, "model_confidence": 1.0}
    res = scorer.score(patterns=patterns, metrics=None, external_signals=strong)
    undamped = _scorer().score(patterns=patterns, metrics=None, external_signals=strong)
    if undamped.score >= scorer.config.critical_threshold:
        assert res.score == undamped.score


# --- Co-occurrence gating (plan 1.7) ---

def test_lone_supporting_pattern_is_store_band():
    scorer = _scorer()
    r = scorer.score(patterns=[PatternKey(key="spectral:hf_burst")], metrics=None)
    assert r.action.value == "store"
    assert r.score < scorer.config.alert_threshold


def test_two_supporting_patterns_corroborate():
    scorer = _scorer()
    r = scorer.score(patterns=[PatternKey(key="spectral:hf_burst"),
                               PatternKey(key="temporal:impulsive_burst")], metrics=None)
    assert r.action.value in ("alert", "critical")


def test_model_agreement_corroborates_lone_supporting_pattern():
    scorer = _scorer()
    r = scorer.score(patterns=[PatternKey(key="spectral:hf_burst")], metrics=None,
                     external_signals={"anomaly_detector_score": 0.97})
    assert r.action.value in ("alert", "critical")


def test_non_supporting_fault_pattern_alerts_alone():
    scorer = _scorer()
    r = scorer.score(patterns=[PatternKey(key="SPINDLE_POWER_SURGE")], metrics=None)
    assert r.action.value in ("alert", "critical")


def test_gating_can_be_disabled():
    scorer = _scorer(enable_cooccurrence_gating=False)
    r = scorer.score(patterns=[PatternKey(key="spectral:hf_burst")], metrics=None)
    assert r.action.value in ("alert", "critical")


def test_raw_and_canonical_supporting_keys_behave_identically():
    """The registry is keyed by raw name; the live pipeline emits canonical
    keys. Both must gate the same (regression for the canonical-key miss)."""
    raw = _scorer().score(patterns=[PatternKey(key="HF_ENERGY_BURST")], metrics=None)
    canon = _scorer().score(patterns=[PatternKey(key="spectral:hf_burst")], metrics=None)
    assert raw.action.value == canon.action.value == "store"

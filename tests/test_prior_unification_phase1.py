"""Phase-1 prior-unification regression tests.

Covers the consolidated-backlog Phase-1 items:
  P1  one shared estimator (scorer + experiment call prior_math)
  P2  inject ACTUAL counts (preserve evidence volume; no notional_total=20)
  P3  per-context cache holds the hierarchical scoring value
  P4  prior_history is the scorer-faithful value (parity)
  X6  splitter downsample actually reduces partition size
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from backend.agents.memory import prior_math as pm
from backend.agents.memory.scorer import SignificanceScorer, SignificanceConfig
from backend.agents.core.context import CuttingContext


# --------------------------------------------------------------------------
# prior_math — the single source of truth
# --------------------------------------------------------------------------

def test_effective_count_saturates_and_is_bounded():
    # Bounded by 1/(1-decay) ≈ 6.667 at decay=0.85.
    assert pm.effective_feedback_count(0) == 0.0
    assert pm.effective_feedback_count(-5) == 0.0
    assert pm.effective_feedback_count(1) == pytest.approx(1.0)
    big = pm.effective_feedback_count(10_000)
    ceiling = 1.0 / (1.0 - pm.DEFAULT_RECENCY_DECAY)
    assert big <= ceiling  # converges to the ceiling, never exceeds it
    assert big == pytest.approx(ceiling, rel=1e-6)


def test_effective_count_no_saturation_when_decay_ge_one():
    assert pm.effective_feedback_count(42, recency_decay=1.0) == 42.0


def test_prior_from_counts_matches_documented_numbers():
    prior, total = pm.prior_from_counts(16, 4)
    assert prior == pytest.approx(0.6314, abs=1e-4)
    assert total == 20.0  # raw (un-saturated) volume is preserved


def test_prior_from_counts_neutral_with_no_feedback():
    prior, total = pm.prior_from_counts(0, 0)
    assert prior == pytest.approx(0.5)
    assert total == 0.0


def test_prior_confidence_ceiling():
    # Even with overwhelming confirmation the prior cannot exceed the ceiling.
    prior, _ = pm.prior_from_counts(1_000_000, 0)
    ceiling = (1.0 / (1.0 - pm.DEFAULT_RECENCY_DECAY) + 1) / (
        1.0 / (1.0 - pm.DEFAULT_RECENCY_DECAY) + 2
    )
    assert prior <= ceiling + 1e-9
    assert prior == pytest.approx(0.8846, abs=1e-3)


# --------------------------------------------------------------------------
# P1 / P4 — scorer uses the shared estimator; reported prior == derived prior
# --------------------------------------------------------------------------

def _scorer(tmp_path):
    return SignificanceScorer(
        config=SignificanceConfig(),
        priors_path=str(tmp_path / "priors.json"),
    )


@pytest.mark.parametrize("c,d", [(0, 0), (1, 0), (16, 4), (3, 7), (100, 0), (50, 50)])
def test_scorer_prior_equals_shared_estimator(tmp_path, c, d):
    """The live scorer derives its prior via prior_math — so the experiment,
    which now also calls prior_math on the same counts, gets the identical
    value (P1). This is also what makes prior_history scorer-faithful (P4)."""
    s = _scorer(tmp_path)
    pk = "SPINDLE_POWER_SURGE"
    s._local_feedback_counts[pk] = {"confirm": float(c), "dismiss": float(d)}
    expected, _ = pm.prior_from_counts(c, d)
    assert s.get_pattern_prior(pk) == pytest.approx(expected)


# --------------------------------------------------------------------------
# P2 — inject ACTUAL counts; evidence volume is preserved (no notional 20)
# --------------------------------------------------------------------------

def test_unified_scorer_injection_preserves_true_volume(tmp_path):
    from backend.agents.experiment.evaluator import _score_via_unified_scorer

    s = _scorer(tmp_path)
    pk = "SPINDLE_POWER_SURGE"
    _score_via_unified_scorer(
        s, [pk], raw_model_score=0.9, anomaly_z=0.0,
        feedback_counts={pk: {"confirm": 100.0, "dismiss": 0.0}},
    )
    # True volume kept — NOT collapsed to the old notional_total=20.
    assert s._local_feedback_counts[pk] == {"confirm": 100.0, "dismiss": 0.0}


def test_unified_scorer_value_fallback_when_no_counts(tmp_path):
    from backend.agents.experiment.evaluator import _score_via_unified_scorer

    s = _scorer(tmp_path)
    pk = "SPINDLE_POWER_SURGE"
    _score_via_unified_scorer(
        s, [pk], raw_model_score=0.9, anomaly_z=0.0,
        pattern_priors={pk: 0.8},  # no counts available → approximate path
    )
    counts = s._local_feedback_counts[pk]
    assert counts["confirm"] + counts["dismiss"] == pytest.approx(20.0)
    assert counts["confirm"] > counts["dismiss"]  # 0.8 prior → mostly confirms


def test_unified_scorer_preserves_seeded_priors_when_feedback_exists(tmp_path):
    """Regression (review finding #1): a pattern with a *seeded* prior but no
    feedback counts must keep its prior once feedback exists for *other*
    patterns. The old if/elif dropped it to neutral 0.5 the moment
    feedback_counts became non-empty."""
    from backend.agents.experiment.evaluator import _score_via_unified_scorer

    s = _scorer(tmp_path)
    fed, seeded = "SPINDLE_POWER_SURGE", "VIBRATION_REGIME_SHIFT"
    _score_via_unified_scorer(
        s, [fed, seeded], raw_model_score=0.9, anomaly_z=0.0,
        pattern_priors={fed: 0.6, seeded: 0.8},          # seeded value for both
        feedback_counts={fed: {"confirm": 5.0, "dismiss": 0.0}},  # real counts for `fed` only
    )
    # `seeded` keeps a non-neutral prior derived from its seeded value...
    assert s.get_pattern_prior(seeded) == pytest.approx(pm.prior_from_counts(16, 4)[0])
    assert s.get_pattern_prior(seeded) > 0.5
    # ...and `fed` uses its ACTUAL counts (5,0), not the 0.6 seed.
    assert s.get_pattern_prior(fed) == pytest.approx(pm.prior_from_counts(5, 0)[0])


def test_seed_feedback_counts_helper(tmp_path):
    s = _scorer(tmp_path)
    pk = "SPINDLE_POWER_SURGE"
    s.seed_feedback_counts({pk: {"confirm": 8.0, "dismiss": 2.0}})
    assert s._local_feedback_counts[pk] == {"confirm": 8.0, "dismiss": 2.0}
    assert s.get_pattern_prior(pk) == pytest.approx(pm.prior_from_counts(8, 2)[0])


def test_context_seeding_preserves_aggregate_prior(tmp_path):
    """De-blinding regression: with a context passed, the prior lookup tries
    context-scoped keys ONLY (no global fallback). Seeding under the
    least-specific (shared machine) key must therefore reproduce the exact
    aggregate prior the global path gives — otherwise context-awareness would
    silently zero out the feedback signal."""
    from backend.agents.experiment.evaluator import _build_cutting_context

    s = _scorer(tmp_path)
    pk = "SPINDLE_POWER_SURGE"
    ctx_a = _build_cutting_context("OF00003", "2462.0")
    # least-specific key is shared (machine_type), per-tool key is most-specific
    assert s._candidate_context_keys(ctx_a) == [
        "machine_type=CNC_5axis|tool_type=T2462",
        "machine_type=CNC_5axis",
    ]
    s.seed_feedback_counts({pk: {"confirm": 16.0, "dismiss": 4.0}}, context=ctx_a)
    expected = pm.prior_from_counts(16, 4)[0]
    # same tool, the seeding tool, and a DIFFERENT tool on the same machine all
    # resolve to the shared machine-level aggregate (no fragmentation, no 0.5).
    assert s.get_pattern_prior(pk, context=ctx_a) == pytest.approx(expected)
    assert s.get_pattern_prior(pk, context=None) == pytest.approx(expected)
    ctx_b = _build_cutting_context("OF00004", "9999")
    assert s.get_pattern_prior(pk, context=ctx_b) == pytest.approx(expected)


def test_counterfactual_scorer_isolated_from_primary(tmp_path):
    """A1: the counterfactual must differ from the primary path by FEEDBACK
    alone. A separate frozen scorer seeded with the initial counts gives the
    same score as the primary when both hold the initial counts, and stays put
    when the primary's counts grow — so the delta is feedback, not a different
    scoring formula."""
    from backend.agents.experiment.evaluator import _score_via_unified_scorer

    primary = _scorer(tmp_path / "p")
    cf = _scorer(tmp_path / "cf")
    pk = "SPINDLE_POWER_SURGE"
    init = {pk: {"confirm": 2.0, "dismiss": 2.0}}

    # Both at initial counts → identical score.
    r0_primary = _score_via_unified_scorer(primary, [pk], 0.65, 0.0, feedback_counts=init)
    r0_cf = _score_via_unified_scorer(cf, [pk], 0.65, 0.0, feedback_counts=init)
    assert r0_primary.score == pytest.approx(r0_cf.score)

    # Feedback accumulates on the primary; the counterfactual stays frozen.
    grown = {pk: {"confirm": 40.0, "dismiss": 2.0}}
    r1_primary = _score_via_unified_scorer(primary, [pk], 0.65, 0.0, feedback_counts=grown)
    r1_cf = _score_via_unified_scorer(cf, [pk], 0.65, 0.0, feedback_counts=init)  # frozen
    assert r1_cf.score == pytest.approx(r0_cf.score)           # cf unchanged
    assert r1_primary.score >= r1_cf.score                      # feedback raised primary


def test_leaky_feature_guard_rejects_severity():
    """A3: the forbidden-column guard must reject label-derived columns."""
    from backend.agents.experiment.config import (
        assert_features_safe,
        FORBIDDEN_FEATURE_COLUMNS,
    )

    assert "severity" in FORBIDDEN_FEATURE_COLUMNS
    assert "stop_type" in FORBIDDEN_FEATURE_COLUMNS
    assert_features_safe(["power_spindle_mean", "vib_severity_x_mean"])  # ok
    with pytest.raises(ValueError):
        assert_features_safe(["power_spindle_mean", "severity"])


def test_faithful_pipeline_disables_experiment_only_layers():
    """A2: the faithful SUT disables supervised blend, tool prior, threshold
    adaptation; the ablation flag re-enables them."""
    from backend.agents.experiment.config import ExperimentConfig

    f = ExperimentConfig()
    assert f.faithful_pipeline is True
    assert f.use_supervised_model is False
    assert f.use_tool_priors is False
    assert f.weight_supervised == 0.0 and f.weight_unsupervised == 1.0
    assert f.threshold_adaptation_rate == 0.0

    a = ExperimentConfig(faithful_pipeline=False, use_supervised_model=True, use_tool_priors=True)
    assert a.use_supervised_model is True and a.use_tool_priors is True


def test_context_prior_pooling(tmp_path):
    """D1: partial pooling. Flag OFF reproduces first-match; flag ON shrinks a
    sparse leaf context toward its parent, while an abundant leaf dominates."""
    from backend.agents.memory.scorer import SignificanceScorer, SignificanceConfig
    from backend.agents.core.context import CuttingContext

    ctx = CuttingContext(machine_type="cnc", tool_type="T1")

    def _mk(pooling, kappa=8.0):
        s = SignificanceScorer(
            config=SignificanceConfig(context_prior_pooling=pooling,
                                      context_prior_pooling_kappa=kappa),
            priors_path=str(tmp_path / f"p_{pooling}_{kappa}.json"),
        )
        keys = s._candidate_context_keys(ctx)          # [leaf, parent]
        leaf, parent = keys[0], keys[-1]
        return s, leaf, parent

    pk = "SPINDLE_POWER_SURGE"

    # OFF: first-match -> uses the most-specific level that has data (the leaf).
    s_off, leaf, parent = _mk(False)
    s_off._context_feedback_counts[parent] = {pk: {"confirm": 30.0, "dismiss": 0.0}}
    s_off._context_feedback_counts[leaf] = {pk: {"confirm": 0.0, "dismiss": 2.0}}
    p_off = s_off.get_pattern_prior(pk, context=ctx)
    assert p_off < 0.3  # leaf (dismiss-heavy) wins under first-match

    # ON, sparse leaf: pooled estimate is pulled UP toward the confirm-heavy parent.
    s_on, leaf, parent = _mk(True)
    s_on._context_feedback_counts[parent] = {pk: {"confirm": 30.0, "dismiss": 0.0}}
    s_on._context_feedback_counts[leaf] = {pk: {"confirm": 0.0, "dismiss": 2.0}}
    p_on = s_on.get_pattern_prior(pk, context=ctx)
    assert p_on > p_off          # shrinkage toward parent raised it
    assert 0.4 < p_on < 0.8      # between local (~0.12) and parent (~0.8)

    # ON, abundant leaf: local evidence dominates, estimate returns near local.
    s_ab, leaf, parent = _mk(True)
    s_ab._context_feedback_counts[parent] = {pk: {"confirm": 30.0, "dismiss": 0.0}}
    s_ab._context_feedback_counts[leaf] = {pk: {"confirm": 0.0, "dismiss": 60.0}}
    p_ab = s_ab.get_pattern_prior(pk, context=ctx)
    assert p_ab < p_on           # more leaf evidence -> closer to local (low)


def test_phase_result_surfaces_scorer_fallbacks():
    """X7: the fallback count must be observable in the serialised phase."""
    from backend.agents.experiment.evaluator import PhaseResult

    pr = PhaseResult(phase="test", n_samples=10, n_scorer_fallbacks=3)
    assert pr.to_dict()["n_scorer_fallbacks"] == 3


# --------------------------------------------------------------------------
# P3 — per-context cache stores the hierarchical scoring value
# --------------------------------------------------------------------------

def test_context_cache_matches_scoring_value(tmp_path):
    s = _scorer(tmp_path)
    pk = "SPINDLE_POWER_SURGE"
    ctx = CuttingContext(machine_type="cnc_mill", tool_type="end_mill")

    for _ in range(3):
        s.update_pattern_prior(pk, was_significant=True, context=ctx)

    ctx_key = next(k for k in s._candidate_context_keys(ctx) if k)
    cached = s._context_pattern_priors[ctx_key][pk]
    # The cache must equal what scoring actually uses under this context.
    assert cached == pytest.approx(s.get_pattern_prior(pk, context=ctx))
    # And the global cache stays global (>0.5 after 3 confirms, derived globally).
    assert s._pattern_priors[pk] == pytest.approx(s.get_pattern_prior(pk, context=None))


def test_update_pattern_prior_moves_in_right_direction(tmp_path):
    s = _scorer(tmp_path)
    pk = "SPINDLE_POWER_SURGE"
    base = s.get_pattern_prior(pk)
    s.update_pattern_prior(pk, was_significant=True)
    after_confirm = s.get_pattern_prior(pk)
    s.update_pattern_prior(pk, was_significant=False)
    s.update_pattern_prior(pk, was_significant=False)
    after_dismiss = s.get_pattern_prior(pk)
    assert after_confirm > base
    assert after_dismiss < after_confirm


# --------------------------------------------------------------------------
# X6 — splitter downsample is no longer a silent no-op
# --------------------------------------------------------------------------

def test_downsample_actually_reduces_partitions(tmp_path):
    from backend.agents.experiment.config import ExperimentConfig
    from backend.agents.experiment.splitter import create_split

    rows = []
    for op in ("op_a", "op_b", "op_t", "op_e"):
        for i in range(40):
            rows.append({
                "operation_id": op,
                "label": "normal" if i % 2 else "pre_stoppage",
                "event_timestamp": i,
                "feat": float(i),
            })
    csv = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    cfg = ExperimentConfig(
        features_csv=csv,
        train_ops=["op_a", "op_b"],
        test_op="op_t",
        eval_op="op_e",
        downsample_max=10,
        output_dir=tmp_path / "out",
    )
    split = create_split(cfg)
    # Each partition had 40 (train 80); downsample_max=10 must cap them.
    assert len(split.test_df) == 10
    assert len(split.eval_df) == 10
    assert len(split.train_df) == 10

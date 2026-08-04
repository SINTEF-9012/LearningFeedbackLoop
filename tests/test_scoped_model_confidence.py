"""Context-scoped model trust (plan 1.1).

The point: sustained false alarms in one context must quiet that context's model
contribution *without* quieting a different context that catches real breaks.
Plus backward compatibility with the flat (pre-scoping) file format.
"""
from __future__ import annotations

import json

from backend.agents.model_confidence import (
    GLOBAL_SCOPE,
    ScopedModelConfidence,
    current_model_confidence,
    load_model_confidence_state,
    record_model_feedback_outcome,
)


def _confirms(store, key, n):
    for _ in range(n):
        store.record(model_fired=True, was_confirmed=True, context_key=key)


def _false_alarms(store, key, n):
    for _ in range(n):
        store.record(model_fired=True, was_confirmed=False, context_key=key)


def test_no_context_matches_global():
    store = ScopedModelConfidence()
    _false_alarms(store, None, 10)  # None -> global only
    assert store.confidence(None) == store.global_state.feedback_confidence()


def test_suppression_is_selective_across_contexts():
    store = ScopedModelConfidence()
    _false_alarms(store, "regime=finishing", 15)  # FP-prone context
    _confirms(store, "regime=roughing", 15)        # break-catching context

    fp_conf = store.confidence("regime=finishing")
    tp_conf = store.confidence("regime=roughing")
    assert fp_conf < 0.4          # quieted
    assert tp_conf > 0.6          # kept loud
    assert tp_conf - fp_conf > 0.3  # genuinely selective, not uniform


def test_unseen_context_falls_back_to_global():
    store = ScopedModelConfidence()
    _false_alarms(store, "regime=finishing", 12)
    # A context with no evidence of its own uses the global aggregate.
    assert store.confidence("regime=brand_new") == store.confidence(GLOBAL_SCOPE)


def test_thin_context_evidence_shrinks_toward_global():
    store = ScopedModelConfidence()
    _confirms(store, GLOBAL_SCOPE, 30)          # strong positive global
    _false_alarms(store, "regime=x", 1)          # one FP in context x
    glob = store.confidence(GLOBAL_SCOPE)
    ctx = store.confidence("regime=x")
    ctx_only = store.scopes["regime=x"].feedback_confidence()
    # One event should not swing the context all the way to its raw value.
    assert ctx_only < ctx < glob


def test_backward_compatible_flat_file(tmp_path):
    path = tmp_path / "model_confidence.json"
    # Old-format flat write via the legacy path…
    record_model_feedback_outcome(model_fired=True, was_confirmed=False, path=path)
    # …loads as the global scope and reads back through both APIs.
    flat = load_model_confidence_state(path)
    assert flat.false_positives == 1
    assert current_model_confidence(path) == current_model_confidence(path, context_key=None)


def test_persisted_file_has_scopes_and_flat_mirror(tmp_path):
    path = tmp_path / "model_confidence.json"
    record_model_feedback_outcome(model_fired=True, was_confirmed=True,
                                  path=path, context_key="regime=roughing")
    payload = json.loads(path.read_text())
    assert "scopes" in payload
    assert GLOBAL_SCOPE in payload["scopes"]
    assert "regime=roughing" in payload["scopes"]
    # Flat mirror at top level for pre-scoping readers.
    assert payload["true_positives"] == 1


def test_scorer_read_write_selective_suppression(tmp_path):
    """End-to-end through the scorer: dismissing alerts in one regime lowers the
    next alert's model contribution in THAT regime only, via record_model_feedback
    (write) + score() (context-scoped read)."""
    from backend.agents.memory.scorer import SignificanceConfig, SignificanceScorer
    from backend.agents.core.schemas import PatternKey
    from backend.agents.core.context import CuttingContext, OperatingRegime

    cfg = SignificanceConfig()
    scorer = SignificanceScorer(cfg, model_confidence_path=str(tmp_path / "mc.json"))

    finishing = CuttingContext(operating_regime=OperatingRegime.FINISHING)
    roughing = CuttingContext(operating_regime=OperatingRegime.ROUGHING)
    signals = {"anomaly_detector_score": 0.95}  # model fires

    # Dismiss a run of model-fired alerts in finishing; confirm them in roughing.
    # (Both update the global aggregate, so selectivity must come from the scopes.)
    for _ in range(15):
        scorer.record_model_feedback(
            triggered_rules=["classical_alert"], was_confirmed=False,
            external_signals=signals, cutting_context=finishing,
        )
        scorer.record_model_feedback(
            triggered_rules=["classical_alert"], was_confirmed=True,
            external_signals=signals, cutting_context=roughing,
        )

    fin = scorer.score(patterns=[PatternKey(key="HF_ENERGY_BURST")], metrics=None,
                       context=finishing, external_signals=dict(signals))
    rou = scorer.score(patterns=[PatternKey(key="HF_ENERGY_BURST")], metrics=None,
                       context=roughing, external_signals=dict(signals))
    # Finishing (many dismissals) scores lower than roughing (untouched).
    assert fin.score < rou.score


def test_diagnostics_surface_scopes(tmp_path):
    """get_model_confidence_diagnostics exposes per-context scopes (plan 1.6),
    least-trusted first, plus the flat global view for back-compat."""
    from backend.agents.model_confidence import get_model_confidence_diagnostics

    path = tmp_path / "mc.json"
    for _ in range(10):
        record_model_feedback_outcome(model_fired=True, was_confirmed=False,
                                      path=path, context_key="regime=finishing")
    for _ in range(10):
        record_model_feedback_outcome(model_fired=True, was_confirmed=True,
                                      path=path, context_key="regime=roughing")

    diag = get_model_confidence_diagnostics(path)
    assert diag["scope_count"] == 2
    # Least-trusted (finishing, all dismissed) first.
    assert diag["scopes"][0]["context"] == "regime=finishing"
    assert diag["scopes"][0]["model_confidence"] < diag["scopes"][1]["model_confidence"]
    # Flat global view still present.
    assert "model_confidence" in diag and "smoothed_precision" in diag

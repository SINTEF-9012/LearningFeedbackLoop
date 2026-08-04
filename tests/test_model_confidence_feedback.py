from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.model_confidence import (
    current_model_confidence,
    fingerprint_model_artifact,
    load_model_confidence_state,
    record_model_feedback_outcome,
    reset_model_confidence_state,
)
from backend.agents.memory.scorer import SignificanceScorer
from backend.agents.memory.retrainer import ModelRetrainer
from backend.agents.processing.classical_models import OnlineAnomalyDetector
from backend.inference_streamer import _model_confidence

import numpy as np


class _DummyWindowModel:
    def __init__(self, *, score: float = 0.83):
        self.is_trained = True
        self._training_stats = {"n_samples": 1}
        self._score = score

    def score_dict(self, _feature_dict):
        return self._score


class _DummyInferenceModel:
    def __init__(self, *, trained: bool = True, n_training_samples: int = 48):
        self.is_trained = trained
        self.n_training_samples = n_training_samples


def test_feedback_model_confidence_defaults_to_neutral_without_feedback(tmp_path):
    path = tmp_path / "model_confidence.json"

    assert current_model_confidence(path) == pytest.approx(0.5, rel=1e-6)
    assert _model_confidence(_DummyInferenceModel(), path) == pytest.approx(0.5, rel=1e-6)


def test_significance_scorer_records_separate_feedback_state_per_model_signal(tmp_path):
    scorer = SignificanceScorer(priors_path=str(tmp_path / "pattern_priors.json"))

    scorer.record_model_feedback(
        triggered_rules=["classical_alert"],
        was_confirmed=True,
        external_signals={
            "anomaly_detector_score": 0.91,
            "harmonic_context_score": 0.76,
            "breakage_prediction": 0.64,
        },
    )
    scorer.record_model_feedback(
        triggered_rules=[],
        was_confirmed=True,
        external_signals={
            "anomaly_detector_score": 0.42,
            "harmonic_context_score": 0.35,
            "breakage_prediction": 0.18,
        },
    )

    classical_state = load_model_confidence_state(scorer._model_confidence_paths["anomaly_detector_score"])
    harmonic_state = load_model_confidence_state(scorer._model_confidence_paths["harmonic_context_score"])
    breakage_state = load_model_confidence_state(scorer._model_confidence_paths["breakage_prediction"])

    assert classical_state.true_positives == 1
    assert classical_state.false_negatives == 1
    assert harmonic_state.true_positives == 1
    assert harmonic_state.false_negatives == 1
    assert breakage_state.true_positives == 1
    assert breakage_state.false_negatives == 1


def test_feedback_model_confidence_increases_with_confirmed_model_alerts(tmp_path):
    path = tmp_path / "model_confidence.json"

    for _ in range(12):
        record_model_feedback_outcome(model_fired=True, was_confirmed=True, path=path)

    state = load_model_confidence_state(path)
    confidence = current_model_confidence(path)

    assert state.true_positives == 12
    assert state.false_positives == 0
    assert confidence > 0.5


def test_feedback_model_confidence_drops_after_false_positives_and_misses(tmp_path):
    path = tmp_path / "model_confidence.json"

    for _ in range(8):
        record_model_feedback_outcome(model_fired=True, was_confirmed=True, path=path)
    improved = current_model_confidence(path)

    for _ in range(10):
        record_model_feedback_outcome(model_fired=True, was_confirmed=False, path=path)
    for _ in range(4):
        record_model_feedback_outcome(model_fired=False, was_confirmed=True, path=path)

    degraded = current_model_confidence(path)
    state = load_model_confidence_state(path)

    assert state.false_positives == 10
    assert state.false_negatives == 4
    assert degraded < improved
    assert degraded < 0.5


def test_online_detector_uses_feedback_driven_model_confidence(tmp_path):
    path = tmp_path / "model_confidence.json"
    for _ in range(10):
        record_model_feedback_outcome(model_fired=True, was_confirmed=True, path=path)

    detector = OnlineAnomalyDetector(
        seed_model=_DummyWindowModel(score=0.91),
        model_confidence_path=path,
    )
    signals = detector.score_window({"power_spindle_mean": 5.0})

    assert signals["anomaly_detector_score"] == pytest.approx(0.91, rel=1e-6)
    assert signals["model_confidence"] == pytest.approx(current_model_confidence(path), rel=1e-6)


def test_reset_model_confidence_state_returns_to_neutral_and_tracks_fingerprint(tmp_path):
    path = tmp_path / "model_confidence.json"
    model_path = tmp_path / "seed_model.pkl"
    model_path.write_bytes(b"dummy-model-v2")

    for _ in range(6):
        record_model_feedback_outcome(model_fired=True, was_confirmed=True, path=path)

    reset_model_confidence_state(
        path=path,
        model_fingerprint=fingerprint_model_artifact(model_path),
        reason="model_retrained",
    )
    state = load_model_confidence_state(path)

    assert current_model_confidence(path) == pytest.approx(0.5, rel=1e-6)
    assert state.feedback_count == 0
    assert state.evidence_count == 0
    assert state.last_reset_reason == "model_retrained"
    assert isinstance(state.model_fingerprint, str)
    assert state.model_fingerprint.startswith("sha256:")


def test_retrainer_resets_model_confidence_after_successful_save(tmp_path, monkeypatch):
    path = tmp_path / "model_confidence.json"
    model_path = tmp_path / "seed_model.pkl"

    for _ in range(8):
        record_model_feedback_outcome(model_fired=True, was_confirmed=True, path=path)

    import backend.agents.processing.classical_models as classical_models

    class _DummySeedModelConfig:
        pass

    class _DummySeedModel:
        def __init__(self, config=None):
            self.config = config

        def load(self, _path):
            return None

        def score(self, features):
            return 0.9 if float(np.mean(features)) > 0.5 else 0.1

        def train(self, _features):
            return None

        def save(self, path_to_save):
            path_to_save = path_to_save if hasattr(path_to_save, "write_bytes") else Path(path_to_save)
            path_to_save.parent.mkdir(parents=True, exist_ok=True)
            path_to_save.write_bytes(b"dummy-trained-model")

    monkeypatch.setattr(classical_models, "SeedModel", _DummySeedModel)
    monkeypatch.setattr(classical_models, "SeedModelConfig", _DummySeedModelConfig)

    retrainer = ModelRetrainer(
        model_path=model_path,
        model_confidence_path=path,
        min_positive_samples=1,
        min_negative_samples=1,
    )
    retrainer.record_feedback(np.ones(4), is_significant=True, memory_id="m1")
    retrainer.record_feedback(np.zeros(4), is_significant=False, memory_id="m2")

    result = retrainer.retrain()
    state = load_model_confidence_state(path)

    assert result.success is True
    assert model_path.exists()
    assert current_model_confidence(path) == pytest.approx(0.5, rel=1e-6)
    assert state.feedback_count == 0
    assert state.last_reset_reason == "model_retrained"
    assert isinstance(state.model_fingerprint, str)
    assert state.model_fingerprint.startswith("sha256:")
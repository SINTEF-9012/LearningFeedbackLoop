"""Tests for lazy seed-model training (classical_models.create_seed_model)."""

from __future__ import annotations

import time

import pytest

from backend.agents.processing.classical_models import (
    SeedModel,
    create_seed_model,
    create_online_detector,
)


def test_lazy_returns_untrained_immediately(tmp_path):
    """When no cached model exists and lazy_training=True, function returns
    fast with an untrained model."""
    start = time.perf_counter()
    model = create_seed_model(
        casedata_path="data/casedata",
        model_path=str(tmp_path / "seed.pkl"),
        lazy_training=True,
    )
    elapsed = time.perf_counter() - start
    assert isinstance(model, SeedModel)
    # Must return in well under the synchronous-train budget. On a slow
    # system a few hundred ms is tolerable for thread spin-up; 5 s is a
    # generous guard-rail.
    assert elapsed < 5.0, f"lazy return took {elapsed:.2f}s"
    assert hasattr(model, "_lazy_training_thread")


def test_lazy_online_detector_skips_seed_score_when_untrained():
    """OnlineAnomalyDetector.score_window must not crash when the seed
    model is still untrained, and must omit anomaly_detector_score."""
    model = SeedModel()  # freshly constructed, is_trained=False
    detector = create_online_detector(seed_model=model)

    signals = detector.score_window({"rms": 0.5, "peak": 1.0})

    assert "anomaly_detector_score" not in signals
    assert "model_source" not in signals


def test_lazy_skipped_when_cached_model_exists(tmp_path):
    """If the model file exists, lazy_training is ignored and the
    cached model is loaded synchronously (no thread spawned)."""
    # Train minimally and save so we have a cached model to load.
    model = SeedModel()
    # Feed a small synthetic feature set so training completes in ms.
    import numpy as np
    rng = np.random.default_rng(0)
    features = rng.normal(size=(120, 17))
    model.train(features)
    cache_path = tmp_path / "seed.pkl"
    model.save(cache_path)

    loaded = create_seed_model(
        casedata_path="data/casedata",
        model_path=str(cache_path),
        lazy_training=True,
    )
    assert loaded.is_trained is True
    assert not hasattr(loaded, "_lazy_training_thread")


def test_sync_path_untouched(monkeypatch, tmp_path):
    """Default ``lazy_training=False`` still calls ``train_from_casedata``
    synchronously. We stub out the loader + train to keep the test fast
    and confirm the code path fires."""
    calls: list[str] = []

    def fake_train(self, loader, operation_ids=None):
        calls.append("trained")
        self._is_trained = True
        self._training_stats = {"n_samples": 1}
        return self._training_stats

    monkeypatch.setattr(SeedModel, "train_from_casedata", fake_train)

    class _FakeLoader:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        "backend.agents.processing.dataset_loader.DatasetLoader",
        _FakeLoader,
    )

    model = create_seed_model(
        casedata_path="data/casedata",
        model_path=str(tmp_path / "seed.pkl"),
        lazy_training=False,
    )
    assert calls == ["trained"]
    assert not hasattr(model, "_lazy_training_thread")

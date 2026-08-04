from __future__ import annotations

import numpy as np
import pytest

from backend.agents.processing.harmonic_config import casedata_stoppage_preset
from backend.agents.processing.harmonic_trainer import TORCH_AVAILABLE, HarmonicContextTrainer


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")


def test_train_from_sequence_samples_uses_configured_context_sources(monkeypatch, tmp_path):
    config = casedata_stoppage_preset(model_save_path=str(tmp_path / "harmonic.pt"))
    trainer = HarmonicContextTrainer(config)
    captured = {}

    def fake_train_loop(model, train_loader, val_loader):
        captured["train_windows"] = len(train_loader.dataset)
        captured["val_windows"] = len(val_loader.dataset)
        return {
            "epochs": 3,
            "best_val_loss": 0.12,
            "best_val_acc": 0.88,
            "train_losses": [0.4, 0.3, 0.2],
            "val_losses": [0.3, 0.2, 0.12],
        }

    def fake_save_model(model, n_harm, n_params, harm_cols, ctx_stats, result):
        captured["saved"] = {
            "n_harm": n_harm,
            "n_params": n_params,
            "harm_cols": list(harm_cols),
            "ctx_stats": ctx_stats,
        }
        result.model_path = str(tmp_path / "harmonic.pt")

    monkeypatch.setattr(trainer, "_train_loop", fake_train_loop)
    monkeypatch.setattr(trainer, "_save_model", fake_save_model)

    matrices = [
        np.ones((20, 4), dtype=np.float32),
        np.full((20, 4), 2.0, dtype=np.float32),
        np.full((20, 4), 3.0, dtype=np.float32),
        np.full((20, 4), 4.0, dtype=np.float32),
    ]
    params = [
        np.array([1200.0, 250.0], dtype=np.float32),
        np.array([1250.0, 255.0], dtype=np.float32),
        np.array([1300.0, 260.0], dtype=np.float32),
        np.array([1350.0, 265.0], dtype=np.float32),
    ]
    labels = [0, 1, 0, 1]
    groups = ["OF00001", "OF00002", "OF00001", "OF00003"]

    result = trainer.train_from_sequence_samples(
        harmonic_matrices=matrices,
        param_vectors=params,
        labels=labels,
        sample_groups=groups,
        harmonic_columns=["X_H1", "Y_H1", "X_H2", "Y_H2"],
    )

    assert result.success is True
    assert result.n_samples == 4
    assert result.n_positive == 2
    assert result.n_negative == 2
    assert result.model_path == str(tmp_path / "harmonic.pt")
    assert result.context_param_stats["spindle_speed"]["source_column"] == "spindle_speed_mean"
    assert result.context_param_stats["feed_rate"]["source_column"] == "feed_rate_mean"
    assert captured["saved"]["n_harm"] == 4
    assert captured["saved"]["n_params"] == 2
    assert captured["saved"]["harm_cols"] == ["X_H1", "Y_H1", "X_H2", "Y_H2"]
    assert captured["train_windows"] > 0
    assert captured["val_windows"] > 0


def test_train_from_sequence_samples_rejects_context_param_mismatch(tmp_path):
    config = casedata_stoppage_preset(model_save_path=str(tmp_path / "harmonic.pt"))
    trainer = HarmonicContextTrainer(config)

    result = trainer.train_from_sequence_samples(
        harmonic_matrices=[np.ones((20, 4), dtype=np.float32)],
        param_vectors=[np.array([1200.0, 250.0, 6.0], dtype=np.float32)],
        labels=[1],
        harmonic_columns=["X_H1", "Y_H1", "X_H2", "Y_H2"],
    )

    assert result.success is False
    assert "Expected 2 context params" in (result.error or "")
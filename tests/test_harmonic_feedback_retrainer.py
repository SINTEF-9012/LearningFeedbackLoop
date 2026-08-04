from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.agents.core.schemas import Memory, PatternKey, PatternType
from backend.agents.memory.feedback import FeedbackAction, MemoryFeedbackHandler, MemoryFeedbackRequest
from backend.agents.memory.scorer import SignificanceScorer
from backend.agents.model_confidence import (
    current_model_confidence,
    load_model_confidence_state,
    record_model_feedback_outcome,
)
from backend.agents.processing.harmonic_feedback_retrainer import HarmonicFeedbackRetrainer
from backend.agents.processing.harmonic_trainer import TrainResult
from backend.agents.storage.store import MemoryStore
from backend.routers import harmonic as harmonic_router


def _context_feedback_kwargs() -> dict:
    return {
        "raw_metrics": {
            "Vibration_Harmonic_1_X_Amplitude": 1.23,
            "spindle_speed_mean": 6000.0,
            "feed_rate_mean": 120.0,
        },
        "harmonic_context": {
            "source": "harmonic_context_v1",
            "feature_labels": ["X·H1"],
            "feature_values": [1.23],
            "context_weights": [0.25],
        },
        "harmonic_runtime": {
            "scorer_kind": "context",
            "dataset": "casedata",
        },
        "cutting_context": None,
        "source": "simulated_casedata",
        "casedata": {"operation_id": "OP-1"},
    }


def _pair_feedback_kwargs(dataset: str = "pair_lfl") -> dict:
    return {
        "raw_metrics": {
            "Vibration_Peak_1_X_Amplitude": 1.23,
            "Vibration_Peak_1_X_Frequency": 240.0,
            "Vibration_Peak_1_Y_Amplitude": 0.91,
            "Vibration_Peak_1_Y_Frequency": 241.5,
            "tool_diameter": 10.0,
            "num_teeth": 4.0,
            "spindle_speed_mean": 6000.0,
            "feed_per_tooth": 0.03,
            "feed_rate_mean": 720.0,
        },
        "harmonic_context": None,
        "harmonic_runtime": {
            "scorer_kind": "pair",
            "dataset": dataset,
        },
        "cutting_context": None,
        "source": "simulated_casedata",
        "casedata": {"operation_id": "OP-PAIR-1"},
    }


def test_harmonic_feedback_retrainer_buckets_context_feedback(tmp_path):
    retrainer = HarmonicFeedbackRetrainer(
        model_confidence_path=str(tmp_path / "model_confidence.json"),
        feedback_threshold=2,
        min_positive_samples=1,
        min_negative_samples=1,
    )

    stored = retrainer.record_feedback(
        was_significant=True,
        memory_id="m-1",
        session_id="session-1",
        **_context_feedback_kwargs(),
    )

    assert stored is True
    status = retrainer.get_status()
    bucket = status["buckets"]["context:casedata"]
    assert status["total_feedback"] == 1
    assert bucket["dataset_name"] == "casedata"
    assert bucket["buffer_size"] == 1
    assert bucket["confirmed_in_buffer"] == 1
    assert bucket["dismissed_in_buffer"] == 0


def test_harmonic_feedback_retrainer_buckets_pair_lfl_feedback(tmp_path):
    retrainer = HarmonicFeedbackRetrainer(
        model_confidence_path=str(tmp_path / "model_confidence.json"),
        feedback_threshold=2,
        min_positive_samples=1,
        min_negative_samples=1,
    )

    stored = retrainer.record_feedback(
        was_significant=True,
        memory_id="m-pair-1",
        session_id="session-1",
        **_pair_feedback_kwargs(),
    )

    assert stored is True
    status = retrainer.get_status(dataset_name="pair_lfl", scorer_kind="pair")
    bucket = status["buckets"]["pair:pair_lfl"]
    assert status["total_feedback"] == 1
    assert bucket["dataset_name"] == "pair_lfl"
    assert bucket["scorer_kind"] == "pair"
    assert bucket["buffer_size"] == 1
    assert bucket["confirmed_in_buffer"] == 1
    assert bucket["dismissed_in_buffer"] == 0


def test_harmonic_feedback_retrainer_reset_feedback_clears_pair_lfl_bucket(tmp_path):
    retrainer = HarmonicFeedbackRetrainer(
        model_confidence_path=str(tmp_path / "model_confidence.json"),
        feedback_threshold=2,
        min_positive_samples=1,
        min_negative_samples=1,
    )
    retrainer.record_feedback(
        was_significant=True,
        memory_id="m-pair-reset",
        session_id="session-1",
        **_pair_feedback_kwargs(),
    )

    removed = retrainer.reset_feedback(dataset_name="pair_lfl", scorer_kind="pair")

    assert removed["bucket_key"] == "pair:pair_lfl"
    assert removed["removed_buffer_size"] == 1
    status = retrainer.get_status(dataset_name="pair_lfl", scorer_kind="pair")
    assert status["buckets"] == {}


def test_harmonic_feedback_retrainer_retrains_context_bucket_and_resets_confidence(tmp_path, monkeypatch):
    retrainer = HarmonicFeedbackRetrainer(
        model_confidence_path=str(tmp_path / "model_confidence.json"),
        feedback_threshold=2,
        min_positive_samples=1,
        min_negative_samples=1,
    )
    retrainer.record_feedback(
        was_significant=True,
        memory_id="m-pos",
        session_id="session-1",
        **_context_feedback_kwargs(),
    )
    retrainer.record_feedback(
        was_significant=False,
        memory_id="m-neg",
        session_id="session-1",
        **_context_feedback_kwargs(),
    )

    harmonic_confidence_path = retrainer.model_confidence_path
    for _ in range(5):
        record_model_feedback_outcome(
            model_fired=True,
            was_confirmed=True,
            path=harmonic_confidence_path,
        )

    import backend.agents.processing.harmonic_trainer as harmonic_trainer_module

    class _DummyHarmonicTrainer:
        def __init__(self, config):
            self.config = config

        def train_from_dataframe(self, df, operation_col="operation_id"):
            assert operation_col == "operation_id"
            assert set(df["label"]) == {"pre_stoppage", "feedback_negative"}
            assert "Vibration_Harmonic_1_X_Amplitude" in df.columns

            save_path = Path(self.config.model_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(b"trained-harmonic-model")

            return TrainResult(
                success=True,
                model_path=str(save_path),
                n_samples=len(df),
                n_positive=int((df["label"] == "pre_stoppage").sum()),
                n_negative=int((df["label"] == "feedback_negative").sum()),
                best_val_loss=0.12,
                best_val_acc=0.87,
            )

    monkeypatch.setattr(harmonic_trainer_module, "HarmonicContextTrainer", _DummyHarmonicTrainer)

    result = retrainer.retrain(
        dataset_name="casedata",
        scorer_kind="context",
        config_overrides={"model_save_path": str(tmp_path / "harmonic_context_casedata.pt")},
    )

    state = load_model_confidence_state(harmonic_confidence_path)
    assert result.success is True
    assert Path(result.model_path).exists()
    assert result.best_val_loss == pytest.approx(0.12)
    assert result.best_val_acc == pytest.approx(0.87)
    assert current_model_confidence(harmonic_confidence_path) == pytest.approx(0.5, rel=1e-6)
    assert state.feedback_count == 0
    assert state.last_reset_reason == "model_retrained"


def test_harmonic_feedback_retrainer_retrains_pair_lfl_bucket(tmp_path, monkeypatch):
    retrainer = HarmonicFeedbackRetrainer(
        model_confidence_path=str(tmp_path / "model_confidence.json"),
        feedback_threshold=2,
        min_positive_samples=1,
        min_negative_samples=1,
    )
    retrainer.record_feedback(
        was_significant=True,
        memory_id="m-pair-pos",
        session_id="session-1",
        **_pair_feedback_kwargs(),
    )
    retrainer.record_feedback(
        was_significant=False,
        memory_id="m-pair-neg",
        session_id="session-1",
        **_pair_feedback_kwargs(),
    )

    import backend.agents.processing.harmonic_pair_trainer as harmonic_pair_trainer_module

    class _DummyPairTrainer:
        def __init__(self, config):
            self.config = config

        def train_from_dataframe(self, df, operation_col="operation_id"):
            assert operation_col == "operation_id"
            assert self.config.dataset_name == "pair_lfl"
            assert any("Vibration_Peak" in pattern for pattern in self.config.pair_frequency_column_patterns)
            assert {"Vibration_Peak_1_X_Amplitude", "Vibration_Peak_1_X_Frequency"}.issubset(df.columns)
            assert set(df["label"]) == {"pre_stoppage", "feedback_negative"}

            save_path = Path(self.config.model_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(b"trained-pair-lfl-feedback-model")

            return TrainResult(
                success=True,
                model_path=str(save_path),
                n_samples=len(df),
                n_positive=int((df["label"] == "pre_stoppage").sum()),
                n_negative=int((df["label"] == "feedback_negative").sum()),
                best_val_loss=0.34,
                best_val_acc=0.81,
            )

    monkeypatch.setattr(harmonic_pair_trainer_module, "HarmonicPairTrainer", _DummyPairTrainer)

    result = retrainer.retrain(
        dataset_name="pair_lfl",
        scorer_kind="pair",
        config_overrides={"model_save_path": str(tmp_path / "harmonic_pair_lfl_feedback.pt")},
    )

    assert result.success is True
    assert result.dataset_name == "pair_lfl"
    assert result.scorer_kind == "pair"
    assert Path(result.model_path).exists()
    assert result.best_val_loss == pytest.approx(0.34)
    assert result.best_val_acc == pytest.approx(0.81)


def test_harmonic_feedback_retrainer_treats_site_a_source_as_casedata(tmp_path):
    retrainer = HarmonicFeedbackRetrainer(
        model_confidence_path=str(tmp_path / "model_confidence.json"),
    )

    dataset_name, scorer_kind = retrainer._resolve_runtime(
        harmonic_runtime=None,
        raw_metrics={},
        source="site_a-live",
        casedata=None,
    )

    assert dataset_name == "casedata"
    assert scorer_kind == "context"


@pytest.mark.asyncio
async def test_feedback_handler_records_harmonic_feedback_for_retraining(tmp_path):
    store = MemoryStore(
        db_path=str(tmp_path / "memories.db"),
        enable_ann=False,
        enable_embeddings=False,
    )
    memory = Memory(
        id="mem-harmonic",
        session_id="session-1",
        time_range=(0.0, 1.0),
        annotation_text="test",
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:test")],
        metadata={
            "raw_metrics": {
                "Vibration_Harmonic_1_X_Amplitude": 1.23,
                "spindle_speed_mean": 6000.0,
                "feed_rate_mean": 120.0,
            },
            "external_signals": {
                "harmonic_context_score": 0.66,
                "harmonic_context_source": "harmonic_context_v1",
            },
            "harmonic_runtime": {
                "scorer_kind": "context",
                "dataset": "casedata",
            },
            "harmonic_context": {
                "source": "harmonic_context_v1",
                "feature_labels": ["X·H1"],
                "feature_values": [1.23],
                "context_weights": [0.25],
            },
            "significance_score": 0.7,
            "significance_action": "alert",
            "triggered_rules": ["harmonic_alert"],
            "source": "simulated_casedata",
            "casedata": {"operation_id": "OP-1"},
        },
    )
    store.create(memory)

    scorer = SignificanceScorer(priors_path=str(tmp_path / "priors.json"))
    handler = MemoryFeedbackHandler(memory_store=store, significance_scorer=scorer)
    handler.retrainer = Mock()
    handler.harmonic_retrainer = Mock()

    resp = await handler.process_feedback(
        "mem-harmonic",
        MemoryFeedbackRequest(action=FeedbackAction.CONFIRM, user_id="tester"),
    )

    assert resp.success is True
    handler.harmonic_retrainer.record_feedback.assert_called_once()
    call_kwargs = handler.harmonic_retrainer.record_feedback.call_args.kwargs
    assert call_kwargs["was_significant"] is True
    assert call_kwargs["harmonic_runtime"] == {"scorer_kind": "context", "dataset": "casedata"}


@pytest.mark.asyncio
async def test_harmonic_retrain_status_route_returns_feedback_buckets(monkeypatch):
    dummy_orchestrator = SimpleNamespace(scorer=SimpleNamespace(_model_confidence_path="/tmp/model_confidence.json"))
    monkeypatch.setattr(
        "backend.agents.memory.orchestrator.get_orchestrator",
        lambda: dummy_orchestrator,
    )

    class _DummyRetrainer:
        def get_status(self, *, dataset_name=None, scorer_kind=None):
            assert dataset_name == "casedata"
            assert scorer_kind == "context"
            return {
                "total_feedback": 2,
                "active_bucket": "context:casedata",
                "buckets": {
                    "context:casedata": {
                        "dataset_name": "casedata",
                        "scorer_kind": "context",
                        "total_feedback": 2,
                        "since_last_retrain": 2,
                        "retrain_threshold": 20,
                        "buffer_size": 2,
                        "confirmed_in_buffer": 1,
                        "dismissed_in_buffer": 1,
                        "should_retrain": False,
                        "retrain_count": 0,
                        "last_retrain": None,
                        "model_save_path": "data/models/harmonic_context_casedata.pt",
                    }
                },
            }

    monkeypatch.setattr(
        "backend.agents.processing.harmonic_feedback_retrainer.get_harmonic_feedback_retrainer",
        lambda model_confidence_path=None: _DummyRetrainer(),
    )

    out = await harmonic_router.harmonic_retrain_status(dataset="casedata", scorer_kind="context")

    assert out.total_feedback == 2
    assert out.active_bucket == "context:casedata"
    assert out.buckets["context:casedata"].buffer_size == 2


@pytest.mark.asyncio
async def test_harmonic_routes_smoke_real_feedback_bucket_retrain(tmp_path, monkeypatch):
    retrainer = HarmonicFeedbackRetrainer(
        model_confidence_path=str(tmp_path / "model_confidence.json"),
        feedback_threshold=2,
        min_positive_samples=1,
        min_negative_samples=1,
    )
    retrainer.record_feedback(
        was_significant=True,
        memory_id="m-pos",
        session_id="session-1",
        **_context_feedback_kwargs(),
    )
    retrainer.record_feedback(
        was_significant=False,
        memory_id="m-neg",
        session_id="session-1",
        **_context_feedback_kwargs(),
    )

    dummy_orchestrator = SimpleNamespace(
        scorer=SimpleNamespace(_model_confidence_path=str(tmp_path / "model_confidence.json")),
    )
    monkeypatch.setattr(
        "backend.agents.memory.orchestrator.get_orchestrator",
        lambda: dummy_orchestrator,
    )
    monkeypatch.setattr(
        "backend.agents.processing.harmonic_feedback_retrainer.get_harmonic_feedback_retrainer",
        lambda model_confidence_path=None: retrainer,
    )

    import backend.agents.processing.harmonic_trainer as harmonic_trainer_module

    class _DummyHarmonicTrainer:
        def __init__(self, config):
            self.config = config

        def train_from_dataframe(self, df, operation_col="operation_id"):
            assert operation_col == "operation_id"
            assert set(df["label"]) == {"pre_stoppage", "feedback_negative"}

            save_path = Path(self.config.model_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(b"smoke-harmonic-model")

            return TrainResult(
                success=True,
                model_path=str(save_path),
                n_samples=len(df),
                n_positive=int((df["label"] == "pre_stoppage").sum()),
                n_negative=int((df["label"] == "feedback_negative").sum()),
                best_val_loss=0.21,
                best_val_acc=0.79,
            )

    monkeypatch.setattr(harmonic_trainer_module, "HarmonicContextTrainer", _DummyHarmonicTrainer)

    status = await harmonic_router.harmonic_retrain_status(dataset="casedata", scorer_kind="context")

    assert status.active_bucket == "context:casedata"
    assert status.buckets["context:casedata"].buffer_size == 2
    assert status.buckets["context:casedata"].should_retrain is True

    out = await harmonic_router.harmonic_retrain(
        harmonic_router.HarmonicRetrainRequest(
            dataset="casedata",
            scorer_kind="context",
            random_seed=17,
            checkpoint_suffix="smoke",
            model_save_path=str(tmp_path / "harmonic_context_casedata.smoke.pt"),
        )
    )

    assert out.success is True
    assert out.bucket_key == "context:casedata"
    assert out.dataset_name == "casedata"
    assert out.scorer_kind == "context"
    assert out.n_samples_used == 2
    assert out.n_confirmed == 1
    assert out.n_dismissed == 1
    assert out.best_val_loss == pytest.approx(0.21)
    assert out.best_val_acc == pytest.approx(0.79)
    assert Path(out.model_path).is_file()
    assert out.model_path.endswith("harmonic_context_casedata.smoke.pt")


@pytest.mark.asyncio
async def test_harmonic_dev_seed_feedback_route_populates_pair_lfl_bucket(tmp_path, monkeypatch):
    retrainer = HarmonicFeedbackRetrainer(
        model_confidence_path=str(tmp_path / "model_confidence.json"),
    )
    dummy_orchestrator = SimpleNamespace(
        scorer=SimpleNamespace(_model_confidence_path=str(tmp_path / "model_confidence.json")),
    )

    monkeypatch.setenv("HARMONIC_ENABLE_DEV_SEED", "1")
    monkeypatch.setattr(
        "backend.agents.memory.orchestrator.get_orchestrator",
        lambda: dummy_orchestrator,
    )
    monkeypatch.setattr(
        "backend.agents.processing.harmonic_feedback_retrainer.get_harmonic_feedback_retrainer",
        lambda model_confidence_path=None: retrainer,
    )

    out = await harmonic_router.harmonic_seed_feedback(
        harmonic_router.HarmonicFeedbackSeedRequest(
            dataset="pair_lfl",
            confirmed=12,
            dismissed=8,
            clear_existing=True,
            session_prefix="test-seed",
            operation_prefix="TEST-PAIR",
        )
    )

    assert out.bucket_key == "pair:pair_lfl"
    assert out.dataset_name == "pair_lfl"
    assert out.scorer_kind == "pair"
    assert out.added_confirmed == 12
    assert out.added_dismissed == 8
    assert out.buffer_size == 20
    assert out.confirmed_in_buffer == 12
    assert out.dismissed_in_buffer == 8
    assert out.should_retrain is True
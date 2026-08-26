import json
from types import SimpleNamespace

import pytest

from backend.agents.core.context import CuttingContext
from backend.agents.memory.orchestrator import MemoryEventOrchestrator, OrchestratorConfig
from backend.agents.memory.experiment_results import _build_run_summary
from backend.agents.memory.experiment_routes import (
    _harden_live_breakage_config_overrides,
)
from backend.agents.memory.router import (
    MissedEventRequest,
    ProcessEventRequest,
    process_event,
    report_missed_event,
)
from backend.agents.memory.scorer import SignificanceAction
from backend.agents.memory.breakage_experiment_runner import (
    LiveBreakageExperimentRunner,
    _build_execution_summary,
)
from backend.agents.experiment.config import ExperimentConfig


def test_scoped_feedback_priors_do_not_mix_user_ids():
    orchestrator = MemoryEventOrchestrator(
        config=OrchestratorConfig(
            generate_explanations=False,
            dispatch_alerts=False,
            use_classical_models=False,
        )
    )

    orchestrator.store.add_feedback_event(
        memory_id=None,
        action="confirm",
        user_id="global-operator",
        pattern_keys=["TEST_PATTERN"],
    )
    orchestrator.store.add_feedback_event(
        memory_id=None,
        action="dismiss",
        user_id="experiment:run-1",
        pattern_keys=["TEST_PATTERN"],
    )

    global_prior = orchestrator.scorer.get_pattern_prior("TEST_PATTERN")
    scoped_prior = orchestrator.scorer.get_pattern_prior(
        "TEST_PATTERN",
        context=CuttingContext(extra={"feedback_scope_user_id": "experiment:run-1"}),
    )

    assert global_prior == pytest.approx(0.5)
    assert scoped_prior < 0.5


@pytest.mark.asyncio
async def test_process_event_derives_patterns_server_side(monkeypatch):
    captured = {}

    class DummyOrchestrator:
        async def process_event(self, event):
            captured["pattern_keys"] = [p.key for p in event.patterns]
            captured["feedback_scope"] = (
                (event.cutting_context.extra or {}).get("feedback_scope_user_id")
                if event.cutting_context is not None else None
            )
            return SimpleNamespace(
                processed=True,
                significant=True,
                memory_id="mem-1",
                significance_score=0.77,
                action=SignificanceAction.ALERT,
                explanation=None,
                explanation_source=None,
                alert_line=None,
                alert_line_source=None,
                similar_memories=[],
                alert_dispatched=False,
                error=None,
                model_breakdown={},
            )

    def fake_detect_patterns(features, thresholds=None, include_details=False):
        assert features["power_spindle_delta_max"] == pytest.approx(99.0)
        return {"fired": ["SERVER_PATTERN"]}

    monkeypatch.setattr("backend.agents.memory.router.get_orchestrator", lambda: DummyOrchestrator())
    monkeypatch.setattr("backend.agents.patterns.registry.detect_patterns", fake_detect_patterns)

    response = await process_event(
        ProcessEventRequest(
            session_id="session-1",
            pattern_keys=["CLIENT_PATTERN"],
            derive_patterns=True,
            metrics={"power_spindle_delta_max": 99.0},
            metadata={"feedback_scope_user_id": "experiment:run-1"},
        )
    )

    assert captured["pattern_keys"] == ["SERVER_PATTERN"]
    assert captured["feedback_scope"] == "experiment:run-1"
    assert response.pattern_keys_used == ["SERVER_PATTERN"]


@pytest.mark.asyncio
async def test_missed_event_uses_server_patterns_and_scope(monkeypatch):
    captured = {}

    class DummyStore:
        def add_feedback_event(self, **kwargs):
            captured.update(kwargs)
            return "fb-1"

    class DummyScorer:
        def __init__(self):
            self.updated = []
            self.config = SimpleNamespace(alert_threshold=0.6)
            self._adaptive_thresholds = None

        def update_pattern_prior(self, pattern_key, was_significant):
            self.updated.append((pattern_key, was_significant))

        def record_feedback_for_adaptive_thresholds(self, score, action, was_confirmed):
            captured["threshold_nudge"] = (score, action, was_confirmed)

    class DummyOrchestrator:
        def __init__(self):
            self.scorer = DummyScorer()
            self.store = DummyStore()

    def fake_detect_patterns(features, thresholds=None, include_details=False):
        assert features["power_spindle_delta_max"] == pytest.approx(42.0)
        return {"fired": ["SERVER_MISSED_PATTERN"]}

    dummy = DummyOrchestrator()
    # report_missed_event asks for the scorer and the store directly now.
    monkeypatch.setattr("backend.agents.memory.router.get_scorer", lambda: dummy.scorer)
    monkeypatch.setattr("backend.agents.memory.router.get_store", lambda: dummy.store)
    monkeypatch.setattr("backend.agents.patterns.registry.detect_patterns", fake_detect_patterns)

    response = await report_missed_event(
        MissedEventRequest(
            session_id="session-1",
            pattern_keys=["CLIENT_PATTERN"],
            derive_patterns=True,
            user_id="experiment:run-1",
            raw_metrics={"power_spindle_delta_max": 42.0},
        )
    )

    assert captured["pattern_keys"] == ["SERVER_MISSED_PATTERN"]
    assert captured["user_id"] == "experiment:run-1"
    assert response["patterns_boosted"] == ["SERVER_MISSED_PATTERN"]
    assert dummy.scorer.updated == []


def test_live_breakage_runner_defaults_to_isolated_api_mode():
    runner = LiveBreakageExperimentRunner(run_id="breakage_demo")

    assert runner.config_overrides["api_mode"] is True
    assert runner.config_overrides["api_mode_strict"] is True
    assert runner.config_overrides["experiment_fast_path"] is False
    assert runner.config_overrides["api_use_server_patterns"] is True
    assert runner.config_overrides["api_batch_size"] == 1
    assert runner.config_overrides["persist_shared_priors"] is False
    assert runner.config_overrides["feedback_user_id"] == "experiment:breakage_demo"


def test_server_pattern_mode_forces_single_event_api_batches():
    config = ExperimentConfig(
        api_mode=True,
        api_use_server_patterns=True,
        api_batch_size=32,
    )

    assert config.api_batch_size == 1


def test_run_summary_exposes_execution_summary(tmp_path):
    run_dir = tmp_path / "breakage_run"
    run_dir.mkdir()
    (run_dir / "experiment_results.json").write_text("{}")

    summary = _build_execution_summary(
        config={
            "api_mode": True,
            "api_mode_strict": True,
            "api_use_server_patterns": True,
            "feedback_user_id": "experiment:run-42",
            "persist_shared_priors": False,
        },
        sandbox_priors=True,
        data_source=SimpleNamespace(kind="split", split_name="PART0001_excel"),
        n_folds=3,
        n_feedback_events=7,
        pattern_keys_used=["SERVER_PATTERN", "SPINDLE_LOAD_RAMP"],
    )

    run_summary = _build_run_summary(
        "breakage_run",
        {
            "experiment_type": "breakage",
            "aggregate": {
                "test_f1_mean": 0.5,
                "eval_f1_mean": 0.6,
            },
            "summary": summary,
        },
        run_dir,
    )

    assert run_summary["summary"]["mode"] == "api"
    assert run_summary["summary"]["feedback_scope_user_id"] == "experiment:run-42"
    assert run_summary["summary"]["pattern_keys_used"] == ["SERVER_PATTERN", "SPINDLE_LOAD_RAMP"]


def test_run_summary_sanitizes_non_finite_values(tmp_path):
    run_dir = tmp_path / "stoppage_run"
    run_dir.mkdir()
    (run_dir / "experiment_results.json").write_text("{}")

    run_summary = _build_run_summary(
        "stoppage_run",
        {
            "experiment_type": "stoppage",
            "config": {
                "store_threshold": float("nan"),
                "alert_threshold": float("inf"),
            },
            "comparison": {
                "test": {
                    "f1": float("nan"),
                    "precision": 0.5,
                },
                "eval": {
                    "f1": float("nan"),
                    "precision": float("inf"),
                    "recall": 0.6,
                },
            },
            "summary": {
                "n_feedback_events": float("nan"),
            },
        },
        run_dir,
    )

    assert run_summary["config"]["store_threshold"] is None
    assert run_summary["config"]["alert_threshold"] is None
    assert run_summary["test_metrics"]["f1"] is None
    assert run_summary["eval_metrics"]["f1"] is None
    assert run_summary["eval_metrics"]["precision"] is None
    assert run_summary["summary"]["n_feedback_events"] is None
    json.dumps(run_summary, allow_nan=False)


def test_live_breakage_route_hardens_api_config_against_client_overrides():
    hardened = _harden_live_breakage_config_overrides(
        run_id="breakage_demo",
        request=SimpleNamespace(base_url="http://127.0.0.1:8010/"),
        config_overrides={
            "api_mode": False,
            "api_mode_strict": False,
            "api_base_url": "http://example.invalid",
            "experiment_fast_path": True,
            "api_use_server_patterns": False,
            "api_batch_size": 99,
            "persist_shared_priors": True,
            "feedback_user_id": "operator",
            "feedback_every_n": 7,
        },
    )

    assert hardened["api_mode"] is True
    assert hardened["api_mode_strict"] is True
    assert hardened["experiment_fast_path"] is False
    assert hardened["api_use_server_patterns"] is True
    assert hardened["api_batch_size"] == 1
    assert hardened["persist_shared_priors"] is False
    assert hardened["feedback_user_id"] == "experiment:breakage_demo"
    assert hardened["api_base_url"] == "http://127.0.0.1:8010"
    assert hardened["feedback_every_n"] == 7
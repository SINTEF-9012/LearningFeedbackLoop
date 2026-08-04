"""Session score calibration wired into the orchestrator (plan 1.2), opt-in."""
from __future__ import annotations

from pathlib import Path

from backend.agents.memory.orchestrator import (
    MemoryEventOrchestrator,
    MemoryEvent,
    OrchestratorConfig,
    TimeRange,
)


def _light_orchestrator(tmp_path: Path) -> MemoryEventOrchestrator:
    return MemoryEventOrchestrator(
        config=OrchestratorConfig(
            use_classical_models=False,
            enable_harmonic_scorer=False,
            dispatch_alerts=False,
            priors_path=str(tmp_path / "priors.json"),
            model_confidence_path=str(tmp_path / "mc.json"),
        )
    )


def _event(session_id, score):
    return MemoryEvent(
        session_id=session_id,
        time_range=TimeRange(i0=0, i1=5, t0=0.0, t1=5.0, fs=1.0),
        patterns=[],
        external_signals={"anomaly_detector_score": score, "model_confidence": 1.0},
    )


def test_calibration_off_by_default_leaves_signals_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMORY_CALIBRATE_MODEL_SCORE", raising=False)
    orch = _light_orchestrator(tmp_path)
    ev = _event("s1", 0.9)
    out = orch._maybe_calibrate_model_score(ev)
    assert out is ev.external_signals
    assert out["anomaly_detector_score"] == 0.9


def test_calibration_on_replaces_with_session_percentile(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_CALIBRATE_MODEL_SCORE", "1")
    monkeypatch.setenv("MEMORY_CALIBRATE_WARMUP", "10")
    orch = _light_orchestrator(tmp_path)

    for _ in range(10):  # warm-up: neutral
        out = orch._maybe_calibrate_model_score(_event("s1", 0.30))
        assert out["anomaly_detector_score"] == 0.0
        assert out["anomaly_detector_calibrated"] is True

    hi = orch._maybe_calibrate_model_score(_event("s1", 0.95))
    assert hi["anomaly_detector_score"] > 0.9
    assert hi["anomaly_detector_score_raw"] == 0.95


def test_calibrators_are_per_session(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_CALIBRATE_MODEL_SCORE", "1")
    monkeypatch.setenv("MEMORY_CALIBRATE_WARMUP", "5")
    orch = _light_orchestrator(tmp_path)
    for _ in range(5):
        orch._maybe_calibrate_model_score(_event("sA", 0.30))
    orch._maybe_calibrate_model_score(_event("sB", 0.99))
    assert "sA" in orch._score_calibrators and "sB" in orch._score_calibrators
    assert orch._score_calibrators["sB"].warmed_up is False

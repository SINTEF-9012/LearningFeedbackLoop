"""Counterfactual feedback summary (plan 2.4)."""
from backend.agents.memory.counterfactual import (
    summarize_counterfactual,
    load_counterfactual_summary,
)


def _snap():
    return {
        "session": "all",
        "adjudicate": "episode",
        "off": {"auc": 0.63, "rows": [
            {"operation_id": "s·OF1", "n_pre_break": 100, "alerts": 60, "tp_alerts": 60, "fp_alerts": 0},
            {"operation_id": "s·OF2", "n_pre_break": 0, "alerts": 40, "tp_alerts": 0, "fp_alerts": 40},
            {"operation_id": "s·OF3", "n_pre_break": 0, "alerts": 30, "tp_alerts": 0, "fp_alerts": 30},
        ]},
        "on": {"auc": 0.66, "rows": [
            {"operation_id": "s·OF1", "n_pre_break": 100, "alerts": 20, "tp_alerts": 20, "fp_alerts": 0},
            {"operation_id": "s·OF2", "n_pre_break": 0, "alerts": 10, "tp_alerts": 0, "fp_alerts": 10},
            {"operation_id": "s·OF3", "n_pre_break": 0, "alerts": 0, "tp_alerts": 0, "fp_alerts": 0},
        ]},
    }


def test_summary_burden_and_coverage():
    s = summarize_counterfactual(_snap())
    assert s["off"]["alerts"] == 130 and s["on"]["alerts"] == 30
    assert s["burden_reduction"] == round((130 - 30) / 130, 4)
    assert s["false_alarm_reduction"] == round((70 - 10) / 70, 4)
    # Broken episode still caught in both arms → coverage preserved.
    assert s["off"]["broken_episodes_alerted"] == 1
    assert s["on"]["broken_episodes_alerted"] == 1
    assert s["coverage_preserved"] is True
    # One healthy episode fully silenced (OF3: 30 → 0).
    assert s["off"]["healthy_episodes_alerting"] == 2
    assert s["on"]["healthy_episodes_alerting"] == 1


def test_coverage_not_preserved_when_broken_episode_dropped():
    snap = _snap()
    snap["on"]["rows"][0]["alerts"] = 0  # broken OF1 no longer alerts
    snap["on"]["rows"][0]["tp_alerts"] = 0
    s = summarize_counterfactual(snap)
    assert s["coverage_preserved"] is False


def test_missing_snapshot_returns_none(tmp_path):
    assert load_counterfactual_summary(tmp_path / "nope.json") is None


def test_default_snapshot_loads_if_present():
    # The repo ships the case-study snapshot; if present it summarizes cleanly.
    s = load_counterfactual_summary()
    if s is not None:
        assert s["measured"] is True
        assert s["off"]["alerts"] >= s["on"]["alerts"]

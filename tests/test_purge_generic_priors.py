from __future__ import annotations

import json

from backend.agents.config import SignificanceThresholds
from backend.agents.memory.scorer import SignificanceConfig
from scripts.purge_generic_priors import purge_prior_payload, purge_priors_file, should_purge_key


def test_should_purge_key_matches_generic_and_hypothesis_namespaces() -> None:
    assert should_purge_key("freq:ch0:low") is True
    assert should_purge_key("amp:ch1:loud") is True
    assert should_purge_key("temporal:ch2:sustained") is True
    assert should_purge_key("spectral:ch0:wideband") is True
    assert should_purge_key("kurtosis:ch2:heavy-tails") is True
    assert should_purge_key("RATIO_ch0_ch1:>28.0") is True
    assert should_purge_key("hypothesis:workpiece_slip") is True
    assert should_purge_key("SPINDLE_POWER_SURGE") is False
    assert should_purge_key("suppressed:power_spindle_mean_H+vib_severity_x_mean_H") is False


def test_purge_prior_payload_removes_generic_channel_and_hypothesis_entries() -> None:
    payload = {
        "pattern_priors": {
            "freq:ch0:low": 0.78,
            "amp:ch1:loud": 0.88,
            "RATIO_ch0_ch1:>28.0": 0.66,
            "hypothesis:workpiece_slip": 0.91,
            "SPINDLE_POWER_SURGE": 0.74,
            "suppressed:power_spindle_mean_H+vib_severity_x_mean_H": 0.66,
        },
        "feedback_counts": {
            "freq:ch0:low": {"confirm": 3, "dismiss": 0},
            "RATIO_ch0_ch1:>28.0": {"confirm": 1, "dismiss": 2},
            "hypothesis:workpiece_slip": {"confirm": 2, "dismiss": 0},
            "SPINDLE_POWER_SURGE": {"confirm": 2, "dismiss": 1},
        },
        "prior_source": "feedback_runtime",
    }

    cleaned, summary = purge_prior_payload(payload)

    assert set(cleaned["pattern_priors"].keys()) == {
        "SPINDLE_POWER_SURGE",
        "suppressed:power_spindle_mean_H+vib_severity_x_mean_H",
    }
    assert set(cleaned["feedback_counts"].keys()) == {"SPINDLE_POWER_SURGE"}
    assert summary["removed_pattern_priors"] == 4
    assert summary["removed_feedback_counts"] == 3


def test_purge_priors_file_rewrites_json_in_place(tmp_path) -> None:
    path = tmp_path / "pattern_priors.json"
    path.write_text(json.dumps({
        "pattern_priors": {
            "freq:ch0:low": 0.78,
            "SPINDLE_POWER_SURGE": 0.74,
        },
        "feedback_counts": {
            "freq:ch0:low": {"confirm": 3, "dismiss": 0},
        },
        "prior_source": "feedback_runtime",
    }), encoding="utf-8")

    cleaned, summary = purge_priors_file(path)
    reloaded = json.loads(path.read_text(encoding="utf-8"))

    assert cleaned == reloaded
    assert reloaded["pattern_priors"] == {"SPINDLE_POWER_SURGE": 0.74}
    assert reloaded["feedback_counts"] == {}
    assert summary["removed_pattern_priors"] == 1
    assert summary["removed_feedback_counts"] == 1


def test_prior_damping_defaults_are_raised(monkeypatch) -> None:
    monkeypatch.delenv("SIG_PRIOR_DAMPING_K", raising=False)

    assert SignificanceThresholds().prior_evidence_damping_k == 20.0
    assert SignificanceThresholds.from_env().prior_evidence_damping_k == 20.0
    assert SignificanceConfig().prior_evidence_damping_k == 20.0
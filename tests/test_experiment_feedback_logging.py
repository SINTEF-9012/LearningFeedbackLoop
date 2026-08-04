from backend.agents.experiment.evaluator import (
    PhaseResult,
    SampleResult,
    _integrate_promoted_discoveries,
    _summarize_feedback_events,
)
from backend.agents.patterns.discovery import DiscoveredPattern


def test_phase_result_to_dict_persists_feedback_audit_and_sample_feedback_fields() -> None:
    sample = SampleResult(
        sample_id="sample-1",
        label="pre_stoppage",
        operation_id="OF00004",
        feedback_given=True,
        feedback_action="CONFIRM",
        feedback_source="flagged",
        detected_patterns=["VIBRATION_REGIME_SHIFT"],
        prior_snapshot={"VIBRATION_REGIME_SHIFT": 0.62},
        co_occurring_pairs=[("A", "B")],
        propagated_prior_deltas={"FEED_STALL": 0.03125},
        stored_in_memory=True,
    )
    phase = PhaseResult(
        phase="eval",
        operation="OF00004",
        sample_results=[sample],
        feedback_events=[
            {
                "source_sample_id": "sample-1",
                "feedback_action": "CONFIRM",
                "feedback_source": "flagged",
                "pattern_updates": [
                    {
                        "pattern_key": "VIBRATION_REGIME_SHIFT",
                        "polarity": "protective",
                        "old_prior": 0.5,
                        "new_prior": 0.62,
                        "delta": 0.12,
                    }
                ],
            }
        ],
    )

    out = phase.to_dict()

    assert out["feedback_events"][0]["source_sample_id"] == "sample-1"
    assert out["pattern_feedback_summary"]["VIBRATION_REGIME_SHIFT"]["n_confirms"] == 1
    assert out["sample_results"][0]["prior_snapshot"] == {"VIBRATION_REGIME_SHIFT": 0.62}
    assert out["sample_results"][0]["co_occurring_pairs"] == [["A", "B"]]
    assert out["sample_results"][0]["propagated_prior_deltas"] == {"FEED_STALL": 0.0312}
    assert out["sample_results"][0]["stored_in_memory"] is True


def test_summarize_feedback_events_aggregates_pattern_response() -> None:
    summary = _summarize_feedback_events([
        {
            "feedback_action": "CONFIRM",
            "pattern_updates": [
                {
                    "pattern_key": "VIBRATION_REGIME_SHIFT",
                    "polarity": "protective",
                    "new_prior": 0.62,
                    "delta": 0.12,
                }
            ],
        },
        {
            "feedback_action": "DISMISS",
            "pattern_updates": [
                {
                    "pattern_key": "VIBRATION_REGIME_SHIFT",
                    "polarity": "protective",
                    "new_prior": 0.57,
                    "delta": -0.05,
                },
                {
                    "pattern_key": "FEED_STALL",
                    "polarity": "fault_supporting",
                    "new_prior": 0.45,
                    "delta": -0.05,
                },
            ],
        },
    ])

    assert summary["VIBRATION_REGIME_SHIFT"] == {
        "polarity": "protective",
        "n_feedback_events": 2,
        "n_confirms": 1,
        "n_dismissals": 1,
        "total_prior_delta": 0.07,
        "mean_prior_delta": 0.035,
        "max_abs_prior_delta": 0.12,
        "last_prior": 0.57,
    }
    assert summary["FEED_STALL"]["polarity"] == "fault_supporting"
    assert summary["FEED_STALL"]["n_dismissals"] == 1


def test_integrate_promoted_discoveries_inserts_confirmed_patterns_into_runtime_priors() -> None:
    pattern_priors = {"FEED_STALL": 0.5}
    feedback_counts = {"FEED_STALL": {"confirm": 0, "dismiss": 0}}
    discovered = [
        DiscoveredPattern(
            key="discovered:power_spindle_mean_H+vib_severity_x_mean_H",
            features={"power_spindle_mean": "high", "vib_severity_x_mean": "high"},
            confirmation_count=4,
            promoted=True,
            prior=0.5,
        )
    ]

    inserted = _integrate_promoted_discoveries(
        pattern_priors,
        feedback_counts,
        discovered,
        feedback_action="CONFIRM",
    )

    assert inserted == ["discovered:power_spindle_mean_H+vib_severity_x_mean_H"]
    assert pattern_priors["discovered:power_spindle_mean_H+vib_severity_x_mean_H"] == 0.5
    assert feedback_counts["discovered:power_spindle_mean_H+vib_severity_x_mean_H"] == {
        "confirm": 4,
        "dismiss": 0,
    }


def test_integrate_promoted_discoveries_inserts_suppression_patterns_with_dismiss_counts() -> None:
    pattern_priors = {"FEED_STALL": 0.5}
    feedback_counts = {"FEED_STALL": {"confirm": 0, "dismiss": 0}}
    discovered = [
        DiscoveredPattern(
            key="suppressed:power_spindle_mean_H+vib_severity_x_mean_H",
            features={"power_spindle_mean": "high", "vib_severity_x_mean": "high"},
            confirmation_count=4,
            promoted=True,
            prior=0.3,
        )
    ]

    inserted = _integrate_promoted_discoveries(
        pattern_priors,
        feedback_counts,
        discovered,
        feedback_action="DISMISS",
    )

    assert inserted == ["suppressed:power_spindle_mean_H+vib_severity_x_mean_H"]
    assert pattern_priors["suppressed:power_spindle_mean_H+vib_severity_x_mean_H"] == 0.3
    assert feedback_counts["suppressed:power_spindle_mean_H+vib_severity_x_mean_H"] == {
        "confirm": 0,
        "dismiss": 4,
    }
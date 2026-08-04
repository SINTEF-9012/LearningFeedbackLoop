from backend.agents.memory.experiment_summary_routes import _build_threshold_recommendations


def test_build_threshold_recommendations_uses_heuristics_without_llm():
    key_metrics = {
        "eval": {
            "false_positives": 4,
            "missed": 3,
            "total": 10,
            "precision": 0.5,
            "recall": 0.4,
        }
    }
    all_patterns = {"VIBRATION_SPIKE": 5}
    results = {"config": {"store_threshold": 0.3, "alert_threshold": 0.6}}

    recs = _build_threshold_recommendations(
        key_metrics,
        all_patterns,
        results,
        llm_available=False,
    )

    params = [rec.parameter for rec in recs]
    assert "alert_threshold" in params
    assert "store_threshold" in params
    assert "prior:VIBRATION_SPIKE" in params
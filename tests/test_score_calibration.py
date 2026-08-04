import random

from backend.agents.processing.score_calibration import SessionScoreCalibrator


def test_warmup_returns_neutral():
    # A score is calibrated against the window BEFORE joining it, so the
    # first `warmup` calls are neutral and call warmup+1 is the first
    # calibrated one.
    cal = SessionScoreCalibrator(warmup=10)
    for i in range(10):
        c = cal.update(0.7 + 0.01 * i)
        assert not c.warmed_up
        assert c.z == 0.0
        assert c.percentile == 0.0
    assert cal.update(0.7).warmed_up


def test_spike_stands_out_after_warmup():
    rng = random.Random(42)
    cal = SessionScoreCalibrator(warmup=20)
    for _ in range(100):
        cal.update(0.70 + rng.gauss(0.0, 0.01))
    c = cal.update(0.95)
    assert c.z > 5.0
    assert c.percentile == 1.0


def test_adapts_to_session_shift():
    """A constant offset (the cross-session shift failure mode) is absorbed:
    after the window fills with shifted scores, z returns to ~0."""
    rng = random.Random(7)
    cal = SessionScoreCalibrator(warmup=20, window=100)
    for _ in range(100):
        cal.update(0.30 + rng.gauss(0.0, 0.01))
    first_shifted = cal.update(0.80 + rng.gauss(0.0, 0.01))
    assert first_shifted.z > 10.0  # shift initially looks anomalous
    for _ in range(150):  # window fully turns over at the new level
        cal.update(0.80 + rng.gauss(0.0, 0.01))
    settled = cal.update(0.80)
    assert abs(settled.z) < 2.0


def test_window_eviction_keeps_rank_consistent():
    cal = SessionScoreCalibrator(warmup=5, window=10)
    for i in range(50):
        cal.update(float(i))
    assert len(cal) == 10
    # Window holds 40..49; 100 is above all of it, 39 below all of it.
    assert cal.update(100.0).percentile == 1.0
    assert cal.update(39.0).percentile == 0.0

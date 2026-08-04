"""Tests for the structured PatternPrior record (read-side prior aggregation).

Uses ``cluster:*`` keys: signature/fault keys normalize to ``signature:*`` and are
deliberately excluded from per-pattern priors (forced to 0.5), so priors attach to
discovered/cluster patterns.
"""

from backend.agents.memory.scorer import SignificanceScorer, normalize_pattern_key
from backend.agents.memory.pattern_prior import PatternPrior


def test_record_reflects_counts_and_prior():
    s = SignificanceScorer()
    key = normalize_pattern_key("cluster:test")
    s._local_feedback_counts[key] = {"confirm": 3.0, "dismiss": 1.0}

    rec = s.get_pattern_prior_record("cluster:test")
    assert isinstance(rec, PatternPrior)
    assert rec.pattern_key == key
    assert rec.confirmed == 3.0 and rec.dismissed == 1.0
    assert rec.evidence_count == 4.0
    # prior_strength matches the scorer's own derivation (source of truth)
    assert abs(rec.prior_strength - s.get_pattern_prior(key)) < 1e-9
    # confirm-heavy → prior leans above neutral
    assert rec.prior_strength > 0.5
    # confidence is volume-based and bounded
    assert 0.0 < rec.confidence < 1.0


def test_record_no_feedback_is_neutral_low_confidence():
    s = SignificanceScorer()
    rec = s.get_pattern_prior_record("cluster:fresh")
    assert rec.confirmed == 0.0 and rec.dismissed == 0.0
    assert rec.evidence_count == 0.0
    assert abs(rec.prior_strength - 0.5) < 1e-9
    assert rec.confidence == 0.0  # no evidence -> zero confidence


def test_confidence_monotonic_in_volume():
    s = SignificanceScorer()
    key = normalize_pattern_key("cluster:test")
    s._local_feedback_counts[key] = {"confirm": 1.0, "dismiss": 0.0}
    low = s.get_pattern_prior_record("cluster:test").confidence
    s._local_feedback_counts[key] = {"confirm": 20.0, "dismiss": 0.0}
    high = s.get_pattern_prior_record("cluster:test").confidence
    assert high > low


def test_to_dict_exposes_evidence():
    s = SignificanceScorer()
    key = normalize_pattern_key("cluster:test")
    s._local_feedback_counts[key] = {"confirm": 2.0, "dismiss": 0.0}
    d = s.get_pattern_prior_record("cluster:test").to_dict()
    assert {"pattern_key", "prior_strength", "confidence", "confirmed",
            "dismissed", "evidence_count", "evidence_memory_ids"} <= set(d)
    assert d["confirmed"] == 2.0


def test_list_records_covers_known_priors():
    s = SignificanceScorer()
    ck = normalize_pattern_key("cluster:chatter")
    tk = normalize_pattern_key("cluster:break")
    s._pattern_priors[ck] = 0.7
    s._pattern_priors[tk] = 0.4
    keys = {r.pattern_key for r in s.list_pattern_prior_records()}
    assert ck in keys and tk in keys

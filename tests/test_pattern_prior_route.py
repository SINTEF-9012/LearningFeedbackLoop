"""Route test for GET /memory/priors (structured pattern-prior records)."""

import backend.agents.memory.router as R
from backend.agents.memory.scorer import SignificanceScorer, normalize_pattern_key


class _Orch:
    def __init__(self, scorer):
        self.scorer = scorer


async def test_priors_route_lists_records(monkeypatch):
    s = SignificanceScorer()
    key = normalize_pattern_key("cluster:chatter")
    s._pattern_priors[key] = 0.7
    s._local_feedback_counts[key] = {"confirm": 3.0, "dismiss": 1.0}
    monkeypatch.setattr(R, "get_orchestrator", lambda: _Orch(s))

    resp = await R.list_pattern_priors(pattern_key=None)
    assert resp["count"] >= 1
    rec = next(p for p in resp["priors"] if p["pattern_key"] == key)
    assert rec["confirmed"] == 3.0 and rec["dismissed"] == 1.0
    assert 0.0 < rec["confidence"] < 1.0


async def test_priors_route_single(monkeypatch):
    s = SignificanceScorer()
    key = normalize_pattern_key("cluster:test")
    s._local_feedback_counts[key] = {"confirm": 2.0, "dismiss": 0.0}
    monkeypatch.setattr(R, "get_orchestrator", lambda: _Orch(s))

    resp = await R.list_pattern_priors(pattern_key="cluster:test")
    assert resp["prior"]["pattern_key"] == key
    assert resp["prior"]["confirmed"] == 2.0

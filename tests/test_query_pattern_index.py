"""Tests for Agent Q Round 19 — QueryPatternIndex.

Two concerns:
1. The indexed form must produce **identical** `(score, matched)` output
   to the legacy triple-loop reference for any combination of query
   and memory pattern lists. We assert that over a variety of shapes
   and a random fuzz batch.
2. The indexed form must not regress performance. We time a hot-path
   scenario (1 query × 100 memories × 5 repeats) and assert the
   indexed path is at least as fast as the naive per-call static
   method. This is a soft guard (1.2× tolerance) to absorb CI jitter.
"""

from __future__ import annotations

import random
import time
from typing import List, Tuple

import pytest

from backend.agents.memory.retriever import PatternMatcher, QueryPatternIndex


def _legacy_reference(
    query_patterns: List[str], memory_patterns: List[str]
) -> Tuple[float, List[str]]:
    """Literal port of the pre-Round-19 scoring loop."""
    if not query_patterns or not memory_patterns:
        return (0.0, [])

    def exact(p1: str, p2: str) -> bool:
        return p1.lower() == p2.lower()

    def family(p1: str, p2: str) -> bool:
        def get_family(p: str) -> str:
            if ":" in p:
                return p.rsplit(":", 1)[0].lower()
            return p.lower()
        return get_family(p1) == get_family(p2)

    def type_match(p1: str, p2: str) -> bool:
        def get_type(p: str) -> str:
            return p.split("_")[0].lower()
        return get_type(p1) == get_type(p2)

    matched: List[str] = []
    score_sum = 0.0
    for qp in query_patterns:
        for mp in memory_patterns:
            if exact(qp, mp):
                matched.append(mp)
                score_sum += 1.0
                break
            elif family(qp, mp):
                matched.append(mp)
                score_sum += 0.6
                break
            elif type_match(qp, mp):
                matched.append(mp)
                score_sum += 0.3
                break
    score = score_sum / len(query_patterns) if query_patterns else 0.0
    return (min(1.0, score), matched)


@pytest.mark.parametrize(
    "query,memory",
    [
        ([], []),
        (["RATIO_Fx_Fy:>5"], []),
        ([], ["RATIO_Fx_Fy:>5"]),
        (["RATIO_Fx_Fy:>5"], ["RATIO_Fx_Fy:>5"]),  # exact
        (["RATIO_Fx_Fy:>5"], ["RATIO_Fx_Fy:2-5"]),  # family
        (["RATIO_Fx_Fy:>5"], ["RATIO_Fz_My:>3"]),  # type
        (["RATIO_Fx_Fy:>5"], ["SPECTRAL_PEAK_512Hz"]),  # no match
        (
            ["RATIO_Fx_Fy:>5", "SPECTRAL_PEAK_512Hz", "TREND_DOWN"],
            ["RATIO_Fx_Fy:2-5", "SPECTRAL_PEAK_512Hz", "OUTLIER_HIGH"],
        ),
        # Case insensitivity
        (["Ratio_Fx_Fy:>5"], ["ratio_fx_fy:2-5"]),
    ],
)
def test_indexed_matches_legacy(query, memory):
    ref_score, ref_matched = _legacy_reference(query, memory)
    idx = QueryPatternIndex(query)
    new_score, new_matched = idx.score_against(memory)
    assert new_score == pytest.approx(ref_score)
    assert new_matched == ref_matched


def test_static_api_still_matches_legacy():
    # Backward-compat: PatternMatcher.score_pattern_similarity keeps working.
    q = ["RATIO_Fx_Fy:>5", "TREND_DOWN"]
    m = ["RATIO_Fz_My:2-5", "TREND_DOWN"]
    ref = _legacy_reference(q, m)
    got = PatternMatcher.score_pattern_similarity(q, m)
    assert got == ref


def test_fuzz_equivalence_random():
    rng = random.Random(42)
    types = ["RATIO", "SPECTRAL", "TREND", "OUTLIER"]
    vars_ = ["Fx_Fy", "Fz_My", "peak_512Hz", "low_freq", "bearing_1"]
    ranges = [":>5", ":2-5", ":>3", ":<1", ""]

    def gen_pattern() -> str:
        return rng.choice(types) + "_" + rng.choice(vars_) + rng.choice(ranges)

    for _ in range(200):
        q_len = rng.randint(0, 8)
        m_len = rng.randint(0, 12)
        q = [gen_pattern() for _ in range(q_len)]
        m = [gen_pattern() for _ in range(m_len)]
        ref = _legacy_reference(q, m)
        idx = QueryPatternIndex(q)
        got = idx.score_against(m)
        assert got == ref, f"mismatch for q={q} m={m}"


def test_reuse_across_candidates_is_correct():
    # The whole point of the index: build once, reuse. Verify each call
    # returns the right answer for its own memory list.
    q = ["RATIO_Fx_Fy:>5", "TREND_DOWN", "OUTLIER_HIGH"]
    idx = QueryPatternIndex(q)
    corpus = [
        ["RATIO_Fx_Fy:>5"],
        ["RATIO_Fz_Fy:2-5", "TREND_UP"],
        ["SPECTRAL_PEAK_512Hz"],
        ["OUTLIER_HIGH", "TREND_DOWN"],
    ]
    for mem in corpus:
        assert idx.score_against(mem) == _legacy_reference(q, mem)


def test_perf_indexed_not_worse_than_static_per_call():
    """Soft perf guard: building the index once is at least as fast as
    calling the static method per candidate (which re-tokenises)."""
    rng = random.Random(0)
    types = ["RATIO", "SPECTRAL", "TREND", "OUTLIER", "HARMONIC"]
    suffix = [":>5", ":2-5", ""]

    def pat() -> str:
        return (
            rng.choice(types)
            + "_"
            + rng.choice(["a", "b", "c", "d"])
            + "_"
            + rng.choice(["x", "y", "z"])
            + rng.choice(suffix)
        )

    query = [pat() for _ in range(8)]
    corpus = [[pat() for _ in range(10)] for _ in range(100)]

    # Legacy path: PatternMatcher.score_pattern_similarity called
    # per candidate (re-tokenises query every time).
    iters = 20

    t0 = time.perf_counter()
    for _ in range(iters):
        for mem in corpus:
            PatternMatcher.score_pattern_similarity(query, mem)
    legacy_elapsed = time.perf_counter() - t0

    # Indexed path: build once per outer loop, reuse across candidates.
    t0 = time.perf_counter()
    for _ in range(iters):
        idx = QueryPatternIndex(query)
        for mem in corpus:
            idx.score_against(mem)
    indexed_elapsed = time.perf_counter() - t0

    # Allow 20% slack to absorb CI jitter; expectation is that indexed
    # is strictly faster.
    assert indexed_elapsed <= legacy_elapsed * 1.2, (
        f"Indexed path regressed: legacy={legacy_elapsed*1000:.2f}ms, "
        f"indexed={indexed_elapsed*1000:.2f}ms"
    )

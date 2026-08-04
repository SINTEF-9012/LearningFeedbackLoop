"""LFL pipeline micro-benchmark (Agent Q baseline, 2026-04-24).

Times the key hot paths the refactor aims to speed up, so every later PR
can report before/after numbers for the hot paths it touches.

Covers (when the required subsystems import cleanly):
    1. compute_harmonics on a synthetic multi-channel window
    2. DefaultFeatureExtractor.extract on a synthetic window
    3. Neo4jMemoryStore._ensure_schema (init cost) — optional, skipped if
       Neo4j is not reachable
    4. SignificanceScorer.score on a canned pattern / metrics payload

Usage:
    python scripts/bench_pipeline.py [--iters N] [--json out.json]

Intentionally dependency-light: only numpy. Neo4j and torch-backed code
paths are probed via best-effort imports; unavailable ones are skipped
rather than failing the whole run.

Design notes:
- Reports min / median / p95 / mean in milliseconds.
- No warmup loop by default; set --warmup to enable.
- Output JSON is stable-schema so successive runs diff cleanly:
    {version, timestamp, machine, results: {name: {iters, min_ms, ...}}}
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

BENCH_VERSION = 1


def _time_callable(fn: Callable[[], Any], iters: int, warmup: int) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
    samples_ms: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    samples_ms.sort()
    p95 = samples_ms[min(len(samples_ms) - 1, int(0.95 * len(samples_ms)))]
    return {
        "iters": iters,
        "min_ms": samples_ms[0],
        "median_ms": statistics.median(samples_ms),
        "p95_ms": p95,
        "max_ms": samples_ms[-1],
        "mean_ms": statistics.fmean(samples_ms),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def _bench_compute_harmonics(iters: int, warmup: int) -> Optional[Dict[str, Any]]:
    try:
        # The repo has multiple harmonic utilities; pick the first that works.
        from backend.agents.processing.harmonic_features import compute_harmonics  # type: ignore
    except Exception:
        try:
            from backend.agents.processing.features import compute_harmonics  # type: ignore
        except Exception as exc:
            return {"skipped": f"compute_harmonics unavailable: {exc!r}"}

    fs = 20_000.0
    n = 20_000  # 1s
    t = np.arange(n) / fs
    rng = np.random.default_rng(42)
    # 4 channels: tones + noise
    data = np.stack([
        np.sin(2 * np.pi * 97 * t) + 0.1 * rng.standard_normal(n),
        np.sin(2 * np.pi * 194 * t) + 0.1 * rng.standard_normal(n),
        np.sin(2 * np.pi * 291 * t) + 0.1 * rng.standard_normal(n),
        0.2 * rng.standard_normal(n),
    ], axis=0)

    fundamentals = [97.0, 194.0, 291.0]

    def _call():
        try:
            return compute_harmonics(data, fs=fs, fundamentals=fundamentals)
        except TypeError:
            # Different signature variant
            return compute_harmonics(data, fs, fundamentals)

    try:
        _call()
    except Exception as exc:
        return {"skipped": f"compute_harmonics call failed: {exc!r}"}

    return _time_callable(_call, iters=iters, warmup=warmup)


def _bench_feature_extractor(iters: int, warmup: int) -> Optional[Dict[str, Any]]:
    try:
        from backend.agents.processing.features import DefaultFeatureExtractor  # type: ignore
    except Exception as exc:
        return {"skipped": f"DefaultFeatureExtractor unavailable: {exc!r}"}

    fe = DefaultFeatureExtractor()
    fs = 20_000.0
    n = 4_096
    rng = np.random.default_rng(0)
    window = rng.standard_normal((4, n)).astype(np.float32)

    def _call():
        # Try a few likely signatures without coupling to the real one.
        for args in (
            (window,),
            (window, fs),
            ({"Fx": window[0], "Fy": window[1], "Fz": window[2], "P": window[3]}, fs),
        ):
            try:
                return fe.extract(*args)
            except TypeError:
                continue
        raise RuntimeError("no matching extract() signature")

    try:
        _call()
    except Exception as exc:
        return {"skipped": f"feature extractor call failed: {exc!r}"}

    return _time_callable(_call, iters=iters, warmup=warmup)


def _bench_neo4j_schema(iters: int, warmup: int) -> Optional[Dict[str, Any]]:
    uri = os.environ.get("NEO4J_URI")
    if not uri:
        return {"skipped": "NEO4J_URI not set"}
    try:
        from backend.agents.storage.neo4j_store import Neo4jMemoryStore  # type: ignore
    except Exception as exc:
        return {"skipped": f"Neo4jMemoryStore import failed: {exc!r}"}

    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    pw = os.environ.get("NEO4J_PASSWORD", "password")
    db = os.environ.get("NEO4J_DATABASE", "neo4j")

    def _call():
        # Construct a fresh store each time — this is what the guard
        # protects against re-running.
        store = Neo4jMemoryStore(uri=uri, username=user, password=pw, database=db)
        store._driver.close()

    try:
        _call()
    except Exception as exc:
        return {"skipped": f"Neo4j connection failed: {exc!r}"}

    return _time_callable(_call, iters=iters, warmup=warmup)


def _bench_scorer(iters: int, warmup: int) -> Optional[Dict[str, Any]]:
    try:
        from backend.agents.memory.scorer import (
            SignificanceScorer, SignificanceConfig,
        )
        from backend.agents.core.schemas import PatternKey, PatternType, NumericMetrics
    except Exception as exc:
        return {"skipped": f"Scorer/schemas unavailable: {exc!r}"}

    try:
        scorer = SignificanceScorer(config=SignificanceConfig())
    except Exception as exc:
        return {"skipped": f"Scorer construct failed: {exc!r}"}

    # Build a minimal set of patterns + metrics
    try:
        patterns = [
            PatternKey(key="fault:chatter", type=PatternType.FAULT, confidence=0.8),
        ]
    except Exception as exc:
        return {"skipped": f"PatternKey build failed: {exc!r}"}

    try:
        metrics = NumericMetrics()
    except Exception:
        metrics = None

    session_id = "bench-session"

    def _call():
        try:
            return scorer.score(
                patterns=patterns,
                metrics=metrics,
                context=None,
                session_id=session_id,
                external_signals={"anomaly_detector_score": 0.3},
            )
        except TypeError:
            return scorer.score(patterns, metrics, None, session_id,
                                {"anomaly_detector_score": 0.3})

    try:
        _call()
    except Exception as exc:
        return {"skipped": f"Scorer score() failed: {exc!r}"}

    return _time_callable(_call, iters=iters, warmup=warmup)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

_BENCHES: List[Tuple[str, Callable[[int, int], Optional[Dict[str, Any]]]]] = [
    ("compute_harmonics", _bench_compute_harmonics),
    ("feature_extractor", _bench_feature_extractor),
    ("neo4j_schema_init", _bench_neo4j_schema),
    ("scorer_score", _bench_scorer),
]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=50,
                        help="number of timed iterations per bench (default 50)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="number of untimed warmup iterations (default 5)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="restrict to named benches (default: all)")
    parser.add_argument("--json", type=str, default=None,
                        help="write results to this JSON path")
    args = parser.parse_args(argv)

    # Ensure repo root is on sys.path so `backend.*` imports work when this
    # script is run directly.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    names = set(args.only) if args.only else None
    results: Dict[str, Any] = {}
    for name, fn in _BENCHES:
        if names is not None and name not in names:
            continue
        print(f"[bench] {name} ...", flush=True)
        try:
            results[name] = fn(args.iters, args.warmup) or {"skipped": "returned None"}
        except Exception as exc:  # pragma: no cover — defensive
            results[name] = {"error": repr(exc)}

    report = {
        "version": BENCH_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
        },
        "params": {"iters": args.iters, "warmup": args.warmup},
        "results": results,
    }

    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
        print(f"[bench] wrote {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

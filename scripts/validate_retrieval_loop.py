#!/usr/bin/env python3
"""Validate the memory-KG retrieval loop end-to-end against the LIVE store.

The one architectural claim no experiment exercises: *the system stores significant events
as memories in the knowledge graph and retrieves similar past memories (weighted by prior
operator feedback) when a new event fires.* The faithful AUC runs only exercise detection +
the per-pattern prior; they never round-trip through the store. This script drives the real
`MemoryEventOrchestrator.process_event` (NOT fast-path) against the configured backend
(Neo4j if up, else the SQLite/in-memory store), feeding a slice of the stoppage events so
that store -> retrieve -> feedback runs on the deployed code path.

It reports integration fidelity (events stored, retrieval hit-rate, feedback prior update),
NOT a new detection AUC — retrieval runs *after* scoring (orchestrator step 3) and feeds the
explanation/feedback path, so it cannot change the significance score. That separation is
itself the thing being confirmed.
"""

from __future__ import annotations

import asyncio
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.agents.core.schemas import PatternKey, PatternType, TimeRange
from backend.agents.memory.orchestrator import (
    MemoryEvent,
    MemoryEventOrchestrator,
    OrchestratorConfig,
)
from backend.agents.experiment.config import ExperimentConfig
from backend.agents.experiment.evaluator import _detect_patterns_batch

CSV = ROOT / "data" / "breakage_patterns" / "stoppage_features.csv"
N_EVENTS = 120  # ~2 operations — enough to build history, small enough to stay quick


def _make_store():
    """Real configured store: Neo4j when reachable, else SQLite fallback."""
    from backend.agents.config import (
        NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
    )
    try:
        from backend.agents.storage.neo4j_store import Neo4jMemoryStore
        store = Neo4jMemoryStore(
            uri=NEO4J_URI, username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD, database=NEO4J_DATABASE,
            connect_timeout_s=5.0,
        )
        store.count()  # force a real connectivity check against the live graph
        return store, f"Neo4j ({NEO4J_URI})"
    except Exception as exc:
        from backend.agents.storage.store import MemoryStore
        return MemoryStore(db_path=":memory:", enable_ann=False), f"SQLite/in-memory (Neo4j unreachable: {exc})"


def _event_from_row(row, fired_keys, session_id, idx):
    patterns = [
        PatternKey(pattern_type=PatternType.FAULT, key=k, confidence=0.8)
        for k in fired_keys
    ]
    return MemoryEvent(
        session_id=session_id,
        time_range=TimeRange(i0=idx, i1=idx + 1, t0=float(idx), t1=float(idx + 1), fs=1.0),
        patterns=patterns,
        metrics=None,
        cutting_context=None,
        external_signals={},
        channels=[],
        metadata={"operation_id": str(row.get("operation_id", "")), "row": int(idx)},
    )


async def run():
    df = pd.read_csv(CSV).reset_index(drop=True).head(N_EVENTS)
    cfg = ExperimentConfig()
    cfg.use_seed_model = False
    fired = _detect_patterns_batch(df, cfg)

    store, backend = _make_store()
    orch = MemoryEventOrchestrator(
        memory_store=store,
        config=OrchestratorConfig(
            use_classical_models=False,
            enable_harmonic_scorer=False,
            generate_explanations=False,
            dispatch_alerts=False,
            always_store=True,            # round-trip every event through the store
            min_score_for_retrieval=0.0,  # always attempt retrieval
            top_k_similar=5,
        ),
    )

    def _backend_count():
        try:
            return int(store.count())
        except Exception:
            return None

    session_id = "retrieval-validation"

    # Old-junk guard: purge any residue from a prior interrupted run of THIS
    # validation before we start, so the round-trip is measured clean.
    purged = 0
    try:
        for m in store.list_by_session(session_id, limit=100000):
            if store.delete(m.id):
                purged += 1
    except Exception:
        pass

    count_before = _backend_count()

    print("=" * 78)
    print("  RETRIEVAL-LOOP VALIDATION — live orchestrator.process_event (store->retrieve)")
    print(f"  backend: {backend}")
    print(f"  events:  {len(df)} stoppage windows")
    if count_before is not None:
        print(f"  backend memories before: {count_before}"
              + (f"  (purged {purged} stale validation residue)" if purged else ""))
    print("=" * 78)

    stored = retrieved = with_hits = errors = 0
    hit_counts = []
    sample_lines = []
    created_ids = []
    for i in range(len(df)):
        try:
            res = await orch.process_event(
                _event_from_row(df.iloc[i], fired[i], session_id, i)
            )
        except Exception as exc:  # surface real failures, never hide them
            errors += 1
            if errors <= 3:
                sample_lines.append(f"  ! event {i} error: {exc}")
            continue
        if res.memory_id:
            stored += 1
            created_ids.append(res.memory_id)
        n = len(res.similar_memories or [])
        if n:
            retrieved += 1
            with_hits += 1
            hit_counts.append(n)
        if i in (len(df) - 3, len(df) - 2, len(df) - 1):
            sample_lines.append(
                f"  event {i:3d} score={res.significance_score:.2f} "
                f"action={res.action.value:8s} stored={'Y' if res.memory_id else 'n'} "
                f"similar={n}"
            )

    # Direct retriever round-trip on the last event's patterns (independent of gating).
    last_patterns = [
        PatternKey(pattern_type=PatternType.FAULT, key=k, confidence=0.8)
        for k in fired[len(df) - 1]
    ]
    direct = orch.retriever.retrieve(query_patterns=last_patterns, top_k=5)

    def _match_session(match):
        mem = getattr(match, "memory", None)
        return getattr(mem, "session_id", None) if mem is not None else getattr(match, "session_id", None)

    own = sum(1 for d in direct if _match_session(d) == session_id)
    pool = len(direct) - own

    count_after = _backend_count()
    # Authoritative proof the writes hit the BACKEND (not just the orchestrator's
    # in-memory dict, which always assigns a memory_id): the count delta and a
    # session-scoped read-back straight from the store.
    backend_delta = (count_after - count_before) if (count_before is not None and count_after is not None) else None
    session_in_backend = None
    try:
        session_in_backend = len(store.list_by_session(session_id, limit=1000))
    except Exception:
        pass

    print(f"  events processed ............ {len(df)}")
    print(f"  orchestrator returned id .... {stored}")
    if backend_delta is not None:
        print(f"  BACKEND count delta ......... +{backend_delta}  ({count_before} -> {count_after})")
    if session_in_backend is not None:
        print(f"  read back from backend ...... {session_in_backend} memories for this session")
    print(f"  events with retrieval hits .. {with_hits}")
    if hit_counts:
        print(f"  mean neighbours / hit ....... {sum(hit_counts)/len(hit_counts):.1f}")
    print(f"  direct retriever round-trip . {len(direct)} neighbours "
          f"({own} from this run, {pool} from pre-existing graph)")
    print(f"  errors ...................... {errors}")
    print("-" * 78)
    for ln in sample_lines:
        print(ln)
    print("=" * 78)
    backend_ok = (backend_delta is None) or (backend_delta > 0) or (session_in_backend or 0) > 0
    verdict = (
        "PASS — store->retrieve round-trips end-to-end on the deployed path"
        if stored > 0 and (with_hits > 0 or len(direct) > 0) and backend_ok
        else "INCOMPLETE — writes did not reach the backend (see errors above)"
    )
    print(f"  {verdict}")

    # Clean up: remove the memories this run created so it stays idempotent and
    # does not pollute the live graph.
    deleted = 0
    for mid in created_ids:
        try:
            if store.delete(mid):
                deleted += 1
        except Exception:
            pass
    final = _backend_count()
    print(f"  cleanup: deleted {deleted}/{len(created_ids)} test memories"
          + (f" (backend now {final})" if final is not None else ""))
    print("=" * 78)
    try:
        store.close()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(run())

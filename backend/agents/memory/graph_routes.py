"""Graph-surface endpoints for the memory system.

Split out of ``memory/router.py`` (Phase 2): co-occurrence graph, graph stats,
snapshots, scoped clears, the feedback graph and session graph context — plus
the two Neo4j experiment-record endpoints that were interleaved with them and
read the same graph.

Mounted by ``memory/router.py`` via ``include_router`` **before** the catch-all
``/{memory_id}`` route, so ``/memory/graph/...`` keeps resolving here rather
than being swallowed as a memory id.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from .orchestrator import get_orchestrator, get_scorer, get_store

logger = logging.getLogger(__name__)

router = APIRouter()


def _clear_all_graph_enabled() -> bool:
    return os.environ.get("ALLOW_GRAPH_CLEAR_ALL", "").strip().lower() in {"1", "true", "yes"}


@router.get("/graph/co-occurrence")
async def get_co_occurrence_graph():
    """Return the pattern co-occurrence network.

    Tries three sources in order:
    1. Neo4j ``[:CO_OCCURS_WITH]`` edges (live graph database).
    2. The latest experiment results JSON (``co_occurrence_graph`` field).
    3. Empty graph if neither is available.

    Response format::

        {
          "nodes": [{"id": "BREAKAGE_POWER_SPIKE", "weight": 12, "prior": 0.62}, ...],
          "edges": [{"source": "BREAKAGE_POWER_SPIKE", "target": "BREAKAGE_FEED_OVERRIDE_DROP", "weight": 5}, ...],
          "source": "neo4j" | "experiment_json" | "empty"
        }
    """
    scorer = get_scorer()
    store = get_store()

    # --- Source 1: Neo4j live edges ---------------------------------
    store = store
    if hasattr(store, "_run") and hasattr(store, "_driver"):
        try:
            rows = store._run(
                "MATCH (a:Pattern)-[r:CO_OCCURS_WITH]-(b:Pattern) "
                "WHERE a.key < b.key "
                "RETURN a.key AS a, b.key AS b, r.weight AS w "
                "ORDER BY r.weight DESC LIMIT 100"
            )
            if rows:
                node_set: dict = {}
                edges = []
                for row in rows:
                    a_key, b_key, w = str(row["a"]), str(row["b"]), int(row.get("w", 1))
                    node_set[a_key] = node_set.get(a_key, 0) + w
                    node_set[b_key] = node_set.get(b_key, 0) + w
                    edges.append({"source": a_key, "target": b_key, "weight": w})

                # Compute normalized strength for each edge
                for e in edges:
                    max_count = max(node_set.get(e["source"], 1), node_set.get(e["target"], 1))
                    e["strength"] = round(e["weight"] / max(max_count, 1), 3)

                priors = dict(scorer._pattern_priors or {})
                nodes = [
                    {"id": k, "weight": v, "prior": float(priors.get(k, 0.5))}
                    for k, v in sorted(node_set.items())
                ]
                return {"nodes": nodes, "edges": edges, "source": "neo4j"}
        except Exception as exc:
            logger.debug("Neo4j co-occurrence query failed: %s", exc)

    # --- Source 2: experiment results JSON ---------------------------
    import pathlib
    candidates = [
        pathlib.Path(__file__).resolve().parents[3] / "data" / "breakage_patterns" / "stoppage_experiment" / "experiment_results.json",
        pathlib.Path(__file__).resolve().parents[3] / "data" / "breakage_patterns" / "experiment_results.json",
        pathlib.Path("data/breakage_patterns/experiment_results.json"),
    ]
    for p in candidates:
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            # Try multiple JSON shapes the experiment runner may produce
            co_graph = (
                data.get("co_occurrence_graph")
                or data.get("eval_phase", {}).get("co_occurrence_graph")
                or data.get("eval", {}).get("co_occurrence_graph")
                or {}
            )
            # Also merge from per-fold dicts if top-level is empty
            if not co_graph:
                merged: dict = {}
                for fold in data.get("folds", []):
                    if isinstance(fold, dict):
                        fold_co = fold.get("co_occurrence_graph", {})
                        for k, v in fold_co.items():
                            merged[k] = merged.get(k, 0) + int(v)
                co_graph = merged
            if not co_graph:
                continue

            node_set_j: dict = {}
            edges_j = []
            for pair_key, weight in co_graph.items():
                parts = pair_key.split("|")
                if len(parts) != 2:
                    continue
                a, b = parts
                node_set_j[a] = node_set_j.get(a, 0) + int(weight)
                node_set_j[b] = node_set_j.get(b, 0) + int(weight)
                edges_j.append({"source": a, "target": b, "weight": int(weight)})

            # Compute normalized strength for each edge
            for e in edges_j:
                max_count = max(node_set_j.get(e["source"], 1), node_set_j.get(e["target"], 1))
                e["strength"] = round(e["weight"] / max(max_count, 1), 3)

            priors_j = dict(scorer._pattern_priors or {})
            nodes_j = [
                {"id": k, "weight": v, "prior": float(priors_j.get(k, 0.5))}
                for k, v in sorted(node_set_j.items())
            ]
            return {"nodes": nodes_j, "edges": edges_j, "source": "experiment_json"}
        except Exception as exc:
            logger.debug("Failed to read co-occurrence from %s: %s", p, exc)

    # --- Source 3: empty graph --------------------------------------
    return {"nodes": [], "edges": [], "source": "empty"}


@router.get("/graph/co-occurrence/{run_id}")
async def get_experiment_co_occurrence_graph(run_id: str):
    """Return co-occurrence graph scoped to a single experiment run.

    Only includes patterns and edges from sessions that belong to the
    specified experiment.  Requires Neo4j backend with :Experiment nodes.
    """
    store = get_store()
    store = store
    if hasattr(store, "get_experiment_graph"):
        try:
            graph = store.get_experiment_graph(run_id)
            return {**graph, "source": "neo4j", "run_id": run_id}
        except Exception as exc:
            logger.warning("Experiment graph query failed for %s: %s", run_id, exc)
    return {"nodes": [], "edges": [], "source": "empty", "run_id": run_id}


@router.get("/experiments")
async def list_neo4j_experiments():
    """List all experiments stored in Neo4j as :Experiment nodes.

    Response format::

        {
          "experiments": [
            {
              "run_id": "breakage_2026-04-10_0912",
              "experiment_type": "breakage",
              "test_f1": 0.82,
              ...
              "n_sessions": 12,
              "n_memories": 47
            },
            ...
          ],
          "source": "neo4j" | "none"
        }
    """
    store = get_store()
    store = store
    if hasattr(store, "list_experiments"):
        try:
            experiments = store.list_experiments()
            return {"experiments": experiments, "source": "neo4j"}
        except Exception as exc:
            logger.warning("Experiment list query failed: %s", exc)
    return {"experiments": [], "source": "none"}


@router.post("/graph/co-occurrence/decay")
async def decay_co_occurrence(
    max_age_days: int = 30,
    decay_factor: float = 0.5,
    prune_below: int = 1,
):
    """Apply time-based decay to old CO_OCCURS_WITH edges.

    Edges not updated in *max_age_days* have their weight multiplied by
    *decay_factor*.  Edges with weight ≤ *prune_below* are deleted.
    This keeps the co-occurrence graph focused on the current regime.
    """
    store = get_store()
    store = store
    if hasattr(store, "decay_old_co_occurrence"):
        try:
            pruned = store.decay_old_co_occurrence(
                max_age_days=max_age_days,
                decay_factor=decay_factor,
                prune_below=prune_below,
            )
            return {"pruned_edges": pruned, "max_age_days": max_age_days}
        except Exception as exc:
            logger.warning("Co-occurrence decay failed: %s", exc)
            return {"error": str(exc)}
    return {"error": "Neo4j store not available"}


# -----------------------------------------------------------------------
# Graph management endpoints (snapshots, stats, clear, per-experiment delete)
# -----------------------------------------------------------------------


@router.get("/graph/stats")
async def get_graph_stats():
    """Return node and relationship counts for graph management UI."""
    store = get_store()
    store = store
    if hasattr(store, "graph_stats"):
        try:
            return store.graph_stats()
        except Exception as exc:
            logger.warning("Graph stats failed: %s", exc)
            return {"error": str(exc)}
    return {"error": "Neo4j store not available"}


@router.get("/graph/cleanup-preview")
async def preview_memory_graph_cleanup():
    """Preview what a memory-graph-only cleanup would delete.

    This intentionally excludes the document/entity knowledge graph.
    """
    store = get_store()
    store = store
    if hasattr(store, "preview_memory_graph_cleanup"):
        try:
            return store.preview_memory_graph_cleanup()
        except Exception as exc:
            logger.warning("Memory graph cleanup preview failed: %s", exc)
            return {"error": str(exc)}
    return {"error": "Neo4j store not available"}


@router.post("/graph/snapshot")
async def create_snapshot(body: Dict[str, Any] = Body(default={})):
    """Capture a snapshot of the current pattern priors and co-occurrence state.

    Optional body: ``{label?: string, run_id?: string}``
    """
    store = get_store()
    store = store
    if not hasattr(store, "capture_snapshot"):
        raise HTTPException(status_code=501, detail="Neo4j store not available")
    try:
        snap_id = store.capture_snapshot(
            run_id=body.get("run_id"),
            label=body.get("label"),
        )
        return {"snapshot_id": snap_id, "status": "created"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/graph/snapshots")
async def list_snapshots():
    """List all graph snapshots, newest first."""
    store = get_store()
    store = store
    if hasattr(store, "list_snapshots"):
        try:
            return {"snapshots": store.list_snapshots()}
        except Exception as exc:
            logger.warning("List snapshots failed: %s", exc)
            return {"snapshots": [], "error": str(exc)}
    return {"snapshots": [], "source": "none"}


@router.post("/graph/snapshot/{snapshot_id}/restore")
async def restore_snapshot(snapshot_id: str):
    """Restore pattern priors and co-occurrence edges from a snapshot.

    This overwrites Pattern.prior values and replaces CO_OCCURS_WITH edges.
    Memories and feedback are NOT affected.
    """
    scorer = get_scorer()
    store = get_store()
    store = store
    if not hasattr(store, "restore_snapshot"):
        raise HTTPException(status_code=501, detail="Neo4j store not available")
    try:
        result = store.restore_snapshot(snapshot_id)
        # Also refresh in-memory scorer priors
        try:
            if hasattr(scorer, "refresh_priors"):
                scorer.refresh_priors()
        except Exception:
            pass
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/graph/snapshot/{snapshot_id}")
async def delete_snapshot(snapshot_id: str):
    """Delete a single snapshot."""
    store = get_store()
    store = store
    if not hasattr(store, "delete_snapshot"):
        raise HTTPException(status_code=501, detail="Neo4j store not available")
    ok = store.delete_snapshot(snapshot_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return {"deleted": True, "snapshot_id": snapshot_id}


@router.delete("/experiments/{run_id}")
async def delete_experiment(run_id: str):
    """Delete an experiment and all its owned data (sessions, memories, feedback, traces).

    Shared Pattern nodes and co-occurrence edges are left intact.
    """
    store = get_store()
    store = store
    if not hasattr(store, "delete_experiment"):
        raise HTTPException(status_code=501, detail="Neo4j store not available")
    try:
        counts = store.delete_experiment(run_id)
        return {"deleted": True, "run_id": run_id, "counts": counts}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/graph/clear-all", include_in_schema=False)
async def clear_all_graph_data():
    """Delete ALL graph nodes and relationships.

    This is a destructive operation. All memories, patterns, feedback,
    experiments, snapshots, and metadata nodes are removed.
    """
    if not _clear_all_graph_enabled():
        raise HTTPException(
            status_code=403,
            detail="Full graph clear is disabled; set ALLOW_GRAPH_CLEAR_ALL=1 to enable it",
        )

    orchestrator = get_orchestrator()
    store = orchestrator.store

    if hasattr(store, "clear_all"):
        try:
            counts = store.clear_all()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    elif hasattr(store, "clear"):
        store.clear()
        counts = {"cleared": True}
    else:
        raise HTTPException(status_code=501, detail="Store does not support clearing")

    # Also reset in-memory state
    orchestrator.clear_memory_cache()
    if hasattr(orchestrator.scorer, "reset_feedback_state"):
        orchestrator.scorer.reset_feedback_state()
    else:
        orchestrator.scorer._pattern_priors.clear()

    return {"deleted": True, "counts": counts, "priors_reset": True}


@router.delete("/graph/clear-memory")
async def clear_memory_graph_data():
    """Delete only the memory-domain graph from the shared Neo4j database."""
    orchestrator = get_orchestrator()
    store = orchestrator.store

    if hasattr(store, "clear_memory_graph"):
        try:
            counts = store.clear_memory_graph()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    elif hasattr(store, "clear"):
        store.clear()
        counts = {"cleared": True}
    else:
        raise HTTPException(status_code=501, detail="Store does not support memory-graph clearing")

    orchestrator.clear_memory_cache()
    if hasattr(orchestrator.scorer, "reset_feedback_state"):
        orchestrator.scorer.reset_feedback_state()
    else:
        orchestrator.scorer._pattern_priors.clear()

    return {"deleted": True, "scope": "memory_graph", "counts": counts, "priors_reset": True}


@router.delete("/graph/clear-legacy-candidates")
async def clear_legacy_candidate_memory_data():
    """Delete only memories matched by the legacy-candidate cleanup heuristic."""
    orchestrator = get_orchestrator()
    store = orchestrator.store

    if not hasattr(store, "clear_legacy_candidate_memories"):
        raise HTTPException(status_code=501, detail="Store does not support legacy candidate cleanup")

    try:
        counts = store.clear_legacy_candidate_memories()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    orchestrator.clear_memory_cache()
    priors_refreshed = False
    if hasattr(orchestrator.scorer, "refresh_priors"):
        try:
            orchestrator.scorer.refresh_priors()
            priors_refreshed = True
        except Exception:
            logger.warning("Failed to refresh scorer priors after legacy candidate cleanup", exc_info=True)
    elif hasattr(orchestrator.scorer, "_load_priors"):
        try:
            orchestrator.scorer._load_priors()
            priors_refreshed = True
        except Exception:
            logger.warning("Failed to reload scorer priors after legacy candidate cleanup", exc_info=True)

    return {
        "deleted": True,
        "scope": "legacy_candidates",
        "counts": counts,
        "priors_refreshed": priors_refreshed,
    }


@router.get("/graph/feedback")
async def get_feedback_graph():
    """Return the feedback-driven graph showing how operator confirm/dismiss
    has shifted pattern priors over time.

    Unlike the co-occurrence graph (which shows which patterns fire together),
    the feedback graph shows:
    - Each pattern as a node sized by total feedback events
    - confirm/dismiss counts per pattern
    - current prior value vs base prior (0.5)
    - edges between patterns that were confirmed in the same memory

    Response::

        {
          "nodes": [{"id": "PATTERN_KEY", "confirms": 3, "dismisses": 1,
                      "prior": 0.72, "delta": +0.22}],
          "edges": [{"source": "A", "target": "B", "weight": 2, "type": "co_confirmed"}],
          "source": "scorer" | "empty"
        }
    """
    scorer = get_scorer()
    store = get_store()
    scorer = scorer

    priors = dict(scorer._pattern_priors or {})
    diagnostics = scorer.get_feedback_diagnostics() if hasattr(scorer, "get_feedback_diagnostics") else {}

    if not diagnostics and not priors:
        return {"nodes": [], "edges": [], "source": "empty"}

    # Build nodes from feedback counts + priors
    all_keys = sorted(set(diagnostics.keys()) | set(priors.keys()))
    nodes = []
    for k in all_keys:
        diag = diagnostics.get(k, {}) if isinstance(diagnostics, dict) else {}
        c = float(diag.get("confirm_weight_total", 0.0) or 0.0)
        d = float(diag.get("dismiss_weight_total", 0.0) or 0.0)
        p = float(priors.get(k, scorer.get_pattern_prior(k) if hasattr(scorer, "get_pattern_prior") else 0.5))
        severity = diag.get("severity_calibration") if isinstance(diag.get("severity_calibration"), dict) else {}
        nodes.append({
            "id": k,
            "confirms": round(c, 4),
            "dismisses": round(d, 4),
            "total_feedback": round(c + d, 4),
            "effective_weight_total": round(float(diag.get("effective_weight_total", c + d) or (c + d)), 4),
            "passive_outcome_count": int(diag.get("passive_outcome_count", 0) or 0),
            "passive_outcome_weight_total": round(float(diag.get("passive_outcome_weight_total", 0.0) or 0.0), 4),
            "severity_correction_count": int(diag.get("severity_correction_count", 0) or 0),
            "severity_calibration": {
                "average_delta": round(float(severity.get("average_delta", 0.0) or 0.0), 4),
                "weight_total": round(float(severity.get("weight_total", 0.0) or 0.0), 4),
                "targets": {
                    label: round(float(value or 0.0), 4)
                    for label, value in (severity.get("targets") or {}).items()
                },
            },
            "prior": round(p, 4),
            "delta": round(p - 0.5, 4),
        })

    # Build edges: patterns that appeared together in confirmed memories
    edges: list = []
    try:
        store = store
        memories = store.list_memories(limit=200, offset=0)
        from collections import Counter
        co_confirmed: Counter = Counter()
        for mem in memories:
            pkeys = []
            for pk in (mem.get("pattern_keys") or []):
                pkeys.append(pk.get("key") if isinstance(pk, dict) else str(pk))
            # Check if this memory was confirmed
            confirmed = (mem.get("label") == "confirmed"
                         or mem.get("metadata", {}).get("feedback_action") == "confirm")
            if confirmed and len(pkeys) >= 2:
                pkeys_sorted = sorted(set(pkeys))
                for i in range(len(pkeys_sorted)):
                    for j in range(i + 1, len(pkeys_sorted)):
                        co_confirmed[(pkeys_sorted[i], pkeys_sorted[j])] += 1
        for (a, b), w in co_confirmed.most_common(100):
            edges.append({"source": a, "target": b, "weight": w, "type": "co_confirmed"})
    except Exception as exc:
        logger.debug("Failed to build feedback edges: %s", exc)

    return {"nodes": nodes, "edges": edges, "source": "scorer"}


@router.get("/graph/session-context")
async def get_session_graph_context():
    """Indicate whether graph data is from the current session or a prior experiment."""
    store = get_store()
    store = store
    is_neo4j = hasattr(store, "_driver") and hasattr(store, "_run")

    session_count = 0
    memory_count = 0
    try:
        stats = store.get_stats()
        session_count = stats.get("session_count", 0)
        memory_count = stats.get("total_memories", 0)
    except Exception:
        pass

    return {
        "backend": "neo4j" if is_neo4j else "sqlite",
        "session_count": session_count,
        "memory_count": memory_count,
        "note": (
            "Graph data comes from the current Neo4j database (live session data)."
            if is_neo4j
            else "Graph data comes from experiment results JSON or in-session memories (SQLite)."
        ),
    }



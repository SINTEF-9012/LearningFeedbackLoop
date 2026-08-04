"""SINDIT integration router — health proxy, KG state, bridge control."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Literal, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..agents.memory.init import get_store
from ..agents.sindit.runtime_state import build_runtime_asset
from ..agents.storage.neo4j_store import _operation_node_id
from ..session_active_context import build_active_session_context

from .dependencies import get_session_or_404, get_sessions_dict

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sindit"])


class ToolDatasetDecisionBody(BaseModel):
    dataset_id: str
    machine_family: str
    tool_number: int
    status: Literal["pending", "confirmed", "rejected"] = "pending"
    selection_mode: Literal["default", "master", "reference", "runtime", "sindit", "manual"] = "default"
    reference_tool_number: int | None = None
    manual_num_teeth: int | None = None
    updated_by: str | None = None
    notes: str | None = None


# Short-lived cache for the health probe. Endpoints like /sindit/tools are
# polled every few seconds by the UI; without this, each poll re-ran two HTTP
# probes with a 3s timeout, so a slow/down SINDIT stalled every poll. The cache
# collapses bursts of probes into one live check per TTL window.
_HEALTH_CACHE_TTL_S = float(os.environ.get("SINDIT_HEALTH_CACHE_TTL_S", "10.0"))
_health_cache: Optional[Tuple[float, Tuple[bool, bool, str]]] = None
_health_lock = asyncio.Lock()


async def _probe_sindit_and_graphdb_health_uncached() -> tuple[bool, bool, str]:
    import httpx

    sindit_url = os.environ.get("SINDIT_API_URL", "http://localhost:9017")
    graphdb_url = sindit_url.replace(":9017", ":7200")
    sindit_ok = False
    graphdb_ok = False

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            try:
                response = await client.get(f"{sindit_url}/health/live")
                sindit_ok = response.status_code < 500
            except Exception:
                sindit_ok = False
            try:
                response = await client.get(f"{graphdb_url}/rest/repositories")
                graphdb_ok = response.status_code < 400
            except Exception:
                graphdb_ok = False
    except Exception:
        pass

    return sindit_ok, graphdb_ok, sindit_url


async def _probe_sindit_and_graphdb_health(*, force: bool = False) -> tuple[bool, bool, str]:
    """Probe SINDIT/GraphDB health, memoized for ``SINDIT_HEALTH_CACHE_TTL_S``.

    Pass ``force=True`` to bypass the cache (e.g. an explicit "recheck" action).
    A lock serialises concurrent callers so a burst of polls triggers one probe.
    """
    global _health_cache

    now = time.monotonic()
    cached = _health_cache
    if not force and cached is not None and (now - cached[0]) < _HEALTH_CACHE_TTL_S:
        return cached[1]

    async with _health_lock:
        # Re-check under the lock: another caller may have refreshed it.
        cached = _health_cache
        now = time.monotonic()
        if not force and cached is not None and (now - cached[0]) < _HEALTH_CACHE_TTL_S:
            return cached[1]
        result = await _probe_sindit_and_graphdb_health_uncached()
        _health_cache = (time.monotonic(), result)
        return result


async def _maybe_authenticated_sindit_client():
    sindit_ok, _graphdb_ok, sindit_url = await _probe_sindit_and_graphdb_health()
    if not sindit_ok:
        return None, False, f"SINDIT health check failed at {sindit_url}/health/live"

    username = os.environ.get("SINDIT_USERNAME", "sindit")
    password = os.environ.get("SINDIT_PASSWORD", "sindit")
    try:
        from ..agents.sindit.client import SinditClient
    except Exception:
        return None, False, None

    client = SinditClient(base_url=sindit_url)
    await client.__aenter__()
    try:
        ok = await client.authenticate(username, password)
        if not ok:
            await client.close()
            return None, False, None
        return client, True, None
    except Exception as exc:
        logger.debug("SINDIT tool-audit auth/connect failed", exc_info=True)
        await client.close()
        return None, False, str(exc)


def _session_case_dir(session: Dict[str, Any]) -> str | None:
    metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
    source_config = session.get("source_config") if isinstance(session.get("source_config"), dict) else {}
    casedata = metadata.get("casedata") if isinstance(metadata.get("casedata"), dict) else {}
    for value in (casedata.get("case_dir"), source_config.get("case_dir")):
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_sindit_active_flag(node: Dict[str, Any] | None) -> bool | None:
    if not isinstance(node, dict):
        return None
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    for container in (metadata, node):
        if "active" in container:
            value = container.get("active")
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "active"}
            if value is None:
                return None
            return bool(value)
    return None


def _build_runtime_reconciliation_row(
    *,
    session_id: str,
    session: Dict[str, Any],
    neo4j_store: Any,
    sindit_operation_node: Dict[str, Any] | None,
    sindit_available: bool,
) -> Dict[str, Any] | None:
    active_context = build_active_session_context(session)
    if active_context is None:
        return None

    operation_id = str(active_context.get("operation_id") or "").strip() or None
    dataset_id = str(active_context.get("dataset_id") or "").strip() or None
    case_dir = _session_case_dir(session)
    operation_node_id = _operation_node_id(operation_id, dataset_id, case_dir) if operation_id else None
    operation_uri = build_runtime_asset("operation", operation_id)["uri"] if operation_id else None

    neo4j_snapshot = {
        "available": bool(neo4j_store and hasattr(neo4j_store, "get_runtime_identity_snapshot")),
        "operation_present": False,
        "dataset_present": False,
        "operation_dataset_matches": None,
        "operation_node_id": operation_node_id,
        "operation_node": None,
        "dataset_node": None,
    }
    if neo4j_snapshot["available"]:
        snapshot = neo4j_store.get_runtime_identity_snapshot(
            operation_node_id=operation_node_id,
            dataset_id=dataset_id,
        )
        op_node = snapshot.get("operation_node")
        ds_node = snapshot.get("dataset_node")
        neo4j_snapshot.update(
            {
                "operation_present": op_node is not None,
                "dataset_present": ds_node is not None,
                "operation_node": op_node,
                "dataset_node": ds_node,
            }
        )
        if op_node is not None and dataset_id:
            neo4j_snapshot["operation_dataset_matches"] = str(op_node.get("dataset_id") or "") == str(dataset_id)

    sindit_active = _extract_sindit_active_flag(sindit_operation_node)
    sindit_snapshot = {
        "available": bool(sindit_available),
        "operation_uri": operation_uri,
        "operation_present": sindit_operation_node is not None,
        "operation_active": sindit_active,
        "operation_node": sindit_operation_node,
    }

    issues: list[str] = []
    if not operation_id:
        issues.append("missing_operation_id")
    if not neo4j_snapshot["available"]:
        issues.append("neo4j_unavailable")
    else:
        if operation_node_id and not neo4j_snapshot["operation_present"]:
            issues.append("missing_neo4j_operation")
        if dataset_id and not neo4j_snapshot["dataset_present"]:
            issues.append("missing_neo4j_dataset")
        if neo4j_snapshot["operation_dataset_matches"] is False:
            issues.append("neo4j_operation_dataset_mismatch")

    if not sindit_snapshot["available"]:
        issues.append("sindit_unavailable")
    elif operation_uri:
        if not sindit_snapshot["operation_present"]:
            issues.append("missing_sindit_runtime_operation")
        elif sindit_active is False:
            issues.append("inactive_sindit_runtime_operation")

    return {
        "session_id": session_id,
        "active_context": active_context,
        "expected": {
            "operation_uri": operation_uri,
            "operation_node_id": operation_node_id,
            "dataset_id": dataset_id,
            "case_dir": case_dir,
        },
        "neo4j": neo4j_snapshot,
        "sindit": sindit_snapshot,
        "issues": issues,
    }


def _build_runtime_operation_asset(
    session: Dict[str, Any],
    active_context: Dict[str, Any],
) -> Dict[str, Any] | None:
    operation_id = str(active_context.get("operation_id") or "").strip()
    if not operation_id:
        return None

    dataset_id = str(active_context.get("dataset_id") or "").strip() or None
    machine_id = str(active_context.get("machine_id") or "").strip() or None
    machine_family = str(active_context.get("machine_family") or "").strip() or None
    tool_id = str(active_context.get("tool_id") or "").strip() or None
    case_dir = _session_case_dir(session)
    detail = machine_id or dataset_id or machine_family or "active session"
    metadata: Dict[str, Any] = {
        "label": f"Operation {operation_id}",
        "description": f"Runtime operation for {detail}.",
        "dataset_id": dataset_id,
        "case_dir": case_dir,
        "machine_id": machine_id,
        "machine_family": machine_family,
        "tool_id": tool_id,
        "tool_number": active_context.get("tool_number"),
    }
    return build_runtime_asset("operation", operation_id, metadata=metadata, active=True)


def _runtime_operation_update_fields(
    session: Dict[str, Any],
    active_context: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    fields = {
        "label": payload.get("label"),
        "assetDescription": payload.get("assetDescription"),
        "active": True,
        "lastSeenAt": metadata.get("lastSeenAt"),
        "operationId": active_context.get("operation_id"),
        "datasetId": active_context.get("dataset_id"),
        "caseDir": _session_case_dir(session),
        "machineId": active_context.get("machine_id"),
        "machineFamily": active_context.get("machine_family"),
        "toolId": active_context.get("tool_id"),
        "toolNumber": active_context.get("tool_number"),
    }
    return {key: value for key, value in fields.items() if value is not None}


@router.get("/health/sindit")
async def sindit_health_proxy():
    """Proxy health check for SINDIT and GraphDB services.

    The browser cannot reach these services directly due to CORS,
    so the UI calls this backend endpoint instead.
    """
    sindit_ok, graphdb_ok, sindit_url = await _probe_sindit_and_graphdb_health()

    return {
        "sindit": sindit_ok,
        "graphdb": graphdb_ok,
        "sindit_url": sindit_url,
        "sindit_enabled": os.environ.get("SINDIT_ENABLED", "false").lower()
        in ("1", "true", "yes"),
    }


@router.get("/sindit/state")
async def sindit_current_state():
    """Proxy the SINDIT KG state — returns all assets, properties, and relationships."""
    sindit_url = os.environ.get("SINDIT_API_URL", "http://localhost:9017")
    username = os.environ.get("SINDIT_USERNAME", "sindit")
    password = os.environ.get("SINDIT_PASSWORD", "sindit")

    try:
        from ..agents.sindit.client import SinditClient
    except Exception:
        raise HTTPException(status_code=501, detail="SINDIT client unavailable")

    async with SinditClient(base_url=sindit_url) as client:
        ok = await client.authenticate(username, password)
        if not ok:
            raise HTTPException(status_code=502, detail="SINDIT authentication failed")

        asset_type = "urn:samm:sindit.sintef.no:1.0.0#AbstractAsset"
        property_type = "urn:samm:sindit.sintef.no:1.0.0#AbstractAssetProperty"

        # Query each class by type so a large node class (properties) can't crowd
        # a small one (assets) out of a single global page — the previous
        # get_nodes(limit=500) returned only properties and reported 0 assets.
        assets = await client.get_nodes_by_type(asset_type, limit=5000)
        properties = await client.get_nodes_by_type(property_type, limit=5000)
        node_types = await client.get_node_types()

        # Fetch each asset's relationships concurrently but bounded, so a large
        # asset graph doesn't turn into hundreds of serial round-trips.
        sem = asyncio.Semaphore(8)

        async def _rels_for(node: Dict[str, Any]) -> List[Dict[str, Any]]:
            uri = node.get("uri") or node.get("nodeUri", "")
            if not uri:
                return []
            async with sem:
                try:
                    return await client.get_relationships_for_node(uri)
                except Exception:
                    return []

        rel_lists = await asyncio.gather(*[_rels_for(n) for n in assets])
        relationships = [rel for lst in rel_lists for rel in lst]

    return {
        "assets": assets,
        "properties": properties,
        "relationships": relationships,
        "node_types": node_types,
        "total_nodes": len(assets) + len(properties),
    }


@router.get("/sindit/reconciliation/runtime")
async def sindit_runtime_reconciliation(request: Request):
    """Compare live session runtime identity against Neo4j and SINDIT state."""
    sessions = get_sessions_dict(request)
    store = get_store()
    backend = type(store).__name__.lower() if store is not None else "none"

    client = None
    sindit_available = False
    sindit_detail = None
    try:
        client, sindit_available, sindit_detail = await _maybe_authenticated_sindit_client()

        rows = []
        skipped_sessions: list[str] = []
        for session_id, session in sessions.items():
            active_context = build_active_session_context(session)
            if active_context is None:
                skipped_sessions.append(session_id)
                continue

            operation_id = str(active_context.get("operation_id") or "").strip() or None
            operation_uri = build_runtime_asset("operation", operation_id)["uri"] if operation_id else None
            sindit_node = await client.get_node(operation_uri, depth=0) if client is not None and operation_uri else None

            row = _build_runtime_reconciliation_row(
                session_id=session_id,
                session=session,
                neo4j_store=store,
                sindit_operation_node=sindit_node,
                sindit_available=sindit_available,
            )
            if row is not None:
                rows.append(row)

        return {
            "neo4j_backend": backend,
            "neo4j_available": bool(store and hasattr(store, "get_runtime_identity_snapshot")),
            "sindit_available": bool(sindit_available),
            "sindit_detail": sindit_detail,
            "count": len(rows),
            "sessions_with_issues": sum(1 for row in rows if row.get("issues")),
            "skipped_sessions": skipped_sessions,
            "sessions": rows,
        }
    finally:
        if client is not None:
            await client.close()


@router.post("/sindit/reconciliation/runtime/operation/ensure")
async def ensure_sindit_runtime_operation(request: Request, session_id: str | None = None):
    """Ensure SINDIT runtime operation nodes exist and are active for live sessions."""
    sessions = get_sessions_dict(request)
    selected_sessions = (
        {session_id: get_session_or_404(session_id, sessions)}
        if session_id is not None
        else sessions
    )
    store = get_store()

    client = None
    try:
        client, sindit_available, sindit_detail = await _maybe_authenticated_sindit_client()
        if client is None or not sindit_available:
            raise HTTPException(
                status_code=503,
                detail=sindit_detail or "SINDIT authentication failed",
            )

        rows = []
        repaired = 0
        failed = 0
        for current_session_id, session in selected_sessions.items():
            active_context = build_active_session_context(session)
            if active_context is None:
                rows.append(
                    {
                        "session_id": current_session_id,
                        "action": "skipped",
                        "errors": ["missing_active_context"],
                        "reconciliation": None,
                    }
                )
                continue

            payload = _build_runtime_operation_asset(session, active_context)
            if payload is None:
                rows.append(
                    {
                        "session_id": current_session_id,
                        "action": "skipped",
                        "errors": ["missing_operation_id"],
                        "reconciliation": _build_runtime_reconciliation_row(
                            session_id=current_session_id,
                            session=session,
                            neo4j_store=store,
                            sindit_operation_node=None,
                            sindit_available=True,
                        ),
                    }
                )
                continue

            operation_uri = str(payload.get("uri") or "")
            existing = await client.get_node(operation_uri, depth=0)
            errors: list[str] = []
            created = False
            reactivated = False
            was_active = _extract_sindit_active_flag(existing) is True

            if existing is None:
                created = bool(await client.post_asset(payload))
                if not created:
                    errors.append("create_failed")

            if not errors and not was_active:
                fields = _runtime_operation_update_fields(session, active_context, payload)
                reactivated = bool(
                    await client.update_node(operation_uri, fields=fields, overwrite=False)
                )
                if not reactivated:
                    errors.append("activate_failed")

            current = await client.get_node(operation_uri, depth=0)
            reconciliation = _build_runtime_reconciliation_row(
                session_id=current_session_id,
                session=session,
                neo4j_store=store,
                sindit_operation_node=current,
                sindit_available=True,
            )

            if errors:
                action = "failed"
                failed += 1
            elif created:
                action = "created"
                repaired += 1
            elif reactivated:
                action = "reactivated"
                repaired += 1
            else:
                action = "already_active"

            rows.append(
                {
                    "session_id": current_session_id,
                    "action": action,
                    "errors": errors,
                    "reconciliation": reconciliation,
                }
            )

        return {
            "count": len(rows),
            "repaired": repaired,
            "failed": failed,
            "sessions": rows,
        }
    finally:
        if client is not None:
            await client.close()


@router.get("/sindit/bridge/status")
async def sindit_bridge_status(request: Request):
    """Return the current status of the SINDIT live-data bridge."""
    bridge = getattr(request.app.state, "sindit_bridge", None)
    if bridge is None:
        return {
            "available": False,
            "running": False,
            "detail": "SINDIT bridge not configured. Set SINDIT_ENABLED=true in .env and restart.",
        }
    return {"available": True, **bridge.status()}


@router.post("/sindit/bridge/start")
async def sindit_bridge_start(request: Request):
    """Start the SINDIT live-data bridge (PubSub → SINDIT KG)."""
    bridge = getattr(request.app.state, "sindit_bridge", None)
    if bridge is None:
        raise HTTPException(
            status_code=400, detail="SINDIT bridge not configured"
        )
    await bridge.start()
    return {"ok": True, **bridge.status()}


@router.post("/sindit/bridge/stop")
async def sindit_bridge_stop(request: Request):
    """Stop the SINDIT live-data bridge."""
    bridge = getattr(request.app.state, "sindit_bridge", None)
    if bridge is None:
        raise HTTPException(
            status_code=400, detail="SINDIT bridge not configured"
        )
    await bridge.stop()
    return {"ok": True, **bridge.status()}


@router.get("/sindit/experiments")
async def sindit_experiment_graph():
    """Return the full LFL experiment sub-graph from SINDIT KG.

    Returns all ``urn:lfl:*`` nodes (experiments, phases, operations,
    patterns, machine) with their properties **and** the relationships
    between them — ready to render as a force-directed graph.
    """
    sindit_url = os.environ.get("SINDIT_API_URL", "http://localhost:9017")
    sindit_enabled = os.environ.get("SINDIT_ENABLED", "false").lower() in ("1", "true", "yes")
    if not sindit_enabled:
        return {"nodes": [], "edges": [], "experiments": [], "detail": "SINDIT not enabled"}

    try:
        from ..agents.sindit.client import SinditClient
    except Exception:
        raise HTTPException(status_code=501, detail="SINDIT client unavailable")

    username = os.environ.get("SINDIT_USERNAME", "sindit")
    password = os.environ.get("SINDIT_PASSWORD", "sindit")

    async with SinditClient(base_url=sindit_url) as client:
        ok = await client.authenticate(username, password)
        if not ok:
            raise HTTPException(status_code=502, detail="SINDIT authentication failed")

        all_nodes = await client.get_nodes(skip=0, limit=1000)
        lfl_nodes = []
        lfl_uris: set = set()
        experiments = []

        def _endpoint_uri(value: Any) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for key in ("uri", "nodeUri", "target", "source", "id"):
                    candidate = value.get(key)
                    if isinstance(candidate, str):
                        return candidate
            return ""

        for node in all_nodes:
            uri = _endpoint_uri(node.get("uri") or node.get("nodeUri") or node)
            if not uri.startswith("urn:lfl:"):
                continue
            lfl_uris.add(uri)

            label = node.get("label") or uri.split(":")[-1]
            desc = node.get("assetDescription") or node.get("description", "")

            # Determine node kind from URI pattern
            if ":experiment:" in uri and ":test-phase" in uri:
                kind = "test-phase"
            elif ":experiment:" in uri and ":eval-phase" in uri:
                kind = "eval-phase"
            elif ":experiment:" in uri:
                kind = "experiment"
            elif ":operation:" in uri:
                kind = "operation"
            elif ":pattern:" in uri:
                kind = "pattern"
            elif ":asset:" in uri:
                kind = "machine"
            elif ":property:" in uri:
                kind = "sensor"
            else:
                kind = "other"

            # Collect properties
            props = {}
            pname = node.get("propertyName")
            pval = node.get("propertyValue") or node.get("value")
            if pname:
                try:
                    pval = float(pval)
                except (TypeError, ValueError):
                    pass
                props[pname] = pval

            gnode = {
                "uri": uri,
                "label": label,
                "kind": kind,
                "description": desc,
                "properties": props,
            }
            lfl_nodes.append(gnode)

            if kind == "experiment":
                experiments.append(gnode)

        # Collect property nodes *linked* to each asset via relationships
        # (these are SINDIT-style inline properties)
        for gnode in lfl_nodes:
            uri = gnode["uri"]
            rels = await client.get_relationships_for_node(uri)
            for r in rels:
                target = _endpoint_uri(r.get("relationshipTarget") or r.get("target"))
                if target and target not in lfl_uris:
                    # It's a property node — fetch it inline
                    prop_node = await client.get_node(target, depth=0)
                    if prop_node:
                        pname = prop_node.get("propertyName") or prop_node.get("label", "")
                        pval = prop_node.get("propertyValue") or prop_node.get("value")
                        if pname:
                            try:
                                pval = float(pval)
                            except (TypeError, ValueError):
                                pass
                            gnode["properties"][pname] = pval

        # Collect relationships between LFL nodes
        seen_edges: set = set()
        edges = []
        for gnode in lfl_nodes:
            uri = gnode["uri"]
            rels = await client.get_relationships_for_node(uri)
            for r in rels:
                src = _endpoint_uri(r.get("relationshipSource") or r.get("source"))
                tgt = _endpoint_uri(r.get("relationshipTarget") or r.get("target"))
                rtype = r.get("relationshipType") or r.get("type", "")
                # SINDIT may return the relationship type as a list of type URIs
                # (SAMM multi-typing) rather than a single string — take the
                # first, else the display shortener below crashes with 500.
                if isinstance(rtype, (list, tuple)):
                    rtype = str(rtype[0]) if rtype else ""
                else:
                    rtype = str(rtype)
                # Only include edges between known LFL nodes
                if src in lfl_uris and tgt in lfl_uris:
                    edge_key = f"{src}→{tgt}→{rtype}"
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        # Shorten relationship type for display
                        rel_label = rtype.rsplit(":", 1)[-1].rsplit("#", 1)[-1]
                        edges.append({
                            "source": src,
                            "target": tgt,
                            "type": rtype,
                            "label": rel_label,
                        })

    return {
        "nodes": lfl_nodes,
        "edges": edges,
        "experiments": experiments,
        "count": len(experiments),
    }


@router.get("/sindit/tools/summary")
async def sindit_tools_summary(
    session_id: str | None = None,
    machine_id: str | None = None,
    family: str | None = None,
    tool_number: int | None = None,
    only_discrepancies: bool = False,
):
    from ..agents.sindit.tool_audit import collect_tool_audit_payload

    client, sindit_available, _detail = await _maybe_authenticated_sindit_client()
    try:
        payload = await collect_tool_audit_payload(
            client=client,
            session_id=session_id,
            machine_id=machine_id,
            family=family,
            tool_number=tool_number,
            only_discrepancies=only_discrepancies,
        )
    finally:
        if client is not None:
            await client.close()
    summary = dict(payload["summary"])
    summary["total"] = payload["total"]
    summary["sindit_available"] = sindit_available
    return summary


@router.get("/sindit/tools")
async def sindit_tools(
    session_id: str | None = None,
    machine_id: str | None = None,
    family: str | None = None,
    tool_number: int | None = None,
    only_discrepancies: bool = False,
):
    from ..agents.sindit.tool_audit import collect_tool_audit_payload

    client, sindit_available, detail = await _maybe_authenticated_sindit_client()
    try:
        payload = await collect_tool_audit_payload(
            client=client,
            session_id=session_id,
            machine_id=machine_id,
            family=family,
            tool_number=tool_number,
            only_discrepancies=only_discrepancies,
        )
    finally:
        if client is not None:
            await client.close()
    payload["sindit_available"] = sindit_available
    if detail:
        payload["detail"] = detail
    return payload


@router.get("/sindit/tools/datasets")
async def sindit_tool_datasets(
    dataset_id: str | None = None,
):
    from ..agents.sindit.tool_audit import collect_tool_dataset_overview_payload

    client, sindit_available, detail = await _maybe_authenticated_sindit_client()
    try:
        payload = await collect_tool_dataset_overview_payload(
            client=client,
            dataset_id=dataset_id,
        )
    finally:
        if client is not None:
            await client.close()
    payload["sindit_available"] = sindit_available
    if detail:
        payload["detail"] = detail
    return payload


@router.post("/sindit/tools/datasets/decision")
async def save_sindit_tool_dataset_decision(body: ToolDatasetDecisionBody):
    from ..agents.sindit.tool_audit import (
        build_tool_dataset_decision_snapshot,
        collect_tool_dataset_overview_payload,
    )
    from ..agents.processing.tool_dataset_decisions import save_tool_dataset_decision

    resolved_context = None
    resolved_sources = None
    notes = body.notes
    if body.status == "confirmed":
        if body.selection_mode != "manual" and (body.reference_tool_number is not None or body.manual_num_teeth is not None):
            raise HTTPException(status_code=400, detail="Manual tool inputs require selection_mode='manual'")
        if body.manual_num_teeth is not None and body.manual_num_teeth <= 0:
            raise HTTPException(status_code=400, detail="manual_num_teeth must be positive")

        client, _sindit_available, _detail = await _maybe_authenticated_sindit_client()
        try:
            payload = await collect_tool_dataset_overview_payload(
                client=client,
                dataset_id=body.dataset_id,
            )
        finally:
            if client is not None:
                await client.close()

        tool_row = None
        for dataset in payload.get("datasets") or []:
            for item in dataset.get("tools") or []:
                if item.get("machine_family") == body.machine_family and int(item.get("tool_number") or 0) == body.tool_number:
                    tool_row = item
                    break
            if tool_row is not None:
                break
        if tool_row is None:
            raise HTTPException(status_code=404, detail="Dataset tool row not found")

        reference_row = None
        if body.selection_mode == "manual" and body.reference_tool_number is not None:
            for dataset in payload.get("datasets") or []:
                for item in dataset.get("tools") or []:
                    if item.get("machine_family") == body.machine_family and int(item.get("tool_number") or 0) == body.reference_tool_number:
                        reference_row = item
                        break
                if reference_row is not None:
                    break
            if reference_row is None:
                raise HTTPException(status_code=404, detail="Reference tool row not found")

        snapshot = build_tool_dataset_decision_snapshot(
            tool_row,
            body.selection_mode,
            reference_row=reference_row,
            manual_num_teeth=body.manual_num_teeth,
        )
        resolved_context = snapshot["resolved_context"]
        resolved_sources = snapshot["resolved_sources"]
        if notes is None and body.selection_mode == "manual":
            note_bits: list[str] = []
            if body.reference_tool_number is not None:
                note_bits.append(f"Manual profile copied from T{body.reference_tool_number} default profile.")
            if body.manual_num_teeth is not None:
                note_bits.append(f"Operator-provided tooth count z={body.manual_num_teeth}.")
            notes = " ".join(note_bits) or None

    try:
        decision = save_tool_dataset_decision(
            dataset_id=body.dataset_id,
            machine_family=body.machine_family,
            tool_number=body.tool_number,
            status=body.status,
            selection_mode=body.selection_mode,
            reference_tool_number=body.reference_tool_number,
            updated_by=body.updated_by,
            notes=notes,
            resolved_context=resolved_context,
            resolved_sources=resolved_sources,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "decision": decision}


@router.get("/sindit/tools/{machine_family}/{tool_number}")
async def sindit_tool_detail(
    machine_family: str,
    tool_number: int,
    session_id: str | None = None,
    machine_id: str | None = None,
):
    from ..agents.sindit.tool_audit import collect_tool_audit_payload

    client, sindit_available, detail = await _maybe_authenticated_sindit_client()
    try:
        payload = await collect_tool_audit_payload(
            client=client,
            session_id=session_id,
            machine_id=machine_id,
            family=machine_family,
            tool_number=tool_number,
            only_discrepancies=False,
        )
    finally:
        if client is not None:
            await client.close()
    if not payload["items"]:
        raise HTTPException(status_code=404, detail="Tool audit row not found")
    row = dict(payload["items"][0])
    row["sindit_available"] = sindit_available
    if detail:
        row["detail"] = detail
    return row


@router.get("/graph/unified")
async def unified_graph(
    machine_uri: str = "urn:lfl:asset:cnc-machine-1",
    memory_limit: int = 50,
) -> dict:
    """Return a unified graph merging SINDIT (current-state) nodes with
    Neo4j memory (historical) nodes, joined by ``machine_uri``.

    Per plan point 2 (Agent B refined): the two graphs cross-reference by
    stable IDs, not merge. Each returned node has a ``source`` tag
    (``sindit`` | ``memory``) so the UI can colour by source.

    Both sides degrade gracefully: if SINDIT is disabled/unreachable, only
    memory nodes are returned. If the memory store is unavailable, only
    SINDIT nodes are returned.
    """
    memory_limit = max(0, min(int(memory_limit or 0), 500))

    nodes: list[dict] = []
    edges: list[dict] = []
    source_counts = {"sindit": 0, "memory": 0}
    degraded: list[str] = []

    # ── SINDIT side ────────────────────────────────────────────────────
    sindit_enabled = os.environ.get("SINDIT_ENABLED", "false").lower() in (
        "1", "true", "yes"
    )
    if sindit_enabled:
        try:
            sindit_payload = await sindit_experiment_graph()
            for n in sindit_payload.get("nodes", []):
                uri = n.get("uri", "")
                if not uri:
                    continue
                nodes.append({
                    "id": uri,
                    "label": n.get("label") or uri.split(":")[-1],
                    "source": "sindit",
                    "kind": n.get("kind", "other"),
                    "properties": n.get("properties", {}),
                })
                source_counts["sindit"] += 1
            for e in sindit_payload.get("edges", []):
                edges.append({
                    "source": e.get("source"),
                    "target": e.get("target"),
                    "label": e.get("label", ""),
                    "kind": "sindit",
                })
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("graph/unified: sindit side failed: %s", exc)
            degraded.append(f"sindit:{exc}")
    else:
        degraded.append("sindit:disabled")

    # ── Memory side ────────────────────────────────────────────────────
    try:
        from ..agents.memory.orchestrator import get_orchestrator

        orchestrator = get_orchestrator()
        memories = orchestrator.list_memories()[:memory_limit] if memory_limit else []
        for mem in memories:
            mem_id = getattr(mem, "id", None) or getattr(mem, "memory_id", None)
            if not mem_id:
                continue
            mem_id = str(mem_id)
            label = getattr(mem, "label", None) or getattr(mem, "pattern_key", None) or mem_id[:8]
            nodes.append({
                "id": f"memory:{mem_id}",
                "label": str(label),
                "source": "memory",
                "kind": "memory",
                "properties": {
                    "session_id": getattr(mem, "session_id", None),
                    "pattern_type": str(getattr(mem, "pattern_type", "") or ""),
                    "created_at": str(getattr(mem, "created_at", "") or ""),
                },
            })
            source_counts["memory"] += 1
            # Cross-reference edge: memory → machine (if known URI present in nodes).
            mem_machine_uri = getattr(mem, "machine_uri", None) or machine_uri
            if any(n["id"] == mem_machine_uri for n in nodes):
                edges.append({
                    "source": f"memory:{mem_id}",
                    "target": mem_machine_uri,
                    "label": "OBSERVED_ON",
                    "kind": "cross",
                })
    except Exception as exc:
        logger.warning("graph/unified: memory side failed: %s", exc)
        degraded.append(f"memory:{exc}")

    return {
        "nodes": nodes,
        "edges": edges,
        "machine_uri": machine_uri,
        "source_counts": source_counts,
        "degraded": degraded,
    }

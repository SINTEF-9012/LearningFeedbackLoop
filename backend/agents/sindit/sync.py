"""Synchronization utilities for SINDIT -> machine-centric Neo4j asset graph.

Runtime sync is SINDIT 2.0-first and does not depend on legacy export formats.
Legacy import helpers are retained only for offline migration scripts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .graph_manager import MachineAssetGraph
from .schema import StatusCode

logger = logging.getLogger(__name__)


_STATUS_HINTS = {
    "active": StatusCode.ACTIVE,
    "running": StatusCode.ACTIVE,
    "run": StatusCode.ACTIVE,
    "idle": StatusCode.IDLE,
    "standby": StatusCode.IDLE,
    "stopped": StatusCode.STOPPED,
    "stop": StatusCode.STOPPED,
    "maintenance": StatusCode.MAINTENANCE,
    "service": StatusCode.MAINTENANCE,
    "fault": StatusCode.FAULT,
    "alarm": StatusCode.FAULT,
    "error": StatusCode.FAULT,
}


def normalize_machine_id(*candidates: Optional[str]) -> str:
    """Choose and normalize a machine identifier from possible source fields."""
    for value in candidates:
        if not value:
            continue
        s = str(value).strip()
        if not s:
            continue
        s = s.rsplit("/", 1)[-1]
        return s.upper().replace(" ", "-")
    return "UNKNOWN-MACHINE"


def infer_status(value: Any, *, fallback: StatusCode = StatusCode.IDLE) -> StatusCode:
    """Infer a normalized status code from arbitrary source values."""
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    for hint, code in _STATUS_HINTS.items():
        if hint in text:
            return code
    return fallback


async def sync_assets_from_sindit_client(
    *,
    graph: MachineAssetGraph,
    client: Any,
    production_line_id: str = "SINDIT-LINE",
    production_line_name: str = "SINDIT Production Line",
    default_status: StatusCode = StatusCode.IDLE,
) -> Dict[str, int]:
    """Pull machine assets from SINDIT API and upsert into machine graph.

    Prefers SINDIT 2.0 graph node APIs; falls back to assets endpoint when needed.
    """
    stats = {"assets_seen": 0, "machines_upserted": 0, "descriptions_upserted": 0}

    await graph.initialize_schema()
    await graph.ensure_status_catalog()
    await graph.create_production_line(production_line_id, production_line_name)

    assets = await _get_sindit2_assets(client)
    for asset in assets or []:
        stats["assets_seen"] += 1

        iri = asset.get("iri")
        machine_id = normalize_machine_id(
            asset.get("id_short"),
            asset.get("machine_id"),
            asset.get("label"),
            iri,
        )
        name = str(asset.get("caption") or asset.get("label") or machine_id)
        desc = str(
            asset.get("description")
            or asset.get("assetDescription")
            or asset.get("text")
            or ""
        ).strip()

        machine_type = str(asset.get("type") or asset.get("assetType") or "sindit-asset")
        vendor = str(asset.get("vendor") or asset.get("manufacturer") or "")
        location = str(asset.get("location") or asset.get("site") or "")

        status_raw = asset.get("machine_status") or asset.get("status") or asset.get("state")
        status = infer_status(status_raw, fallback=default_status)

        metadata_json = _safe_json_for_machine(asset)

        await graph.upsert_machine(
            machine_id=machine_id,
            name=name,
            machine_type=machine_type,
            vendor=vendor,
            location=location,
            active=True,
            status=status,
            status_reason="SINDIT 2.0 asset sync",
            status_source="sindit-sync",
            metadata_json=metadata_json,
        )
        stats["machines_upserted"] += 1

        if desc:
            ok = await graph.add_machine_description(
                machine_id=machine_id,
                description_text=desc,
                lang="en",
                source="sindit",
            )
            if ok:
                stats["descriptions_upserted"] += 1

        await graph.add_machine_to_line(production_line_id, machine_id)

    return stats


async def _get_sindit2_assets(client: Any) -> List[Dict[str, Any]]:
    """Fetch assets using SINDIT 2.0 APIs without relying on legacy export structure."""
    # Prefer generic graph nodes endpoint when available.
    try:
        nodes = await client.get_nodes(skip=0, limit=2000)
        assets: List[Dict[str, Any]] = []
        for n in nodes or []:
            t = str(n.get("type") or n.get("nodeType") or n.get("node_type") or "")
            labels = str(n.get("labels") or n.get("label") or "")
            if "AbstractAsset" in t or "ASSET" in labels.upper() or "asset" in t.lower():
                assets.append(n)
        if assets:
            return assets
    except Exception as exc:
        logger.debug("SINDIT 2.0 get_nodes asset discovery failed, fallback to get_assets: %s", exc)

    # Standard endpoint fallback (still SINDIT API, not legacy export format).
    try:
        return await client.get_assets()
    except Exception as exc:
        logger.warning("SINDIT asset fetch failed: %s", exc)
        return []


def _safe_json_for_machine(asset: Dict[str, Any]) -> str:
    """Store extra SINDIT fields as JSON for forward compatibility."""
    keep = {
        k: v for k, v in (asset or {}).items()
        if k not in {"caption", "label", "description", "assetDescription", "iri", "id_short"}
    }
    try:
        return json.dumps(keep, ensure_ascii=False)
    except Exception:
        return "{}"


def _iter_legacy_assets(relationships: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """DEPRECATED: moved to ``scripts.legacy_migration.sindit1_loaders``.

    Re-exported here for backward compatibility with existing callers
    (plan point 6, Agent F, Round 25). New code should import from
    ``scripts.legacy_migration.sindit1_loaders`` directly.
    """
    from scripts.legacy_migration.sindit1_loaders import (
        _iter_legacy_assets as _impl,
    )
    return _impl(relationships)


def load_legacy_relationship_export(path: Path) -> List[Dict[str, Any]]:
    """DEPRECATED: moved to ``scripts.legacy_migration.sindit1_loaders``.

    Re-exported here for backward compatibility with existing callers.
    """
    from scripts.legacy_migration.sindit1_loaders import (
        load_legacy_relationship_export as _impl,
    )
    return _impl(path)


async def sync_legacy_export_to_machine_graph(
    *,
    graph: MachineAssetGraph,
    export_path: Path,
    production_line_id: str = "SINDIT1-LINE",
    production_line_name: str = "SINDIT 1.x Export Line",
) -> Dict[str, int]:
    """DEPRECATED: moved to ``scripts.legacy_migration.sindit1_loaders``.

    Re-exported here for backward compatibility with existing callers.
    """
    from scripts.legacy_migration.sindit1_loaders import (
        sync_legacy_export_to_machine_graph as _impl,
    )
    return await _impl(
        graph=graph,
        export_path=export_path,
        production_line_id=production_line_id,
        production_line_name=production_line_name,
    )

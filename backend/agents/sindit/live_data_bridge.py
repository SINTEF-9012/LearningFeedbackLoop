"""
SINDIT Live-Data Bridge
========================
Subscribes to the backend PubSub ``features`` channel and pushes sensor
values into the SINDIT Knowledge Graph as digital-twin state.

On first event the bridge auto-creates:
  * A machine asset  (``urn:lfl:asset:cnc-machine-1``)
  * Sensor properties for every numeric field in the feature payload

Subsequent events update the property values via the SINDIT API.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.events import bus

logger = logging.getLogger(__name__)

# ── SINDIT URN helpers ────────────────────────────────────────────────
_ASSET_URI = "urn:lfl:asset:cnc-machine-1"
_ASSET_LABEL = "CNC Machine 1"
_PROP_PREFIX = "urn:lfl:property:"

_ASSET_TYPE = "urn:samm:sindit.sintef.no:1.0.0#AbstractAsset"
_PROP_TYPE = "urn:samm:sindit.sintef.no:1.0.0#AbstractAssetProperty"

# Fields from feature payloads we map to SINDIT properties.
# DEPRECATED (Agent F, 2026-04-24): the source of truth is now
# ``backend.agents.sindit.asset_catalog.SENSOR_PROPERTY_CATALOG``. This
# alias is kept for backward compatibility with any external callers
# that imported ``_SENSOR_FIELDS`` directly; all in-tree code paths
# should use ``get_sensor_property_meta`` instead.
from .asset_catalog import (
    SENSOR_PROPERTY_CATALOG as _SENSOR_FIELDS,
    get_sensor_property_meta,
)
from .runtime_context import resolve_payload_machine_asset


class SinditBridge:
    """Streams feature events into SINDIT as live property updates."""

    def __init__(
        self,
        sindit_url: str = "http://localhost:9017",
        username: str = "sindit",
        password: str = "sindit",
        throttle_s: float = 2.0,
        graph_sync_enabled: bool = False,
        graph_machine_id: str = "CNC-MACHINE-1",
        graph_line_id: str = "LFL-LINE",
    ):
        self._sindit_url = sindit_url
        self._username = username
        self._password = password
        self._throttle_s = throttle_s
        self._graph_sync_enabled = graph_sync_enabled
        self._graph_machine_id = graph_machine_id
        self._graph_line_id = graph_line_id

        # Runtime state
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._created_assets: set[str] = set()
        self._created_props: set[str] = set()
        self._graph = None
        self._graph_driver = None

        # Stats
        self.events_received: int = 0
        self.values_pushed: int = 0
        self.errors: int = 0
        self.last_push_at: Optional[str] = None
        self.last_asset_uri: str = _ASSET_URI
        self.started_at: Optional[str] = None
        self.graph_updates: int = 0

    # ── Public API ────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return

        if self._graph_sync_enabled:
            await self._init_machine_graph_sync()

        self._running = True
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("SinditBridge started – pushing features → %s", self._sindit_url)

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

        if self._graph_driver is not None:
            try:
                await self._graph_driver.close()
            except Exception:
                logger.exception("SinditBridge: failed to close graph driver")
            self._graph_driver = None

        logger.info("SinditBridge stopped")

    @property
    def is_running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.is_running,
            "events_received": self.events_received,
            "values_pushed": self.values_pushed,
            "errors": self.errors,
            "last_push_at": self.last_push_at,
            "started_at": self.started_at,
            "asset_uri": self.last_asset_uri,
            "asset_count": len(self._created_assets),
            "properties_created": len(self._created_props),
            "graph_sync_enabled": self._graph_sync_enabled,
            "graph_updates": self.graph_updates,
            "graph_machine_id": self._graph_machine_id,
            "graph_line_id": self._graph_line_id,
        }

    async def _init_machine_graph_sync(self) -> None:
        # DEPRECATED (Agent B, 2026-04-24): MachineAssetGraph mirrors SINDIT's
        # current-state view into Neo4j. SINDIT is now the single source of
        # truth for current state; Neo4j is history-only. This code path is
        # retained for backward compatibility with existing deployments but
        # is expected to be removed in a future release.
        # Run `scripts/migrations/drop_machine_asset_graph.cypher` to drop
        # the mirrored nodes from Neo4j.
        logger.warning(
            "SinditBridge: MachineAssetGraph sync is DEPRECATED — SINDIT is the "
            "single source of truth for current state. Run "
            "scripts/migrations/drop_machine_asset_graph.cypher to clean up."
        )
        try:
            from neo4j import AsyncGraphDatabase
            from backend.agents.sindit.graph_manager import MachineAssetGraph
        except Exception as exc:
            logger.warning("SinditBridge: graph sync unavailable (neo4j deps missing): %s", exc)
            self._graph_sync_enabled = False
            return

        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "changeme")
        database = os.environ.get("NEO4J_DATABASE", "neo4j")

        self._graph_driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._graph = MachineAssetGraph(self._graph_driver)
        ok = await self._graph.initialize_schema()
        if not ok:
            logger.warning("SinditBridge: failed to initialize machine graph schema")
            self._graph_sync_enabled = False
            return
        await self._graph.ensure_status_catalog()
        await self._graph.create_production_line(self._graph_line_id, "LFL Production Line")
        logger.info("SinditBridge: machine graph sync enabled (%s, db=%s)", uri, database)

    # ── Internal loop ─────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        from backend.agents.sindit.client import SinditClient

        async with SinditClient(base_url=self._sindit_url) as client:
            ok = await client.authenticate(self._username, self._password)
            if not ok:
                logger.error("SinditBridge: SINDIT auth failed – stopping")
                self._running = False
                return

            queue = bus.subscribe("features")
            logger.info("SinditBridge: subscribed to features channel")
            last_push = 0.0

            try:
                while self._running:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    self.events_received += 1

                    # Throttle
                    now = time.monotonic()
                    if now - last_push < self._throttle_s:
                        continue

                    try:
                        await self._push(client, payload)
                        await self._push_machine_graph(payload)
                        last_push = now
                        self.last_push_at = datetime.now(timezone.utc).isoformat()
                    except Exception:
                        self.errors += 1
                        logger.exception("SinditBridge: push error")
            except asyncio.CancelledError:
                pass
            finally:
                bus.unsubscribe("features", queue)

    # ── Push logic ────────────────────────────────────────────────────

    async def _push(self, client: Any, payload: Dict[str, Any]) -> None:
        """Ensure asset + properties exist, then update values."""
        asset = resolve_payload_machine_asset(
            payload,
            default_asset_uri=_ASSET_URI,
            default_label=_ASSET_LABEL,
        )
        asset_uri = asset["asset_uri"]
        asset_label = asset["label"]
        self.last_asset_uri = asset_uri

        # 1. Create asset on first push for this machine
        if asset_uri not in self._created_assets:
            await self._ensure_asset(client, asset_uri=asset_uri, asset_label=asset_label)

        # 2. Extract numeric fields from payload
        features = payload.get("features") or payload
        if isinstance(features, dict):
            fields = features
        else:
            fields = payload

        ts = datetime.now(timezone.utc).isoformat()
        pushed = 0

        for key, value in fields.items():
            if not isinstance(value, (int, float)):
                continue
            prop_uri = _property_uri(asset_uri, key)

            # Ensure property node exists
            if prop_uri not in self._created_props:
                await self._ensure_property(
                    client,
                    field_name=key,
                    prop_uri=prop_uri,
                    asset_uri=asset_uri,
                )

            # Update value
            ok = await client.update_node(
                node_uri=prop_uri,
                fields={
                    "propertyValue": str(value),
                    "propertyValueTimestamp": ts,
                },
            )
            if ok is not None:
                pushed += 1

        self.values_pushed += pushed

    async def _push_machine_graph(self, payload: Dict[str, Any]) -> None:
        if not self._graph_sync_enabled or self._graph is None:
            return

        features = payload.get("features") or payload
        if isinstance(features, dict):
            fields = features
        else:
            fields = payload

        asset = resolve_payload_machine_asset(
            payload,
            default_asset_uri=_ASSET_URI,
            default_label=self._graph_machine_id,
        )
        machine_id = asset["machine_id"] or self._graph_machine_id

        machine_name = str(fields.get("machine_name") or machine_id).strip() or machine_id
        status_hint = fields.get("machine_status") or fields.get("status")
        status_code = self._infer_status(status_hint, fields)
        active = status_code.value not in ("STOPPED", "MAINTENANCE")

        await self._graph.upsert_machine(
            machine_id=machine_id.upper().replace(" ", "-"),
            name=machine_name,
            machine_type="cnc-machine",
            vendor=str(fields.get("vendor") or ""),
            location=str(fields.get("location") or ""),
            active=active,
            status=status_code,
            status_reason=str(fields.get("status_reason") or "Live feature ingest"),
            status_source="sindit-bridge",
        )
        await self._graph.add_machine_to_line(self._graph_line_id, machine_id.upper().replace(" ", "-"))

        description = fields.get("machine_description")
        if isinstance(description, str) and description.strip():
            await self._graph.add_machine_description(
                machine_id=machine_id.upper().replace(" ", "-"),
                description_text=description.strip(),
                source="sindit-bridge",
            )

        self.graph_updates += 1

    @staticmethod
    def _infer_status(status_hint: Any, fields: Dict[str, Any]):
        from backend.agents.sindit.schema import StatusCode

        raw = str(status_hint or "").strip().lower()
        if raw:
            if any(k in raw for k in ("fault", "alarm", "error")):
                return StatusCode.FAULT
            if any(k in raw for k in ("maint", "service")):
                return StatusCode.MAINTENANCE
            if any(k in raw for k in ("stop", "halt", "off")):
                return StatusCode.STOPPED
            if any(k in raw for k in ("idle", "standby")):
                return StatusCode.IDLE
            if any(k in raw for k in ("run", "active", "cut")):
                return StatusCode.ACTIVE

        spindle = fields.get("spindle_speed")
        try:
            if spindle is not None and float(spindle) > 0:
                return StatusCode.ACTIVE
        except Exception:
            pass
        return StatusCode.IDLE

    async def _ensure_asset(self, client: Any, *, asset_uri: str, asset_label: str) -> None:
        """Create the machine asset in SINDIT if it doesn't exist."""
        existing = await client.get_node(asset_uri, depth=0)
        if existing:
            self._created_assets.add(asset_uri)
            return

        result = await client.post_asset({
            "uri": asset_uri,
            "label": asset_label,
            "assetType": _ASSET_TYPE,
            "assetDescription": (
                "Lathe / CNC machine monitored by the LFL "
                "tool-breakage detection system."
            ),
        })
        if result is not None:
            self._created_assets.add(asset_uri)
            logger.info("SinditBridge: created asset %s", asset_uri)
        else:
            logger.warning("SinditBridge: failed to create asset")

    async def _ensure_property(self, client: Any, *, field_name: str, prop_uri: str, asset_uri: str) -> None:
        """Create a property node and link it to the asset."""
        meta = get_sensor_property_meta(field_name)
        label = meta["label"]
        unit = meta["unit"]

        result = await client.post_property({
            "uri": prop_uri,
            "label": label,
            "propertyName": field_name,
            "propertyValue": "0",
            "propertyUnit": unit,
            "propertyDataType": "float",
            "propertyValueTimestamp": datetime.now(timezone.utc).isoformat(),
            "assetUri": asset_uri,
        })

        if result is not None:
            self._created_props.add(prop_uri)
            logger.info("SinditBridge: created property %s → %s", label, prop_uri)
        else:
            logger.warning("SinditBridge: failed to create property %s", field_name)


def _property_uri(asset_uri: str, field_name: str) -> str:
    asset_token = asset_uri.rsplit(":", 1)[-1]
    return f"{_PROP_PREFIX}{asset_token}:{field_name}"

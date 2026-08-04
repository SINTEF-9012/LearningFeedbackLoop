"""
Monitoring Agent — real-time asset & machine status from SINDIT.

Wraps the async :class:`SinditClient` to answer queries about machine
state, sensor readings, connection health, and asset properties.

When SINDIT is disabled the agent returns a clear error message instead
of failing silently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.agents.config import (
    SINDIT_API_URL,
    SINDIT_ENABLED,
    SINDIT_TIMEOUT_S,
)

logger = logging.getLogger(__name__)

# Keywords used for auto-routing in the router
MONITORING_KEYWORDS = frozenset([
    "status", "state", "machine", "asset", "sensor", "spindle",
    "temperature", "humidity", "connection", "health", "live",
    "current", "reading", "vibration", "rpm", "feed",
])


class MonitoringAgent:
    """Agent that answers real-time asset / machine-status queries.

    Registered as ``"monitoring"`` in the agent router with actions:

    - ``status``       — overall SINDIT health + asset listing
    - ``asset``        — details for a single asset (by IRI or search)
    - ``properties``   — live property readings for an asset
    - ``connections``  — list asset-to-asset connections
    - ``query``        — free-text query routed to the best sub-action
    """

    def __init__(self) -> None:
        self._sindit_client: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not SINDIT_ENABLED:
            logger.info("MonitoringAgent: SINDIT disabled — agent will return stubs.")
            return
        try:
            from backend.agents.sindit.client import SinditClient

            self._sindit_client = SinditClient(
                base_url=SINDIT_API_URL,
                timeout=SINDIT_TIMEOUT_S,
            )
            await self._sindit_client.__aenter__()
            logger.info("MonitoringAgent: SINDIT client connected.")
        except Exception as exc:
            logger.warning("MonitoringAgent: SINDIT client init failed: %s", exc)

    # ------------------------------------------------------------------
    # Router interface
    # ------------------------------------------------------------------

    async def handle_request(
        self,
        session_id: str,
        action: Optional[str],
        args: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not SINDIT_ENABLED or self._sindit_client is None:
            return {
                "error": "SINDIT integration is not enabled. "
                         "Set SINDIT_ENABLED=true and provide SINDIT_API_URL.",
                "sindit_enabled": False,
            }

        action = action or "query"

        if action == "status":
            return await self._status()
        if action == "asset":
            return await self._asset_detail(args)
        if action == "properties":
            return await self._properties(args)
        if action == "connections":
            return await self._connections()
        if action == "query":
            return await self._free_query(args)

        return {"error": f"Unknown monitoring action: {action}"}

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def _status(self) -> Dict[str, Any]:
        """SINDIT health check + high-level asset listing."""
        reachable = await self._sindit_client.health()
        assets = await self._sindit_client.get_assets() if reachable else []
        return {
            "sindit_reachable": reachable,
            "asset_count": len(assets),
            "assets": [
                {"iri": a.get("iri"), "label": a.get("label")}
                for a in assets
            ],
        }

    async def _asset_detail(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return full detail for a single asset."""
        iri = args.get("iri") or args.get("asset_iri")
        query = args.get("query") or args.get("q")

        if iri:
            asset = await self._sindit_client.get_asset(iri)
            if asset:
                props = await self._sindit_client.get_properties(iri)
                return {"asset": asset, "properties": props}
            return {"error": f"Asset not found: {iri}"}

        if query:
            matches = await self._sindit_client.search_assets(query)
            if matches:
                # Return the first match with its properties
                best = matches[0]
                best_iri = best.get("iri")
                props = await self._sindit_client.get_properties(best_iri) if best_iri else []
                return {"asset": best, "properties": props, "other_matches": len(matches) - 1}
            return {"error": f"No assets matching: {query}"}

        return {"error": "Provide 'iri' or 'query' in args."}

    async def _properties(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Live property readings for an asset."""
        iri = args.get("iri") or args.get("asset_iri")
        if not iri:
            return {"error": "Provide 'iri' (asset IRI) in args."}

        props = await self._sindit_client.get_properties(iri)
        # Enrich with latest values
        enriched: List[Dict[str, Any]] = []
        for p in props:
            entry = dict(p)
            prop_iri = p.get("iri")
            if prop_iri:
                entry["latest_value"] = await self._sindit_client.get_latest_value(prop_iri)
            enriched.append(entry)

        return {"asset_iri": iri, "properties": enriched}

    async def _connections(self) -> Dict[str, Any]:
        """List all asset-to-asset connections."""
        conns = await self._sindit_client.get_connections()
        return {"connections": conns, "count": len(conns)}

    async def _free_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Route a free-text query to the most appropriate sub-action."""
        query = (args.get("query") or args.get("q", "")).lower()
        if not query:
            return await self._status()

        # Simple keyword routing
        if any(w in query for w in ("connection", "mqtt", "opc", "link")):
            return await self._connections()
        if any(w in query for w in ("property", "reading", "value", "live", "current")):
            return await self._properties(args)
        # Default: try asset search
        return await self._asset_detail(args)

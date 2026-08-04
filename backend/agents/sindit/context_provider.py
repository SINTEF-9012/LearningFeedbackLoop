"""
SINDIT Context Provider — enriches CuttingContext with live digital-twin data.

Queries the SINDIT knowledge graph for machine assets, tool properties and
live timeseries readings, then maps them onto ``CuttingContext`` fields so
memory events carry real machining state instead of hardcoded defaults.

When SINDIT is disabled or unreachable every method returns ``None`` /
original context unchanged — the rest of the pipeline continues normally.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .client import SinditClient

logger = logging.getLogger(__name__)

# Default cache location for the offline fallback. A failed live lookup tries
# this file before returning None.
_DEFAULT_CACHE_PATH = Path(os.environ.get(
    "LFL_SINDIT_CACHE_PATH",
    str(Path(__file__).resolve().parents[3] / "data" / "sindit_cache.json"),
))
_DEFAULT_TTL_SECONDS = float(os.environ.get("LFL_SINDIT_CACHE_TTL", "30"))
_DEFAULT_DISK_CACHE_MAX_AGE_SECONDS = float(
    os.environ.get("LFL_SINDIT_CACHE_MAX_AGE_S", "3600")
)


class SinditContextProvider:
    """Bridge between the SINDIT digital-twin API and the LFL memory pipeline.

    Construct once at startup (or per-request) and call
    :meth:`enrich_context` to overlay live machine data onto a
    ``CuttingContext`` dict.

    Parameters
    ----------
    client:
        An initialised :class:`SinditClient`.  The caller is responsible for
        lifecycle management (``async with`` or explicit ``close()``).
    machine_asset_iri:
        Optional default asset IRI.  When set, all enrichment calls use this
        asset unless overridden per-call.
    """

    # ---- Well-known SINDIT property local-names → CuttingContext fields ----
    _PROPERTY_MAP: Dict[str, str] = {
        "SpindleSpeed": "spindle_speed",
        "spindle_speed": "spindle_speed",
        "FeedRate": "feed_rate",
        "feed_rate": "feed_rate",
        "ToolID": "tool_id",
        "tool_id": "tool_id",
        "ToolType": "tool_type",
        "tool_type": "tool_type",
        "ToolDiameter": "tool_diameter",
        "tool_diameter": "tool_diameter",
        "NumberOfTeeth": "num_teeth",
        "num_teeth": "num_teeth",
        "ToolLength": "tool_length",
        "tool_length": "tool_length",
        "ToolMaterial": "tool_material",
        "tool_material": "tool_material",
        "Material": "workpiece_material",
        "workpiece_material": "workpiece_material",
        "AxialDepth": "axial_depth",
        "axial_depth": "axial_depth",
        "RadialDepth": "radial_depth",
        "radial_depth": "radial_depth",
        "CuttingSpeed": "cutting_speed",
        "cutting_speed": "cutting_speed",
        "FeedPerTooth": "feed_per_tooth",
        "feed_per_tooth": "feed_per_tooth",
        "MachineState": "machine_state",
        "machine_state": "machine_state",
    }

    def __init__(
        self,
        client: SinditClient,
        machine_asset_iri: Optional[str] = None,
        *,
        cache_ttl_s: float = _DEFAULT_TTL_SECONDS,
        cache_path: Optional[Path] = None,
        disk_cache_max_age_s: float = _DEFAULT_DISK_CACHE_MAX_AGE_SECONDS,
    ):
        self._client = client
        self._machine_iri = machine_asset_iri
        # In-memory TTL cache, keyed by asset IRI.
        # Value: (expires_at_monotonic, enrichment_dict)
        self._ttl_s = max(0.0, float(cache_ttl_s))
        self._memcache: Dict[str, tuple[float, Dict[str, Any]]] = {}
        self._cache_path = Path(cache_path) if cache_path is not None else _DEFAULT_CACHE_PATH
        self._disk_cache_max_age_s = max(0.0, float(disk_cache_max_age_s))
        # SINDIT's KG endpoints (/kg/nodes_by_type, /kg/node) require a bearer
        # token. This client is created at app startup and never explicitly
        # authenticated (unlike the DT routes), so we lazily obtain a token on
        # first use — otherwise every enrichment query returns 401 (ISS-37).
        self._auth_username = os.environ.get("SINDIT_USERNAME", "sindit")
        self._auth_password = os.environ.get("SINDIT_PASSWORD", "sindit")

    async def _ensure_authenticated(self) -> None:
        """Authenticate the enrichment client on first use.

        Cheap no-op once a token is held. If SINDIT is unreachable the
        underlying query still runs (and degrades to the disk-cache path).
        """
        client = self._client
        try:
            # Enrichment runs in a throwaway event loop per event; drop any
            # httpx client bound to a prior (closed) loop so the token-bearing
            # client is rebuilt in the current loop.
            reset = getattr(client, "reset_connection", None)
            if callable(reset):
                reset()
            if getattr(client, "is_authenticated", False):
                return
            await client.authenticate(self._auth_username, self._auth_password)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("SINDIT enrichment auth attempt failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enrich_context(
        self,
        context_dict: Dict[str, Any],
        asset_iri: Optional[str] = None,
        *,
        populate_machine_id: bool = True,
    ) -> Dict[str, Any]:
        """Overlay live SINDIT data onto a CuttingContext-like dict.

        Parameters
        ----------
        context_dict:
            A dict representation of :class:`CuttingContext`.  Missing fields
            are filled in; existing non-``None`` fields are **not** overwritten
            (user/sensor data takes priority).
        asset_iri:
            The machine asset IRI to query.  Falls back to the instance default.
        populate_machine_id:
            When ``True`` (default), missing ``machine_id`` is filled from the
            queried asset's label. Tool-only enrichment should pass ``False`` so
            a tool asset label cannot clobber the real machine id.

        Returns
        -------
        The same ``context_dict`` reference, mutated in-place, with any SINDIT
        values merged in.
        """
        iri = asset_iri or self._machine_iri
        if iri is None:
            logger.debug("No asset IRI — skipping SINDIT enrichment")
            return context_dict

        # 1. Serve from memory cache if fresh.
        cached = self._memcache_get(iri)
        if cached is not None:
            self._apply_enrichment(
                context_dict,
                cached,
                populate_machine_id=populate_machine_id,
            )
            return context_dict

        # 2. Live fetch. On any failure, fall back to disk cache.
        await self._ensure_authenticated()
        try:
            properties = await self._client.get_properties(iri)
        except Exception as exc:
            logger.warning("SINDIT property fetch failed: %s — trying disk cache", exc)
            disk = self._disk_cache_get(iri)
            if disk is not None:
                self._apply_enrichment(
                    context_dict,
                    disk,
                    populate_machine_id=populate_machine_id,
                )
            return context_dict

        enrichment: Dict[str, Any] = {}
        for prop in properties:
            ctx_field = self._resolve_field(prop)
            if ctx_field is None:
                continue

            # Don't overwrite existing values
            if context_dict.get(ctx_field) is not None:
                continue

            # Try to get a live value; fall back to the property's static value
            value = await self._read_value(prop)
            if value is not None:
                enrichment[ctx_field] = value
                context_dict[ctx_field] = value
                logger.debug("SINDIT enriched %s = %s", ctx_field, value)

        # Populate machine_id from asset metadata
        if populate_machine_id and context_dict.get("machine_id") is None:
            try:
                asset = await self._client.get_asset(iri)
            except Exception:
                asset = None
            if asset:
                machine_id = (
                    asset.get("label") or asset.get("iri") or iri
                )
                enrichment["machine_id"] = machine_id
                context_dict["machine_id"] = machine_id

        # 3. Populate caches for next time. Always memcache the result — even an
        #    empty one — so we do NOT re-hit SINDIT on every event for the same
        #    asset (this per-event refetch was the dominant live-lag source,
        #    ISS-34). The TTL still lets later SINDIT imports appear. Only
        #    non-empty enrichments are written to the durable disk cache.
        self._memcache_set(iri, enrichment)
        if enrichment:
            self._disk_cache_set(iri, enrichment)

        return context_dict

    async def enrich_tool_properties(
        self,
        context_dict: Dict[str, Any],
        *,
        tool_iri: str,
    ) -> Dict[str, Any]:
        """Overlay tool-asset properties without touching ``machine_id``."""
        return await self.enrich_context(
            context_dict,
            asset_iri=tool_iri,
            populate_machine_id=False,
        )

    async def get_machine_state(
        self, asset_iri: Optional[str] = None
    ) -> Optional[str]:
        """Return a human-readable machine state string, or ``None``."""
        iri = asset_iri or self._machine_iri
        if iri is None:
            return None
        await self._ensure_authenticated()
        try:
            properties = await self._client.get_properties(iri)
            for prop in properties:
                name = self._prop_local_name(prop)
                if name and name.lower() in ("machinestate", "machine_state", "status"):
                    val = await self._read_value(prop)
                    return str(val) if val is not None else None
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_field(self, prop: Dict[str, Any]) -> Optional[str]:
        """Map a SINDIT property dict to a CuttingContext field name."""
        property_name = prop.get("propertyName")
        if isinstance(property_name, str):
            ctx_field = self._PROPERTY_MAP.get(property_name)
            if ctx_field is not None:
                return ctx_field
        name = self._prop_local_name(prop)
        if name is None:
            return None
        return self._PROPERTY_MAP.get(name)

    @staticmethod
    def _prop_local_name(prop: Dict[str, Any]) -> Optional[str]:
        """Extract the local (short) name from a SINDIT property."""
        # SINDIT properties expose "label" and/or full IRI "iri"
        # SINDIT exposes a machine-readable short name in ``propertyName``;
        # ``label`` is the human display name.
        prop_name = prop.get("propertyName")
        if isinstance(prop_name, str) and prop_name:
            return prop_name
        label = prop.get("label") or prop.get("name")
        if label:
            return label

        iri: Optional[str] = prop.get("uri") or prop.get("iri")
        if iri:
            # Take the fragment or last path segment
            if "#" in iri:
                return iri.rsplit("#", 1)[-1]
            return iri.rsplit("/", 1)[-1]
        return None

    async def _read_value(self, prop: Dict[str, Any]) -> Optional[Any]:
        """Read the latest value for a property, trying timeseries first."""
        prop_iri = prop.get("uri") or prop.get("iri")
        if prop_iri:
            live = await self._client.get_latest_value(prop_iri)
            if live is not None:
                return live
        # Fallback: value embedded in the property node. SINDIT serves this as
        # ``propertyValue`` (older shapes used ``value``/``staticValue``).
        for key in ("propertyValue", "value", "staticValue"):
            if prop.get(key) is not None:
                return prop[key]
        return None

    # ------------------------------------------------------------------
    # Cache helpers (TTL in-memory + JSON disk fallback)
    # ------------------------------------------------------------------

    def _apply_enrichment(
        self,
        context_dict: Dict[str, Any],
        enrichment: Dict[str, Any],
        *,
        populate_machine_id: bool = True,
    ) -> None:
        """Merge cached enrichment into context_dict without overwriting."""
        for field, value in enrichment.items():
            if field == "machine_id" and not populate_machine_id:
                continue
            if context_dict.get(field) is None:
                context_dict[field] = value

    def _memcache_get(self, iri: str) -> Optional[Dict[str, Any]]:
        entry = self._memcache.get(iri)
        if entry is None:
            return None
        expires_at, payload = entry
        if time.monotonic() >= expires_at:
            # Stale: drop it but let caller try live fetch next.
            self._memcache.pop(iri, None)
            return None
        return payload

    def _memcache_set(self, iri: str, enrichment: Dict[str, Any]) -> None:
        if self._ttl_s <= 0:
            return
        self._memcache[iri] = (time.monotonic() + self._ttl_s, dict(enrichment))

    def _disk_cache_get(self, iri: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path
        try:
            if not path.is_file():
                return None
            with path.open("r", encoding="utf-8") as fh:
                blob = json.load(fh)
        except Exception as exc:
            logger.debug("SINDIT disk cache read failed (%s): %s", path, exc)
            return None
        if not isinstance(blob, dict):
            return None
        entry = blob.get(iri)
        if not isinstance(entry, dict):
            return None
        cached_at = entry.get("cached_at")
        if not isinstance(cached_at, (int, float)):
            try:
                cached_at = path.stat().st_mtime
            except Exception:
                cached_at = None
        if (
            self._disk_cache_max_age_s > 0
            and isinstance(cached_at, (int, float))
            and (time.time() - float(cached_at)) > self._disk_cache_max_age_s
        ):
            logger.warning(
                "SINDIT disk cache entry is stale for %s (age=%.0fs, max=%.0fs) — ignoring fallback",
                iri,
                time.time() - float(cached_at),
                self._disk_cache_max_age_s,
            )
            return None
        payload = entry.get("enrichment")
        return payload if isinstance(payload, dict) else None

    def _disk_cache_set(self, iri: str, enrichment: Dict[str, Any]) -> None:
        path = self._cache_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                try:
                    with path.open("r", encoding="utf-8") as fh:
                        blob = json.load(fh)
                    if not isinstance(blob, dict):
                        blob = {}
                except Exception:
                    blob = {}
            else:
                blob = {}
            blob[iri] = {
                "enrichment": dict(enrichment),
                "cached_at": time.time(),
            }
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(blob, fh, indent=2, sort_keys=True)
            tmp.replace(path)
        except Exception as exc:
            logger.debug("SINDIT disk cache write failed (%s): %s", path, exc)

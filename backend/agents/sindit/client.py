"""
Async SINDIT REST API client for the LFL backend.

Wraps SINDIT's knowledge-graph API (port 9017 by default) to query
digital-twin assets, properties, connections and timeseries readings.

All methods are async (httpx) so they can be called from FastAPI handlers
or the memory bridge without blocking the event loop.

Configuration is driven by the module-level constants imported from
``backend.agents.config`` (SINDIT_API_URL, SINDIT_TIMEOUT_S).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# SAMM node-type URIs served by SINDIT (see /kg/node_types). Assets and their
# properties are both nodes; there is no /kg/assets or /kg/properties endpoint.
_ASSET_TYPE_URI = "urn:samm:sindit.sintef.no:1.0.0#AbstractAsset"
_PROPERTY_TYPE_URI = "urn:samm:sindit.sintef.no:1.0.0#AbstractAssetProperty"
_CONNECTION_TYPE_URI = "urn:samm:sindit.sintef.no:1.0.0#AbstractAssetConnection"

# Default page size when the SINDIT API supports pagination.
_DEFAULT_LIMIT = 100

_ASSET_FIELDS = frozenset({
    "class_uri",
    "uri",
    "label",
    "assetProperties",
    "assetDescription",
    "assetType",
})
_PROPERTY_FIELDS = frozenset({
    "class_uri",
    "uri",
    "label",
    "propertyUnit",
    "propertySemanticID",
    "propertyDescription",
    "propertyDataType",
    "propertyValue",
    "propertyName",
    "propertyValueTimestamp",
    "propertyConnection",
})
_RELATIONSHIP_FIELDS = frozenset({
    "class_uri",
    "uri",
    "label",
    "relationshipDescription",
    "relationshipType",
    "relationshipSemanticID",
    "relationshipValue",
    "relationshipUnit",
    "relationshipSource",
    "relationshipTarget",
})


def _uri_ref_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return {"uri": value}
    return value


def _normalize_asset_properties(values: Any) -> Any:
    if not isinstance(values, list):
        return values
    return [_uri_ref_payload(item) if isinstance(item, str) else item for item in values]


class SinditClient:
    """Lightweight async wrapper around the SINDIT REST API.

    Usage::

        async with SinditClient("http://localhost:9017") as client:
            assets = await client.get_assets()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:9017",
        timeout: float = 5.0,
        token: Optional[str] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._token = token
        self._client: Optional[httpx.AsyncClient] = None
        # Stored so an expired bearer token can be transparently refreshed on a
        # 401 (Keycloak tokens are short-lived; the live bridge writes for hours).
        self._username: Optional[str] = None
        self._password: Optional[str] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "SinditClient":
        self._client = self._build_client()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _build_client(self) -> httpx.AsyncClient:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=headers,
            trust_env=False,
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = self._build_client()
        return self._client

    def reset_connection(self) -> None:
        """Drop the cached httpx client without awaiting close.

        Enrichment runs each event in a throwaway ``asyncio.run`` loop; an
        httpx client built in a previous (now-closed) loop cannot be reused.
        Dropping the reference forces :meth:`_ensure_client` to rebuild in the
        current loop — the bearer token in ``self._token`` is preserved, so the
        rebuilt client stays authenticated (ISS-37).
        """
        self._client = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def authenticate(
        self,
        username: str = "sindit",
        password: str = "sindit",
    ) -> bool:
        """Obtain a bearer token via the OAuth2 password flow.

        Returns ``True`` on success.  The token is stored and used for
        all subsequent requests.
        """
        # Retain credentials so ``_post`` can silently re-auth on token expiry.
        self._username = username
        self._password = password
        try:
            client = self._ensure_client()
            resp = await client.post(
                "/token",
                data={"username": username, "password": password},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code == 200:
                token_data = resp.json()
                self._token = token_data.get("access_token", "")
                # Rebuild client with the new token header
                await self.close()
                self._client = self._build_client()
                logger.info("SINDIT auth succeeded for user=%s", username)
                return True
            logger.warning("SINDIT auth failed: %s %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.warning("SINDIT auth error: %s", exc)
        return False

    @property
    def is_authenticated(self) -> bool:
        return bool(self._token)

    # ------------------------------------------------------------------
    # Health / connectivity
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        """Return ``True`` if the SINDIT API is reachable."""
        try:
            resp = await self._ensure_client().get("/health/live")
            return resp.status_code < 500
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Assets
    # ------------------------------------------------------------------

    async def get_assets(self) -> List[Dict[str, Any]]:
        """List all assets in the knowledge graph.

        SINDIT has no ``/kg/assets`` endpoint — assets are nodes of the
        AbstractAsset SAMM type, served via ``/kg/nodes_by_type``. ISS-34 fixed
        the singular ``get_asset``; this plural path was still wrong and 404'd on
        every catalog sync (log spam).
        """
        return await self._get_list(
            "/kg/nodes_by_type",
            params={"type_uri": _ASSET_TYPE_URI, "skip": 0, "limit": 5000},
        )

    async def get_asset(self, asset_iri: str) -> Optional[Dict[str, Any]]:
        """Get a single asset node by IRI.

        SINDIT serves single nodes at ``/kg/node?node_uri=`` — there is no
        ``/kg/assets`` endpoint (an asset is just a node). Using the wrong path
        made this 404 on every enrichment call (ISS-34).
        """
        return await self.get_node(asset_iri, depth=0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    async def get_properties(self, asset_iri: str) -> List[Dict[str, Any]]:
        """List AbstractAssetProperty nodes attached to an asset.

        SINDIT has no ``/kg/properties`` endpoint (ISS-34). Property nodes carry
        their value in ``propertyValue`` and reference their asset via
        ``propertyConnection.uri``, so fetch the property-typed nodes and filter.
        The result is cached by the context provider (per asset, TTL), so the
        one-time full fetch does not repeat per event.
        """
        nodes = await self._get_list(
            "/kg/nodes_by_type",
            params={"type_uri": _PROPERTY_TYPE_URI, "skip": 0, "limit": 5000},
            quiet_statuses={404},
        )
        out: List[Dict[str, Any]] = []
        for node in nodes:
            conn = node.get("propertyConnection")
            conn_uri = conn.get("uri") if isinstance(conn, dict) else None
            if conn_uri == asset_iri:
                out.append(node)
        return out

    async def get_property(self, property_iri: str) -> Optional[Dict[str, Any]]:
        """Get a single property by IRI."""
        return await self._get_one(
            "/kg/properties", params={"iri": property_iri}
        )

    async def get_latest_value(self, property_iri: str) -> Optional[Any]:
        """Get the most recent timeseries value for a property.

        Returns the raw value (float, str, dict) or ``None``.
        """
        try:
            resp = await self._ensure_client().get(
                "/kg/properties/latest",
                params={"iri": property_iri},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("value", data)
        except Exception as exc:
            logger.debug("get_latest_value failed for %s: %s", property_iri, exc)
        return None

    # ------------------------------------------------------------------
    # Connections (asset-to-asset relationships)
    # ------------------------------------------------------------------

    async def get_connections(self) -> List[Dict[str, Any]]:
        """List all connection nodes.

        SINDIT has no ``/kg/connections`` endpoint (ISS-34); connections are
        typed nodes. Fail fast (quiet 404) rather than warn+retry on the hot path.
        """
        return await self._get_list(
            "/kg/nodes_by_type",
            params={"type_uri": _CONNECTION_TYPE_URI, "skip": 0, "limit": 1000},
            quiet_statuses={404},
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_assets(self, query: str) -> List[Dict[str, Any]]:
        """Text search across asset labels / descriptions."""
        return await self._get_list(
            "/kg/search", params={"q": query}
        )

    # ------------------------------------------------------------------
    # Extended graph queries
    # ------------------------------------------------------------------

    async def get_nodes(
        self, skip: int = 0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all nodes in the knowledge graph (paginated)."""
        return await self._get_list(
            "/kg/nodes", params={"skip": skip, "limit": limit}
        )

    async def get_nodes_by_type(
        self, type_uri: str, skip: int = 0, limit: int = 5000
    ) -> List[Dict[str, Any]]:
        """Get all nodes of a given SAMM type (paginated).

        Unlike :meth:`get_nodes`, this filters server-side by type, so a small
        node class (e.g. assets) is not crowded out of a global page by a much
        larger one (e.g. properties).
        """
        return await self._get_list(
            "/kg/nodes_by_type",
            params={"type_uri": type_uri, "skip": skip, "limit": limit},
        )

    async def get_node(
        self, node_uri: str, depth: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Get a single node by URI with optional relationship depth."""
        return await self._get_one(
            "/kg/node", params={"node_uri": node_uri, "depth": depth}
        )

    async def get_node_types(self) -> List[Dict[str, Any]]:
        """List all node types in the knowledge graph."""
        return await self._get_list("/kg/node_types")

    async def get_relationships_for_node(
        self, node_uri: str
    ) -> List[Dict[str, Any]]:
        """List all relationships connected to a node."""
        return await self._get_list(
            "/kg/relationship_by_node", params={"node_uri": node_uri}
        )

    # ------------------------------------------------------------------
    # Write operations (require authentication)
    # ------------------------------------------------------------------

    async def post_asset(
        self, asset_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create or replace an asset in the knowledge graph."""
        return await self._post(
            "/kg/asset",
            json_body=self._normalize_asset_payload(asset_data),
        )

    async def post_property(
        self, property_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create or replace an asset property."""
        payload, parent_uri = self._normalize_property_payload(property_data)
        result = await self._post("/kg/asset_property", json_body=payload)
        if result is None or not parent_uri:
            return result

        link_result = await self.update_node(
            node_uri=parent_uri,
            fields={"assetProperties": [{"uri": payload.get("uri")}]},
            overwrite=False,
        )
        if link_result is None:
            logger.warning(
                "SINDIT property %s was created but could not be linked to %s",
                payload.get("uri"),
                parent_uri,
            )
            return None
        return result

    async def update_node(
        self, node_uri: str, fields: Dict[str, Any], *, overwrite: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Update fields on an existing node (PATCH-like via POST)."""
        body = {"uri": node_uri, **fields}
        return await self._post(
            "/kg/node",
            json_body=body,
            params={"overwrite": "True" if overwrite else "False"},
        )

    async def post_streaming_property(
        self, property_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create or replace a streaming property."""
        return await self._post("/kg/streaming_property", json_body=property_data)

    async def post_timeseries_property(
        self, property_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create or replace a timeseries property."""
        return await self._post("/kg/timeseries_property", json_body=property_data)

    async def post_relationship(
        self, relationship_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create a typed relationship between two assets."""
        return await self._post(
            "/kg/relationship",
            json_body=self._normalize_relationship_payload(relationship_data),
        )

    def _normalize_asset_payload(self, asset_data: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            key: value
            for key, value in asset_data.items()
            if key in _ASSET_FIELDS and value is not None
        }
        if "assetProperties" in payload:
            payload["assetProperties"] = _normalize_asset_properties(payload["assetProperties"])
        return payload

    def _normalize_property_payload(
        self,
        property_data: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Optional[str]]:
        payload = {
            key: value
            for key, value in property_data.items()
            if key in _PROPERTY_FIELDS and value is not None
        }
        if "propertyConnection" in payload:
            payload["propertyConnection"] = _uri_ref_payload(payload["propertyConnection"])
        parent_uri = property_data.get("assetUri") or property_data.get("parentUri")
        return payload, parent_uri

    def _normalize_relationship_payload(
        self,
        relationship_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            key: value
            for key, value in relationship_data.items()
            if key in _RELATIONSHIP_FIELDS and value is not None
        }
        source = payload.get("relationshipSource")
        if source is None:
            source = relationship_data.get("sourceUri")
        target = payload.get("relationshipTarget")
        if target is None:
            target = relationship_data.get("targetUri")
        if source is not None:
            payload["relationshipSource"] = _uri_ref_payload(source)
        if target is not None:
            payload["relationshipTarget"] = _uri_ref_payload(target)
        return payload

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    async def _post(
        self,
        path: str,
        json_body: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        *,
        _retry: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """POST JSON to a SINDIT endpoint. Returns the response body.

        On a 401/403 (expired bearer token) the stored credentials are used to
        re-authenticate once and the request is retried — so the long-running
        live bridge self-heals instead of spamming 401s.
        """
        try:
            resp = await self._ensure_client().post(
                path, json=json_body, params=params
            )
            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                return {"status": "ok", "code": resp.status_code}
        except httpx.HTTPStatusError as exc:
            if (
                exc.response.status_code in (401, 403)
                and _retry
                and self._username is not None
            ):
                if await self.authenticate(self._username, self._password or ""):
                    return await self._post(path, json_body, params=params, _retry=False)
            logger.warning(
                "SINDIT POST %s returned %s: %s",
                path,
                exc.response.status_code,
                exc.response.text[:300],
            )
        except Exception as exc:
            logger.warning("SINDIT POST %s failed: %s", path, exc)
        return None

    async def _get_list(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        quiet_statuses: Optional[set[int]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            resp = await self._ensure_client().get(path, params=params)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            # Some endpoints wrap lists in {"items": [...]}
            if isinstance(data, dict) and "items" in data:
                return data["items"]
            return [data]
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if quiet_statuses and status_code in quiet_statuses:
                logger.debug(
                    "SINDIT %s returned %s for params=%s",
                    path,
                    status_code,
                    params,
                )
            else:
                logger.warning(
                    "SINDIT %s returned %s: %s", path, status_code, exc
                )
        except Exception as exc:
            logger.debug("SINDIT request to %s failed: %s", path, exc)
        return []

    async def _get_one(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            resp = await self._ensure_client().get(path, params=params)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return data
            if isinstance(data, list) and data:
                return data[0]
        except Exception as exc:
            logger.debug("SINDIT GET %s failed: %s", path, exc)
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

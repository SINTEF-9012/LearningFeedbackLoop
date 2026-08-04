from __future__ import annotations

import logging

import httpx
import pytest

from backend.agents.sindit.client import SinditClient


class _Response404:
    status_code = 404

    def raise_for_status(self):
        request = httpx.Request("GET", "http://localhost:9017/kg/nodes_by_type")
        raise httpx.HTTPStatusError("not found", request=request, response=self)

    def json(self):
        return {"detail": "not found"}


class _FakeHttpClient:
    """GET stub that records the path and returns a fixed body (or 404)."""

    def __init__(self, body=None, status=200):
        self.calls = 0
        self.paths: list[str] = []
        self.is_closed = False
        self._body = body if body is not None else []
        self._status = status

    async def get(self, path: str, params=None):
        self.calls += 1
        self.paths.append(path)
        if self._status == 404:
            return _Response404()
        return _Response200(self._body)


@pytest.mark.asyncio
async def test_get_properties_uses_nodes_by_type_and_downgrades_404(caplog):
    # ISS-34: get_properties now queries /kg/nodes_by_type (there is no
    # /kg/properties endpoint) and still fails quietly on 404.
    client = SinditClient()
    fake_http = _FakeHttpClient(status=404)
    client._client = fake_http  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING):
        properties = await client.get_properties("urn:lfl:asset:machine-1")

    assert properties == []
    assert fake_http.paths == ["/kg/nodes_by_type"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_get_properties_filters_by_property_connection():
    # ISS-34: property nodes reference their asset via propertyConnection.uri.
    client = SinditClient()
    body = [
        {"propertyName": "spindle_speed", "propertyValue": 1200,
         "propertyConnection": {"uri": "urn:lfl:asset:machine-1"}},
        {"propertyName": "delta_f1", "propertyValue": 0.03,
         "propertyConnection": {"uri": "urn:lfl:experiment:other"}},
    ]
    client._client = _FakeHttpClient(body=body)  # type: ignore[assignment]

    props = await client.get_properties("urn:lfl:asset:machine-1")
    assert len(props) == 1 and props[0]["propertyName"] == "spindle_speed"


@pytest.mark.asyncio
async def test_get_asset_uses_node_endpoint():
    # ISS-34: get_asset now resolves via /kg/node (there is no /kg/assets).
    client = SinditClient()
    fake_http = _FakeHttpClient(body={"uri": "urn:lfl:asset:machine-1", "label": "Machine 1"})
    client._client = fake_http  # type: ignore[assignment]

    asset = await client.get_asset("urn:lfl:asset:machine-1")
    assert asset and asset.get("label") == "Machine 1"
    assert fake_http.paths == ["/kg/node"]


class _Response200:
    status_code = 200

    def __init__(self, body=None):
        self._body = body or {"ok": True}

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class _RecordingHttpClient:
    def __init__(self):
        self.is_closed = False
        self.posts: list[dict] = []

    async def post(self, path: str, json=None, params=None):
        self.posts.append({"path": path, "json": json, "params": params})
        return _Response200()


@pytest.mark.asyncio
async def test_post_property_normalizes_legacy_payload_and_links_asset():
    client = SinditClient()
    fake_http = _RecordingHttpClient()
    client._client = fake_http  # type: ignore[assignment]

    result = await client.post_property(
        {
            "uri": "urn:lfl:property:machine-1:spindle-speed",
            "label": "Spindle Speed",
            "propertyName": "spindle_speed",
            "propertyValue": "1200",
            "propertyUnit": "rpm",
            "propertyDataType": "float",
            "propertyValueTimestamp": "2026-05-18T10:00:00+00:00",
            "assetUri": "urn:lfl:asset:machine-1",
        }
    )

    assert result == {"ok": True}
    assert fake_http.posts[0] == {
        "path": "/kg/asset_property",
        "json": {
            "uri": "urn:lfl:property:machine-1:spindle-speed",
            "label": "Spindle Speed",
            "propertyName": "spindle_speed",
            "propertyValue": "1200",
            "propertyUnit": "rpm",
            "propertyDataType": "float",
            "propertyValueTimestamp": "2026-05-18T10:00:00+00:00",
        },
        "params": None,
    }
    assert fake_http.posts[1] == {
        "path": "/kg/node",
        "json": {
            "uri": "urn:lfl:asset:machine-1",
            "assetProperties": [{"uri": "urn:lfl:property:machine-1:spindle-speed"}],
        },
        "params": {"overwrite": "False"},
    }


@pytest.mark.asyncio
async def test_post_relationship_normalizes_legacy_source_target_keys():
    client = SinditClient()
    fake_http = _RecordingHttpClient()
    client._client = fake_http  # type: ignore[assignment]

    result = await client.post_relationship(
        {
            "sourceUri": "urn:lfl:asset:machine-1",
            "targetUri": "urn:lfl:tool:t1",
            "relationshipType": "HAS_TOOL",
        }
    )

    assert result == {"ok": True}
    assert fake_http.posts == [
        {
            "path": "/kg/relationship",
            "json": {
                "relationshipType": "HAS_TOOL",
                "relationshipSource": {"uri": "urn:lfl:asset:machine-1"},
                "relationshipTarget": {"uri": "urn:lfl:tool:t1"},
            },
            "params": None,
        }
    ]


@pytest.mark.asyncio
async def test_post_asset_drops_unsupported_legacy_fields():
    client = SinditClient()
    fake_http = _RecordingHttpClient()
    client._client = fake_http  # type: ignore[assignment]

    result = await client.post_asset(
        {
            "uri": "urn:lfl:asset:machine-1",
            "label": "Machine 1",
            "assetType": "urn:samm:sindit.sintef.no:1.0.0#AbstractAsset",
            "assetDescription": "Machine.",
            "lflAssetKind": "Machine",
            "metadata": {"make": "DMG MORI"},
        }
    )

    assert result == {"ok": True}
    assert fake_http.posts == [
        {
            "path": "/kg/asset",
            "json": {
                "uri": "urn:lfl:asset:machine-1",
                "label": "Machine 1",
                "assetType": "urn:samm:sindit.sintef.no:1.0.0#AbstractAsset",
                "assetDescription": "Machine.",
            },
            "params": None,
        }
    ]
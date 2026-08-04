"""Agent B — tests for SinditContextProvider TTL + disk-cache fallback."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from backend.agents.core.context import CuttingContext
from backend.agents.sindit.context_provider import SinditContextProvider


class _FakeClient:
    """Minimal SinditClient stand-in. Counts calls and can raise on demand."""

    def __init__(
        self,
        *,
        properties: Optional[List[Dict[str, Any]]] = None,
        asset: Optional[Dict[str, Any]] = None,
        values: Optional[Dict[str, Any]] = None,
        raise_on_properties: bool = False,
    ):
        self.properties = properties or []
        self.asset = asset
        self.values = values or {}
        self.raise_on_properties = raise_on_properties
        self.get_properties_calls = 0
        self.get_asset_calls = 0

    async def get_properties(self, iri: str):
        self.get_properties_calls += 1
        if self.raise_on_properties:
            raise RuntimeError("simulated SINDIT outage")
        return self.properties

    async def get_asset(self, iri: str):
        self.get_asset_calls += 1
        return self.asset

    async def get_latest_value(self, prop_iri: str):
        return self.values.get(prop_iri)


def _provider(tmp_path: Path, client: _FakeClient, ttl: float = 30.0):
    return SinditContextProvider(
        client=client,  # type: ignore[arg-type]
        machine_asset_iri="urn:test:asset:cnc1",
        cache_ttl_s=ttl,
        cache_path=tmp_path / "sindit_cache.json",
    )


@pytest.mark.asyncio
async def test_enrich_populates_fields_and_fills_cache(tmp_path):
    client = _FakeClient(
        properties=[
            {"iri": "urn:p:spindle", "label": "SpindleSpeed"},
            {"iri": "urn:p:feed", "label": "FeedRate"},
        ],
        asset={"iri": "urn:test:asset:cnc1", "label": "CNC-1"},
        values={"urn:p:spindle": 12000, "urn:p:feed": 850},
    )
    prov = _provider(tmp_path, client)
    ctx: Dict[str, Any] = {}
    out = await prov.enrich_context(ctx)
    assert out["spindle_speed"] == 12000
    assert out["feed_rate"] == 850
    assert out["machine_id"] == "CNC-1"
    # Disk cache written
    cache_path = tmp_path / "sindit_cache.json"
    assert cache_path.is_file()
    blob = json.loads(cache_path.read_text())
    assert "urn:test:asset:cnc1" in blob
    assert blob["urn:test:asset:cnc1"]["enrichment"]["spindle_speed"] == 12000


@pytest.mark.asyncio
async def test_ttl_cache_avoids_second_live_call(tmp_path):
    client = _FakeClient(
        properties=[{"iri": "urn:p:spindle", "label": "SpindleSpeed"}],
        asset={"iri": "urn:test:asset:cnc1", "label": "CNC-1"},
        values={"urn:p:spindle": 9000},
    )
    prov = _provider(tmp_path, client, ttl=30.0)
    await prov.enrich_context({})
    assert client.get_properties_calls == 1
    # Second call should be served from memcache — no new client calls.
    ctx2: Dict[str, Any] = {}
    await prov.enrich_context(ctx2)
    assert client.get_properties_calls == 1
    assert ctx2["spindle_speed"] == 9000


@pytest.mark.asyncio
async def test_disk_fallback_on_outage(tmp_path):
    # Prime the disk cache via one successful call
    good_client = _FakeClient(
        properties=[{"iri": "urn:p:spindle", "label": "SpindleSpeed"}],
        asset={"iri": "urn:test:asset:cnc1", "label": "CNC-1"},
        values={"urn:p:spindle": 7500},
    )
    prov1 = _provider(tmp_path, good_client)
    await prov1.enrich_context({})

    # New provider simulating a process restart with SINDIT down.
    bad_client = _FakeClient(raise_on_properties=True)
    prov2 = _provider(tmp_path, bad_client)
    ctx: Dict[str, Any] = {}
    out = await prov2.enrich_context(ctx)
    # Live call raised → disk cache served the last-known value.
    assert out["spindle_speed"] == 7500


@pytest.mark.asyncio
async def test_outage_without_disk_cache_returns_unchanged(tmp_path):
    bad_client = _FakeClient(raise_on_properties=True)
    prov = _provider(tmp_path, bad_client)
    ctx: Dict[str, Any] = {"existing_field": "preserved"}
    out = await prov.enrich_context(ctx)
    assert out == {"existing_field": "preserved"}


@pytest.mark.asyncio
async def test_existing_values_not_overwritten(tmp_path):
    client = _FakeClient(
        properties=[{"iri": "urn:p:spindle", "label": "SpindleSpeed"}],
        values={"urn:p:spindle": 10000},
    )
    prov = _provider(tmp_path, client)
    ctx: Dict[str, Any] = {"spindle_speed": 5555}
    out = await prov.enrich_context(ctx)
    # User-provided value wins.
    assert out["spindle_speed"] == 5555


def test_cutting_context_tool_fields_default_to_none():
    ctx = CuttingContext()
    assert ctx.tool_length is None
    assert ctx.tool_material is None


def test_resolve_field_prefers_property_name_over_mangled_label(tmp_path):
    prov = _provider(tmp_path, _FakeClient())
    prop = {
        "propertyName": "ToolDiameter",
        "label": "Tooldiameter",
    }
    assert prov._resolve_field(prop) == "tool_diameter"


@pytest.mark.asyncio
async def test_enrich_tool_properties_skips_machine_id_population(tmp_path):
    client = _FakeClient(
        properties=[
            {"iri": "urn:p:length", "propertyName": "ToolLength"},
            {"iri": "urn:p:material", "propertyName": "ToolMaterial"},
        ],
        asset={"iri": "urn:test:tool:t7", "label": "Builder_b12 T7"},
        values={"urn:p:length": 113.0, "urn:p:material": "carbide"},
    )
    prov = _provider(tmp_path, client)
    ctx: Dict[str, Any] = {"machine_id": "CNC-1"}

    out = await prov.enrich_tool_properties(ctx, tool_iri="urn:test:tool:t7")

    assert out["tool_length"] == 113.0
    assert out["tool_material"] == "carbide"
    assert out["machine_id"] == "CNC-1"
    assert client.get_asset_calls == 0


@pytest.mark.asyncio
async def test_empty_tool_property_result_is_negative_cached(tmp_path):
    client = _FakeClient(properties=[])
    prov = _provider(tmp_path, client)

    first: Dict[str, Any] = {}
    second: Dict[str, Any] = {}

    out1 = await prov.enrich_tool_properties(first, tool_iri="urn:lfl:tool:builder_b12-t55")
    out2 = await prov.enrich_tool_properties(second, tool_iri="urn:lfl:tool:builder_b12-t55")

    assert out1 == {}
    assert out2 == {}
    assert client.get_properties_calls == 1


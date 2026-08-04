from __future__ import annotations

import pytest

from backend.agents.sindit.asset_catalog import build_machine_asset
from backend.agents.sindit.live_data_bridge import SinditBridge


class _FakeSinditClient:
    def __init__(self):
        self.nodes = {}
        self.assets = []
        self.properties = []
        self.updates = []

    async def get_node(self, uri: str, depth: int = 0):
        return self.nodes.get(uri)

    async def post_asset(self, payload):
        self.nodes[payload["uri"]] = dict(payload)
        self.assets.append(dict(payload))
        return dict(payload)

    async def post_property(self, payload):
        self.properties.append(dict(payload))
        return dict(payload)

    async def update_node(self, node_uri: str, fields):
        self.updates.append((node_uri, dict(fields)))
        return {"ok": True}


@pytest.mark.asyncio
async def test_sindit_bridge_routes_payloads_to_resolved_machine_assets():
    client = _FakeSinditClient()
    bridge = SinditBridge(throttle_s=0.0)

    site_b_id = "Site_b - MACHINE_B1 - CASE_B1"
    site_c_id = "SITE_C - MACHINE_C1 - CASE_C1"
    site_b_uri = build_machine_asset(site_b_id, label=site_b_id)["uri"]
    site_c_uri = build_machine_asset(site_c_id, label=site_c_id)["uri"]

    await bridge._push(
        client,
        {
            "features": {"spindle_speed": 1200.0},
            "metadata": {
                "source": "simulated_casedata",
                "casedata": {"case_dir": site_b_id},
            },
        },
    )
    await bridge._push(
        client,
        {
            "features": {"spindle_speed": 900.0},
            "metadata": {
                "source": "mqtt",
                "machine_id": site_c_id,
            },
        },
    )

    assert [asset["uri"] for asset in client.assets] == [site_b_uri, site_c_uri]
    assert {prop["assetUri"] for prop in client.properties} == {site_b_uri, site_c_uri}
    assert bridge.status()["asset_uri"] == site_c_uri
    assert bridge.status()["asset_count"] == 2
    assert any(node_uri.startswith("urn:lfl:property:site_b") for node_uri, _fields in client.updates)
    assert any(node_uri.startswith("urn:lfl:property:site_c") for node_uri, _fields in client.updates)
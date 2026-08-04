"""Tests for the /graph/unified endpoint.

Agent B (2026-04-24, Round 24): verifies that the unified graph endpoint
- is reachable (router is included in app),
- degrades gracefully when SINDIT is disabled,
- tags each node with a ``source`` key,
- returns a ``source_counts`` dict.
"""
from __future__ import annotations

import asyncio
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    # Force SINDIT disabled so the test doesn't touch the network.
    monkeypatch.setenv("SINDIT_ENABLED", "false")
    from backend.app import app
    return TestClient(app)


def test_graph_unified_endpoint_exists(client):
    r = client.get("/graph/unified")
    assert r.status_code == 200, r.text


def test_graph_unified_shape(client):
    r = client.get("/graph/unified")
    body = r.json()
    assert "nodes" in body
    assert "edges" in body
    assert "source_counts" in body
    assert "machine_uri" in body
    assert body["source_counts"].keys() == {"sindit", "memory"}


def test_graph_unified_degrades_when_sindit_disabled(client):
    r = client.get("/graph/unified")
    body = r.json()
    # SINDIT disabled → must be surfaced in degraded list and zero sindit nodes.
    assert any(d.startswith("sindit:") for d in body.get("degraded", []))
    assert body["source_counts"]["sindit"] == 0


def test_graph_unified_every_node_tagged(client):
    r = client.get("/graph/unified")
    body = r.json()
    for node in body["nodes"]:
        assert node.get("source") in {"sindit", "memory"}


def test_graph_unified_accepts_custom_machine_uri(client):
    r = client.get("/graph/unified", params={"machine_uri": "urn:lfl:asset:xyz"})
    body = r.json()
    assert body["machine_uri"] == "urn:lfl:asset:xyz"


def test_sindit_router_is_mounted(client):
    # Sanity-check that the previously-orphaned sindit router is now wired.
    r = client.get("/health/sindit")
    assert r.status_code == 200, r.text


def test_sindit_experiment_graph_accepts_dict_relationship_endpoints(monkeypatch):
    monkeypatch.setenv("SINDIT_ENABLED", "true")

    class _FakeSinditClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def authenticate(self, username: str, password: str) -> bool:
            return True

        async def get_nodes(self, skip: int = 0, limit: int = 1000):
            return [
                {"uri": "urn:lfl:asset:machine_a1", "label": "MACHINE_A1"},
                {"uri": "urn:lfl:operation:op-1", "label": "op-1"},
            ]

        async def get_relationships_for_node(self, uri: str):
            if uri == "urn:lfl:asset:machine_a1":
                return [
                    {
                        "relationshipSource": {"uri": "urn:lfl:asset:machine_a1"},
                        "relationshipTarget": {"uri": "urn:lfl:property:temperature"},
                        "relationshipType": "urn:rel:hasProperty",
                    },
                    {
                        "relationshipSource": {"uri": "urn:lfl:asset:machine_a1"},
                        "relationshipTarget": {"uri": "urn:lfl:operation:op-1"},
                        "relationshipType": "urn:rel:feeds",
                    },
                ]
            return []

        async def get_node(self, uri: str, depth: int = 0):
            if uri == "urn:lfl:property:temperature":
                return {"propertyName": "temperature", "propertyValue": "42.0"}
            return None

    monkeypatch.setattr("backend.agents.sindit.client.SinditClient", _FakeSinditClient)

    from backend.routers import sindit as sindit_router

    payload = asyncio.run(sindit_router.sindit_experiment_graph())

    assert {node["uri"] for node in payload["nodes"]} == {
        "urn:lfl:asset:machine_a1",
        "urn:lfl:operation:op-1",
    }
    assert payload["nodes"][0]["properties"]["temperature"] == 42.0
    assert payload["edges"] == [
        {
            "source": "urn:lfl:asset:machine_a1",
            "target": "urn:lfl:operation:op-1",
            "type": "urn:rel:feeds",
            "label": "feeds",
        }
    ]

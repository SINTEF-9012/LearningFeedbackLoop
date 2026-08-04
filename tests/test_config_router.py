"""Tests for backend.routers.config — Agent K."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers.config import router


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    (tmp_path / "pattern_priors.json").write_text(
        json.dumps({"pattern_priors": {"A": 0.8, "B": 0.7}})
    )
    return tmp_path


def test_config_urls_returns_defaults(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ["SINDIT_URL", "GRAPHDB_URL", "INFLUXDB_URL", "NEO4J_URL", "UPSTREAM_KNOWLEDGE_URL", "UI_URL"]:
        monkeypatch.delenv(k, raising=False)
    resp = client.get("/config/urls")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sindit"].startswith("http://localhost:")
    assert body["graphdb"] == "http://localhost:7200"
    assert body["influxdb"] == "http://localhost:8086"
    assert body["neo4j"].startswith("bolt://")
    assert body["upstream_knowledge"] is None


def test_config_urls_respects_env(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SINDIT_URL", "https://sindit.prod.example/")
    monkeypatch.setenv("UPSTREAM_KNOWLEDGE_URL", "https://hub.example/ingest")
    resp = client.get("/config/urls")
    body = resp.json()
    assert body["sindit"] == "https://sindit.prod.example/"
    assert body["upstream_knowledge"] == "https://hub.example/ingest"


def test_config_learnings_defaults_to_local_only(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "MQTT_LEARNINGS_ENABLED",
        "MQTT_LEARNINGS_TOPIC",
        "MQTT_BROKER_HOST",
        "MQTT_BROKER_PORT",
        "MQTT_LEARNINGS_QOS",
        "MQTT_LEARNINGS_RETAIN",
    ]:
        monkeypatch.delenv(key, raising=False)

    resp = client.get("/config/learnings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["mqtt_enabled"] is False
    assert body["mqtt_configured"] is False
    assert body["mqtt_forwarding_active"] is False
    assert body["mqtt_state"] == "disabled"
    assert body["websocket_global"] == "/learnings/ws"


def test_config_learnings_reports_runtime_status(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.mqtt_learning_publisher = SimpleNamespace(
        status=lambda: {
            "state": "connected",
            "task_active": True,
            "published_count": 14,
            "connected_at": 111.0,
            "last_published_at": 222.0,
            "last_error": None,
        }
    )
    client = TestClient(app)

    monkeypatch.setenv("MQTT_LEARNINGS_ENABLED", "true")
    monkeypatch.setenv("MQTT_LEARNINGS_TOPIC", "factory/learnings")
    monkeypatch.setenv("MQTT_BROKER_HOST", "broker.local")
    monkeypatch.setenv("MQTT_BROKER_PORT", "1884")
    monkeypatch.setenv("MQTT_LEARNINGS_QOS", "1")

    resp = client.get("/config/learnings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["mqtt_enabled"] is True
    assert body["mqtt_configured"] is True
    assert body["mqtt_forwarding_active"] is True
    assert body["mqtt_state"] == "connected"
    assert body["mqtt_topic"] == "factory/learnings"
    assert body["mqtt_broker_host"] == "broker.local"
    assert body["mqtt_broker_port"] == 1884
    assert body["published_count"] == 14


def test_knowledge_push_file_sink_writes_pack(
    client: TestClient,
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "packs"
    monkeypatch.setenv("KNOWLEDGE_PACK_DIR", str(out_dir))
    monkeypatch.setenv("KNOWLEDGE_PACK_PREFIX", "test_pack")

    resp = client.post(
        "/knowledge/push",
        json={
            "site": "CNC-7",
            "data_dir": str(data_dir),
            "context": {
                "machine_type": "cnc",
                "tool_type": "endmill",
                "material": "al",
                "regime": "rough",
            },
            "sinks": ["file"],
            "notes": ["hello"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["site"] == "CNC-7"
    assert body["sinks"] == {"file": True}
    assert body["summary"]["priors"] >= 0
    files = list(out_dir.glob("test_pack_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["tenant_id"] == "CNC-7"
    assert payload["signer"] == "knowledge_push"
    assert payload["license"] == "internal-only"
    assert payload["pii_scrub_level"] == "symbolic_only"


def test_knowledge_push_requires_complete_context(
    client: TestClient,
    data_dir: Path,
) -> None:
    resp = client.post(
        "/knowledge/push",
        json={
            "site": "CNC-7",
            "data_dir": str(data_dir),
            "context": {"machine_type": "cnc", "tool_type": "endmill"},
            "sinks": ["file"],
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert "missing_context_keys" in body["detail"]
    assert body["detail"]["missing_context_keys"] == ["material", "regime"]


def test_knowledge_push_mqtt_disabled_returns_false(
    client: TestClient,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MQTT_ENABLED", raising=False)
    resp = client.post(
        "/knowledge/push",
        json={
            "site": "CNC-7",
            "data_dir": str(data_dir),
            "context": {
                "machine_type": "cnc",
                "tool_type": "endmill",
                "material": "al",
                "regime": "rough",
            },
            "sinks": ["mqtt"],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["sinks"] == {"mqtt": False}


def test_knowledge_push_unknown_sink_ignored(
    client: TestClient,
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KNOWLEDGE_PACK_DIR", str(tmp_path / "p"))
    resp = client.post(
        "/knowledge/push",
        json={
            "site": "s",
            "data_dir": str(data_dir),
            "context": {
                "machine_type": "cnc",
                "tool_type": "endmill",
                "material": "al",
                "regime": "rough",
            },
            "sinks": ["file", "nonsense"],
        },
    )
    assert resp.status_code == 200
    # "nonsense" dropped; only "file" result present.
    assert "file" in resp.json()["sinks"]
    assert "nonsense" not in resp.json()["sinks"]


def test_knowledge_push_http_sink_uses_upstream_url(
    client: TestClient,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response()

    monkeypatch.setenv("UPSTREAM_KNOWLEDGE_URL", "https://hub.example/ingest")
    monkeypatch.setattr("backend.agents.knowledge.sinks.urlopen", fake_urlopen)

    resp = client.post(
        "/knowledge/push",
        json={
            "site": "CNC-7",
            "data_dir": str(data_dir),
            "context": {
                "machine_type": "cnc",
                "tool_type": "endmill",
                "material": "al",
                "regime": "rough",
            },
            "sinks": ["http"],
        },
    )

    assert resp.status_code == 200
    assert resp.json()["sinks"] == {"http": True}
    assert captured["url"] == "https://hub.example/ingest"
    assert captured["payload"]["site"] == "CNC-7"

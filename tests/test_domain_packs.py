"""Tests for backend.agents.domain_pack_loader + backend.routers.domain — Agent J."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.agents.domain_config import (
    get_active_domain,
    list_domains,
    reset_active_domain,
)
from backend.agents.domain_pack_loader import (
    DEFAULT_PACK_DIR,
    DomainPackError,
    get_threshold,
    load_pack,
    load_packs,
    register_packs,
)


# ── load_pack ──────────────────────────────────────────────────────────


def test_load_real_cnc_pack() -> None:
    pack = load_pack(DEFAULT_PACK_DIR / "cnc.yaml")
    assert pack.name == "cnc_machining"
    assert pack.display_name == "CNC Machining"
    assert "Spindle_Power" in pack.signature_channels
    assert any(ft.name == "tool_breakage" for ft in pack.fault_types)
    assert get_threshold(pack, "chatter_ratio_threshold", 0.0) == pytest.approx(5.0)


def test_real_cnc_pack_is_data_aligned() -> None:
    """The cnc pack must use real casedata channel names (it overrides the Python default)."""
    pack = load_pack(DEFAULT_PACK_DIR / "cnc.yaml")
    # Canonical casedata channel names appear as signature channels.
    for ch in ("Power_Spindle", "Vibration_Severity_X", "Chatter_Detection_Amplitude_X",
               "Spindle_Speed_Actual", "Feed_Rate_Actual"):
        assert ch in pack.signature_channels
    # Roles map to real channels, with aliases covering SITE_C/Site_b/SITE_A variants.
    assert pack.channel_roles["primary_power"] == "Power_Spindle"
    assert "Spindle_Power" in pack.channel_role_aliases["primary_power"]
    # All four CNC fault types are present.
    assert {"tool_breakage", "chatter", "chip_adhesion", "workpiece_slip"} <= {
        ft.name for ft in pack.fault_types
    }


def test_load_pack_missing_required(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("display_name: no name field\n")
    with pytest.raises(DomainPackError):
        load_pack(bad)


def test_load_pack_invalid_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: [valid: yaml\n")
    with pytest.raises(DomainPackError):
        load_pack(bad)


def test_load_pack_fault_missing_required(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: x\n"
        "fault_types:\n"
        "  - name: f1\n"
        "    severity: 0.5\n"  # missing pattern_key
    )
    with pytest.raises(DomainPackError):
        load_pack(bad)


def test_load_packs_tolerates_missing_dir(tmp_path: Path) -> None:
    result = load_packs(tmp_path / "nonexistent")
    assert result == {}


def test_load_packs_skips_bad_files(tmp_path: Path) -> None:
    (tmp_path / "good.yaml").write_text("name: good_one\n")
    (tmp_path / "bad.yaml").write_text("display_name: no_name\n")
    result = load_packs(tmp_path)
    assert "good_one" in result
    assert len(result) == 1


def test_get_threshold_fallback() -> None:
    pack = load_pack(DEFAULT_PACK_DIR / "cnc.yaml")
    assert get_threshold(pack, "nonexistent_key", 1.5) == pytest.approx(1.5)


def test_register_packs_returns_names(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("name: alpha\n")
    (tmp_path / "b.yaml").write_text("name: beta\n")
    names = register_packs(tmp_path)
    assert names == ["alpha", "beta"]
    assert "alpha" in list_domains()
    assert "beta" in list_domains()


# ── /domain router ─────────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    # Import module so its top-level register_packs runs at least once.
    from backend.routers.domain import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_domain_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DOMAIN_PROFILE", raising=False)
    reset_active_domain()
    yield
    monkeypatch.delenv("DOMAIN_PROFILE", raising=False)
    reset_active_domain()


def test_domain_active_returns_current(client: TestClient) -> None:
    resp = client.get("/domain/active")
    assert resp.status_code == 200
    body = resp.json()
    assert "name" in body
    assert "fault_types" in body
    assert "thresholds" in body
    # Every fault type has an indicator_count.
    for ft in body["fault_types"]:
        assert "indicator_count" in ft


def test_domain_list_includes_yaml_packs(client: TestClient) -> None:
    resp = client.get("/domain/list")
    body = resp.json()
    # cnc.yaml was registered at router import time (overriding the Python default).
    assert "cnc_machining" in body["domains"]
    # Built-in Python domains remain available.
    assert "generic" in body["domains"]


def test_domain_set_active_switches(client: TestClient) -> None:
    # cnc_machining is now YAML-backed (cnc.yaml overrides the Python default).
    resp = client.post("/domain/active", json={"name": "cnc_machining"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "cnc_machining"
    assert body["source"] == "yaml"
    assert body["thresholds"]["chatter_ratio_threshold"] == pytest.approx(5.0)
    # Verify backend state reflects the switch.
    assert get_active_domain().name == "cnc_machining"


def test_domain_set_active_rejects_unknown(client: TestClient) -> None:
    resp = client.post("/domain/active", json={"name": "galaxy_brain"})
    assert resp.status_code == 404

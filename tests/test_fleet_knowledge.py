from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.agents.knowledge import (
    FleetFamilyReview,
    KnowledgePack,
    aggregate_fleet_packs,
    apply_family_reviews,
)
from backend.routers.fleet_knowledge import router


def _pack(
    *,
    site: str,
    tenant_id: str,
    prior: float,
    discovery_key: str = "discovered:cnc_endmill_alarm",
    discovery_features: dict[str, str] | None = None,
    evidence_count: int | None = None,
) -> KnowledgePack:
    features = discovery_features or {
        "power_spindle_mean": "high",
        "vibration_rms_mean": "high",
    }
    pattern_priors = {"pattern_priors": {"CUSTOM:test": prior}}
    if evidence_count is not None:
        pattern_priors["pattern_evidence_counts"] = {"CUSTOM:test": evidence_count}
    return KnowledgePack(
        site=site,
        tenant_id=tenant_id,
        context={
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
        },
        pattern_priors=pattern_priors,
        discovered_patterns={
            "patterns": {
                discovery_key: {
                    "key": discovery_key,
                    "features": features,
                    "promoted": True,
                    "prior": prior,
                }
            }
        },
    )


def test_aggregate_fleet_packs_enforces_k_anonymity() -> None:
    result = aggregate_fleet_packs(
        [_pack(site="site-a", tenant_id="tenant-a", prior=0.9)],
        {
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
        },
        min_sites=2,
    )

    assert result.k_anonymity_met is False
    assert result.pattern_priors == {}
    assert result.discovered_patterns == {}


def test_aggregate_fleet_packs_averages_exact_context_priors_and_discoveries() -> None:
    result = aggregate_fleet_packs(
        [
            _pack(site="site-a", tenant_id="tenant-a", prior=0.9),
            _pack(site="site-b", tenant_id="tenant-b", prior=0.7),
            KnowledgePack(
                site="site-c",
                tenant_id="tenant-c",
                context={
                    "machine_type": "cnc",
                    "tool_type": "drill",
                    "material": "al",
                    "regime": "rough",
                },
                pattern_priors={"pattern_priors": {"CUSTOM:test": 0.99}},
                discovered_patterns={"patterns": {}},
            ),
        ],
        {
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
        },
        min_sites=2,
    )

    assert result.k_anonymity_met is True
    assert result.pack_count == 2
    assert result.site_count == 2
    assert result.pattern_priors["CUSTOM:test"]["prior"] == 0.8
    assert result.pattern_priors["CUSTOM:test"]["site_count"] == 2
    assert result.discovered_patterns["patterns"]["discovered:cnc_endmill_alarm"]["site_count"] == 2


def test_aggregate_fleet_packs_weights_priors_by_exported_evidence() -> None:
    result = aggregate_fleet_packs(
        [
            _pack(site="site-a", tenant_id="tenant-a", prior=0.9, evidence_count=9),
            _pack(site="site-b", tenant_id="tenant-b", prior=0.5, evidence_count=1),
        ],
        {
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
        },
        min_sites=2,
    )

    assert result.pattern_priors["CUSTOM:test"]["prior"] == 0.86
    assert result.pattern_priors["CUSTOM:test"]["evidence_count"] == 10


def test_aggregate_fleet_packs_filters_individual_artifacts_below_threshold() -> None:
    result = aggregate_fleet_packs(
        [
            _pack(site="site-a", tenant_id="tenant-a", prior=0.9, discovery_key="discovered:shared"),
            _pack(site="site-b", tenant_id="tenant-b", prior=0.7, discovery_key="discovered:shared"),
            KnowledgePack(
                site="site-c",
                tenant_id="tenant-c",
                context={
                    "machine_type": "cnc",
                    "tool_type": "endmill",
                    "material": "al",
                    "regime": "rough",
                },
                pattern_priors={"pattern_priors": {"CUSTOM:rare": 0.95}},
                discovered_patterns={
                    "patterns": {
                        "discovered:rare": {"key": "discovered:rare", "promoted": True, "prior": 0.95}
                    }
                },
            ),
        ],
        {
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
        },
        min_sites=2,
    )

    assert "CUSTOM:rare" not in result.pattern_priors
    assert "discovered:rare" not in result.discovered_patterns["patterns"]
    assert "discovered:shared" in result.discovered_patterns["patterns"]


def test_aggregate_fleet_packs_clusters_near_equivalent_discoveries_into_families() -> None:
    result = aggregate_fleet_packs(
        [
            _pack(
                site="site-a",
                tenant_id="tenant-a",
                prior=0.9,
                discovery_key="discovered:spindle-load-a",
                discovery_features={
                    "power_spindle_mean": "high",
                    "vibration_rms_mean": "high",
                },
            ),
            _pack(
                site="site-b",
                tenant_id="tenant-b",
                prior=0.7,
                discovery_key="discovered:spindle-load-b",
                discovery_features={
                    "power_spindle_mean": "high",
                    "vibration_rms_mean": "high",
                    "acoustic_rms_mean": "high",
                },
            ),
        ],
        {
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
        },
        min_sites=2,
    )

    assert "discovered:spindle-load-a" not in result.discovered_patterns.get("patterns", {})
    assert "discovered:spindle-load-b" not in result.discovered_patterns.get("patterns", {})
    assert len(result.discovery_families) == 1
    family = result.discovery_families[0]
    assert family["site_count"] == 2
    assert family["pattern_count"] == 2
    assert family["source"] == "discovered"
    assert family["status"] == "candidate"
    assert family["member_keys"] == ["discovered:spindle-load-a", "discovered:spindle-load-b"]
    assert family["representative_features"] == {
        "power_spindle_mean": "high",
        "vibration_rms_mean": "high",
    }


def test_apply_family_reviews_promotes_matching_family() -> None:
    aggregate = aggregate_fleet_packs(
        [
            _pack(
                site="site-a",
                tenant_id="tenant-a",
                prior=0.9,
                discovery_key="discovered:spindle-load-a",
                discovery_features={
                    "power_spindle_mean": "high",
                    "vibration_rms_mean": "high",
                },
            ),
            _pack(
                site="site-b",
                tenant_id="tenant-b",
                prior=0.7,
                discovery_key="discovered:spindle-load-b",
                discovery_features={
                    "power_spindle_mean": "high",
                    "vibration_rms_mean": "high",
                    "acoustic_rms_mean": "high",
                },
            ),
        ],
        {
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
        },
        min_sites=2,
    )
    family_key = aggregate.discovery_families[0]["family_key"]

    reviewed = apply_family_reviews(
        aggregate,
        [
            FleetFamilyReview(
                family_key=family_key,
                context={
                    "machine_type": "cnc",
                    "tool_type": "endmill",
                    "material": "al",
                    "regime": "rough",
                },
                canonical_name="SPINDLE_LOAD_RAMP",
                status="promoted",
                reviewer="fleet-qa",
                reason="Consistent spindle-load pattern across sites",
            )
        ],
    )

    family = reviewed.discovery_families[0]
    assert family["canonical_name"] == "SPINDLE_LOAD_RAMP"
    assert family["status"] == "promoted"
    assert family["review"]["reviewer"] == "fleet-qa"


def test_fleet_router_persists_family_review_and_applies_it(tmp_path: Path, monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    store_path = tmp_path / "fleet_packs.jsonl"
    review_path = tmp_path / "fleet_family_reviews.jsonl"
    monkeypatch.setenv("FLEET_KNOWLEDGE_STORE_PATH", str(store_path))
    monkeypatch.setenv("FLEET_KNOWLEDGE_REVIEWS_PATH", str(review_path))

    client.post("/fleet/pack", json=_pack(site="site-a", tenant_id="tenant-a", prior=0.9).to_dict())
    client.post("/fleet/pack", json=_pack(site="site-b", tenant_id="tenant-b", prior=0.7).to_dict())

    family_resp = client.get(
        "/fleet/pack",
        params={
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
            "min_sites": 2,
        },
    )
    family_key = family_resp.json()["discovery_families"][0]["family_key"]

    review_resp = client.post(
        "/fleet/family/review",
        json={
            "family_key": family_key,
            "canonical_name": "SPINDLE_LOAD_RAMP",
            "status": "promoted",
            "reviewer": "fleet-qa",
            "reason": "Consistent signature across two endmill sites",
            "context": {
                "machine_type": "cnc",
                "tool_type": "endmill",
                "material": "al",
                "regime": "rough",
            },
        },
    )

    assert review_resp.status_code == 200
    assert review_resp.json()["canonical_name"] == "SPINDLE_LOAD_RAMP"

    get_resp = client.get(
        "/fleet/pack",
        params={
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
            "min_sites": 2,
        },
    )

    assert get_resp.status_code == 200
    family = get_resp.json()["discovery_families"][0]
    assert family["canonical_name"] == "SPINDLE_LOAD_RAMP"
    assert family["status"] == "promoted"
    assert family["review"]["reason"] == "Consistent signature across two endmill sites"
    assert review_path.exists()


def test_fleet_router_rejects_family_review_without_complete_context(tmp_path: Path, monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    monkeypatch.setenv("FLEET_KNOWLEDGE_REVIEWS_PATH", str(tmp_path / "fleet_family_reviews.jsonl"))

    resp = client.post(
        "/fleet/family/review",
        json={
            "family_key": "family:test",
            "canonical_name": "TEST",
            "status": "promoted",
            "context": {"machine_type": "cnc", "tool_type": "endmill"},
        },
    )

    assert resp.status_code == 422
    assert resp.json()["detail"]["missing_context_keys"] == ["material", "regime"]


def test_fleet_router_accepts_pack_and_returns_aggregated_context(tmp_path: Path, monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    store_path = tmp_path / "fleet_packs.jsonl"
    monkeypatch.setenv("FLEET_KNOWLEDGE_STORE_PATH", str(store_path))

    payload_a = _pack(site="site-a", tenant_id="tenant-a", prior=0.9).to_dict()
    payload_b = _pack(site="site-b", tenant_id="tenant-b", prior=0.7).to_dict()

    resp_a = client.post("/fleet/pack", json=payload_a)
    resp_b = client.post("/fleet/pack", json=payload_b)

    assert resp_a.status_code == 200
    assert resp_b.status_code == 200
    assert resp_b.json()["stored_count"] == 2

    get_resp = client.get(
        "/fleet/pack",
        params={
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
            "min_sites": 2,
        },
    )

    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["k_anonymity_met"] is True
    assert body["pattern_priors"]["CUSTOM:test"]["prior"] == 0.8
    assert body["discovered_patterns"]["patterns"]["discovered:cnc_endmill_alarm"]["site_count"] == 2
    assert store_path.exists()
    assert len(store_path.read_text().strip().splitlines()) == 2


def test_fleet_router_rejects_pack_without_complete_context(tmp_path: Path, monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    monkeypatch.setenv("FLEET_KNOWLEDGE_STORE_PATH", str(tmp_path / "fleet_packs.jsonl"))

    resp = client.post(
        "/fleet/pack",
        json={
            "site": "site-a",
            "context": {"machine_type": "cnc", "tool_type": "endmill"},
        },
    )

    assert resp.status_code == 422
    assert resp.json()["detail"]["missing_context_keys"] == ["material", "regime"]
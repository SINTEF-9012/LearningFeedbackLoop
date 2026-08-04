from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers.reconfig import router


def _proposal_payload() -> dict:
    return {
        "triggered_by": ["SPINDLE_LOAD_RAMP", "memory:123"],
        "context": {
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
        },
        "batch": {
            "batch_id": "batch-42",
            "unit_index": 3,
            "unit_count": 12,
            "recipe_id": "recipe-a",
        },
        "parameter_deltas": [
            {
                "parameter": "feed_rate",
                "direction": "decrease",
                "magnitude_pct": -10.0,
                "confidence": 0.82,
                "evidence": ["SPINDLE_LOAD_RAMP", "doc:tool-wear"],
                "rationale": "Observed spindle-load ramp suggests reducing feed to limit wear progression.",
            }
        ],
        "tool_actions": [
            {
                "action": "inspect",
                "tool_number": 12,
                "reason_code": "wear_indicated_by_power_ramp",
                "confidence": 0.76,
                "evidence": ["SPINDLE_LOAD_RAMP"],
            }
        ],
        "recipe_edits": [],
        "risk": "medium",
    }


def test_reconfig_router_creates_and_lists_proposal(tmp_path: Path, monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    outbox_path = tmp_path / "reconfig_outbox.jsonl"
    monkeypatch.setenv("RECONFIG_OUTBOX_PATH", str(outbox_path))

    create_resp = client.post("/reconfig/proposal", json=_proposal_payload())

    assert create_resp.status_code == 200
    created = create_resp.json()
    assert created["proposal_id"]
    assert created["requires_operator_confirmation"] is True
    assert created["operator_decision"] is None

    list_resp = client.get(
        "/reconfig/proposal",
        params={
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
        },
    )

    assert list_resp.status_code == 200
    items = list_resp.json()["items"]
    assert len(items) == 1
    assert items[0]["proposal_id"] == created["proposal_id"]
    assert outbox_path.exists()
    assert len(outbox_path.read_text().strip().splitlines()) == 1


def test_reconfig_router_accepts_existing_proposal(tmp_path: Path, monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    monkeypatch.setenv("RECONFIG_OUTBOX_PATH", str(tmp_path / "reconfig_outbox.jsonl"))

    created = client.post("/reconfig/proposal", json=_proposal_payload()).json()
    accept_resp = client.post(
        f"/reconfig/{created['proposal_id']}/accept",
        json={
            "operator_id": "operator-7",
            "applied_via": "manual",
        },
    )

    assert accept_resp.status_code == 200
    accepted = accept_resp.json()
    assert accepted["operator_decision"] == "accept"
    assert accepted["operator_decision_by"] == "operator-7"
    assert accepted["applied"] is True
    assert accepted["applied_via"] == "manual"


def test_reconfig_router_reject_requires_reason(tmp_path: Path, monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    monkeypatch.setenv("RECONFIG_OUTBOX_PATH", str(tmp_path / "reconfig_outbox.jsonl"))

    created = client.post("/reconfig/proposal", json=_proposal_payload()).json()
    reject_resp = client.post(
        f"/reconfig/{created['proposal_id']}/reject",
        json={
            "operator_id": "operator-7",
        },
    )

    assert reject_resp.status_code == 422
    assert reject_resp.json()["detail"]["message"] == "reject requires a reason"


def test_reconfig_router_logs_manual_operator_action(tmp_path: Path, monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    outbox_path = tmp_path / "reconfig_outbox.jsonl"
    monkeypatch.setenv("RECONFIG_OUTBOX_PATH", str(outbox_path))

    manual_resp = client.post(
        "/reconfig/manual",
        json={
            "triggered_by": [],
            "context": {
                "machine_type": "cnc",
                "tool_type": "endmill",
                "material": "al",
                "regime": "rough",
            },
            "parameter_deltas": [
                {
                    "parameter": "feed_override",
                    "direction": "decrease",
                    "magnitude_pct": -8.0,
                    "confidence": 1.0,
                    "evidence": ["operator:manual"],
                    "rationale": "Operator reduced feed override based on audible chatter.",
                }
            ],
            "tool_actions": [],
            "recipe_edits": [],
            "risk": "low",
            "operator_id": "operator-9",
            "reason": "Audible chatter at tool entry",
        },
    )

    assert manual_resp.status_code == 200
    manual = manual_resp.json()
    assert manual["operator_decision"] == "manual"
    assert manual["operator_decision_by"] == "operator-9"
    assert manual["requires_operator_confirmation"] is False

    stored_rows = [json.loads(line) for line in outbox_path.read_text().strip().splitlines()]
    assert stored_rows[-1]["operator_decision"] == "manual"
    assert stored_rows[-1]["triggered_by"] == []


def test_reconfig_router_modify_updates_existing_proposal(tmp_path: Path, monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    monkeypatch.setenv("RECONFIG_OUTBOX_PATH", str(tmp_path / "reconfig_outbox.jsonl"))

    created = client.post("/reconfig/proposal", json=_proposal_payload()).json()
    modify_resp = client.post(
        f"/reconfig/{created['proposal_id']}/modify",
        json={
            "operator_id": "operator-7",
            "reason": "Prefer smaller first step",
            "parameter_deltas": [
                {
                    "parameter": "feed_rate",
                    "direction": "decrease",
                    "magnitude_pct": -5.0,
                    "confidence": 0.82,
                    "evidence": ["SPINDLE_LOAD_RAMP", "doc:tool-wear"],
                    "rationale": "Start with a smaller feed reduction before tool change.",
                }
            ],
        },
    )

    assert modify_resp.status_code == 200
    modified = modify_resp.json()
    assert modified["operator_decision"] == "modify"
    assert modified["parameter_deltas"][0]["magnitude_pct"] == -5.0
    assert modified["notes"][-1] == "operator modify reason: Prefer smaller first step"


def test_reconfig_router_composes_known_pattern_into_proposal(tmp_path: Path, monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    outbox_path = tmp_path / "reconfig_outbox.jsonl"
    monkeypatch.setenv("RECONFIG_OUTBOX_PATH", str(outbox_path))

    compose_resp = client.post(
        "/reconfig/compose",
        json={
            "triggered_by": ["SPINDLE_LOAD_RAMP"],
            "pattern_scores": {"SPINDLE_LOAD_RAMP": 0.74},
            "context": {
                "machine_type": "cnc",
                "tool_type": "endmill",
                "material": "al",
                "regime": "rough",
            },
            "batch": {
                "batch_id": "batch-42",
                "unit_index": 3,
                "unit_count": 12,
                "recipe_id": "recipe-a",
            },
        },
    )

    assert compose_resp.status_code == 200
    proposal = compose_resp.json()
    assert proposal["triggered_by"] == ["SPINDLE_LOAD_RAMP"]
    assert proposal["parameter_deltas"][0]["parameter"] == "feed_rate"
    assert proposal["parameter_deltas"][0]["direction"] == "decrease"
    assert proposal["parameter_deltas"][0]["magnitude_pct"] == -5.0
    assert proposal["tool_actions"][0]["action"] == "inspect"
    assert proposal["recipe_edits"][0]["target"] == "next_unit"
    assert proposal["recipe_edits"][0]["recipe_id"] == "recipe-a"
    assert proposal["recipe_edits"][0]["edits"][0]["parameter"] == "feed_rate"
    assert proposal["risk"] == "medium"
    assert proposal["notes"][-1] == "composed via deterministic reconfig prompt"
    assert len(outbox_path.read_text().strip().splitlines()) == 1
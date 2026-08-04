"""Read-only MaaS capability-evidence loader for the UI (plan §4.2)."""
import json

from backend.agents.maas.evidence_summary import load_evidence_summary


def test_missing_artifact_returns_none(tmp_path):
    assert load_evidence_summary(tmp_path / "nope.json") is None


def test_loads_list_artifact(tmp_path):
    p = tmp_path / "ev.json"
    p.write_text(json.dumps([
        {"plant_id": "PLANT-004", "capability": "Tool-wear monitoring",
         "declared": True, "confirm_rate": 0.75, "confidence": 0.167,
         "co2_avoided_kg_total": 2763.0},
    ]))
    s = load_evidence_summary(p)
    assert s is not None
    assert s["count"] == 1 and s["illustrative"] is True
    assert s["records"][0]["capability"] == "Tool-wear monitoring"


def test_loads_single_object_artifact(tmp_path):
    p = tmp_path / "ev.json"
    p.write_text(json.dumps({"plant_id": "PLANT-004", "capability": "X"}))
    s = load_evidence_summary(p)
    assert s is not None and s["count"] == 1


def test_empty_or_bad_returns_none(tmp_path):
    p = tmp_path / "ev.json"
    p.write_text("[]")
    assert load_evidence_summary(p) is None
    p.write_text("not json")
    assert load_evidence_summary(p) is None


def test_default_artifact_if_present():
    # The repo ships the tool-wear evidence; if present it loads cleanly.
    s = load_evidence_summary()
    if s is not None:
        assert s["count"] >= 1
        assert "capability" in s["records"][0]

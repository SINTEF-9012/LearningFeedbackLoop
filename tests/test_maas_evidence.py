"""Tests for the MaaS capability-evidence layer (agents/maas)."""

import json

from backend.agents.maas import (
    CAPABILITY_PATTERN_MAP,
    DPPRegistry,
    PlantCatalogue,
    build_availability_evidence,
    build_evidence,
    build_fault_lead_time_evidence,
    capability_for_pattern,
)
from backend.agents.maas.evidence_exporter import CONFIDENCE_PRIOR_N, write_evidence


def _plants_file(tmp_path):
    p = tmp_path / "plants.json"
    p.write_text(json.dumps([
        {"plantId": "PLANT-004", "supplierId": "SUP-003",
         "machiningStabilityCapabilities": ["Tool-wear monitoring", "Vibration control"],
         "averageEnergyConsumptionPerPartKwh": 3100, "co2FactorKgPerKwh": 0.22,
         "availabilityNext6WeeksPercent": 68},
        {"plantId": "PLANT-001", "supplierId": "SUP-001",
         "machiningStabilityCapabilities": ["Vibration control"],
         "averageEnergyConsumptionPerPartKwh": 420, "co2FactorKgPerKwh": 0.30},
    ]))
    return p


def _dpp_dir(tmp_path):
    d = tmp_path / "dpp"
    d.mkdir()
    (d / "DPP_demo.json").write_text(json.dumps({"DPP": [{
        "company": "site_a", "event_id": "OF00005",
        "carbonFootprint": {"ProductCarbonFootprints": [
            {"PCF in kg CO2eq": "921.0", "ProcessName": "SITE_A total A3"},
            {"PCF in kg CO2eq": "17968.3", "ProcessName": "SITE_A total A1-A3"},
        ]},
    }]}))
    return d


def test_capability_pattern_lookup():
    assert capability_for_pattern("fault:tool_breakage") == "Tool-wear monitoring"
    assert capability_for_pattern("fault:chatter") == "Vibration control"
    assert capability_for_pattern("nonexistent") is None
    # every mapped pattern resolves back to its capability
    for cap, patterns in CAPABILITY_PATTERN_MAP.items():
        for pat in patterns:
            assert capability_for_pattern(pat) == cap


def test_plant_catalogue(tmp_path):
    cat = PlantCatalogue.from_file(_plants_file(tmp_path))
    assert cat.supplier_of("PLANT-004") == "SUP-003"
    assert cat.declares("PLANT-004", "Tool-wear monitoring") is True
    assert cat.declares("PLANT-001", "Tool-wear monitoring") is False
    assert cat.energy_kwh_per_part("PLANT-004") == 3100.0
    assert cat.co2_factor("PLANT-004") == 0.22


def test_dpp_registry(tmp_path):
    reg = DPPRegistry.from_dir(_dpp_dir(tmp_path))
    impact = reg.resolve("OF00005")
    assert impact is not None
    assert impact.pcf_total_kg == 17968.3
    assert impact.pcf_processing_kg == 921.0
    assert impact.co2_avoided_per_scrap_kg == 921.0
    # unknown event falls back to first available
    assert reg.resolve("missing") is impact


def test_dpp_registry_missing_dir(tmp_path):
    reg = DPPRegistry.from_dir(tmp_path / "nope")
    assert reg.resolve("x") is None
    assert reg.all() == []


def test_build_evidence_full(tmp_path):
    cat = PlantCatalogue.from_file(_plants_file(tmp_path))
    dpp = DPPRegistry.from_dir(_dpp_dir(tmp_path))
    agg = {"plant_id": "PLANT-004",
           "context": {"machine_family": "gantry_mill", "tool_type": "face_mill"},
           "capability": "Tool-wear monitoring",
           "confirmed": 3, "dismissed": 1, "lead_times_s": [30, 40, 50]}
    [e] = build_evidence([agg], catalogue=cat, dpp=dpp, window_days=90)
    assert e.supplier_id == "SUP-003"
    assert e.declared is True
    assert e.confirm_rate == 0.75
    assert e.confidence == round(4 / (4 + CONFIDENCE_PRIOR_N), 3)
    assert e.lead_time_s_median == 40.0
    assert e.realised_co2_kg_per_good_part == round(3100 * 0.22, 1)
    assert e.co2_avoided_kg_per_confirmed_catch == 921.0
    assert e.co2_avoided_kg_total == round(921.0 * 3, 1)


def test_build_evidence_degrades_without_catalogue_or_dpp():
    agg = {"plant_id": "PLANT-X", "capability": "Tool-wear monitoring",
           "confirmed": 2, "dismissed": 0, "context": {}}
    [e] = build_evidence([agg])
    assert e.supplier_id == "UNKNOWN"
    assert e.declared is False
    assert e.realised_co2_kg_per_good_part is None
    assert e.co2_avoided_kg_per_confirmed_catch is None
    assert e.confidence > 0  # still produces a (low) confidence claim


def test_build_evidence_from_pattern_key(tmp_path):
    cat = PlantCatalogue.from_file(_plants_file(tmp_path))
    agg = {"plant_id": "PLANT-004", "pattern": "fault:tool_breakage",
           "confirmed": 1, "dismissed": 0, "context": {}}
    [e] = build_evidence([agg], catalogue=cat)
    assert e.capability == "Tool-wear monitoring"


def test_write_evidence_roundtrip(tmp_path):
    agg = {"plant_id": "P", "capability": "Vibration control",
           "confirmed": 1, "dismissed": 0, "context": {}}
    recs = build_evidence([agg])
    out = tmp_path / "ev.json"
    assert write_evidence(recs, out) == 1
    loaded = json.loads(out.read_text())
    assert loaded[0]["capability"] == "Vibration control"


# ── Fault-and-lead-time evidence (skeleton facet) ──────────────────────────────

def test_build_fault_lead_time_evidence(tmp_path):
    cat = PlantCatalogue.from_file(_plants_file(tmp_path))
    agg = {"plant_id": "PLANT-004", "context": {"material": "steel"},
           "capability": "Tool-wear monitoring",
           "faults": [
               {"fault": "tool_breakage", "confirmed": 41, "dismissed": 3,
                "lead_times_s": [50, 52, 54]},
               {"fault": "chatter", "confirmed": 77, "dismissed": 12, "lead_times_s": []},
           ]}
    [e] = build_fault_lead_time_evidence([agg], catalogue=cat, window_days=90)
    assert e.supplier_id == "SUP-003"
    assert e.window == "90d"
    assert e.faults[0]["lead_time_s_median"] == 52.0  # median of [50, 52, 54]
    assert e.faults[1]["lead_time_s_median"] is None   # no lead times -> null, not fabricated
    # confidence reflects total adjudicated events (41+3+77+12 = 133)
    assert e.confidence == round(133 / (133 + CONFIDENCE_PRIOR_N), 3)


def test_build_fault_lead_time_resolves_capability_from_pattern():
    agg = {"plant_id": "PLANT-004", "pattern": "fault:tool_breakage", "context": {},
           "faults": [{"fault": "tool_breakage", "confirmed": 1, "dismissed": 0}]}
    [e] = build_fault_lead_time_evidence([agg])
    assert e.capability == "Tool-wear monitoring"
    assert e.supplier_id == "UNKNOWN"  # degrades without catalogue


# ── Availability-adjustment evidence (skeleton facet) ──────────────────────────

def test_build_availability_evidence_full(tmp_path):
    cat = PlantCatalogue.from_file(_plants_file(tmp_path))
    agg = {"plant_id": "PLANT-004", "context": {"material": "Casting steel"},
           "confirmed_stoppages": 6, "operating_hours": 1040}
    [e] = build_availability_evidence(
        [agg], catalogue=cat, assumed_downtime_h_per_stoppage=7, window_days=90)
    assert e.supplier_id == "SUP-003"
    assert e.declared_availability_pct == 68.0
    assert e.mean_hours_between_stoppages == 173.3   # 1040 / 6
    assert e.availability_adjustment_pct == -4.0     # -(6*7/1040)*100, rounded
    assert e.confidence == round(6 / (6 + CONFIDENCE_PRIOR_N), 3)


def test_build_availability_evidence_degrades_without_catalogue_or_downtime():
    agg = {"plant_id": "PLANT-X", "context": {}, "confirmed_stoppages": 3,
           "operating_hours": 300}
    [e] = build_availability_evidence([agg])
    assert e.supplier_id == "UNKNOWN"
    assert e.declared_availability_pct is None        # no catalogue
    assert e.mean_hours_between_stoppages == 100.0     # still a direct observable
    assert e.availability_adjustment_pct is None       # no downtime assumption -> not fabricated


def test_build_availability_evidence_per_record_downtime():
    # downtime supplied on the record overrides the (absent) function default
    agg = {"plant_id": "PLANT-X", "context": {}, "confirmed_stoppages": 2,
           "operating_hours": 200, "assumed_downtime_h_per_stoppage": 5}
    [e] = build_availability_evidence([agg])
    assert e.availability_adjustment_pct == -5.0       # -(2*5/200)*100


def test_write_evidence_accepts_new_facets(tmp_path):
    cat = PlantCatalogue.from_file(_plants_file(tmp_path))
    recs = build_availability_evidence(
        [{"plant_id": "PLANT-004", "confirmed_stoppages": 1, "operating_hours": 100}],
        catalogue=cat)
    out = tmp_path / "avail.json"
    assert write_evidence(recs, out) == 1
    loaded = json.loads(out.read_text())
    assert loaded[0]["plant_id"] == "PLANT-004"
    assert "mean_hours_between_stoppages" in loaded[0]

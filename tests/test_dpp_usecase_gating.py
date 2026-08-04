"""Tests for usecase-scoped DPP resolution.

The operator-facing carbon context must never attribute one usecase's embodied
carbon to another (e.g. surface SITE_A's part footprint on a SITE_C event). These
tests lock the ``resolve_for_usecase`` contract: strict within-usecase match,
never a cross-usecase fallback.
"""

import json

from backend.agents.maas import DPPRegistry


def _write_dpp(directory, name, company, usecase, event_id, a3, total):
    pcfs = [
        {"PCF in kg CO2eq": a3, "ProcessName": "part A3"},
        {"PCF in kg CO2eq": total, "ProcessName": "part A1-A3"},
    ]
    rec = {"company": company, "event_id": event_id,
           "carbonFootprint": {"ProductCarbonFootprints": pcfs}}
    if usecase is not None:
        rec["usecase"] = usecase
    (directory / name).write_text(json.dumps({"DPP": [rec]}))


def _registry(tmp_path):
    d = tmp_path / "dpp"
    d.mkdir()
    _write_dpp(d, "DPP_site_a.json", "site_a", None, "SITE_A-PART", 921.0, 17968.3)
    _write_dpp(d, "DPP_site_c.json", "site_c", "SITE_C", "SITE_C-PART", 784.0, 13460.0)
    return DPPRegistry.from_dir(d)


def test_resolve_for_usecase_matches_own_usecase(tmp_path):
    reg = _registry(tmp_path)
    site_c = reg.resolve_for_usecase("SITE_C")
    assert site_c is not None
    assert site_c.usecase == "SITE_C"
    assert site_c.pcf_processing_kg == 784.0


def test_company_name_normalises_to_usecase(tmp_path):
    # SITE_A DPP has no explicit `usecase`; company "site_a" must map to SITE_A.
    reg = _registry(tmp_path)
    site_a = reg.resolve_for_usecase("SITE_A")
    assert site_a is not None
    assert site_a.usecase == "SITE_A"
    assert site_a.pcf_processing_kg == 921.0


def test_no_cross_usecase_leak(tmp_path):
    # A usecase with no DPP must return None — never another usecase's part.
    reg = _registry(tmp_path)
    assert reg.resolve_for_usecase("SITE_B") is None
    assert reg.resolve_for_usecase(None) is None
    assert reg.resolve_for_usecase("") is None


def test_event_id_honoured_only_within_usecase(tmp_path):
    reg = _registry(tmp_path)
    # A SITE_A event id must not resolve when asking within SITE_C.
    within_site_c = reg.resolve_for_usecase("SITE_C", event_id="SITE_A-PART")
    assert within_site_c is not None
    assert within_site_c.usecase == "SITE_C"  # falls back to the SITE_C part, not SITE_A


def test_legacy_resolve_unscoped_still_works(tmp_path):
    # The feedback impact-weighting path relies on unscoped resolve(); keep it.
    reg = _registry(tmp_path)
    assert reg.resolve("SITE_A-PART") is not None
    assert reg.resolve() is not None  # first-available fallback preserved

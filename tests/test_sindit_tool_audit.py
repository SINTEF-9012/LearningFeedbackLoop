from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.agents.processing.tool_lookup import ToolSpec
from backend.agents.sindit.tool_audit import (
    _runtime_snapshot,
    build_tool_dataset_decision_snapshot,
    build_tool_dataset_overview,
    build_tool_audit_rows,
    build_tool_audit_summary,
    clear_runtime_observations,
    filter_tool_audit_rows,
    load_tool_dataset_decisions,
    record_tool_anomaly,
    record_tool_feedback,
    record_tool_observation,
    save_tool_dataset_decision,
)


def test_record_tool_observation_and_filtering():
    clear_runtime_observations()
    record_tool_observation(
        "sess-1",
        {
            "machine_id": "Site_b - MACHINE_B1 - CASE_B1",
            "tool_id": "T6",
            "tool_diameter": 65.0,
            "num_teeth": 1,
            "spindle_speed": 1800.0,
            "feed_rate": 900.0,
            "extra": {
                "machine_family": "builder_b12",
                "tool_number": 6,
                "sindit_tool_iri": "urn:lfl:tool:builder_b12-t6",
            },
        },
    )

    rows = build_tool_audit_rows(
        master={},
        graph={},
        runtime={
            ("builder_b12", 6): {
                "session_ids": ["sess-1"],
                "machine_ids": ["Site_b - MACHINE_B1 - CASE_B1"],
                "tool_id": "T6",
                "tool_uri": "urn:lfl:tool:builder_b12-t6",
                "seen_count": 1,
                "first_seen_at": "2026-05-13T00:00:00+00:00",
                "last_seen_at": "2026-05-13T00:00:00+00:00",
                "effective_ctx": {
                    "tool_diameter": 65.0,
                    "num_teeth": 1,
                    "spindle_speed": 1800.0,
                    "feed_rate": 900.0,
                },
            }
        },
        family_registry={"builder_b12": ["Site_b - MACHINE_B1 - CASE_B1"]},
        sindit_available=False,
    )

    filtered = filter_tool_audit_rows(
        rows,
        session_id="sess-1",
        machine_id="Site_b - MACHINE_B1 - CASE_B1",
    )
    assert len(filtered) == 1
    assert filtered[0]["harmonic_ready"] is True


def test_record_tool_observation_preserves_last_nonzero_cutting_state():
    clear_runtime_observations()
    base_ctx = {
        "machine_id": "Site_b - MACHINE_B1 - CASE_B1",
        "tool_id": "T55",
        "extra": {
            "machine_family": "builder_b12",
            "tool_number": 55,
            "sindit_tool_iri": "urn:lfl:tool:builder_b12-t55",
        },
    }

    record_tool_observation(
        "sess-55",
        {
            **base_ctx,
            "tool_diameter": 20.0,
            "num_teeth": 4,
            "spindle_speed": 1800.0,
            "feed_rate": 900.0,
        },
    )
    record_tool_observation(
        "sess-55",
        {
            **base_ctx,
            "tool_diameter": 20.0,
            "num_teeth": 4,
            "spindle_speed": 0.0,
            "feed_rate": 0.0,
        },
    )

    rows = build_tool_audit_rows(
        master={},
        graph={},
        runtime=_runtime_snapshot(),
        family_registry={"builder_b12": ["Site_b - MACHINE_B1 - CASE_B1"]},
        sindit_available=False,
    )

    row = next(item for item in rows if item["machine_family"] == "builder_b12" and item["tool_number"] == 55)
    assert row["runtime"]["effective_ctx"]["spindle_speed"] == 1800.0
    assert row["runtime"]["effective_ctx"]["feed_rate"] == 900.0


def test_runtime_snapshot_preserves_nonzero_cutting_state_across_sessions():
    clear_runtime_observations()
    base_ctx = {
        "machine_id": "Site_b - MACHINE_B1 - CASE_B1",
        "tool_id": "T55",
        "extra": {
            "machine_family": "builder_b12",
            "tool_number": 55,
            "sindit_tool_iri": "urn:lfl:tool:builder_b12-t55",
        },
    }

    record_tool_observation(
        "sess-55-a",
        {
            **base_ctx,
            "tool_diameter": 20.0,
            "num_teeth": 4,
            "spindle_speed": 1800.0,
            "feed_rate": 900.0,
        },
    )
    record_tool_observation(
        "sess-55-b",
        {
            **base_ctx,
            "tool_diameter": 20.0,
            "num_teeth": 4,
            "spindle_speed": 0.0,
            "feed_rate": 0.0,
        },
    )

    rows = build_tool_audit_rows(
        master={},
        graph={},
        runtime=_runtime_snapshot(),
        family_registry={"builder_b12": ["Site_b - MACHINE_B1 - CASE_B1"]},
        sindit_available=False,
    )

    row = next(item for item in rows if item["machine_family"] == "builder_b12" and item["tool_number"] == 55)
    assert sorted(row["runtime"]["session_ids"]) == ["sess-55-a", "sess-55-b"]
    assert row["runtime"]["effective_ctx"]["spindle_speed"] == 1800.0
    assert row["runtime"]["effective_ctx"]["feed_rate"] == 900.0


def test_record_tool_anomaly_and_feedback_roll_up_into_runtime_layer():
    clear_runtime_observations()
    ctx = {
        "machine_id": "Site_b - MACHINE_B1 - CASE_B1",
        "tool_id": "T55",
        "tool_diameter": 20.0,
        "num_teeth": 4,
        "spindle_speed": 1800.0,
        "feed_rate": 900.0,
        "extra": {
            "machine_family": "builder_b12",
            "tool_number": 55,
            "sindit_tool_iri": "urn:lfl:tool:builder_b12-t55",
        },
    }

    record_tool_observation("sess-55", ctx)
    snapshot = record_tool_anomaly(
        "sess-55",
        ctx,
        memory_id="mem-55",
        significance_score=0.82,
        significant=True,
        alert_dispatched=True,
        pattern_keys=["signature:modulated_tooth_passing_vibration"],
        triggered_rules=["CLASSICAL_ALERT"],
        pattern_priors={"signature:modulated_tooth_passing_vibration": 0.71},
    )
    assert snapshot is not None
    assert snapshot["anomaly_stats"]["scored_count"] == 1
    assert snapshot["anomaly_stats"]["alerted_count"] == 1

    feedback_snapshot = record_tool_feedback(
        "sess-55",
        ctx,
        action="confirm",
        memory_id="mem-55",
        pattern_keys=["signature:modulated_tooth_passing_vibration"],
        operator_id="ui",
    )
    assert feedback_snapshot is not None
    assert feedback_snapshot["anomaly_stats"]["confirmed_count"] == 1

    rows = build_tool_audit_rows(
        master={},
        graph={},
        runtime=_runtime_snapshot(),
        family_registry={"builder_b12": ["Site_b - MACHINE_B1 - CASE_B1"]},
        sindit_available=False,
    )

    row = next(item for item in rows if item["machine_family"] == "builder_b12" and item["tool_number"] == 55)
    stats = row["runtime"]["anomaly_stats"]
    assert stats["scored_count"] == 1
    assert stats["significant_count"] == 1
    assert stats["alerted_count"] == 1
    assert stats["confirmed_count"] == 1
    assert stats["dismissed_count"] == 0
    assert stats["last_memory_id"] == "mem-55"
    assert stats["last_feedback_action"] == "confirm"
    assert stats["pattern_counts"]["signature:modulated_tooth_passing_vibration"] == 1


def test_build_tool_audit_rows_flags_missing_and_mismatched_fields():
    rows = build_tool_audit_rows(
        master={
            ("builder_b12", 6): ToolSpec(
                machine_family="builder_b12",
                tool_number=6,
                tool_id="T06",
                description="FINISH BORE 65MM DIA",
                tool_type="bore",
                diameter_mm=65.0,
                teeth=1,
                tool_length_mm=128.125,
                source="site_b/Builder_b1 2 Tooling Database.xlsx",
            )
        },
        graph={
            ("builder_b12", 6): [
                {
                    "asset_uri": "urn:lfl:tool:builder_b12-t6",
                    "label": "Builder_b12 T6",
                    "tool_diameter": 63.0,
                    "num_teeth": 2,
                    "tool_type": "mill",
                    "tool_length": 120.0,
                    "last_imported_at": "2026-05-10T00:00:00+00:00",
                    "source_workbook": "site_b/Builder_b1 2 Tooling Database.xlsx",
                    "machine_uris": ["urn:lfl:asset:site_b---machine_b1---case_b1"],
                }
            ]
        },
        runtime={
            ("builder_b12", 6): {
                "session_ids": ["sess-1"],
                "machine_ids": ["Site_b - MACHINE_B1 - CASE_B1"],
                "tool_id": "T6",
                "tool_uri": "urn:lfl:tool:builder_b12-t6",
                "seen_count": 1,
                "first_seen_at": "2026-05-13T00:00:00+00:00",
                "last_seen_at": "2026-05-13T00:00:00+00:00",
                "effective_ctx": {
                    "spindle_speed": 1800.0,
                },
            }
        },
        family_registry={"builder_b12": ["Site_b - MACHINE_B1 - CASE_B1"]},
        sindit_available=True,
        now=datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc),
    )

    row = rows[0]
    assert "diameter_mismatch_mm" in row["flags"]
    assert "teeth_mismatch" in row["flags"]
    assert "tool_type_mismatch" in row["flags"]
    assert "tool_length_mismatch_mm" in row["flags"]
    assert "stale_import" in row["flags"]
    assert "harmonic_not_ready" in row["flags"]


def test_build_tool_audit_summary_counts_flags():
    summary = build_tool_audit_summary(
        [
            {"flags": ["missing_tool_diameter", "missing_sindit_asset"], "harmonic_ready": False},
            {"flags": [], "harmonic_ready": True},
            {"flags": ["family_resolution_miss", "missing_num_teeth"], "harmonic_ready": False},
        ],
        sindit_available=True,
    )

    assert summary["tools_seen"] == 3
    assert summary["discrepancies"] == 2
    assert summary["harmonic_ready"] == 1
    assert summary["missing_diameter"] == 1
    assert summary["missing_teeth"] == 1
    assert summary["missing_sindit_asset"] == 1
    assert summary["family_resolution_miss"] == 1


def test_build_tool_audit_rows_merges_master_geometry_with_runtime_cutting_state_for_harmonics():
    rows = build_tool_audit_rows(
        master={
            ("builder_b12", 55): ToolSpec(
                machine_family="builder_b12",
                tool_number=55,
                tool_id="T55",
                description="CARBIDE MILL 20MM DIA",
                tool_type="mill",
                diameter_mm=20.0,
                teeth=4,
                tool_length_mm=120.0,
                source="site_b/Builder_b1 2 Tooling Database.xlsx",
            )
        },
        graph={},
        runtime={
            ("builder_b12", 55): {
                "session_ids": ["sess-55"],
                "machine_ids": ["Site_b - MACHINE_B1 - CASE_B1"],
                "tool_id": "T55",
                "tool_uri": "urn:lfl:tool:builder_b12-t55",
                "seen_count": 1,
                "first_seen_at": "2026-05-13T00:00:00+00:00",
                "last_seen_at": "2026-05-13T00:00:00+00:00",
                "effective_ctx": {
                    "spindle_speed": 1800.0,
                    "feed_rate": 900.0,
                },
            }
        },
        family_registry={"builder_b12": ["Site_b - MACHINE_B1 - CASE_B1"]},
        sindit_available=False,
    )

    row = rows[0]
    assert row["harmonic_ready"] is True
    assert "harmonic_not_ready" not in row["flags"]


def test_build_tool_audit_rows_includes_reference_and_process_plan_layers():
    rows = build_tool_audit_rows(
        master={
            ("builder_b12", 64): ToolSpec(
                machine_family="builder_b12",
                tool_number=64,
                tool_id="T64",
                description="FINISH CARBIDE MILL 20MM DIA",
                tool_type="mill",
                diameter_mm=20.0,
                teeth=4,
                tool_length_mm=293.0,
                source="site_b/Builder_b1 2 Tooling Database.xlsx",
            )
        },
        graph={},
        runtime={},
        family_registry={"builder_b12": ["Site_b - MACHINE_B1 - CASE_B1"]},
        reference_index={
            ("builder_b12", 64): {
                "tool_number": 64,
                "tool_label": "64",
                "description": "FINISH CARBIDE MILL 20MM DIA",
                "drawing_required": False,
                "operations": [{"operation_id": "OP045", "title": "FINISH CUT OUT"}],
                "reference_lines": ["ISC5620750", "ECXL-B-4/6"],
                "notes": [],
                "dimensions": {"overall_length_mm": 293.0},
                "sources": ["site_b/Critical tool list.docx"],
            }
        },
        process_plan_index={
            ("builder_b12", 64): {
                "use_case_ids": [1],
                "use_case_titles": ["Extension boom"],
                "operation_ids": ["OP045"],
                "setups": ["SETUP 1.2"],
                "entries": [
                    {
                        "use_case_id": 1,
                        "use_case_title": "Extension boom",
                        "setup": "SETUP 1.2",
                        "operation_id": "OP045",
                        "head": None,
                        "op_type": "FINISH MILLING",
                        "description": "FINISH CUT OUT",
                        "tool_raw": "64",
                        "slide_number": 8,
                        "slide_row_index": 6,
                    }
                ],
            }
        },
        sindit_available=False,
    )

    row = rows[0]
    assert row["reference"]["description"] == "FINISH CARBIDE MILL 20MM DIA"
    assert row["reference"]["dimensions"]["overall_length_mm"] == 293.0
    assert row["process_plan"]["use_case_titles"] == ["Extension boom"]
    assert row["process_plan"]["entries"][0]["operation_id"] == "OP045"


def test_build_tool_dataset_overview_separates_casedata_and_olddata():
    payload = build_tool_dataset_overview(
        coverage_items=[
            {
                "dataset": "site_b_olddata",
                "machine_id": "Site_b - MACHINE_B1 - CASE_B1",
                "machine_family": "builder_b12",
                "operation_id": "OF00001",
                "valid_tool_rows": 20,
                "resolved_dz_rows": 18,
                "harmonic_ready_rows": 18,
                "unique_tools": [64],
                "resolved_master_tools": [64],
                "resolved_d_tools": [64],
                "resolved_z_tools": [64],
                "resolved_dz_tools": [64],
                "harmonic_ready_tools": [64],
            },
            {
                "dataset": "site_b_olddata",
                "machine_id": "olddata",
                "machine_family": "builder_b12",
                "operation_id": "OF00002",
                "valid_tool_rows": 10,
                "resolved_dz_rows": 0,
                "harmonic_ready_rows": 0,
                "unique_tools": [1],
                "resolved_master_tools": [],
                "resolved_d_tools": [],
                "resolved_z_tools": [],
                "resolved_dz_tools": [],
                "harmonic_ready_tools": [],
            },
        ],
        audit_rows=[
            {
                "machine_family": "builder_b12",
                "tool_number": 64,
                "master": {
                    "tool_id": "T64",
                    "description": "FINISH CARBIDE MILL 20MM DIA",
                    "tool_type": "mill",
                    "diameter_mm": 20.0,
                    "teeth": 4,
                    "tool_length_mm": 293.0,
                },
                "reference": None,
                "runtime": None,
                "sindit": None,
                "process_plan": None,
                "flags": [],
                "harmonic_ready": True,
            },
            {
                "machine_family": "builder_b12",
                "tool_number": 1,
                "master": None,
                "reference": None,
                "runtime": None,
                "sindit": None,
                "process_plan": None,
                "flags": ["missing_master_spec", "missing_tool_diameter", "missing_num_teeth"],
                "harmonic_ready": False,
            },
        ],
        decisions={
            ("site_b_casedata", "builder_b12", 64): {
                "dataset_id": "site_b_casedata",
                "machine_family": "builder_b12",
                "tool_number": 64,
                "selection_mode": "master",
                "status": "confirmed",
            }
        },
    )

    assert payload["total_datasets"] == 2
    assert [dataset["dataset_id"] for dataset in payload["datasets"]] == ["site_b_casedata", "site_b_olddata"]
    assert payload["datasets"][0]["shared_workpiece"] is True
    assert "two machines cutting the same workpiece" in payload["datasets"][0]["workpiece_note"]
    assert payload["datasets"][1]["shared_workpiece"] is False
    assert payload["datasets"][0]["harmonic_summary"]["harmonic_ready_tool_pct"] == 100.0
    assert payload["datasets"][0]["harmonic_summary"]["harmonic_ready_row_pct"] == 90.0
    assert payload["datasets"][0]["part_summaries"][0]["label"] == "Site_b - MACHINE_B1 - CASE_B1 / OF00001"
    assert payload["datasets"][1]["harmonic_summary"]["harmonic_ready_tool_pct"] == 0.0
    assert payload["datasets"][1]["harmonic_summary"]["harmonic_ready_row_pct"] == 0.0

    casedata_tool = payload["datasets"][0]["tools"][0]
    assert casedata_tool["decision_status"] == "confirmed"
    assert casedata_tool["selected_profile"] == "master"
    assert casedata_tool["certainty"] == "certain"

    olddata_tool = payload["datasets"][1]["tools"][0]
    assert olddata_tool["certainty"] == "needs_review"
    assert "missing_master_spec" in olddata_tool["review_flags"]


def test_save_and_load_tool_dataset_decisions(tmp_path):
    target = tmp_path / "dataset_tool_decisions.json"

    saved = save_tool_dataset_decision(
        dataset_id="site_a_line2",
        machine_family="machine_a1",
        tool_number=49,
        status="rejected",
        selection_mode="manual",
        reference_tool_number=44,
        updated_by="pytest",
        resolved_context={"tool_diameter": 42.0, "num_teeth": 8},
        resolved_sources={"tool_diameter": "runtime", "num_teeth": "runtime"},
        path=target,
    )

    loaded = load_tool_dataset_decisions(target)
    assert saved["status"] == "rejected"
    assert loaded[("site_a_line2", "machine_a1", 49)]["selection_mode"] == "manual"
    assert loaded[("site_a_line2", "machine_a1", 49)]["reference_tool_number"] == 44
    assert loaded[("site_a_line2", "machine_a1", 49)]["updated_by"] == "pytest"
    assert loaded[("site_a_line2", "machine_a1", 49)]["resolved_context"] == {}


def test_build_tool_dataset_overview_applies_confirmed_decision_snapshot():
    payload = build_tool_dataset_overview(
        coverage_items=[
            {
                "dataset": "site_c",
                "machine_id": "SITE_C - MACHINE_C1 - CASE_C1",
                "machine_family": "press_c-20-0482-010",
                "operation_id": "OF00001",
                "valid_tool_rows": 20,
                "resolved_dz_rows": 20,
                "harmonic_ready_rows": 20,
                "unique_tools": [2467],
                "resolved_master_tools": [2467],
                "resolved_d_tools": [2467],
                "resolved_z_tools": [2467],
                "resolved_dz_tools": [2467],
                "harmonic_ready_tools": [2467],
            },
        ],
        audit_rows=[
            {
                "machine_family": "press_c-20-0482-010",
                "tool_number": 2467,
                "master": {
                    "description": "Eck-Messerkopf Ingersoll",
                    "tool_type": "mill",
                    "diameter_mm": 84.0,
                    "teeth": None,
                },
                "reference": None,
                "runtime": None,
                "sindit": None,
                "process_plan": None,
                "flags": ["missing_num_teeth"],
                "harmonic_ready": False,
            }
        ],
        decisions={
            ("site_c_casedata", "press_c-20-0482-010", 2467): {
                "dataset_id": "site_c_casedata",
                "machine_family": "press_c-20-0482-010",
                "tool_number": 2467,
                "selection_mode": "default",
                "status": "confirmed",
                "notes": "Best guess: Ø84 Ingersoll shoulder heads typically carry 5 inserts.",
                "resolved_context": {
                    "tool_diameter": 84.0,
                    "num_teeth": 5,
                    "tool_type": "mill",
                },
                "resolved_sources": {
                    "tool_diameter": "master",
                    "num_teeth": "guess",
                    "tool_type": "master",
                },
            }
        },
    )

    row = payload["datasets"][0]["tools"][0]
    assert row["profiles"]["default"]["teeth"]["value"] == 5
    assert row["profiles"]["default"]["teeth"]["source"] == "guess"
    assert row["profiles"]["default"]["notes"] == "Best guess: Ø84 Ingersoll shoulder heads typically carry 5 inserts."
    assert row["decision_status"] == "confirmed"
    assert row["certainty"] == "defaulted"
    assert "missing_num_teeth" not in row["review_flags"]


def test_build_tool_dataset_decision_snapshot_supports_manual_reference_and_tooth_override():
    reference_row = {
        "profiles": {
            "default": {
                "label": "Default recommendation",
                "available": True,
                "tool_id": {"value": "PART0003", "source": "master"},
                "diameter_mm": {"value": 125.0, "source": "master"},
                "teeth": {"value": 7, "source": "guess"},
                "tool_type": {"value": "mill", "source": "master"},
                "tool_length_mm": {"value": 113.0, "source": "master"},
            }
        }
    }

    snapshot = build_tool_dataset_decision_snapshot(
        {"profiles": {"default": {"label": "Default recommendation", "available": True}}},
        "manual",
        reference_row=reference_row,
        manual_num_teeth=9,
    )

    assert snapshot["selection_mode"] == "manual"
    assert snapshot["resolved_context"]["tool_diameter"] == 125.0
    assert snapshot["resolved_context"]["num_teeth"] == 9
    assert snapshot["resolved_sources"]["tool_diameter"] == "master"
    assert snapshot["resolved_sources"]["num_teeth"] == "manual"


def test_sindit_tool_dataset_routes(tmp_path, monkeypatch):
    from backend.agents.processing import tool_dataset_decisions as tool_decisions_module
    from backend.routers import sindit as sindit_router

    async def fake_client_factory():
        return None, False, None

    async def fake_collect(*, client=None, dataset_id=None):
        return {
            "datasets": [
                {
                    "dataset_id": dataset_id or "site_b_casedata",
                    "label": "Site_b casedata",
                    "machine_ids": ["Site_b - MACHINE_B1 - CASE_B1"],
                    "machine_families": ["builder_b12"],
                    "operation_count": 1,
                    "summary": {
                        "tool_count": 1,
                        "certain_count": 1,
                        "defaulted_count": 0,
                        "needs_review_count": 0,
                        "confirmed_count": 0,
                        "rejected_count": 0,
                        "pending_count": 1,
                        "master_backed_count": 1,
                    },
                    "tools": [
                        {
                            "dataset_id": dataset_id or "site_b_casedata",
                            "dataset_label": "Site_b casedata",
                            "machine_family": "builder_b12",
                            "tool_number": 64,
                            "machine_ids": ["Site_b - MACHINE_B1 - CASE_B1"],
                            "operation_ids": ["OF00001"],
                            "operation_count": 1,
                            "coverage": {
                                "observed": True,
                                "master": True,
                                "diameter": True,
                                "teeth": True,
                                "diameter_and_teeth": True,
                                "harmonic_ready": True,
                            },
                            "profiles": {
                                "default": {
                                    "label": "Default recommendation",
                                    "available": True,
                                    "tool_id": {"value": "T64", "source": "master"},
                                    "diameter_mm": {"value": 20.0, "source": "master"},
                                    "teeth": {"value": 4, "source": "master"},
                                    "tool_type": {"value": "mill", "source": "master"},
                                    "tool_length_mm": {"value": 293.0, "source": "master"},
                                    "tool_material": {"value": "carbide", "source": "master"},
                                },
                                "manual": {
                                    "label": "Manual override",
                                    "available": True,
                                    "tool_id": {"value": "T64", "source": "master"},
                                    "diameter_mm": {"value": 20.0, "source": "master"},
                                    "teeth": {"value": 4, "source": "master"},
                                    "tool_type": {"value": "mill", "source": "master"},
                                    "tool_length_mm": {"value": 293.0, "source": "master"},
                                    "tool_material": {"value": "carbide", "source": "master"},
                                },
                            },
                            "available_profiles": ["default", "manual"],
                            "recommended_profile": "default",
                            "selected_profile": "default",
                            "decision": None,
                            "decision_status": "pending",
                            "certainty": "certain",
                            "certainty_reasons": ["Workbook master covers the core tool parameters."],
                            "review_flags": [],
                            "evidence_sources": ["master"],
                            "audit": {
                                "machine_family": "builder_b12",
                                "tool_number": 64,
                                "master": {
                                    "tool_id": "T64",
                                    "tool_type": "mill",
                                    "diameter_mm": 20.0,
                                    "teeth": 4,
                                    "tool_length_mm": 293.0,
                                    "tool_material": "carbide",
                                },
                                "flags": [],
                                "harmonic_ready": True,
                            },
                        },
                        {
                            "dataset_id": dataset_id or "site_b_casedata",
                            "dataset_label": "Site_b casedata",
                            "machine_family": "builder_b12",
                            "tool_number": 65,
                            "machine_ids": ["Site_b - MACHINE_B1 - CASE_B1"],
                            "operation_ids": ["OF00001"],
                            "operation_count": 1,
                            "coverage": {
                                "observed": True,
                                "master": False,
                                "diameter": False,
                                "teeth": False,
                                "diameter_and_teeth": False,
                                "harmonic_ready": False,
                            },
                            "profiles": {
                                "default": {
                                    "label": "Default recommendation",
                                    "available": True,
                                    "tool_id": {"value": "T65", "source": None},
                                    "diameter_mm": {"value": None, "source": None},
                                    "teeth": {"value": None, "source": None},
                                    "tool_type": {"value": None, "source": None},
                                    "tool_length_mm": {"value": None, "source": None},
                                    "tool_material": {"value": None, "source": None},
                                },
                                "manual": {
                                    "label": "Manual override",
                                    "available": True,
                                    "tool_id": {"value": "T65", "source": None},
                                    "diameter_mm": {"value": None, "source": None},
                                    "teeth": {"value": None, "source": None},
                                    "tool_type": {"value": None, "source": None},
                                    "tool_length_mm": {"value": None, "source": None},
                                    "tool_material": {"value": None, "source": None},
                                },
                            },
                            "available_profiles": ["default", "manual"],
                            "recommended_profile": "default",
                            "selected_profile": "default",
                            "decision": None,
                            "decision_status": "pending",
                            "certainty": "needs_review",
                            "certainty_reasons": ["Tooth count still unresolved."],
                            "review_flags": ["missing_tool_diameter", "missing_num_teeth"],
                            "evidence_sources": [],
                            "audit": {
                                "machine_family": "builder_b12",
                                "tool_number": 65,
                                "master": None,
                                "flags": ["missing_tool_diameter", "missing_num_teeth"],
                                "harmonic_ready": False,
                            },
                        },
                    ],
                }
            ],
            "total_datasets": 1,
            "total_tools": 2,
            "sindit_available": False,
        }

    monkeypatch.setattr(sindit_router, "_maybe_authenticated_sindit_client", fake_client_factory)
    monkeypatch.setattr("backend.agents.sindit.tool_audit.collect_tool_dataset_overview_payload", fake_collect)
    monkeypatch.setattr(tool_decisions_module, "DATASET_TOOL_DECISIONS_PATH", tmp_path / "dataset_tool_decisions.json")

    app = FastAPI()
    app.include_router(sindit_router.router)
    client = TestClient(app)

    response = client.get("/sindit/tools/datasets", params={"dataset_id": "site_b_casedata"})
    assert response.status_code == 200
    assert response.json()["datasets"][0]["dataset_id"] == "site_b_casedata"

    response = client.post(
        "/sindit/tools/datasets/decision",
        json={
            "dataset_id": "site_b_casedata",
            "machine_family": "builder_b12",
            "tool_number": 64,
            "status": "confirmed",
            "selection_mode": "default",
            "updated_by": "pytest",
        },
    )
    assert response.status_code == 200
    stored = load_tool_dataset_decisions(tmp_path / "dataset_tool_decisions.json")
    assert stored[("site_b_casedata", "builder_b12", 64)]["status"] == "confirmed"
    assert stored[("site_b_casedata", "builder_b12", 64)]["resolved_context"]["tool_diameter"] == 20.0
    assert stored[("site_b_casedata", "builder_b12", 64)]["resolved_context"]["num_teeth"] == 4

    manual_response = client.post(
        "/sindit/tools/datasets/decision",
        json={
            "dataset_id": "site_b_casedata",
            "machine_family": "builder_b12",
            "tool_number": 65,
            "status": "confirmed",
            "selection_mode": "manual",
            "reference_tool_number": 64,
            "manual_num_teeth": 6,
            "updated_by": "pytest",
        },
    )
    assert manual_response.status_code == 200
    stored = load_tool_dataset_decisions(tmp_path / "dataset_tool_decisions.json")
    assert stored[("site_b_casedata", "builder_b12", 65)]["selection_mode"] == "manual"
    assert stored[("site_b_casedata", "builder_b12", 65)]["reference_tool_number"] == 64
    assert stored[("site_b_casedata", "builder_b12", 65)]["resolved_context"]["tool_diameter"] == 20.0
    assert stored[("site_b_casedata", "builder_b12", 65)]["resolved_context"]["num_teeth"] == 6
    assert stored[("site_b_casedata", "builder_b12", 65)]["resolved_sources"]["num_teeth"] == "manual"
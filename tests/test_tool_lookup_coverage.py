from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.agents.processing.tool_lookup import ToolSpec
from backend.agents.processing import tool_lookup_coverage as coverage


def test_summarize_coverage_items_applies_thresholds() -> None:
    summary = coverage.summarize_coverage_items(
        [
            {
                "dataset": "site_b_olddata",
                "valid_tool_rows": 100,
                "resolved_master_rows": 100,
                "resolved_d_rows": 100,
                "resolved_z_rows": 95,
                "resolved_dz_rows": 95,
                "harmonic_ready_rows": 90,
            },
            {
                "dataset": "site_a_casedata",
                "valid_tool_rows": 10,
                "resolved_master_rows": 10,
                "resolved_d_rows": 10,
                "resolved_z_rows": 10,
                "resolved_dz_rows": 10,
                "harmonic_ready_rows": 10,
            },
            {
                "dataset": "site_a_line2",
                "valid_tool_rows": 10,
                "resolved_master_rows": 10,
                "resolved_d_rows": 10,
                "resolved_z_rows": 6,
                "resolved_dz_rows": 6,
                "harmonic_ready_rows": 6,
            },
            {
                "dataset": "site_c",
                "valid_tool_rows": 10,
                "resolved_master_rows": 3,
                "resolved_d_rows": 10,
                "resolved_z_rows": 3,
                "resolved_dz_rows": 3,
                "harmonic_ready_rows": 3,
            },
        ]
    )

    assert summary["datasets"]["site_b_olddata"]["passes"] is True
    assert summary["datasets"]["site_a_casedata"]["passes"] is True
    assert summary["datasets"]["site_a_line2"]["passes"] is True
    assert summary["datasets"]["site_c"]["passes"] is False
    assert coverage.build_threshold_failures(summary) == [
        "site_c: coverage_master=0.300 < threshold=0.400"
    ]


def test_collect_site_a_line2_coverage_uses_raw_teeth_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    session_dir = repo_root / "data" / "Site_a_line2" / "Monitored data" / "2026_03_03-04"
    session_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "Date": [
                "2026-03-03T00:00:00Z",
                "2026-03-03T00:00:01Z",
            ],
            "UF5-Numero_de_pieza_OF": [100003888, 100003888],
            "Cnc_Tool_Number_RT": [201, 201],
            "SpindleSpeedActual": [1800.0, 1810.0],
            "Axis_FeedRate_actual": [900.0, 910.0],
        }
    ).to_csv(session_dir / "A_TYZBPS.csv", index=False)

    pd.DataFrame(
        {
            "Date": [
                "2026-03-03T00:00:00Z",
                "2026-03-03T00:00:01Z",
            ],
            "CNC_parameters_teeth_num": [2, 2],
        }
    ).to_csv(session_dir / "B_DLG6CF.csv", index=False, sep=";")

    def fake_lookup(machine_family: str, tool_number: int | float | str | None):
        assert machine_family == "machine_a1"
        assert int(tool_number or 0) == 201
        return ToolSpec(
            machine_family="machine_a1",
            tool_number=201,
            tool_id="PART0001",
            description="FEEDMILL 125",
            tool_type="mill",
            diameter_mm=125.0,
            teeth=None,
            source="site_a/Machine_a1.xlsx",
        )

    monkeypatch.setattr(coverage, "lookup_tool_spec", fake_lookup)

    items = coverage.collect_site_a_line2_coverage_items(repo_root / "data" / "Site_a_line2")

    assert len(items) == 1
    item = items[0]
    assert item["dataset"] == "site_a_line2"
    assert item["machine_family"] == "machine_a1"
    assert item["operation_id"] == "OF00011"
    assert item["coverage_diameter"] == 1.0
    assert item["coverage_teeth"] == 1.0
    assert item["coverage_dz"] == 1.0
    assert item["harmonic_ready"] == 1.0
    assert item["tools_missing_master_teeth"] == [201]


def test_collect_casedata_coverage_handles_direct_of_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    olddata_root = tmp_path / "data" / "olddata" / "OF00001"
    olddata_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": [
                "2026-03-03T00:00:00Z",
                "2026-03-03T00:00:01Z",
            ],
            "Tool_Number": [6, 6],
            "Spindle_Speed_Actual": [628.0, 628.0],
            "Feed_Rate_Actual": [1034.0, 1034.0],
        }
    ).to_csv(olddata_root / "OF00001_G_FAKE_TYZBPS.csv", index=False)

    def fake_lookup(machine_family: str, tool_number: int | float | str | None):
        assert machine_family == "builder_b12"
        assert int(tool_number or 0) == 6
        return ToolSpec(
            machine_family="builder_b12",
            tool_number=6,
            tool_id="T06",
            description="FINISH BORE 65MM DIA",
            tool_type="bore",
            diameter_mm=65.0,
            teeth=1,
            source="site_b/Builder_b1 2 Tooling Database.xlsx",
        )

    monkeypatch.setattr(coverage, "lookup_tool_spec", fake_lookup)

    items = coverage.collect_casedata_coverage_items(
        tmp_path / "data" / "olddata",
        machine_id_override="olddata",
    )

    assert len(items) == 1
    item = items[0]
    assert item["dataset"] == "site_b_olddata"
    assert item["machine_id"] == "olddata"
    assert item["machine_family"] == "builder_b12"
    assert item["operation_id"] == "OF00001"
    assert item["coverage_dz"] == 1.0
    assert item["harmonic_ready"] == 1.0


def test_collect_casedata_coverage_classifies_site_a_case_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    site_a_root = tmp_path / "data" / "site_a" / "Site_a - MACHINE_A1 - CASE_A1" / "OF00001"
    site_a_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": [
                "2026-03-03T00:00:00Z",
                "2026-03-03T00:00:01Z",
            ],
            "Tool_Number": [44, 44],
            "Spindle_Speed_Actual": [1800.0, 1800.0],
            "Feed_Rate_Actual": [900.0, 900.0],
        }
    ).to_csv(site_a_root / "OF00001_G_FAKE_TYZBPS.csv", index=False)

    def fake_lookup(machine_family: str, tool_number: int | float | str | None):
        assert machine_family == "machine_a1"
        assert int(tool_number or 0) == 44
        return ToolSpec(
            machine_family="machine_a1",
            tool_number=44,
            tool_id="M000974",
            description="FRESA Ø125X90º T490",
            tool_type="mill",
            diameter_mm=125.0,
            teeth=7,
            source="site_a/Machine_a1.xlsx",
        )

    monkeypatch.setattr(coverage, "lookup_tool_spec", fake_lookup)

    items = coverage.collect_coverage_items(tmp_path)

    assert len(items) == 1
    item = items[0]
    assert item["dataset"] == "site_a_casedata"
    assert item["machine_id"] == "Site_a - MACHINE_A1 - CASE_A1"
    assert item["machine_family"] == "machine_a1"
    assert item["coverage_dz"] == 1.0
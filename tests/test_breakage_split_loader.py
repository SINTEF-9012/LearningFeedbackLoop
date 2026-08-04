from pathlib import Path

import pandas as pd

from backend.agents.memory.breakage_experiment_runner import LiveBreakageExperimentRunner
from backend.agents.processing import tool_dataset_decisions as tool_dataset_decisions_module
from backend.agents.processing.tool_lookup import FAMILY_MACHINE_A1


def test_load_split_dataset_normalizes_operations_and_labels(tmp_path, monkeypatch) -> None:
    split_dir = tmp_path / "splits" / "PART0001_excel"
    split_dir.mkdir(parents=True)

    pd.DataFrame([
        {
            "window_start": "2026-03-04T19:54:00Z",
            "Spdl_actual_power": 10.0,
            "SpindleSpeedActual": 560.0,
            "Axis_FeedRate_actual": 6200.0,
            "Accel_Severity_Acc_1_range1_Severity": 1.2,
            "Accel_Severity_Acc_2_range2_Severity": 0.7,
            "Spindle_Temperature_degreeCelsius_d1": 21.5,
            "Cnc_Tool_Number_RT": 2,
            "CNC_parameters_teeth_num": 4,
            "session": "2026_03_03-04",
            "uf5_of": 100003889,
            "label": "tool_wear",
            "embargoed": False,
        },
        {
            "window_start": "2026-03-04T19:55:00Z",
            "Spdl_actual_power": 0.0,
            "SpindleSpeedActual": 0.0,
            "Axis_FeedRate_actual": 0.0,
            "Accel_Severity_Acc_1_range1_Severity": 0.1,
            "Accel_Severity_Acc_2_range2_Severity": 0.05,
            "Spindle_Temperature_degreeCelsius_d1": 21.0,
            "Cnc_Tool_Number_RT": 2,
            "CNC_parameters_teeth_num": 4,
            "session": "2026_03_03-04",
            "uf5_of": 100003892,
            "label": "normal",
            "embargoed": False,
        },
        {
            "window_start": "2026-03-04T19:56:00Z",
            "Spdl_actual_power": 0.0,
            "SpindleSpeedActual": 0.0,
            "Axis_FeedRate_actual": 0.0,
            "Accel_Severity_Acc_1_range1_Severity": 0.1,
            "Accel_Severity_Acc_2_range2_Severity": 0.05,
            "Spindle_Temperature_degreeCelsius_d1": 21.0,
            "Cnc_Tool_Number_RT": 2,
            "CNC_parameters_teeth_num": 4,
            "session": "2026_03_03-04",
            "uf5_of": 100003892,
            "label": "pre_break",
            "embargoed": True,
        },
    ]).to_csv(split_dir / "site_a_line2_PART0001_labeled.csv", index=False)
    (split_dir / "overall_summary.json").write_text("{}")

    monkeypatch.setattr(
        "backend.agents.memory.breakage_experiment_runner._SPLITS_ROOT",
        tmp_path / "splits",
    )

    runner = LiveBreakageExperimentRunner(dataset="site_a_line2", label_scheme="conservative")
    df, source = runner._load_dataset_frame()

    assert source.kind == "split"
    assert source.split_name == "PART0001_excel"
    assert len(df) == 2
    assert sorted(df["operation_id"].unique().tolist()) == ["OF00012", "OF00014"]
    assert sorted(df["label"].unique().tolist()) == ["normal", "pre_stoppage"]
    assert df["machine_family"].dropna().unique().tolist() == [FAMILY_MACHINE_A1]
    assert int(df.loc[df["operation_id"] == "OF00012", "tool_number"].iloc[0]) == 2
    assert df.loc[df["operation_id"] == "OF00012", "tool_id"].iloc[0] == "PART0003"
    assert df.loc[df["operation_id"] == "OF00012", "tool_type"].iloc[0] == "mill"
    assert float(df.loc[df["operation_id"] == "OF00012", "tool_diameter"].iloc[0]) == 125.0
    assert int(df.loc[df["operation_id"] == "OF00012", "num_teeth"].iloc[0]) == 4
    assert float(df.loc[df["operation_id"] == "OF00012", "tool_length"].iloc[0]) == 113.0
    assert df.loc[df["operation_id"] == "OF00012", "sindit_tool_iri"].iloc[0] == "urn:lfl:tool:machine_a1-t2"
    assert float(df.loc[df["operation_id"] == "OF00012", "power_spindle_mean"].iloc[0]) == 10.0
    assert float(df.loc[df["operation_id"] == "OF00012", "vib_severity_x_mean"].iloc[0]) == 1.2


def test_load_split_dataset_original_scheme_keeps_only_hard_break_positive(tmp_path, monkeypatch) -> None:
    split_dir = tmp_path / "splits" / "PART0001_excel"
    split_dir.mkdir(parents=True)

    pd.DataFrame([
        {
            "window_start": "2026-03-04T19:54:00Z",
            "Spdl_actual_power": 10.0,
            "SpindleSpeedActual": 560.0,
            "Axis_FeedRate_actual": 6200.0,
            "Accel_Severity_Acc_1_range1_Severity": 1.2,
            "Accel_Severity_Acc_2_range2_Severity": 0.7,
            "Spindle_Temperature_degreeCelsius_d1": 21.5,
            "Cnc_Tool_Number_RT": 2,
            "CNC_parameters_teeth_num": 4,
            "session": "2026_03_03-04",
            "uf5_of": 100003889,
            "label": "tool_wear",
            "embargoed": False,
        },
        {
            "window_start": "2026-03-04T19:55:00Z",
            "Spdl_actual_power": 12.0,
            "SpindleSpeedActual": 560.0,
            "Axis_FeedRate_actual": 6200.0,
            "Accel_Severity_Acc_1_range1_Severity": 1.5,
            "Accel_Severity_Acc_2_range2_Severity": 0.8,
            "Spindle_Temperature_degreeCelsius_d1": 21.7,
            "Cnc_Tool_Number_RT": 2,
            "CNC_parameters_teeth_num": 4,
            "session": "2026_03_03-04",
            "uf5_of": 100003891,
            "label": "pre_break",
            "embargoed": False,
        },
    ]).to_csv(split_dir / "site_a_line2_PART0001_labeled.csv", index=False)
    (split_dir / "overall_summary.json").write_text("{}")

    monkeypatch.setattr(
        "backend.agents.memory.breakage_experiment_runner._SPLITS_ROOT",
        tmp_path / "splits",
    )

    runner = LiveBreakageExperimentRunner(dataset="site_a_line2", label_scheme="original")
    df, _ = runner._load_dataset_frame()

    labels_by_of = dict(zip(df["operation_id"], df["label"]))
    assert labels_by_of["OF00012"] == "normal"
    assert labels_by_of["OF00013"] == "pre_stoppage"


def test_load_split_dataset_applies_confirmed_dataset_tool_decision(tmp_path, monkeypatch) -> None:
    target = tmp_path / "dataset_tool_decisions.json"
    monkeypatch.setattr(tool_dataset_decisions_module, "DATASET_TOOL_DECISIONS_PATH", target)
    tool_dataset_decisions_module.save_tool_dataset_decision(
        dataset_id="site_a_line2",
        machine_family=FAMILY_MACHINE_A1,
        tool_number=2,
        status="confirmed",
        selection_mode="default",
        resolved_context={
            "tool_id": "CONFIRMED-T2",
            "tool_type": "boring_head",
            "tool_diameter": 126.5,
            "num_teeth": 7,
            "tool_length": 115.0,
        },
        resolved_sources={
            "tool_id": "runtime",
            "tool_type": "reference",
            "tool_diameter": "reference",
            "num_teeth": "runtime",
            "tool_length": "reference",
        },
    )

    split_dir = tmp_path / "splits" / "PART0001_excel"
    split_dir.mkdir(parents=True)

    pd.DataFrame([
        {
            "window_start": "2026-03-04T19:54:00Z",
            "Spdl_actual_power": 10.0,
            "SpindleSpeedActual": 560.0,
            "Axis_FeedRate_actual": 6200.0,
            "Accel_Severity_Acc_1_range1_Severity": 1.2,
            "Accel_Severity_Acc_2_range2_Severity": 0.7,
            "Spindle_Temperature_degreeCelsius_d1": 21.5,
            "Cnc_Tool_Number_RT": 2,
            "CNC_parameters_teeth_num": 4,
            "session": "2026_03_03-04",
            "uf5_of": 100003889,
            "label": "normal",
            "embargoed": False,
        },
    ]).to_csv(split_dir / "site_a_line2_PART0001_labeled.csv", index=False)
    (split_dir / "overall_summary.json").write_text("{}")

    monkeypatch.setattr(
        "backend.agents.memory.breakage_experiment_runner._SPLITS_ROOT",
        tmp_path / "splits",
    )

    runner = LiveBreakageExperimentRunner(dataset="site_a_line2", label_scheme="conservative")
    df, _ = runner._load_dataset_frame()

    row = df.iloc[0]
    assert row["tool_id"] == "CONFIRMED-T2"
    assert row["tool_type"] == "boring_head"
    assert float(row["tool_diameter"]) == 126.5
    assert int(row["num_teeth"]) == 7
    assert float(row["tool_length"]) == 115.0

def test_v2_v3_schemes_bypass_split_and_load_csv(tmp_path, monkeypatch) -> None:
    """v2/v3 label schemes must load their own CSV even when a split bundle exists."""
    split_dir = tmp_path / "splits" / "PART0001_excel"
    split_dir.mkdir(parents=True)
    (split_dir / "site_a_line2_PART0001_labeled.csv").write_text("window_start,label\n")
    (split_dir / "overall_summary.json").write_text("{}")

    pd.DataFrame([
        {"power_spindle_mean": 1.0, "operation_id": "OF889", "label": "pre_break",
         "timestamp": "2026-03-04T19:54:00Z", "session": "2026_03_03-04"},
        {"power_spindle_mean": 0.5, "operation_id": "OF888", "label": "normal",
         "timestamp": "2026-03-04T19:55:00Z", "session": "2026_03_03-04"},
    ]).to_csv(tmp_path / "site_a_line2_breakage_v2.csv", index=False)

    monkeypatch.setattr(
        "backend.agents.memory.breakage_experiment_runner._SPLITS_ROOT",
        tmp_path / "splits",
    )
    monkeypatch.setattr(
        "backend.agents.memory.breakage_experiment_runner._FEATURES_ROOT",
        tmp_path,
    )

    runner = LiveBreakageExperimentRunner(dataset="site_a_line2", label_scheme="v2")
    assert runner._resolve_split_dir() is None

    df, source = runner._load_dataset_frame()
    assert source.source_path.endswith("site_a_line2_breakage_v2.csv")
    assert sorted(df["label"].unique()) == ["normal", "pre_stoppage"]

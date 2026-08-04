from __future__ import annotations

import pandas as pd

from backend.agents.processing.dataset_loader import DatasetLoader


def _write_csv(path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_loader_detects_vibration_by_header_and_falls_back_to_machine_state_power(tmp_path) -> None:
    op_dir = tmp_path / "site_a" / "Site_a - MACHINE_A1 - CASE_A1" / "OF00001"
    timestamps = pd.to_datetime([
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:01Z",
    ])

    _write_csv(
        op_dir / "sample_BXCZ3M.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "Operation_Status": [3, 3],
                "Power_Y": [4.0, 5.0],
                "Power_Z": [6.0, 7.0],
            }
        ),
    )
    _write_csv(
        op_dir / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "Feed_Rate_Actual": [120.0, 121.0],
                "Spindle_Speed_Actual": [5000.0, 5001.0],
                "Temperature_Head": [30.0, 31.0],
                "Tool_Number": [2, 2],
                "Power_Spindle": [10.0, 14.0],
                "Power_Active": [20.0, 24.0],
            }
        ),
    )
    _write_csv(
        op_dir / "sample_7N4ZJ8.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "Vibration_Severity_X": [1.0, 3.0],
                "Vibration_Severity_Y": [2.0, 4.0],
                "Chatter_Detection_OnOff_X": [0, 1],
                "Chatter_Detection_OnOff_Y": [0, 0],
                "Chatter_Detection_Amplitude_X": [0.0, 1.5],
                "Chatter_Detection_Amplitude_Y": [0.0, 0.0],
                "Chatter_Detection_Frequency_X": [0.0, 250.0],
                "Chatter_Detection_Frequency_Y": [0.0, 0.0],
            }
        ),
    )

    loader = DatasetLoader(tmp_path / "site_a")
    operation = loader.list_operations()[0]

    assert sorted(operation.channel_files.keys()) == ["axis_power", "machine_state", "vibration"]

    window = loader.extract_window(
        "OF00001",
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:01+00:00",
    )
    features = window.compute_features()

    assert window.vibration is not None
    assert features["power_spindle_mean"] == 12.0
    assert features["power_active_mean"] == 22.0
    assert features["vib_severity_x_mean"] == 2.0
    assert features["chatter_x_count"] == 1.0


def test_loader_supports_flat_olddata_layout(tmp_path) -> None:
    op_dir = tmp_path / "olddata" / "OF00001"
    timestamps = pd.to_datetime([
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:01Z",
    ])

    _write_csv(
        op_dir / "OF00001_G_CASE_A1_BXCZ3M.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "Operation_Status": [3, 3],
                "Power_Y": [4.0, 5.0],
            }
        ),
    )
    _write_csv(
        op_dir / "OF00001_G_CASE_A1_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "Feed_Rate_Actual": [120.0, 121.0],
                "Spindle_Speed_Actual": [5000.0, 5001.0],
                "Tool_Number": [2, 2],
            }
        ),
    )
    _write_csv(
        op_dir / "OF00001_G_CASE_A1_7N4ZJ8.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "Vibration_Severity_X": [1.0, 3.0],
                "Vibration_Severity_Y": [2.0, 4.0],
            }
        ),
    )

    loader = DatasetLoader(tmp_path / "olddata")
    cases = loader.list_cases()
    operations = loader.list_operations()

    assert cases == ["G_CASE_A1"]
    assert len(operations) == 1
    assert operations[0].case_dir == "G_CASE_A1"
    assert operations[0].tool_id == "A1"
    assert sorted(operations[0].channel_files.keys()) == ["axis_power", "machine_state", "vibration"]


def test_loader_preserves_duplicate_operation_ids_across_cases(tmp_path) -> None:
    timestamps = pd.to_datetime([
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:01Z",
    ])

    for case_name in (
        "Site_b - MACHINE_B1 - CASE_B1",
        "Site_b - MACHINE_B2 - CASE_B2",
    ):
        op_dir = tmp_path / "casedata" / case_name / "OF00001"
        _write_csv(
            op_dir / "sample_TYZBPS.csv",
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "Feed_Rate_Actual": [120.0, 121.0],
                    "Spindle_Speed_Actual": [5000.0, 5001.0],
                    "Tool_Number": [2, 2],
                }
            ),
        )

    loader = DatasetLoader(tmp_path / "casedata")

    assert loader.list_cases() == [
        "Site_b - MACHINE_B1 - CASE_B1",
        "Site_b - MACHINE_B2 - CASE_B2",
    ]

    all_operations = loader.list_operations()
    assert len(all_operations) == 2
    assert [operation.case_dir for operation in all_operations] == [
        "Site_b - MACHINE_B1 - CASE_B1",
        "Site_b - MACHINE_B2 - CASE_B2",
    ]

    assert [operation.operation_id for operation in loader.list_operations("Site_b - MACHINE_B1 - CASE_B1")] == ["OF00001"]
    assert [operation.operation_id for operation in loader.list_operations("Site_b - MACHINE_B2 - CASE_B2")] == ["OF00001"]
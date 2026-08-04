from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd

from scripts.detect_premature_stoppage import analyse, write_json_report
from scripts.extract_pre_stoppage_patterns import run_extraction


CASE_DIR = "Site_a - MACHINE_A1 - CASE_A1"


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _build_operation(op_dir: Path, *, seconds: int, stop_at: int | None = None) -> None:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=seconds, freq="s")

    spindle = [500.0] * seconds
    feed = [150.0] * seconds
    override = [100.0] * seconds
    status = [3.0] * seconds
    if stop_at is not None:
        for index in range(stop_at, seconds):
            spindle[index] = 0.0
            feed[index] = 0.0
            override[index] = 0.0
            status[index] = 0.0

    _write_csv(
        op_dir / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "Feed_Override": override,
                "Feed_Rate_Actual": feed,
                "Feed_Rate_Commanded": [150.0] * seconds,
                "Spindle_Speed_Actual": spindle,
                "Spindle_Speed_Commanded": [500.0] * seconds,
                "Spindle_Speed_Override": [100.0] * seconds,
                "Tool_Number": [2] * seconds,
                "Program_Name": ["SITE_A_PLAN"] * seconds,
                "Position_MCS_X": list(range(seconds)),
                "Position_MCS_Y": list(range(seconds)),
                "Position_MCS_Z": list(range(seconds)),
                "Temperature_Head": [25.0] * seconds,
                "Temperature_Room": [20.0] * seconds,
            }
        ),
    )
    _write_csv(
        op_dir / "sample_BXCZ3M.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "Operation_Status": status,
                "Power_Spindle": [12.0] * seconds,
                "Power_X1": [2.0] * seconds,
                "Power_X2": [2.0] * seconds,
                "Power_Y": [2.0] * seconds,
                "Power_Z": [2.0] * seconds,
            }
        ),
    )


def _archive_operation(op_dir: Path) -> None:
    archive_path = op_dir.parent / f"{op_dir.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for csv_path in sorted(op_dir.glob("*.csv")):
            archive.add(csv_path, arcname=csv_path.name)
    shutil.rmtree(op_dir)


def test_run_extraction_uses_site_a_weak_label_report_with_archives(tmp_path: Path) -> None:
    root = tmp_path / "site_a" / CASE_DIR
    _build_operation(root / "OF00001", seconds=120)
    _build_operation(root / "OF00002", seconds=120)
    _build_operation(root / "OF00003", seconds=30, stop_at=25)
    _archive_operation(root / "OF00002")
    _archive_operation(root / "OF00003")

    events, operator_events, baselines = analyse(tmp_path / "site_a", case=CASE_DIR)
    report_path = tmp_path / "out" / "site_a_unexpected_stops.json"
    write_json_report(report_path, tmp_path / "site_a", events, operator_events, baselines)

    samples, _, loaded_stops = run_extraction(
        data_dir=tmp_path / "site_a",
        case=CASE_DIR,
        output_dir=tmp_path / "features",
        save_raw=False,
        weak_label_report=report_path,
        include_candidate_labels=True,
    )

    features_path = tmp_path / "features" / "stoppage_features.csv"
    metadata_path = tmp_path / "features" / "extraction_metadata.json"
    frame = pd.read_csv(features_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    timestamped_candidates = [
        event
        for event in events
        if isinstance(getattr(event, "timestamp", None), str) is False or getattr(event, "timestamp", None) != "n/a (aggregate)"
    ]

    assert features_path.is_file()
    assert metadata_path.is_file()
    assert len(samples) == len(frame)
    assert set(frame["case_dir"]) == {CASE_DIR}
    assert set(frame["operation_id"]).issuperset({"OF00001", "OF00003"})
    assert set(frame["label"]) == {"normal", "pre_stoppage"}
    assert metadata["weak_label_report"] == str(report_path)
    assert metadata["include_candidate_labels"] is True
    assert metadata["cases"] == [CASE_DIR]
    assert sum(len(events_for_op) for events_for_op in loaded_stops.values()) == len(operator_events) + len(timestamped_candidates)
    assert any(frame["operation_id"] == "OF00003")


def test_run_extraction_relative_root_matches_case_aware_report(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "site_a" / CASE_DIR
    _build_operation(root / "OF00001", seconds=120)
    _build_operation(root / "OF00002", seconds=120)
    _build_operation(root / "OF00003", seconds=30, stop_at=25)
    _archive_operation(root / "OF00002")
    _archive_operation(root / "OF00003")

    events, operator_events, baselines = analyse(tmp_path / "site_a", case=CASE_DIR)
    report_path = tmp_path / "out" / "site_a_unexpected_stops.json"
    write_json_report(report_path, tmp_path / "site_a", events, operator_events, baselines)

    monkeypatch.chdir(tmp_path)
    samples, _, _ = run_extraction(
        data_dir=Path("site_a"),
        case=CASE_DIR,
        output_dir=tmp_path / "features_rel",
        save_raw=False,
        weak_label_report=report_path,
    )

    positives = [sample for sample in samples if sample.label == "pre_stoppage"]
    assert positives
    assert all(sample.case_dir == CASE_DIR for sample in positives)
    assert any(sample.operation_id == "OF00003" for sample in positives)
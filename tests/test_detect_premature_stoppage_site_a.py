from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd

from scripts.detect_premature_stoppage import analyse, write_json_report


CASE_DIR = "Site_a - MACHINE_A1 - CASE_A1"


def _write_csv(path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _build_operation(op_dir, *, seconds: int, stop_at: int | None = None) -> None:
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
                "Tool_Number": [2] * seconds,
                "Program_Name": ["SITE_A_PLAN"] * seconds,
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
            }
        ),
    )


def _archive_operation(op_dir) -> None:
    archive_path = op_dir.parent / f"{op_dir.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for csv_path in sorted(op_dir.glob("*.csv")):
            archive.add(csv_path, arcname=csv_path.name)
    shutil.rmtree(op_dir)


def test_analyse_site_a_supports_loader_discovery_and_process_plan(tmp_path) -> None:
    root = tmp_path / "site_a" / CASE_DIR
    _build_operation(root / "OF00001", seconds=120)
    _build_operation(root / "OF00002", seconds=120)
    _build_operation(root / "OF00003", seconds=30, stop_at=25)

    events, operator_events, baselines = analyse(
        tmp_path / "site_a",
        case=CASE_DIR,
    )

    assert 2.0 in baselines
    short_event = next(event for event in events if event.operation_id == "OF00003")
    assert short_event.case_dir == CASE_DIR
    assert short_event.expected_duration_s > 100.0
    assert short_event.deficit_pct > 70.0
    assert short_event.context["machine_family"] == "machine_a1"
    assert "OP20" in short_event.context["process_plan"]["operation_ids"]

    operator_event = next(event for event in operator_events if event.operation_id == "OF00003")
    assert operator_event.case_dir == CASE_DIR
    assert operator_event.context["machine_family"] == "machine_a1"
    assert "OP20" in operator_event.context["process_plan"]["operation_ids"]


def test_write_json_report_exports_weak_labels_with_plan_context(tmp_path) -> None:
    root = tmp_path / "site_a" / CASE_DIR
    _build_operation(root / "OF00001", seconds=120)
    _build_operation(root / "OF00002", seconds=120)
    _build_operation(root / "OF00003", seconds=30, stop_at=25)

    events, operator_events, baselines = analyse(
        tmp_path / "site_a",
        case=CASE_DIR,
    )
    report_path = tmp_path / "out" / "site_a_unexpected_stops.json"

    write_json_report(
        report_path,
        tmp_path / "site_a",
        events,
        operator_events,
        baselines,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path.is_file()
    assert payload["weak_labels"]
    assert any(label["label"] == "unexpected_stop" for label in payload["weak_labels"])
    assert any(
        "OP20" in ((label.get("context") or {}).get("process_plan") or {}).get("operation_ids", [])
        for label in payload["weak_labels"]
    )


def test_analyse_site_a_extracts_archived_operations_for_baselines(tmp_path) -> None:
    root = tmp_path / "site_a" / CASE_DIR
    _build_operation(root / "OF00001", seconds=120)
    _build_operation(root / "OF00002", seconds=120)
    _build_operation(root / "OF00003", seconds=30, stop_at=25)
    _archive_operation(root / "OF00002")
    _archive_operation(root / "OF00003")

    events, operator_events, baselines = analyse(
        tmp_path / "site_a",
        case=CASE_DIR,
    )

    assert 2.0 in baselines
    assert any(event.operation_id == "OF00003" for event in events)
    assert any(event.operation_id == "OF00003" for event in operator_events)


def test_analyse_site_a_relative_root_keeps_unpacked_operations(tmp_path, monkeypatch) -> None:
    root = tmp_path / "site_a" / CASE_DIR
    _build_operation(root / "OF00001", seconds=120)
    _build_operation(root / "OF00002", seconds=120)
    _build_operation(root / "OF00003", seconds=30, stop_at=25)
    _archive_operation(root / "OF00002")
    _archive_operation(root / "OF00003")

    monkeypatch.chdir(tmp_path)
    events, operator_events, baselines = analyse(
        Path("site_a"),
        case=CASE_DIR,
    )

    assert 2.0 in baselines
    assert any("OF00001" in op_key for op_key in baselines[2.0].total_durations)
    assert any(event.operation_id == "OF00003" for event in events)
    assert any(event.operation_id == "OF00003" for event in operator_events)
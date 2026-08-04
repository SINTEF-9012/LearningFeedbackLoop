from __future__ import annotations

import asyncio
import time

import pandas as pd
import pytest

from backend.ingestion.registry import create_source, registered_sources
from backend.ingestion.schema import FrameEnvelope
from backend.ingestion.simulated_casedata import SimulatedCasedataSource


def _write_case_csv(path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


@pytest.mark.asyncio
async def test_simulated_casedata_source_merges_operation_rows_and_publishes_envelopes(tmp_path, monkeypatch):
    root = tmp_path / "casedata"
    op_dir = root / "Case A - Tool 1" / "OF0001"
    op_dir.mkdir(parents=True)

    _write_case_csv(
        op_dir / "sample_BXCZ3M.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime([
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                ]),
                "Operation_Status": [1, 1],
                "Power_Spindle": [10.0, 11.0],
                "Power_Y": [2.0, 2.5],
            }
        ),
    )
    _write_case_csv(
        op_dir / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime([
                    "2026-01-01T00:00:00.200Z",
                    "2026-01-01T00:00:01.200Z",
                ]),
                "Feed_Rate_Actual": [120.0, 121.0],
                "Spindle_Speed_Actual": [5000.0, 5001.0],
                "Temperature_Head": [31.0, 31.5],
                "Tool_Number": [7, 7],
            }
        ),
    )

    published: list[tuple[str, FrameEnvelope]] = []

    async def fake_publish(session_id: str, payload: FrameEnvelope) -> None:
        published.append((session_id, payload))

    monkeypatch.setattr("backend.ingestion.simulated_casedata.publish_feature", fake_publish)

    queue: asyncio.Queue = asyncio.Queue()
    sessions = {
        "session-1": {
            "session_id": "session-1",
            "config": {"speed": 1000.0},
            "data": {},
            "metadata": {},
            "running": True,
            "paused": False,
            "subscribers": [queue],
            "task": None,
        }
    }

    source = SimulatedCasedataSource(
        sessions,
        casedata_root=root,
        operation_id="OF0001",
    )

    await source.run("session-1")

    first_frame = await asyncio.wait_for(queue.get(), timeout=1.0)
    second_frame = await asyncio.wait_for(queue.get(), timeout=1.0)
    eos = await asyncio.wait_for(queue.get(), timeout=1.0)

    assert first_frame["Power_Spindle"] == 10.0
    assert first_frame["Feed_Rate_Actual"] == 120.0
    assert second_frame["Spindle_Speed_Actual"] == 5001.0
    assert eos["eos"] is True
    assert len(published) == 2
    assert all(isinstance(item[1], FrameEnvelope) for item in published)
    assert published[0][1].signals["Power_Spindle"] == 10.0
    assert published[0][1].metadata["operation_id"] == "OF0001"
    assert source.status("session-1")["operation_id"] == "OF0001"


def test_source_registry_exposes_simulated_casedata():
    assert "simulated_casedata" in registered_sources()

    source = create_source(
        "simulated_casedata",
        {},
        casedata_root="data/casedata",
        operation_id="OF0001",
    )

    assert isinstance(source, SimulatedCasedataSource)


def test_simulated_casedata_source_resolves_duplicate_operation_ids_by_case(tmp_path):
    root = tmp_path / "casedata"
    case_a = root / "Site_b - MACHINE_B1 - CASE_B1" / "OF00001"
    case_b = root / "SITE_C - MACHINE_C1 - CASE_C1" / "OF00001"
    case_a.mkdir(parents=True)
    case_b.mkdir(parents=True)

    _write_case_csv(
        case_a / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"]),
                "Feed_Rate_Actual": [120.0],
                "Spindle_Speed_Actual": [5000.0],
                "Temperature_Head": [31.0],
                "Tool_Number": [7],
            }
        ),
    )
    _write_case_csv(
        case_b / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"]),
                "Feed_Rate_Actual": [80.0],
                "Spindle_Speed_Actual": [2000.0],
                "Temperature_Head": [27.0],
                "Tool_Number": [4],
            }
        ),
    )

    sessions = {
        "session-1": {
            "session_id": "session-1",
            "config": {"speed": 1.0},
            "data": {},
            "metadata": {},
            "running": False,
            "paused": False,
            "subscribers": [],
            "task": None,
        }
    }

    resolved = SimulatedCasedataSource.resolve_operation_id(
        root,
        "OF00001",
        case_dir="Site_b - MACHINE_B1 - CASE_B1",
    )
    source = SimulatedCasedataSource(
        sessions,
        casedata_root=root,
        operation_id=resolved,
        case_dir="Site_b - MACHINE_B1 - CASE_B1",
    )

    data, metadata = source.session_data()

    assert metadata["casedata"]["case_dir"] == "Site_b - MACHINE_B1 - CASE_B1"
    assert data["Feed_Rate_Actual"] == [120.0]
    assert source.status("session-1")["case_dir"] == "Site_b - MACHINE_B1 - CASE_B1"


def test_simulated_casedata_session_data_keeps_vibration_feature_columns(tmp_path):
    root = tmp_path / "casedata"
    op_dir = root / "SITE_C - MACHINE_C1 - CASE_C1" / "OF00001"
    op_dir.mkdir(parents=True)

    _write_case_csv(
        op_dir / "sample_7DTZHE.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime([
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                ]),
                "Vibration_Harmonic_1_X_Amplitude": [0.0, 0.3],
                "Vibration_Harmonic_1_X_Frequency": [100.0, 101.0],
                "Vibration_Peak_1_X_Frequency": [40.0, 42.0],
                "Vibration_Peak_1_X_Amplitude": [1.2, 1.5],
                "Chatter_Detection_Amplitude_X": [0.1, 0.2],
            }
        ),
    )
    _write_case_csv(
        op_dir / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime([
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                ]),
                "Feed_Rate_Actual": [120.0, 121.0],
                "Spindle_Speed_Actual": [5000.0, 5001.0],
                "Tool_Number": [7, 7],
            }
        ),
    )

    sessions = {
        "session-1": {
            "session_id": "session-1",
            "config": {"speed": 1.0},
            "data": {},
            "metadata": {},
            "running": False,
            "paused": False,
            "subscribers": [],
            "task": None,
        }
    }

    source = SimulatedCasedataSource(
        sessions,
        casedata_root=root,
        operation_id="OF00001",
        case_dir="SITE_C - MACHINE_C1 - CASE_C1",
    )

    data, _ = source.session_data()

    assert data["Vibration_Harmonic_1_X_Amplitude"] == [0.0, 0.3]
    assert data["Vibration_Peak_1_X_Frequency"] == [40.0, 42.0]
    assert data["Vibration_Peak_1_X_Amplitude"] == [1.2, 1.5]


@pytest.mark.asyncio
async def test_simulated_casedata_source_warm_starts_first_inference_window(tmp_path, monkeypatch):
    root = tmp_path / "casedata"
    op_dir = root / "SITE_C - MACHINE_C1 - CASE_C1" / "OF00001"
    op_dir.mkdir(parents=True)

    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=12, freq="1s")
    _write_case_csv(
        op_dir / "sample_BXCZ3M.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "Operation_Status": [1] * 12,
                "Power_Spindle": [10.0 + idx for idx in range(12)],
                "Power_Y": [2.0 + idx * 0.1 for idx in range(12)],
            }
        ),
    )
    _write_case_csv(
        op_dir / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "Feed_Rate_Actual": [120.0 + idx for idx in range(12)],
                "Spindle_Speed_Actual": [5000.0 + idx for idx in range(12)],
                "Tool_Number": [7] * 12,
            }
        ),
    )

    published: list[tuple[str, FrameEnvelope]] = []

    async def fake_publish(session_id: str, payload: FrameEnvelope) -> None:
        published.append((session_id, payload))

    monkeypatch.setattr("backend.ingestion.simulated_casedata.publish_feature", fake_publish)

    queue: asyncio.Queue = asyncio.Queue()
    sessions = {
        "session-1": {
            "session_id": "session-1",
            "config": {"speed": 1.0},
            "inference_config": {"window_samples": 10},
            "data": {},
            "metadata": {},
            "running": True,
            "paused": False,
            "subscribers": [queue],
            "task": None,
        }
    }

    source = SimulatedCasedataSource(
        sessions,
        casedata_root=root,
        operation_id="OF00001",
        case_dir="SITE_C - MACHINE_C1 - CASE_C1",
    )

    task = source.start("session-1")
    try:
        start = time.perf_counter()
        frames = [await asyncio.wait_for(queue.get(), timeout=0.5) for _ in range(10)]
        elapsed = time.perf_counter() - start
    finally:
        sessions["session-1"]["running"] = False
        await task

    assert frames[0]["i"] == 0
    assert frames[-1]["i"] == 9
    assert sessions["session-1"]["position"] >= 10
    assert elapsed < 0.5
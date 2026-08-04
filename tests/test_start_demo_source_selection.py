from __future__ import annotations

import time
import warnings

import pandas as pd
from fastapi.testclient import TestClient

from backend.app import app
from backend.routers import sessions as sessions_router


def _write_case_csv(path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def _write_harmonic_ready_vibration_csv(op_dir, *, amplitudes: tuple[float, float] = (0.2, 0.4)) -> None:
    _write_case_csv(
        op_dir / "sample_7DTZHE.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime([
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                ]),
                "Vibration_Harmonic_1_X_Amplitude": list(amplitudes),
                "Vibration_Harmonic_1_X_Frequency": [100.0, 101.0],
            }
        ),
    )


def _wait_for_session_loaded(session_id: str, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        session = app.state.sessions[session_id]
        if session.get("last_error"):
            raise AssertionError(session["last_error"])
        if not session.get("loading", False) and session.get("data"):
            return session
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for session {session_id} to finish loading")


def test_start_demo_can_launch_simulated_casedata_source(tmp_path):
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

    client = TestClient(app)
    app.state.sessions.clear()

    try:
        response = client.post(
            "/sessions/start-demo",
            json={
                "source": "simulated_casedata",
                "casedata_root": str(root),
                "operation_id": "OF0001",
                "valid_tools_only": False,
                "speed": 1.0,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["source"] == "simulated_casedata"
        assert payload["n_events"] == 0
        assert payload["status"] == "loading"

        session_id = payload["session_id"]
        session = app.state.sessions[session_id]
        assert session["source_name"] == "simulated_casedata"
        assert session["source_config"]["operation_id"] == "OF0001"
        assert session["config"]["speed"] == 1.0
        session = _wait_for_session_loaded(session_id)
        assert session["config"]["samples_per_tick"] == 1
        assert "Power_Spindle" in session["data"]
        assert session["metadata"]["sample_frequency"] == 1.0
        assert session["inference_config"]["window_seconds"] == 10.0
        assert session["inference_config"]["stride_samples"] == 1

        time.sleep(0.3)
        source_resp = client.get(f"/sessions/{session_id}/source")
        assert source_resp.status_code == 200
        assert source_resp.json()["kind"] == "simulated_casedata"
    finally:
        for session_id in list(app.state.sessions.keys()):
            client.delete(f"/sessions/{session_id}")
        app.state.sessions.clear()


def test_start_demo_defaults_site_c_sessions_to_pair_lfl(tmp_path):
    root = tmp_path / "casedata"
    case_dir = "SITE_C - MACHINE_C1 - CASE_C1"
    op_dir = root / case_dir / "OF00001"
    op_dir.mkdir(parents=True)

    timestamps = pd.to_datetime([
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:01Z",
    ])

    _write_case_csv(
        op_dir / "sample_BXCZ3M.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
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
                "timestamp": timestamps,
                "Feed_Rate_Actual": [120.0, 121.0],
                "Spindle_Speed_Actual": [5000.0, 5001.0],
                "Temperature_Head": [31.0, 31.5],
                "Tool_Number": [7, 7],
            }
        ),
    )
    _write_case_csv(
        op_dir / "sample_7DTZHE.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "Vibration_Peak_1_X_Frequency": [40.0, 41.0],
                "Vibration_Peak_1_X_Amplitude": [1.0, 1.1],
                "Vibration_Peak_1_Y_Frequency": [55.0, 56.0],
                "Vibration_Peak_1_Y_Amplitude": [0.8, 0.9],
            }
        ),
    )

    client = TestClient(app)
    app.state.sessions.clear()

    try:
        response = client.post(
            "/sessions/start-demo",
            json={
                "source": "simulated_casedata",
                "casedata_root": str(root),
                "case_dir": case_dir,
                "operation_id": "OF00001",
                "speed": 1.0,
            },
        )
        assert response.status_code == 200

        session_id = response.json()["session_id"]
        session = app.state.sessions[session_id]

        assert session["config"]["harmonic_scorer_kind"] == "pair"
        assert session["config"]["harmonic_dataset"] == "pair_lfl"
        assert session["metadata"]["harmonic_scorer_kind"] == "pair"
        assert session["metadata"]["harmonic_dataset"] == "pair_lfl"

        _wait_for_session_loaded(session_id)
    finally:
        for session_id in list(app.state.sessions.keys()):
            client.delete(f"/sessions/{session_id}")
        app.state.sessions.clear()


def test_start_demo_respects_explicit_pair_casedata_for_site_c_sessions(tmp_path):
    root = tmp_path / "casedata"
    case_dir = "SITE_C - MACHINE_C1 - CASE_C1"
    op_dir = root / case_dir / "OF00001"
    op_dir.mkdir(parents=True)

    timestamps = pd.to_datetime([
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:01Z",
    ])

    _write_case_csv(
        op_dir / "sample_BXCZ3M.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
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
                "timestamp": timestamps,
                "Feed_Rate_Actual": [120.0, 121.0],
                "Spindle_Speed_Actual": [5000.0, 5001.0],
                "Temperature_Head": [31.0, 31.5],
                "Tool_Number": [7, 7],
            }
        ),
    )
    _write_case_csv(
        op_dir / "sample_7DTZHE.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "Vibration_Peak_1_X_Frequency": [40.0, 41.0],
                "Vibration_Peak_1_X_Amplitude": [1.0, 1.1],
                "Vibration_Peak_1_Y_Frequency": [55.0, 56.0],
                "Vibration_Peak_1_Y_Amplitude": [0.8, 0.9],
            }
        ),
    )

    client = TestClient(app)
    app.state.sessions.clear()

    try:
        response = client.post(
            "/sessions/start-demo",
            json={
                "source": "simulated_casedata",
                "casedata_root": str(root),
                "case_dir": case_dir,
                "operation_id": "OF00001",
                "speed": 1.0,
                "harmonic_scorer_kind": "pair",
                "harmonic_dataset": "pair_casedata",
            },
        )
        assert response.status_code == 200

        session_id = response.json()["session_id"]
        session = app.state.sessions[session_id]

        assert session["config"]["harmonic_scorer_kind"] == "pair"
        assert session["config"]["harmonic_dataset"] == "pair_casedata"
        assert session["metadata"]["harmonic_scorer_kind"] == "pair"
        assert session["metadata"]["harmonic_dataset"] == "pair_casedata"

        _wait_for_session_loaded(session_id)
    finally:
        for session_id in list(app.state.sessions.keys()):
            client.delete(f"/sessions/{session_id}")
        app.state.sessions.clear()


def test_start_demo_can_start_casedata_at_first_cutting_row(tmp_path):
    root = tmp_path / "casedata"
    op_dir = root / "Case A - Tool 1" / "OF0001"
    op_dir.mkdir(parents=True)

    timestamps = pd.to_datetime([
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:01Z",
        "2026-01-01T00:00:02Z",
        "2026-01-01T00:00:03Z",
    ])

    _write_case_csv(
        op_dir / "sample_BXCZ3M.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "Operation_Status": [1, 1, 1, 1],
                "Power_Spindle": [0.0, 0.0, 4.0, 4.5],
                "Power_Y": [0.0, 0.0, 2.0, 2.5],
            }
        ),
    )
    _write_case_csv(
        op_dir / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "Feed_Rate_Actual": [0.0, 0.0, 120.0, 121.0],
                "Spindle_Speed_Actual": [0.0, 0.0, 5000.0, 5001.0],
                "Temperature_Head": [31.0, 31.0, 31.5, 31.7],
                "Tool_Number": [7, 7, 7, 7],
            }
        ),
    )

    client = TestClient(app)
    app.state.sessions.clear()

    try:
        response = client.post(
            "/sessions/start-demo",
            json={
                "source": "simulated_casedata",
                "casedata_root": str(root),
                "operation_id": "OF0001",
                "speed": 1.0,
                "start_at_first_cutting_row": True,
            },
        )
        assert response.status_code == 200

        session_id = response.json()["session_id"]
        session = _wait_for_session_loaded(session_id)

        assert session["position"] == 2
        assert session["source_config"]["start_at_first_cutting_row"] is True
        assert session["source_config"]["start_position"] == 2
        assert session["source_config"]["requested_start_position"] == 0

        sessions_response = client.get("/sessions")
        assert sessions_response.status_code == 200
        session_summary = next(
            item
            for item in sessions_response.json()["session_summaries"]
            if item["session_id"] == session_id
        )
        assert session_summary["start_at_first_cutting_row"] is True
        assert session_summary["resolved_start_position"] == 2
        assert session_summary["requested_start_position"] == 0
    finally:
        for session_id in list(app.state.sessions.keys()):
            client.delete(f"/sessions/{session_id}")
        app.state.sessions.clear()


def test_start_demo_can_disambiguate_duplicate_operation_ids_by_case(tmp_path):
    root = tmp_path / "casedata"
    case_a = root / "Site_b - MACHINE_B1 - CASE_B1" / "OF00001"
    case_b = root / "SITE_C - MACHINE_C1 - CASE_C1" / "OF00001"
    case_a.mkdir(parents=True)
    case_b.mkdir(parents=True)

    _write_case_csv(
        case_a / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime([
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                ]),
                "Feed_Rate_Actual": [120.0, 121.0],
                "Spindle_Speed_Actual": [5000.0, 5001.0],
                "Temperature_Head": [31.0, 31.5],
                "Tool_Number": [6, 6],
            }
        ),
    )
    _write_case_csv(
        case_b / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime([
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                ]),
                "Feed_Rate_Actual": [80.0, 81.0],
                "Spindle_Speed_Actual": [2000.0, 2001.0],
                "Temperature_Head": [27.0, 27.5],
                "Tool_Number": [4, 4],
            }
        ),
    )

    client = TestClient(app)
    app.state.sessions.clear()

    try:
        response = client.post(
            "/sessions/start-demo",
            json={
                "source": "simulated_casedata",
                "casedata_root": str(root),
                "case_dir": "Site_b - MACHINE_B1 - CASE_B1",
                "operation_id": "OF00001",
                "speed": 1.0,
            },
        )
        assert response.status_code == 200
        payload = response.json()

        session = _wait_for_session_loaded(payload["session_id"])
        assert session["source_config"]["case_dir"] == "Site_b - MACHINE_B1 - CASE_B1"
        assert session["source_config"]["operation_id"] == "OF00001"
        assert session["data"]["Feed_Rate_Actual"][0] == 120.0
        assert session["metadata"]["casedata"]["case_dir"] == "Site_b - MACHINE_B1 - CASE_B1"

        info_resp = client.get(f"/sessions/{payload['session_id']}")
        assert info_resp.status_code == 200
        active_context = info_resp.json()["active_context"]
        assert active_context["operation_id"] == "OF00001"
        assert active_context["machine_id"] == "Site_b - MACHINE_B1 - CASE_B1"
        assert active_context["tool_label"] == "T06"
        assert active_context["tool_ready"] is True
        assert active_context["missing_fields"] == []
    finally:
        for session_id in list(app.state.sessions.keys()):
            client.delete(f"/sessions/{session_id}")
        app.state.sessions.clear()


def test_start_demo_can_seed_casedata_start_position(tmp_path):
    root = tmp_path / "casedata"
    op_dir = root / "Site_b - MACHINE_B1 - CASE_B1" / "OF00001"
    op_dir.mkdir(parents=True)

    _write_case_csv(
        op_dir / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime([
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                    "2026-01-01T00:00:02Z",
                ]),
                "Feed_Rate_Actual": [120.0, 121.0, 122.0],
                "Spindle_Speed_Actual": [5000.0, 5001.0, 5002.0],
                "Temperature_Head": [31.0, 31.5, 32.0],
                "Tool_Number": [6, 6, 6],
            }
        ),
    )

    client = TestClient(app)
    app.state.sessions.clear()

    try:
        response = client.post(
            "/sessions/start-demo",
            json={
                "source": "simulated_casedata",
                "casedata_root": str(root),
                "case_dir": "Site_b - MACHINE_B1 - CASE_B1",
                "operation_id": "OF00001",
                "start_position": 2,
                "start_paused": True,
                "speed": 1.0,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        session = app.state.sessions[payload["session_id"]]
        assert session["source_config"]["start_position"] == 2
        session = _wait_for_session_loaded(payload["session_id"])
        assert session["position"] == 2
    finally:
        for session_id in list(app.state.sessions.keys()):
            client.delete(f"/sessions/{session_id}")
        app.state.sessions.clear()


def test_casedata_catalog_marks_harmonic_ready_operations(tmp_path, monkeypatch):
    root = tmp_path / "casedata"
    valid_dir = root / "Site_b - MACHINE_B1 - CASE_B1" / "OF00001"
    invalid_dir = root / "Site_b - MACHINE_B1 - CASE_B1" / "OF00002"
    valid_dir.mkdir(parents=True)
    invalid_dir.mkdir(parents=True)

    for op_dir, tool_number in ((valid_dir, 6), (invalid_dir, 999)):
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
                    "Temperature_Head": [31.0, 31.5],
                    "Tool_Number": [tool_number, tool_number],
                }
            ),
        )
    _write_harmonic_ready_vibration_csv(valid_dir)

    monkeypatch.setenv("SIMULATED_CASEDATA_ROOT", str(root))
    client = TestClient(app)

    response = client.get("/sessions/casedata/catalog")
    assert response.status_code == 200
    payload = response.json()
    case = payload["cases"][0]
    assert case["default_valid_operation_id"] == "OF00001"

    operations = {item["operation_id"]: item for item in case["operations"]}
    assert operations["OF00001"]["harmonic_ready"] is True
    assert operations["OF00002"]["harmonic_ready"] is False
    assert "tool diameter" in operations["OF00002"]["missing_fields"]
    assert "number of teeth" in operations["OF00002"]["missing_fields"]


def test_casedata_catalog_ignores_unrelated_mixed_type_machine_state_columns(tmp_path, monkeypatch):
    root = tmp_path / "casedata"
    op_dir = root / "Site_b - MACHINE_B1 - CASE_B1" / "OF00005"
    op_dir.mkdir(parents=True)

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
                "Tool_Number": [6, 6],
                "Mixed_Type_Column": [1, "bad-value"],
            }
        ),
    )
    _write_harmonic_ready_vibration_csv(op_dir)

    monkeypatch.setenv("SIMULATED_CASEDATA_ROOT", str(root))
    client = TestClient(app)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = client.get("/sessions/casedata/catalog")

    assert response.status_code == 200
    assert not any(issubclass(item.category, pd.errors.DtypeWarning) for item in caught)


def test_casedata_catalog_uses_cutting_rows_for_tool_preview(tmp_path, monkeypatch):
    root = tmp_path / "casedata"
    op_dir = root / "Site_b - MACHINE_B1 - CASE_B1" / "OF00003"
    op_dir.mkdir(parents=True)

    _write_case_csv(
        op_dir / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime([
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                    "2026-01-01T00:00:02Z",
                    "2026-01-01T00:00:03Z",
                ]),
                "Feed_Rate_Actual": [0.0, 120.0, 121.0, 122.0],
                "Spindle_Speed_Actual": [0.0, 5000.0, 5001.0, 5002.0],
                "Temperature_Head": [31.0, 31.5, 32.0, 32.5],
                "Tool_Number": [55, 6, 6, 6],
            }
        ),
    )
    _write_harmonic_ready_vibration_csv(op_dir)

    monkeypatch.setenv("SIMULATED_CASEDATA_ROOT", str(root))
    client = TestClient(app)

    response = client.get("/sessions/casedata/catalog")
    assert response.status_code == 200
    payload = response.json()
    operation = payload["cases"][0]["operations"][0]

    assert operation["tool_label"] == "T06"
    assert operation["tool_number"] == 6
    assert operation["harmonic_ready"] is True
    assert operation["missing_fields"] == []


def test_casedata_catalog_prefers_dominant_cutting_tool_for_preview(tmp_path, monkeypatch):
    root = tmp_path / "casedata"
    op_dir = root / "Site_b - MACHINE_B1 - CASE_B1" / "OF00004"
    op_dir.mkdir(parents=True)

    _write_case_csv(
        op_dir / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime([
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                    "2026-01-01T00:00:02Z",
                    "2026-01-01T00:00:03Z",
                ]),
                "Feed_Rate_Actual": [120.0, 121.0, 122.0, 123.0],
                "Spindle_Speed_Actual": [5000.0, 5001.0, 5002.0, 5003.0],
                "Temperature_Head": [31.0, 31.5, 32.0, 32.5],
                "Tool_Number": [1, 6, 6, 6],
            }
        ),
    )
    _write_harmonic_ready_vibration_csv(op_dir)

    monkeypatch.setenv("SIMULATED_CASEDATA_ROOT", str(root))
    client = TestClient(app)

    response = client.get("/sessions/casedata/catalog")
    assert response.status_code == 200
    payload = response.json()
    operation = payload["cases"][0]["operations"][0]

    assert operation["tool_label"] == "T06"
    assert operation["tool_number"] == 6
    assert operation["harmonic_ready"] is True
    assert operation["missing_fields"] == []


def test_casedata_catalog_marks_pair_preview_operations_as_ready(tmp_path, monkeypatch):
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
                "Vibration_Harmonic_1_X_Amplitude": [0.0, 0.0],
                "Vibration_Harmonic_1_X_Frequency": [100.0, 100.0],
                "Vibration_Peak_1_X_Frequency": [40.0, 42.0],
                "Vibration_Peak_1_X_Amplitude": [1.0, 1.4],
                "Vibration_Peak_1_Y_Frequency": [55.0, 57.0],
                "Vibration_Peak_1_Y_Amplitude": [0.8, 1.1],
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
                "Temperature_Head": [31.0, 31.5],
                "Tool_Number": [6, 6],
            }
        ),
    )

    monkeypatch.setattr(
        sessions_router,
        "build_active_session_context",
        lambda session: {
            "tool_ready": True,
            "missing_fields": [],
            "tool_label": "T06",
            "tool_number": 6,
        },
    )
    monkeypatch.setenv("SIMULATED_CASEDATA_ROOT", str(root))
    client = TestClient(app)

    response = client.get("/sessions/casedata/catalog")
    assert response.status_code == 200
    payload = response.json()
    operation = payload["cases"][0]["operations"][0]

    assert operation["harmonic_ready"] is True
    assert operation["harmonic_preview_available"] is False
    assert operation["pair_preview_available"] is True
    assert operation["pair_column_count"] == 4
    assert operation["missing_fields"] == []


def test_casedata_catalog_merges_default_roots_and_exposes_case_root(tmp_path, monkeypatch):
    casedata_root = tmp_path / "casedata"
    site_a_root = tmp_path / "site_a"

    for root, case_dir, operation_id, tool_number in (
        (casedata_root, "Site_b - MACHINE_B1 - CASE_B1", "OF00001", 6),
        (site_a_root, "Site_a - MACHINE_A1 - CASE_A1", "OF20001", 2),
    ):
        op_dir = root / case_dir / operation_id
        op_dir.mkdir(parents=True)
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
                    "Temperature_Head": [31.0, 31.5],
                    "Tool_Number": [tool_number, tool_number],
                }
            ),
        )

    monkeypatch.delenv("SIMULATED_CASEDATA_ROOT", raising=False)
    monkeypatch.setattr(
        sessions_router,
        "_default_casedata_roots",
        lambda: [casedata_root, site_a_root],
    )
    client = TestClient(app)

    response = client.get("/sessions/casedata/catalog")
    assert response.status_code == 200
    payload = response.json()
    cases = {item["case_dir"]: item for item in payload["cases"]}

    assert payload["roots"] == [str(casedata_root), str(site_a_root)]
    assert cases["Site_b - MACHINE_B1 - CASE_B1"]["casedata_root"] == str(casedata_root)
    assert cases["Site_a - MACHINE_A1 - CASE_A1"]["casedata_root"] == str(site_a_root)


def test_start_demo_inferrs_site_a_root_from_case_dir(tmp_path, monkeypatch):
    casedata_root = tmp_path / "casedata"
    site_a_root = tmp_path / "site_a"

    site_b_dir = casedata_root / "Site_b - MACHINE_B1 - CASE_B1" / "OF00001"
    site_b_dir.mkdir(parents=True)
    _write_case_csv(
        site_b_dir / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime([
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                ]),
                "Feed_Rate_Actual": [120.0, 121.0],
                "Spindle_Speed_Actual": [5000.0, 5001.0],
                "Temperature_Head": [31.0, 31.5],
                "Tool_Number": [6, 6],
            }
        ),
    )

    site_a_dir = site_a_root / "Site_a - MACHINE_A1 - CASE_A1" / "OF20001"
    site_a_dir.mkdir(parents=True)
    _write_case_csv(
        site_a_dir / "sample_TYZBPS.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime([
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                ]),
                "Feed_Rate_Actual": [80.0, 81.0],
                "Spindle_Speed_Actual": [2000.0, 2001.0],
                "Temperature_Head": [27.0, 27.5],
                "Tool_Number": [2, 2],
            }
        ),
    )

    monkeypatch.delenv("SIMULATED_CASEDATA_ROOT", raising=False)
    monkeypatch.setattr(
        sessions_router,
        "_default_casedata_roots",
        lambda: [casedata_root, site_a_root],
    )
    client = TestClient(app)
    app.state.sessions.clear()

    try:
        response = client.post(
            "/sessions/start-demo",
            json={
                "source": "simulated_casedata",
                "case_dir": "Site_a - MACHINE_A1 - CASE_A1",
                "operation_id": "OF20001",
                "start_paused": True,
                "speed": 1.0,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        session = app.state.sessions[payload["session_id"]]

        assert session["source_config"]["casedata_root"] == str(site_a_root)
        session = _wait_for_session_loaded(payload["session_id"])
        assert session["metadata"]["casedata"]["root"] == str(site_a_root)
        assert session["metadata"]["casedata"]["case_dir"] == "Site_a - MACHINE_A1 - CASE_A1"
    finally:
        for session_id in list(app.state.sessions.keys()):
            client.delete(f"/sessions/{session_id}")
        app.state.sessions.clear()


def test_start_demo_valid_tools_only_skips_invalid_first_operation(tmp_path):
    root = tmp_path / "casedata"
    invalid_dir = root / "Site_b - MACHINE_B1 - CASE_B1" / "OF00001"
    valid_dir = root / "Site_b - MACHINE_B1 - CASE_B1" / "OF00002"
    invalid_dir.mkdir(parents=True)
    valid_dir.mkdir(parents=True)

    for op_dir, tool_number in ((invalid_dir, 999), (valid_dir, 6)):
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
                    "Temperature_Head": [31.0, 31.5],
                    "Tool_Number": [tool_number, tool_number],
                }
            ),
        )
    _write_harmonic_ready_vibration_csv(valid_dir)

    client = TestClient(app)
    app.state.sessions.clear()

    try:
        response = client.post(
            "/sessions/start-demo",
            json={
                "source": "simulated_casedata",
                "casedata_root": str(root),
                "case_dir": "Site_b - MACHINE_B1 - CASE_B1",
                "valid_tools_only": True,
                "start_paused": True,
                "speed": 1.0,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        session = app.state.sessions[payload["session_id"]]
        assert session["source_config"]["operation_id"] == "OF00002"
        assert session["source_config"]["valid_tools_only"] is True

        _wait_for_session_loaded(payload["session_id"])
        info_resp = client.get(f"/sessions/{payload['session_id']}")
        assert info_resp.status_code == 200
        active_context = info_resp.json()["active_context"]
        assert active_context["operation_id"] == "OF00002"
        assert active_context["tool_ready"] is True
    finally:
        for session_id in list(app.state.sessions.keys()):
            client.delete(f"/sessions/{session_id}")
        app.state.sessions.clear()


def test_start_demo_casedata_defaults_to_harmonic_ready_operation(tmp_path):
    root = tmp_path / "casedata"
    invalid_dir = root / "Site_b - MACHINE_B1 - CASE_B1" / "OF00001"
    valid_dir = root / "Site_b - MACHINE_B1 - CASE_B1" / "OF00002"
    invalid_dir.mkdir(parents=True)
    valid_dir.mkdir(parents=True)

    for op_dir, tool_number in ((invalid_dir, 999), (valid_dir, 6)):
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
                    "Temperature_Head": [31.0, 31.5],
                    "Tool_Number": [tool_number, tool_number],
                }
            ),
        )
    _write_harmonic_ready_vibration_csv(valid_dir)

    client = TestClient(app)
    app.state.sessions.clear()

    try:
        response = client.post(
            "/sessions/start-demo",
            json={
                "source": "simulated_casedata",
                "casedata_root": str(root),
                "case_dir": "Site_b - MACHINE_B1 - CASE_B1",
                "valid_tools_only": True,
                "start_paused": True,
                "speed": 1.0,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        session = app.state.sessions[payload["session_id"]]
        assert session["source_config"]["operation_id"] == "OF00002"
        assert session["source_config"]["valid_tools_only"] is True

        _wait_for_session_loaded(payload["session_id"])
        info_resp = client.get(f"/sessions/{payload['session_id']}")
        assert info_resp.status_code == 200
        active_context = info_resp.json()["active_context"]
        assert active_context["operation_id"] == "OF00002"
        assert active_context["tool_ready"] is True
    finally:
        for session_id in list(app.state.sessions.keys()):
            client.delete(f"/sessions/{session_id}")
        app.state.sessions.clear()
from __future__ import annotations

import asyncio

import pytest

from backend.ingestion.simulated_file import SimulatedFileSource


@pytest.mark.asyncio
async def test_simulated_file_source_run_emits_frames_and_eos(monkeypatch):
    published: list[tuple[str, dict]] = []

    async def fake_publish(session_id: str, payload: dict) -> None:
        published.append((session_id, payload))

    monkeypatch.setattr("backend.ingestion.simulated_file.publish_feature", fake_publish)

    queue: asyncio.Queue = asyncio.Queue()
    sessions = {
        "session-1": {
            "session_id": "session-1",
            "config": {"channels": ["A", "B"], "speed": 1000.0, "samples_per_tick": 1},
            "data": {"A": [1.0, 2.0], "B": [3.0, 4.0]},
            "metadata": {
                "sample_frequency": 2.0,
                "source": "site_a_line2-demo",
                "casedata": {
                    "operation_id": "OF00013",
                    "cutting_context": {
                        "tool_id": "T12",
                        "extra": {"tool_number": 12},
                    },
                },
            },
            "running": True,
            "paused": False,
            "subscribers": [queue],
            "task": None,
        }
    }

    source = SimulatedFileSource(sessions)

    await source.run("session-1")

    frames = [await asyncio.wait_for(queue.get(), timeout=1.0) for _ in range(3)]
    assert frames[0]["A"] == 1.0
    assert frames[1]["B"] == 4.0
    assert frames[2]["eos"] is True
    assert [payload[1]["position"] for payload in published] == [1, 2]
    assert all(payload[1]["metadata"]["source"] == "site_a_line2-demo" for payload in published)
    assert published[0][1]["source"] == "site_a_line2-demo"
    assert published[0][1]["metadata"]["casedata"]["cutting_context"]["tool_id"] == "T12"
    assert sessions["session-1"]["position"] == 2
    assert sessions["session-1"]["running"] is False
    assert sessions["session-1"]["task"] is None
    assert source.status("session-1")["kind"] == "simulated_file"
    assert source.status("session-1")["connected"] is False
    assert source.status("session-1")["last_frame_ts"] is not None


@pytest.mark.asyncio
async def test_simulated_file_source_start_sets_task_and_source_name(monkeypatch):
    async def fake_publish(session_id: str, payload: dict) -> None:
        return None

    monkeypatch.setattr("backend.ingestion.simulated_file.publish_feature", fake_publish)

    sessions = {
        "session-2": {
            "session_id": "session-2",
            "config": {"channels": ["A"], "speed": 1000.0, "samples_per_tick": 1},
            "data": {"A": [5.0]},
            "metadata": {"sample_frequency": 1.0},
            "running": True,
            "paused": False,
            "subscribers": [],
            "task": None,
        }
    }

    source = SimulatedFileSource(sessions)

    task = source.start("session-2")
    await task

    assert sessions["session-2"]["source_name"] == "simulated_file"
    assert sessions["session-2"]["task"] is None
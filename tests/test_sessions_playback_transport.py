from __future__ import annotations

import asyncio

import pytest

from backend.routers.sessions import playback_task


@pytest.mark.asyncio
async def test_playback_task_publishes_session_metadata_with_feature_frames(monkeypatch):
    published: list[tuple[str, dict]] = []

    async def fake_publish(session_id: str, payload: dict) -> None:
        published.append((session_id, payload))

    monkeypatch.setattr("backend.routers.sessions.publish_feature", fake_publish)

    queue: asyncio.Queue = asyncio.Queue()
    sessions = {
        "session-demo": {
            "session_id": "session-demo",
            "config": {"channels": ["A"], "speed": 1000.0, "samples_per_tick": 1},
            "data": {"A": [1.0, 2.0]},
            "metadata": {
                "sample_frequency": 2.0,
                "source": "site_a_line2-demo",
                "machine_family": "machine_a1",
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

    await playback_task("session-demo", sessions)

    assert [payload[1]["position"] for payload in published] == [1, 2]
    assert all(payload[1]["metadata"]["source"] == "site_a_line2-demo" for payload in published)
    assert published[0][1]["source"] == "site_a_line2-demo"
    assert published[0][1]["metadata"]["casedata"]["cutting_context"]["tool_id"] == "T12"

    frames = [await asyncio.wait_for(queue.get(), timeout=1.0) for _ in range(3)]
    assert frames[0]["A"] == 1.0
    assert frames[2]["eos"] is True
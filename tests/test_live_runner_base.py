import pytest

from backend.agents.memory.live_runner_base import LiveRunnerBase


class _DummyRunner(LiveRunnerBase):
    def __init__(self):
        super().__init__("dummy-run")

    def _run_sync(self):
        self._emit_sync("setup", "started", "Preparing", pct=5)
        return {"disk_run_id": "disk-1"}

    def _success_message(self) -> str:
        return "Dummy finished"

    def _success_detail(self, result):
        return {"success": True, "disk_run_id": result["disk_run_id"]}


class _FailingRunner(LiveRunnerBase):
    def __init__(self):
        super().__init__("failing-run")

    def _run_sync(self):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_live_runner_base_emits_progress_and_done(monkeypatch):
    events = []

    async def fake_publish(channel, payload):
        events.append((channel, payload))

    monkeypatch.setattr("backend.agents.memory.live_runner_base.bus.publish", fake_publish)

    runner = _DummyRunner()
    result = await runner.execute()

    assert result == {"disk_run_id": "disk-1"}
    assert events[0][0] == "experiment.dummy-run"
    assert events[0][1]["phase"] == "setup"
    assert events[-1][1]["phase"] == "done"
    assert events[-1][1]["message"] == "Dummy finished"
    assert events[-1][1]["detail"] == {"success": True, "disk_run_id": "disk-1"}


@pytest.mark.asyncio
async def test_live_runner_base_wraps_errors(monkeypatch):
    events = []

    async def fake_publish(channel, payload):
        events.append((channel, payload))

    monkeypatch.setattr("backend.agents.memory.live_runner_base.bus.publish", fake_publish)

    runner = _FailingRunner()
    result = await runner.execute()

    assert result["success"] is False
    assert result["error"] == "boom"
    assert events[-1][0] == "experiment.failing-run"
    assert events[-1][1]["phase"] == "error"
    assert events[-1][1]["detail"]["traceback"]
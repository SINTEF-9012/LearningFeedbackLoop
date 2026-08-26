from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.agents.config import MemorySystemConfig
from backend.agents.memory import init as init_mod
from backend.agents.memory import router as router_mod


def test_runtime_overrides_persist_and_load(tmp_path):
    config = MemorySystemConfig(db_path=str(tmp_path / "memories.db"))

    persisted = init_mod.persist_runtime_overrides(
        {"generate_explanations": True, "dispatch_alerts": False},
        config=config,
    )

    assert persisted == {
        "dispatch_alerts": False,
        "generate_explanations": True,
    }
    assert init_mod.load_runtime_overrides(config) == persisted


def test_apply_runtime_overrides_returns_replaced_config(tmp_path):
    config = MemorySystemConfig(
        db_path=str(tmp_path / "memories.db"),
        generate_explanations=False,
        dispatch_alerts=True,
    )

    effective = init_mod.apply_runtime_overrides(
        config,
        {"generate_explanations": True, "dispatch_alerts": False},
    )

    assert effective.generate_explanations is True
    assert effective.dispatch_alerts is False
    assert config.generate_explanations is False
    assert config.dispatch_alerts is True


@pytest.mark.asyncio
async def test_patch_memory_config_persists_changes(monkeypatch):
    persisted_calls = []
    orchestrator = SimpleNamespace(
        config=SimpleNamespace(generate_explanations=False, dispatch_alerts=True)
    )

    def fake_persist_runtime_overrides(changes):
        persisted_calls.append(dict(changes))
        return {
            "generate_explanations": bool(changes.get("generate_explanations", False)),
            "dispatch_alerts": bool(changes.get("dispatch_alerts", True)),
        }

    # The handler asks for the config directly now rather than reaching through
    # the orchestrator, so patch that accessor.
    monkeypatch.setattr(router_mod, "get_orchestrator_config", lambda: orchestrator.config)
    monkeypatch.setattr(router_mod, "persist_runtime_overrides", fake_persist_runtime_overrides)

    response = await router_mod.patch_memory_config(
        {"generate_explanations": True, "dispatch_alerts": False}
    )

    assert persisted_calls == [{"generate_explanations": True, "dispatch_alerts": False}]
    assert orchestrator.config.generate_explanations is True
    assert orchestrator.config.dispatch_alerts is False
    assert response["persisted"] == {
        "generate_explanations": True,
        "dispatch_alerts": False,
    }
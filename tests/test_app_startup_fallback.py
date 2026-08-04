from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

import backend.app as app_module
from backend.agents.config import MemorySystemConfig
from backend.agents.memory.init import get_store, initialize_memory_system, shutdown_memory_system
from backend.agents.storage.store import MemoryStore


def test_memory_startup_config_enables_lazy_seed_training_when_unset(
    monkeypatch,
):
    config = MemorySystemConfig(
        use_classical_models=True,
        lazy_seed_training=False,
        generate_explanations=False,
        dispatch_alerts=False,
    )

    monkeypatch.delenv("LAZY_SEED_TRAINING", raising=False)
    monkeypatch.setattr("backend.agents.config.get_config", lambda: config)

    startup_config = app_module._memory_startup_config()

    assert startup_config.lazy_seed_training is True
    assert config.lazy_seed_training is False


def test_app_startup_skips_lazy_orchestrator_creation_after_memory_init_failure(
    monkeypatch,
):
    def fail_memory_init(*args, **kwargs):
        raise RuntimeError("neo4j unavailable")

    async def noop_stop_memory_processor():
        return None

    monkeypatch.setattr(app_module, "initialize_memory_system", fail_memory_init)
    monkeypatch.setattr(app_module, "stop_memory_processor", noop_stop_memory_processor)
    monkeypatch.setattr(app_module, "shutdown_memory_system", lambda: None)
    monkeypatch.setattr(
        "backend.agents.memory.feedback_async.build_default_pipeline",
        lambda: SimpleNamespace(callback=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        "backend.agents.memory.init.get_memory_components",
        lambda: (None, None),
    )
    monkeypatch.setattr(
        "backend.agents.memory.orchestrator.get_orchestrator",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("startup should not create a fallback orchestrator")
        ),
    )

    with TestClient(app_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_initialize_memory_system_raises_when_neo4j_unavailable_and_fallback_disabled(
    monkeypatch,
    tmp_path,
):
    class FailingNeo4jStore:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("neo4j unavailable")

    monkeypatch.setattr(
        "backend.agents.storage.neo4j_store.Neo4jMemoryStore",
        FailingNeo4jStore,
    )
    monkeypatch.delenv("ALLOW_SQLITE_FALLBACK", raising=False)

    config = MemorySystemConfig(
        storage_backend="neo4j",
        db_path=str(tmp_path / "memories.db"),
        pattern_index_path=str(tmp_path / "pattern_index.json"),
        pattern_priors_path=str(tmp_path / "pattern_priors.json"),
        use_classical_models=False,
        generate_explanations=False,
        dispatch_alerts=False,
    )

    try:
        try:
            initialize_memory_system(config=config, force=True)
            assert False, "expected initialize_memory_system to fail"
        except RuntimeError as exc:
            assert "ALLOW_SQLITE_FALLBACK" in str(exc)
        assert get_store() is None
    finally:
        shutdown_memory_system()


def test_initialize_memory_system_falls_back_to_sqlite_when_opted_in(
    monkeypatch,
    tmp_path,
):
    class FailingNeo4jStore:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("neo4j unavailable")

    monkeypatch.setattr(
        "backend.agents.storage.neo4j_store.Neo4jMemoryStore",
        FailingNeo4jStore,
    )
    monkeypatch.setenv("ALLOW_SQLITE_FALLBACK", "1")

    config = MemorySystemConfig(
        storage_backend="neo4j",
        db_path=str(tmp_path / "memories.db"),
        pattern_index_path=str(tmp_path / "pattern_index.json"),
        pattern_priors_path=str(tmp_path / "pattern_priors.json"),
        use_classical_models=False,
        generate_explanations=False,
        dispatch_alerts=False,
    )

    try:
        store, orchestrator = initialize_memory_system(config=config, force=True)
        assert isinstance(store, MemoryStore)
        assert get_store() is store
        assert orchestrator is not None
    finally:
        shutdown_memory_system()
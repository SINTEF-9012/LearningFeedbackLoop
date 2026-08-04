import importlib

import pytest

from backend.agents import config as config_module


def test_storage_backend_defaults_to_neo4j_when_unset():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("STORAGE_BACKEND", raising=False)

        reloaded = importlib.reload(config_module)

        assert reloaded.STORAGE_BACKEND == "neo4j"
        assert reloaded.MemorySystemConfig().storage_backend == "neo4j"
        assert reloaded.MemorySystemConfig.from_env().storage_backend == "neo4j"

    importlib.reload(config_module)


def test_llm_provider_defaults_to_groq_when_unset():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        reloaded = importlib.reload(config_module)

        assert reloaded.LLM_PROVIDER == "groq"
        assert reloaded.MemorySystemConfig().llm_provider == "groq"
        assert reloaded.MemorySystemConfig.from_env().llm_provider == "groq"

    importlib.reload(config_module)
from __future__ import annotations

from types import SimpleNamespace

from backend.agents.llm.docs_backend import Neo4jDocsBackend
from backend.agents.llm.document_graph import DocumentGraphStore


class _RecordingSession:
    def __init__(self, commands):
        self._commands = commands

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cypher, **kwargs):
        self._commands.append((cypher, kwargs))
        return SimpleNamespace(consume=lambda: None)


class _RecordingDriver:
    def __init__(self, commands):
        self._commands = commands

    def session(self, database=None):
        return _RecordingSession(self._commands)


def _joined_commands(commands) -> str:
    return "\n".join(cypher for cypher, _ in commands)


def test_document_graph_schema_bootstraps_entity_indexes():
    commands = []
    store = object.__new__(DocumentGraphStore)
    store._driver = _RecordingDriver(commands)
    store._database = "neo4j"

    DocumentGraphStore._ensure_schema(store)

    command_text = _joined_commands(commands)
    assert "entity_id_unique" in command_text
    assert "entity_usecase_idx" in command_text
    assert "entity_type_idx" in command_text
    assert "entity_name_idx" in command_text
    assert "entity_vector_index" in command_text
    assert any(kwargs.get("version") == 3 for _, kwargs in commands)


def test_docs_backend_schema_bootstraps_entity_indexes():
    commands = []
    backend = object.__new__(Neo4jDocsBackend)
    backend._driver = _RecordingDriver(commands)
    backend._database = "neo4j"

    Neo4jDocsBackend._ensure_schema(backend)

    command_text = _joined_commands(commands)
    assert "entity_id_unique" in command_text
    assert "entity_usecase_idx" in command_text
    assert "entity_type_idx" in command_text
    assert "entity_name_idx" in command_text
    assert "entity_vector_index" in command_text
    assert any(kwargs.get("version") == 3 for _, kwargs in commands)


def test_docs_backend_demotes_spreadsheet_match_scores():
    backend = object.__new__(Neo4jDocsBackend)

    spreadsheet = backend._format_match(
        {
            "id": "doc-1",
            "text": "bearing chatter inspection record",
            "source": "site_a",
            "usecase": "SITE_A",
            "file_name": "cor.xlsx",
            "page": 1,
            "machine": "MACHINE_A1",
            "document_type": "spreadsheet",
        },
        0.5,
    )
    manual = backend._format_match(
        {
            "id": "doc-2",
            "text": "bearing chatter troubleshooting guidance",
            "source": "site_a",
            "usecase": "SITE_A",
            "file_name": "manual.pdf",
            "page": 4,
            "machine": "MACHINE_A1",
            "document_type": "manual",
        },
        0.5,
    )

    assert spreadsheet["score"] == 0.3
    assert manual["score"] == 0.5
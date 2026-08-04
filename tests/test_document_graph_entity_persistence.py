from __future__ import annotations

from types import SimpleNamespace

from backend.agents.llm.document_graph import (
    DocumentChunk,
    DocumentFileRecord,
    DocumentGraphStore,
    SemanticEntityRecord,
    SemanticGraphRecords,
    SemanticMentionRecord,
    SemanticRelationRecord,
)


class _RecordingTx:
    def __init__(self):
        self.commands = []

    def run(self, cypher, **kwargs):
        self.commands.append((cypher, kwargs))
        return SimpleNamespace(consume=lambda: None)


def test_tx_upsert_file_persists_semantic_graph_records():
    tx = _RecordingTx()
    file_record = DocumentFileRecord(
        id="file-1",
        source_id="source-1",
        usecase="SITE_A",
        source="site_a",
        dataset_id="site_a",
        file_name="manual.pdf",
        relative_path="manual.pdf",
        subdirectory=None,
        document_type="manual",
        machine="MACHINE_A1",
        machine_ids=["MACHINE_A1"],
        machine_uri="urn:lfl:asset:machine_a1",
        language_code="en",
        original_language="en",
        page_count=1,
        chunk_count=1,
        updated_at="2026-05-18T00:00:00+00:00",
    )
    chunks = [
        DocumentChunk(
            id="doc-1",
            file_id="file-1",
            source_id="source-1",
            usecase="SITE_A",
            source="site_a",
            dataset_id="site_a",
            file_name="manual.pdf",
            relative_path="manual.pdf",
            subdirectory=None,
            text="MACHINE_A1 chatter guidance",
            page=1,
            section=None,
            chunk_index=0,
            document_type="manual",
            machine="MACHINE_A1",
            machine_ids=["MACHINE_A1"],
            machine_uri="urn:lfl:asset:machine_a1",
            language_code="en",
            original_language="en",
            updated_at="2026-05-18T00:00:00+00:00",
        )
    ]
    semantic_records = SemanticGraphRecords(
        entities=[
            SemanticEntityRecord(
                id="entity-1",
                usecase="SITE_A",
                type="Machine",
                name="MACHINE_A1",
                name_norm="machine_a1",
                aliases=["MACHINE_A1"],
                description="MACHINE_A1 chatter guidance",
                updated_at="2026-05-18T00:00:00+00:00",
                canonical_id="urn:lfl:asset:machine_a1",
                embedding=[0.1, 0.2, 0.3],
            )
        ],
        mentions=[
            SemanticMentionRecord(
                chunk_id="doc-1",
                entity_id="entity-1",
                confidence=1.0,
                updated_at="2026-05-18T00:00:00+00:00",
            )
        ],
        relations=[
            SemanticRelationRecord(
                source_entity_id="entity-1",
                target_entity_id="entity-1",
                relation_type="DESCRIBES",
                source_chunk_id="doc-1",
                confidence=0.8,
                updated_at="2026-05-18T00:00:00+00:00",
            )
        ],
        new_entity_count=1,
    )

    DocumentGraphStore._tx_upsert_file(tx, file_record, chunks, semantic_records)

    cypher_text = "\n".join(command for command, _ in tx.commands)
    assert "MATCH ()-[r:REL]->()" in cypher_text
    assert "MERGE (d)-[r:MENTIONS]->(e)" in cypher_text
    assert "MERGE (a)-[r:REL {type: row.relation_type, source_chunk_id: row.source_chunk_id}]->(b)" in cypher_text
    assert "MATCH (e:Entity)" in cypher_text

    entity_write = next(kwargs for command, kwargs in tx.commands if "MERGE (e:Entity {id: row.id})" in command)
    assert entity_write["rows"][0]["props"]["name"] == "MACHINE_A1"
    assert entity_write["rows"][0]["props"]["canonical_id"] == "urn:lfl:asset:machine_a1"
    assert entity_write["rows"][0]["props"]["embedding"] == [0.1, 0.2, 0.3]
from __future__ import annotations

from pathlib import Path

import backend.agents.llm.document_graph as document_graph
from backend.agents.llm.entity_extractor import ExtractedEntity, ExtractedRelation, ExtractionResult
from backend.agents.llm.document_graph import (
    DocumentChunk,
    DocumentFileRecord,
    _DocumentEntityCanonicalIdResolver,
    ingest_machinedocs,
)


class _FakeExtractor:
    def __init__(self, *args, **kwargs):
        self.enabled = kwargs.get("enabled", False)

    def is_enabled(self) -> bool:
        return bool(self.enabled)

    def extract_from_chunk(self, chunk_text: str, *, usecase: str, machine_hint=None, source_hint=None):
        if "failure" in chunk_text:
            return ExtractionResult(entities=[], relations=[], warnings=["extractor_error:boom"])
        if "chatter" in chunk_text:
            return ExtractionResult(
                entities=[
                    ExtractedEntity(name="MACHINE_A1", type="Machine", aliases=["MACHINE_A1"]),
                    ExtractedEntity(name="Chatter", type="Symptom", aliases=[]),
                ],
                relations=[
                    ExtractedRelation(
                        src_name="Chatter",
                        src_type="Symptom",
                        dst_name="MACHINE_A1",
                        dst_type="Machine",
                        rel_type="APPLIES_TO",
                        confidence=0.9,
                    )
                ],
            )
        return ExtractionResult(
            entities=[ExtractedEntity(name="MACHINE_A1", type="Machine", aliases=[])],
            relations=[],
        )


def test_ingest_machinedocs_tracks_semantic_preview_counts(monkeypatch, tmp_path: Path):
    docs_root = tmp_path / "machinedocs"
    docs_root.mkdir()
    manual_path = docs_root / "manual.pdf"
    manual_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(document_graph, "EntityExtractor", _FakeExtractor)

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
        page_count=2,
        chunk_count=3,
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
            text="This chunk mentions chatter on MACHINE_A1 during roughing.",
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
        ),
        DocumentChunk(
            id="doc-2",
            file_id="file-1",
            source_id="source-1",
            usecase="SITE_A",
            source="site_a",
            dataset_id="site_a",
            file_name="manual.pdf",
            relative_path="manual.pdf",
            subdirectory=None,
            text="This failure chunk forces an extractor error for coverage.",
            page=2,
            section=None,
            chunk_index=1,
            document_type="manual",
            machine="MACHINE_A1",
            machine_ids=["MACHINE_A1"],
            machine_uri="urn:lfl:asset:machine_a1",
            language_code="en",
            original_language="en",
            updated_at="2026-05-18T00:00:00+00:00",
        ),
        DocumentChunk(
            id="doc-3",
            file_id="file-1",
            source_id="source-1",
            usecase="SITE_A",
            source="site_a",
            dataset_id="site_a",
            file_name="manual.pdf",
            relative_path="manual.pdf",
            subdirectory=None,
            text="A short mention of MACHINE_A1 appears again in the procedure notes.",
            page=3,
            section=None,
            chunk_index=2,
            document_type="manual",
            machine="MACHINE_A1",
            machine_ids=["MACHINE_A1"],
            machine_uri="urn:lfl:asset:machine_a1",
            language_code="en",
            original_language="en",
            updated_at="2026-05-18T00:00:00+00:00",
        ),
    ]
    monkeypatch.setattr(
        document_graph,
        "parse_document_file",
        lambda *args, **kwargs: (file_record, chunks),
    )

    summary = ingest_machinedocs(
        root=docs_root,
        dry_run=True,
        embed_documents=False,
        extract_entities=True,
    )

    assert summary.files_seen == 1
    assert summary.files_parsed == 1
    assert summary.files_ingested == 0
    assert summary.entities_extracted == 2
    assert summary.relations_extracted == 1
    assert summary.extractor_failures == 1


def test_document_entity_canonical_id_resolver_matches_machine_separator_variants():
    resolver = _DocumentEntityCanonicalIdResolver([])

    resolved = resolver.resolve(
        name="MACHINE_A1",
        entity_type="Machine",
        aliases=["machine machine_a1"],
        usecase="SITE_A",
        machine_hint="MACHINE_A1",
        machine_uri="urn:lfl:asset:machine_a1",
    )

    assert resolved == "urn:lfl:asset:machine_a1"


def test_document_entity_canonical_id_resolver_matches_asset_label_separator_variants():
    resolver = _DocumentEntityCanonicalIdResolver(
        [{"iri": "urn:lfl:asset:machine_b1", "label": "MACHINE_B1"}]
    )

    resolved = resolver.resolve(
        name="MACHINE_B1",
        entity_type="Machine",
        aliases=[],
        usecase="SITE_B",
        machine_hint=None,
        machine_uri=None,
    )

    assert resolved == "urn:lfl:asset:machine_b1"
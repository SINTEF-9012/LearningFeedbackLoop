#!/usr/bin/env python3
"""Ingest `data/machinedocs/` into the Neo4j document domain.

The script is intentionally label-scoped: it only writes the documentation
graph (`DocumentSource`, `DocumentFile`, `Document`, `Entity`) and never
deletes memory-domain nodes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.llm.document_graph import ingest_machinedocs  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(PROJECT_ROOT / "data" / "machinedocs"),
        help="Path to the machine-document root (default: data/machinedocs)",
    )
    parser.add_argument(
        "--usecase",
        action="append",
        choices=["SITE_A", "SITE_B", "SITE_C", "GENERIC"],
        help="Restrict ingestion to one or more usecases.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Stop after parsing this many supported files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk files without writing to Neo4j.",
    )
    parser.add_argument(
        "--clear-documents",
        action="store_true",
        help="Delete only the document-domain nodes before ingesting.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Write document nodes without generating vector embeddings.",
    )
    parser.add_argument(
        "--extract-entities",
        action="store_true",
        help="Run the closed-vocabulary semantic extractor during ingest preview or write mode.",
    )
    parser.add_argument(
        "--extractor-model",
        default=None,
        help="Override the Groq model used for entity extraction.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=3500,
        help="Approximate max characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=400,
        help="Approximate overlap in characters between adjacent chunks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    summary = ingest_machinedocs(
        root=Path(args.root),
        usecases=args.usecase,
        dry_run=args.dry_run,
        clear_documents=args.clear_documents,
        embed_documents=not args.skip_embeddings,
        extract_entities=args.extract_entities,
        entity_extractor_model=args.extractor_model,
        limit_files=args.max_files,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
    )

    print({
        "dry_run": args.dry_run,
        "files_seen": summary.files_seen,
        "files_parsed": summary.files_parsed,
        "files_ingested": summary.files_ingested,
        "chunks_created": summary.chunks_created,
        "entities_extracted": summary.entities_extracted,
        "relations_extracted": summary.relations_extracted,
        "extractor_failures": summary.extractor_failures,
        "skipped_unsupported": summary.skipped_unsupported,
        "skipped_empty": summary.skipped_empty,
        "warnings": summary.warnings,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
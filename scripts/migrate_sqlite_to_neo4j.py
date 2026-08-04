#!/usr/bin/env python3
"""
SQLite → Neo4j Migration Script

Reads all memories, patterns, feedback events, and traces from the SQLite
database, re-embeds everything with the multilingual model, and writes
into Neo4j using idempotent MERGE operations.

Usage:
    python scripts/migrate_sqlite_to_neo4j.py [--sqlite-path memories.db]

Environment variables (or .env):
    NEO4J_URI       bolt://localhost:7687
    NEO4J_USERNAME  neo4j
    NEO4J_PASSWORD  changeme
    NEO4J_DATABASE  neo4j
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dotenv_if_available() -> None:
    """Best-effort load of .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass


def get_embedding_model():
    """Load the sentence-transformers model for re-embedding."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "sentence-transformers is required.  pip install sentence-transformers"
        )
        sys.exit(1)

    model_name = os.getenv(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def embed_texts(model, texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Batch-embed a list of texts, returning list-of-float vectors."""
    if not texts:
        return []
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return [e.tolist() for e in embeddings]


# ---------------------------------------------------------------------------
# SQLite reader
# ---------------------------------------------------------------------------

def read_sqlite(db_path: str) -> dict:
    """Read all data from the SQLite memory store.

    Returns a dict with keys: memories, feedback_events, traces.
    Each value is a list of dicts.
    """
    import sqlite3

    if not Path(db_path).exists():
        logger.error("SQLite database not found: %s", db_path)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # -- Memories --
    memories = []
    try:
        for row in conn.execute("SELECT * FROM memories"):
            memories.append(dict(row))
    except sqlite3.OperationalError:
        logger.warning("No 'memories' table found — skipping.")

    # -- Feedback events --
    feedback_events = []
    try:
        for row in conn.execute("SELECT * FROM feedback_events"):
            feedback_events.append(dict(row))
    except sqlite3.OperationalError:
        logger.info("No 'feedback_events' table — skipping.")

    # -- Traces --
    traces = []
    try:
        for row in conn.execute("SELECT * FROM traces"):
            traces.append(dict(row))
    except sqlite3.OperationalError:
        logger.info("No 'traces' table — skipping.")

    conn.close()
    logger.info(
        "Read from SQLite: %d memories, %d feedback events, %d traces",
        len(memories), len(feedback_events), len(traces),
    )
    return {
        "memories": memories,
        "feedback_events": feedback_events,
        "traces": traces,
    }


# ---------------------------------------------------------------------------
# Neo4j writer
# ---------------------------------------------------------------------------

def get_neo4j_driver():
    """Create a Neo4j driver from environment variables."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.error("neo4j driver required.  pip install neo4j>=5.0")
        sys.exit(1)

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "changeme")

    logger.info("Connecting to Neo4j at %s as %s", uri, user)
    return GraphDatabase.driver(uri, auth=(user, password))


def setup_constraints(driver, database: str) -> None:
    """Create uniqueness constraints and vector index."""
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Pattern) REQUIRE p.key IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
    ]
    with driver.session(database=database) as session:
        for stmt in constraints:
            session.run(stmt)
            logger.info("  %s", stmt.split("FOR")[0].strip())

        # Vector index (idempotent via IF NOT EXISTS in Neo4j 5.x)
        try:
            session.run(
                """
                CREATE VECTOR INDEX memory_embedding_index IF NOT EXISTS
                FOR (m:Memory) ON (m.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """
            )
            logger.info("  Vector index ensured.")
        except Exception as exc:
            logger.warning("  Vector index creation skipped: %s", exc)


def write_memories(driver, database: str, memories: list[dict], embeddings: list[list[float]]) -> None:
    """Write memories to Neo4j using batched MERGE."""

    def _batch(tx, batch: list[tuple[dict, list[float]]]):
        for mem, emb in batch:
            raw_data = mem.get("data")
            if isinstance(raw_data, str):
                try:
                    raw_data = json.loads(raw_data)
                except json.JSONDecodeError:
                    raw_data = {}
            elif raw_data is None:
                raw_data = {}

            mem_id = mem.get("id") or mem.get("memory_id", "")
            session_id = (
                raw_data.get("session_id")
                or mem.get("session_id")
                or "migrated"
            )
            label = raw_data.get("label") or mem.get("label", "")
            tags_raw = raw_data.get("tags") or mem.get("tags")
            tags = json.dumps(tags_raw) if isinstance(tags_raw, list) else "[]"
            metadata = json.dumps(raw_data.get("metadata") or {})

            # Extract pattern keys
            pattern_keys = []
            pk_raw = raw_data.get("pattern_keys") or []
            for pk in pk_raw:
                if isinstance(pk, dict):
                    pattern_keys.append(pk.get("key", ""))
                elif isinstance(pk, str):
                    pattern_keys.append(pk)

            created_at = (
                raw_data.get("created_at")
                or mem.get("created_at")
                or mem.get("timestamp", "")
            )

            # MERGE Memory
            tx.run(
                """
                MERGE (m:Memory {id: $id})
                SET m.session_id    = $session_id,
                    m.label         = $label,
                    m.tags          = $tags,
                    m.metadata      = $metadata,
                    m.embedding     = $embedding,
                    m.created_at    = $created_at
                """,
                id=mem_id,
                session_id=session_id,
                label=label,
                tags=tags,
                metadata=metadata,
                embedding=emb,
                created_at=created_at,
            )

            # MERGE Session
            tx.run(
                """
                MERGE (s:Session {id: $sid})
                MERGE (m:Memory {id: $mid})
                MERGE (m)-[:IN_SESSION]->(s)
                """,
                sid=session_id,
                mid=mem_id,
            )

            # MERGE Pattern nodes + HAS_PATTERN
            for pk in pattern_keys:
                if pk:
                    tx.run(
                        """
                        MERGE (p:Pattern {key: $key})
                        ON CREATE SET p.prior = 0.5
                        MERGE (m:Memory {id: $mid})
                        MERGE (m)-[:HAS_PATTERN]->(p)
                        """,
                        key=pk,
                        mid=mem_id,
                    )

    # Process in batches
    batch_size = 100
    with driver.session(database=database) as session:
        for i in range(0, len(memories), batch_size):
            batch = list(zip(
                memories[i : i + batch_size],
                embeddings[i : i + batch_size],
            ))
            session.execute_write(_batch, batch)
            logger.info("  Wrote memories %d–%d", i + 1, min(i + batch_size, len(memories)))


def write_feedback_events(driver, database: str, events: list[dict]) -> None:
    """Write feedback events to Neo4j."""

    def _batch(tx, batch: list[dict]):
        for ev in batch:
            fb_id = ev.get("id") or ev.get("feedback_id", "")
            memory_id = ev.get("memory_id", "")
            action = ev.get("action", "")
            user_id = ev.get("user_id", "operator")
            timestamp = ev.get("timestamp", "")
            data_raw = ev.get("data")
            data_str = json.dumps(data_raw) if isinstance(data_raw, (dict, list)) else str(data_raw or "")

            tx.run(
                """
                MERGE (f:Feedback {id: $id})
                SET f.action    = $action,
                    f.user_id   = $user_id,
                    f.timestamp = $timestamp,
                    f.data      = $data
                WITH f
                MATCH (m:Memory {id: $mid})
                MERGE (f)-[:ABOUT]->(m)
                """,
                id=fb_id,
                action=action,
                user_id=user_id,
                timestamp=timestamp,
                data=data_str,
                mid=memory_id,
            )

    batch_size = 200
    with driver.session(database=database) as session:
        for i in range(0, len(events), batch_size):
            batch = events[i : i + batch_size]
            session.execute_write(_batch, batch)
        logger.info("  Wrote %d feedback events.", len(events))


def write_traces(driver, database: str, traces: list[dict]) -> None:
    """Write trace records to Neo4j."""

    def _batch(tx, batch: list[dict]):
        for tr in batch:
            tr_id = tr.get("id") or tr.get("trace_id", "")
            tx.run(
                """
                MERGE (t:Trace {id: $id})
                SET t.agent     = $agent,
                    t.action    = $action,
                    t.detail    = $detail,
                    t.timestamp = $timestamp
                """,
                id=tr_id,
                agent=tr.get("agent", ""),
                action=tr.get("action", ""),
                detail=json.dumps(tr.get("detail") or ""),
                timestamp=tr.get("timestamp", ""),
            )

    batch_size = 200
    with driver.session(database=database) as session:
        for i in range(0, len(traces), batch_size):
            batch = traces[i : i + batch_size]
            session.execute_write(_batch, batch)
        logger.info("  Wrote %d traces.", len(traces))


def build_co_occurrence_edges(driver, database: str) -> None:
    """Build [:CO_OCCURS_WITH] edges from session co-occurrence."""
    logger.info("Building co-occurrence edges from session data...")
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (m1:Memory)-[:HAS_PATTERN]->(p1:Pattern),
                  (m1)-[:IN_SESSION]->(s:Session)<-[:IN_SESSION]-(m2:Memory),
                  (m2)-[:HAS_PATTERN]->(p2:Pattern)
            WHERE p1.key < p2.key
            WITH p1, p2, count(DISTINCT s) AS sessions
            MERGE (p1)-[r:CO_OCCURS_WITH]-(p2)
            SET r.weight = sessions
            RETURN count(r) AS edges
            """
        )
        record = result.single()
        edge_count = record["edges"] if record else 0
        logger.info("  Created/updated %d co-occurrence edges.", edge_count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_embedding_texts(memories: list[dict]) -> list[str]:
    """Build embedding-source text for each memory."""
    texts: list[str] = []
    for mem in memories:
        raw_data = mem.get("data")
        if isinstance(raw_data, str):
            try:
                raw_data = json.loads(raw_data)
            except json.JSONDecodeError:
                raw_data = {}
        elif raw_data is None:
            raw_data = {}

        parts: list[str] = []

        # Pattern keys
        for pk in raw_data.get("pattern_keys") or []:
            if isinstance(pk, dict):
                parts.append(pk.get("key", ""))
            elif isinstance(pk, str):
                parts.append(pk)

        # Label
        label = raw_data.get("label") or mem.get("label", "")
        if label:
            parts.append(label)

        # Annotation
        ann = raw_data.get("annotation_text") or ""
        if ann:
            parts.append(ann)

        texts.append(" ".join(parts) if parts else "memory")
    return texts


def main() -> None:
    load_dotenv_if_available()

    parser = argparse.ArgumentParser(description="Migrate SQLite memories to Neo4j")
    parser.add_argument(
        "--sqlite-path",
        default="memories.db",
        help="Path to the SQLite database (default: memories.db)",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name (default: neo4j)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip re-embedding (keep existing vectors as-is — NOT recommended)",
    )
    args = parser.parse_args()

    # 1. Read SQLite
    data = read_sqlite(args.sqlite_path)
    memories = data["memories"]
    if not memories:
        logger.warning("No memories to migrate.")
        return

    # 2. Re-embed
    if args.skip_embeddings:
        logger.warning("Skipping re-embedding. Vectors will be empty.")
        embeddings = [[0.0] * 384 for _ in memories]
    else:
        model = get_embedding_model()
        texts = build_embedding_texts(memories)
        logger.info("Re-embedding %d memories...", len(texts))
        embeddings = embed_texts(model, texts)
        logger.info("Embedding complete (dim=%d).", len(embeddings[0]) if embeddings else 0)

    # 3. Write to Neo4j
    driver = get_neo4j_driver()
    db = args.database

    logger.info("Setting up constraints...")
    setup_constraints(driver, db)

    logger.info("Writing memories...")
    write_memories(driver, db, memories, embeddings)

    logger.info("Writing feedback events...")
    write_feedback_events(driver, db, data["feedback_events"])

    logger.info("Writing traces...")
    write_traces(driver, db, data["traces"])

    logger.info("Building co-occurrence graph...")
    build_co_occurrence_edges(driver, db)

    driver.close()
    logger.info("Migration complete.")


if __name__ == "__main__":
    main()

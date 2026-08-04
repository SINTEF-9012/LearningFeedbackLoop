from __future__ import annotations

import json
from types import SimpleNamespace
from datetime import datetime, timezone

import neo4j
import pytest

from backend.agents.core.schemas import Memory
from backend.agents.storage.graph_boundary import (
    ALLOWED_CROSS_GRAPH_RELATIONSHIPS,
    KNOWLEDGE_GRAPH_LABELS,
    MEMORY_GRAPH_LABELS,
)
from backend.agents.storage.graph_write_outbox import GraphWriteIntent, GraphWriteOutbox
from backend.agents.storage.neo4j_store import Neo4jMemoryStore, _SCHEMA_INITIALIZED
from backend.agents.patterns.signatures import infer_pattern_kind


class _DummySession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, *args, **kwargs):
        return SimpleNamespace(consume=lambda: None)


class _DummyDriver:
    def session(self, database=None):
        return _DummySession()


class _SchemaCaptureSession:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cypher, **kwargs):
        self.calls.append((cypher, kwargs))
        return SimpleNamespace(consume=lambda: None)


class _SchemaCaptureDriver:
    def __init__(self, session):
        self._session = session

    def session(self, database=None):
        return self._session


class _CaptureTx:
    def __init__(self, rows=None):
        self.calls = []
        self._rows = list(rows or [])

    def run(self, cypher, **kwargs):
        self.calls.append((cypher, kwargs))
        rows = self._rows.pop(0) if self._rows else []
        return SimpleNamespace(consume=lambda: None, data=lambda: rows)


class _CaptureSession:
    def __init__(self, tx):
        self._tx = tx

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, *args, **kwargs):
        return self._tx.run(*args, **kwargs)

    def execute_write(self, func):
        return func(self._tx)


class _CaptureDriver:
    def __init__(self, tx):
        self._tx = tx

    def session(self, database=None):
        return _CaptureSession(self._tx)


def test_neo4j_store_uses_configured_driver_limits(monkeypatch):
    captured = {}
    previous_initialized = set(_SCHEMA_INITIALIZED)

    def fake_driver(uri, auth, **kwargs):
        captured["uri"] = uri
        captured["auth"] = auth
        captured["kwargs"] = kwargs
        return _DummyDriver()

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", fake_driver)
    _SCHEMA_INITIALIZED.clear()

    try:
        Neo4jMemoryStore(
            uri="bolt://example:7687",
            username="neo4j",
            password="secret",
            database="neo4j",
            connect_timeout_s=1.25,
            max_pool_size=17,
            max_transaction_retry_s=9.5,
        )
    finally:
        _SCHEMA_INITIALIZED.clear()
        _SCHEMA_INITIALIZED.update(previous_initialized)

    assert captured["uri"] == "bolt://example:7687"
    assert captured["auth"] == ("neo4j", "secret")
    assert captured["kwargs"]["connection_timeout"] == 1.25
    assert captured["kwargs"]["connection_acquisition_timeout"] == 1.25
    assert captured["kwargs"]["max_connection_pool_size"] == 17
    assert captured["kwargs"]["max_transaction_retry_time"] == 9.5


def test_ensure_schema_creates_pattern_prior_and_cooccurrence_indexes():
    session = _SchemaCaptureSession()
    store = object.__new__(Neo4jMemoryStore)
    store._driver = _SchemaCaptureDriver(session)
    store._database = "neo4j"

    Neo4jMemoryStore._ensure_schema(store)

    cyphers = [call[0] for call in session.calls]
    assert any("CREATE INDEX pattern_prior_idx" in cypher for cypher in cyphers)
    assert any("CREATE INDEX co_occurs_weight_idx" in cypher for cypher in cyphers)
    assert any("CREATE CONSTRAINT co_occurrence_update_id_unique" in cypher for cypher in cyphers)


def test_infer_pattern_kind_matches_phase_i_namespaces():
    assert infer_pattern_kind("freq:high") == "generic_physics"
    assert infer_pattern_kind("signature:spindle_shift_phase_change") == "signature"
    assert infer_pattern_kind("hypothesis:workpiece_slip") == "signature"
    assert infer_pattern_kind("discovered:cluster_1") == "discovered"
    assert infer_pattern_kind("SPINDLE_POWER_SURGE") == "domain_rule"
    assert infer_pattern_kind("ANOMALY_HIGH:0.9", pattern_type="anomaly") == "model_score"


def test_serialize_memory_promotes_curated_metadata_fields():
    memory = Memory(
        id="mem-serialize",
        session_id="session-1",
        time_range=(0.0, 1.0),
        created_at=datetime.now(timezone.utc),
        machine_uri="urn:lfl:asset:machine_a1",
        metadata={
            "source": "SITE_A",
            "machine_family": "machine_a1",
            "dataset_id": "site_a_casedata",
            "source_dataset_id": "site_a_line2",
            "machine_iri": "urn:test:machine-iri",
            "sindit_asset_iri": "urn:test:asset-iri",
            "cutting_context": {
                "machine_id": "MACHINE_A1",
                "extra": {
                    "machine_family": "machine_a1",
                    "sindit_tool_iri": "urn:lfl:tool:machine_a1-t7",
                },
            },
            "casedata": {
                "operation_id": "OF00011",
                "dataset_id": "site_a_casedata",
                "case_dir": "Site_b - MACHINE_B1 - CASE_B1",
                "tool_id": "tool-7",
            },
        },
    )

    props = Neo4jMemoryStore._serialize_memory(memory)

    assert props["operation_id"] == "OF00011"
    assert props["dataset_id"] == "site_a_casedata"
    assert props["source_dataset_id"] == "site_a_line2"
    assert props["machine_family"] == "machine_a1"
    assert props["machine_iri"] == "urn:test:machine-iri"
    assert props["sindit_asset_iri"] == "urn:test:asset-iri"
    assert props["sindit_tool_iri"] == "urn:lfl:tool:machine_a1-t7"
    assert props["case_dir"] == "Site_b - MACHINE_B1 - CASE_B1"
    assert props["operation_node_id"] == "site_a_casedata::Site_b - MACHINE_B1 - CASE_B1::OF00011"


def test_store_creates_operation_and_dataset_links():
    tx = _CaptureTx()
    store = object.__new__(Neo4jMemoryStore)
    store._driver = _CaptureDriver(tx)
    store._database = "neo4j"

    memory = Memory(
        id="mem-op-dataset",
        session_id="session-1",
        time_range=(0.0, 1.0),
        created_at=datetime.now(timezone.utc),
        metadata={
            "dataset_id": "site_a_casedata",
            "source_dataset_id": "site_a_line2",
            "casedata": {
                "operation_id": "OF00011",
                "case_dir": "Site_a - MACHINE_A1 - CASE_A1",
                "dataset_id": "site_a_casedata",
            },
        },
    )

    Neo4jMemoryStore.store(store, memory)

    cyphers = [call[0] for call in tx.calls]
    assert any("MERGE (ds:Dataset {id: $dataset_id})" in cypher for cypher in cyphers)
    assert any("MERGE (m)-[:DURING]->(op)" in cypher for cypher in cyphers)
    assert any("MERGE (op)-[:OF_DATASET]->(ds)" in cypher for cypher in cyphers)
    op_call = next(kwargs for cypher, kwargs in tx.calls if "MERGE (m)-[:DURING]->(op)" in cypher)
    assert op_call["operation_node_id"] == "site_a_casedata::Site_a - MACHINE_A1 - CASE_A1::OF00011"


def test_persist_doc_links_stores_links_on_memory_metadata():
    tx = _CaptureTx(rows=[[{"metadata_json": "{}"}], []])
    store = object.__new__(Neo4jMemoryStore)
    store._driver = _CaptureDriver(tx)
    store._database = "neo4j"

    linked = Neo4jMemoryStore.persist_doc_links(
        store,
        memory_id="mem-1",
        pattern_keys=["fault:chatter"],
        doc_links=[
            {
                "id": "doc-1",
                "score": 0.99,
                "page": 330,
                "citation": "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1",
                "query_used": "regenerative chatter harmonic vibration tooth passing",
                "pattern_key": "fault:chatter",
            }
        ],
    )

    assert linked == 1
    read_call, write_call = tx.calls
    assert "RETURN m.metadata_json AS metadata_json" in read_call[0]
    assert "SET m.metadata_json = $metadata_json" in write_call[0]
    assert write_call[1]["doc_link_ids"] == ["doc-1"]
    metadata = json.loads(write_call[1]["metadata_json"])
    assert metadata["doc_links"][0]["pattern_key"] == "fault:chatter"
    assert metadata["doc_links"][0]["evidence_entities"] == []


def test_persist_doc_links_returns_zero_when_memory_missing():
    tx = _CaptureTx(rows=[[]])
    store = object.__new__(Neo4jMemoryStore)
    store._driver = _CaptureDriver(tx)
    store._database = "neo4j"

    linked = Neo4jMemoryStore.persist_doc_links(
        store,
        memory_id="missing-memory",
        pattern_keys=["fault:chatter"],
        doc_links=[
            {
                "id": "missing-doc",
                "score": 0.99,
                "page": 330,
                "citation": "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1",
                "query_used": "regenerative chatter harmonic vibration tooth passing",
                "pattern_key": "fault:chatter",
            }
        ],
    )

    assert linked == 0
    assert len(tx.calls) == 1
    assert "RETURN m.metadata_json AS metadata_json" in tx.calls[0][0]


def test_get_doc_links_reads_memory_metadata():
    store = object.__new__(Neo4jMemoryStore)
    captured = {}

    def fake_run(cypher, **kwargs):
        captured["cypher"] = cypher
        captured["kwargs"] = kwargs
        return [
            {
                "metadata_json": json.dumps(
                    {
                        "doc_links": [
                            {
                                "id": "doc-1",
                                "citation": "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1",
                                "score": 0.99,
                                "page": 330,
                                "file_name": "chatter.pdf",
                                "source": "machinedocs",
                                "usecase": "SITE_A",
                                "machine": "MACHINE_A1",
                                "text": "Regenerative chatter guidance.",
                                "document_type": "pdf",
                                "language": "en",
                                "query_used": "regenerative chatter harmonic vibration tooth passing",
                                "pattern_key": "fault:chatter",
                                "doc_feedback": "helpful",
                                "helpful_count": 2,
                                "not_helpful_count": 1,
                                "feedback_score": 1.0,
                                "evidence_entities": [{"id": "e-1", "name": "Chatter", "type": "Symptom"}],
                            }
                        ]
                    }
                )
            }
        ]

    store._run = fake_run
    store._run_single = lambda cypher, **kwargs: fake_run(cypher, **kwargs)[0]

    links = Neo4jMemoryStore.get_doc_links(store, "mem-1", score_floor=0.6, limit=5)

    assert "RETURN m.metadata_json AS metadata_json" in captured["cypher"]
    assert captured["kwargs"] == {"memory_id": "mem-1"}
    assert links[0]["pattern_key"] == "fault:chatter"
    assert links[0]["doc_feedback"] == "helpful"
    assert links[0]["helpful_count"] == 2
    assert links[0]["evidence_entities"][0]["name"] == "Chatter"


def test_set_doc_link_feedback_updates_memory_metadata():
    tx = _CaptureTx(
        rows=[
            [
                {
                    "metadata_json": json.dumps(
                        {
                            "doc_links": [
                                {
                                    "id": "doc-1",
                                    "citation": "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1",
                                    "score": 0.99,
                                    "page": 330,
                                    "file_name": "chatter.pdf",
                                    "source": "machinedocs",
                                    "usecase": "SITE_A",
                                    "machine": "MACHINE_A1",
                                    "text": "Regenerative chatter guidance.",
                                    "document_type": "pdf",
                                    "language": "en",
                                    "query_used": "regenerative chatter harmonic vibration tooth passing",
                                    "pattern_key": "fault:chatter",
                                    "helpful_count": 0,
                                    "not_helpful_count": 0,
                                    "feedback_score": 0.0,
                                    "evidence_entities": [],
                                }
                            ]
                        }
                    )
                }
            ],
            [],
        ]
    )
    store = object.__new__(Neo4jMemoryStore)
    store._driver = _CaptureDriver(tx)
    store._database = "neo4j"

    updated = Neo4jMemoryStore.set_doc_link_feedback(
        store,
        memory_id="mem-1",
        doc_id="doc-1",
        feedback="helpful",
        user_id="operator",
        reason="Most actionable citation",
    )

    assert updated is not None
    assert updated["doc_feedback"] == "helpful"
    assert updated["feedback_score"] == 1.0
    read_call, write_call = tx.calls
    assert "RETURN m.metadata_json AS metadata_json" in read_call[0]
    assert "SET m.metadata_json = $metadata_json" in write_call[0]
    assert write_call[1]["doc_link_ids"] == ["doc-1"]
    metadata = json.loads(write_call[1]["metadata_json"])
    assert metadata["doc_links"][0]["doc_feedback"] == "helpful"
    assert metadata["doc_links"][0]["helpful_count"] == 1


def test_trace_write_is_retained_in_outbox_and_replayed(tmp_path):
    outbox_path = tmp_path / "neo4j_graph_outbox.jsonl"

    failing_store = object.__new__(Neo4jMemoryStore)
    failing_store._graph_write_outbox = GraphWriteOutbox(outbox_path)
    failing_store._apply_graph_write_intent = lambda intent: (_ for _ in ()).throw(RuntimeError("neo4j unavailable"))

    trace_id = Neo4jMemoryStore.add_trace(
        failing_store,
        session_id="session-1",
        memory_id="mem-1",
        trace_type="score",
        payload={"score": 0.84},
    )

    assert failing_store._graph_write_outbox.pending_count() == 1

    tx = _CaptureTx(rows=[[], [{"linked": 1}], []])
    replay_store = object.__new__(Neo4jMemoryStore)
    replay_store._driver = _CaptureDriver(tx)
    replay_store._database = "neo4j"
    replay_store._graph_write_outbox = GraphWriteOutbox(outbox_path)

    processed = Neo4jMemoryStore._flush_graph_write_outbox(replay_store)

    assert processed == 1
    assert replay_store._graph_write_outbox.pending_count() == 0
    cypher, kwargs = tx.calls[0]
    assert "MERGE (t:Trace {id: $tid})" in cypher
    assert kwargs["tid"] == trace_id
    assert kwargs["tt"] == "score"


def test_add_feedback_event_queues_on_failure_and_replays(tmp_path):
    outbox_path = tmp_path / "neo4j_graph_outbox.jsonl"

    failing_store = object.__new__(Neo4jMemoryStore)
    failing_store._graph_write_outbox = GraphWriteOutbox(outbox_path)
    failing_store._flush_graph_write_outbox = lambda *args, **kwargs: 0

    def fail_feedback_event(payload):
        raise RuntimeError("neo4j unavailable")

    failing_store._apply_feedback_event_intent = fail_feedback_event

    with pytest.raises(RuntimeError, match="neo4j unavailable"):
        Neo4jMemoryStore.add_feedback_event(
            failing_store,
            memory_id="mem-1",
            action="confirm",
            user_id="operator",
            pattern_keys=["CUSTOM:test"],
            data={"source": "confirm_explicit"},
            weight=1.0,
        )

    assert failing_store._graph_write_outbox.pending_count() == 1

    tx = _CaptureTx(rows=[[], [{"linked": 1}], []])
    replay_store = object.__new__(Neo4jMemoryStore)
    replay_store._driver = _CaptureDriver(tx)
    replay_store._database = "neo4j"
    replay_store._graph_write_outbox = GraphWriteOutbox(outbox_path)

    processed = Neo4jMemoryStore._flush_graph_write_outbox(replay_store)

    assert processed == 1
    assert replay_store._graph_write_outbox.pending_count() == 0
    cyphers = [call[0] for call in tx.calls]
    assert any("MERGE (f:Feedback {id: $eid})" in cypher for cypher in cyphers)
    assert any("MERGE (f)-[:ABOUT]->(m)" in cypher for cypher in cyphers)
    assert any("MERGE (f)-[:ON_PATTERN]->(p)" in cypher for cypher in cyphers)
    feedback_call = next(kwargs for cypher, kwargs in tx.calls if "MERGE (f:Feedback {id: $eid})" in cypher)
    assert feedback_call["mid"] == "mem-1"
    assert feedback_call["action"] == "confirm"
    assert feedback_call["uid"] == "operator"


def test_discovered_pattern_write_is_retained_in_outbox_and_replayed(tmp_path):
    outbox_path = tmp_path / "neo4j_graph_outbox.jsonl"

    failing_store = object.__new__(Neo4jMemoryStore)
    failing_store._graph_write_outbox = GraphWriteOutbox(outbox_path)
    failing_store._apply_graph_write_intent = lambda intent: (_ for _ in ()).throw(RuntimeError("neo4j unavailable"))

    Neo4jMemoryStore.store_discovered_pattern(
        failing_store,
        key="discovered:cluster_1",
        features={"power_spindle_mean": "high"},
        confirmation_count=3,
        promoted=True,
        prior=0.77,
        first_seen="2026-05-20T20:00:00+00:00",
        last_seen="2026-05-20T20:05:00+00:00",
        source_memory_ids=["mem-1"],
    )

    assert failing_store._graph_write_outbox.pending_count() == 1

    tx = _CaptureTx()
    replay_store = object.__new__(Neo4jMemoryStore)
    replay_store._driver = _CaptureDriver(tx)
    replay_store._database = "neo4j"
    replay_store._graph_write_outbox = GraphWriteOutbox(outbox_path)

    processed = Neo4jMemoryStore._flush_graph_write_outbox(replay_store)

    assert processed == 1
    assert replay_store._graph_write_outbox.pending_count() == 0
    cyphers = [call[0] for call in tx.calls]
    assert any("MERGE (dp:DiscoveredPattern {key: $key})" in cypher for cypher in cyphers)
    assert any("MERGE (dp)-[:DISCOVERED_FROM]->(m)" in cypher for cypher in cyphers)
    assert any("MERGE (p:Pattern {key: $key})" in cypher for cypher in cyphers)


def test_persist_doc_links_queues_on_failure_and_replays(tmp_path):
    outbox_path = tmp_path / "neo4j_graph_outbox.jsonl"

    failing_store = object.__new__(Neo4jMemoryStore)
    failing_store._graph_write_outbox = GraphWriteOutbox(outbox_path)
    failing_store._flush_graph_write_outbox = lambda *args, **kwargs: 0

    def fail_doc_links(payload):
        raise RuntimeError("neo4j unavailable")

    failing_store._apply_doc_links_intent = fail_doc_links

    linked = Neo4jMemoryStore.persist_doc_links(
        failing_store,
        memory_id="mem-1",
        pattern_keys=["fault:chatter"],
        doc_links=[
            {
                "id": "doc-1",
                "score": 0.99,
                "page": 330,
                "citation": "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1",
                "query_used": "regenerative chatter harmonic vibration tooth passing",
                "pattern_key": "fault:chatter",
            }
        ],
    )

    assert linked == 1
    assert failing_store._graph_write_outbox.pending_count() == 1

    tx = _CaptureTx(rows=[[{"metadata_json": "{}"}], []])
    replay_store = object.__new__(Neo4jMemoryStore)
    replay_store._driver = _CaptureDriver(tx)
    replay_store._database = "neo4j"
    replay_store._graph_write_outbox = GraphWriteOutbox(outbox_path)

    processed = Neo4jMemoryStore._flush_graph_write_outbox(replay_store)

    assert processed == 1
    assert replay_store._graph_write_outbox.pending_count() == 0
    cyphers = [call[0] for call in tx.calls]
    assert any("RETURN m.metadata_json AS metadata_json" in cypher for cypher in cyphers)
    assert any("SET m.metadata_json = $metadata_json" in cypher for cypher in cyphers)


def test_store_experiment_queues_on_failure_and_replays(tmp_path):
    outbox_path = tmp_path / "neo4j_graph_outbox.jsonl"

    failing_store = object.__new__(Neo4jMemoryStore)
    failing_store._graph_write_outbox = GraphWriteOutbox(outbox_path)
    failing_store._flush_graph_write_outbox = lambda *args, **kwargs: 0

    def fail_experiment(payload):
        raise RuntimeError("neo4j unavailable")

    failing_store._apply_experiment_intent = fail_experiment

    Neo4jMemoryStore.store_experiment(
        failing_store,
        run_id="exp-1",
        experiment_type="stoppage",
        config={"test_op": "OF00011"},
        test_metrics={"f1": 0.81, "precision": 0.79, "recall": 0.84},
        eval_metrics={"f1": 0.76, "precision": 0.75, "recall": 0.78},
        comparison={"delta_f1": -0.05, "pct_f1_improvement": -6.17},
        session_ids=["session-1", "session-2"],
    )

    assert failing_store._graph_write_outbox.pending_count() == 1

    tx = _CaptureTx()
    replay_store = object.__new__(Neo4jMemoryStore)
    replay_store._driver = _CaptureDriver(tx)
    replay_store._database = "neo4j"
    replay_store._graph_write_outbox = GraphWriteOutbox(outbox_path)

    processed = Neo4jMemoryStore._flush_graph_write_outbox(replay_store)

    assert processed == 1
    assert replay_store._graph_write_outbox.pending_count() == 0
    cyphers = [call[0] for call in tx.calls]
    assert any("MERGE (e:Experiment {run_id: $rid})" in cypher for cypher in cyphers)
    assert any("MERGE (e)-[:HAS_SESSION]->(s)" in cypher for cypher in cyphers)
    assert any("MERGE (e)-[:TESTED_PATTERN]->(p)" in cypher for cypher in cyphers)
    experiment_call = next(kwargs for cypher, kwargs in tx.calls if "MERGE (e:Experiment {run_id: $rid})" in cypher)
    assert experiment_call["rid"] == "exp-1"
    assert experiment_call["etype"] == "stoppage"


def test_co_occurrence_write_queues_on_failure_and_replays(tmp_path):
    outbox_path = tmp_path / "neo4j_graph_outbox.jsonl"

    failing_store = object.__new__(Neo4jMemoryStore)
    failing_store._graph_write_outbox = GraphWriteOutbox(outbox_path)
    failing_store._flush_graph_write_outbox = lambda *args, **kwargs: 0

    def fail_run(*args, **kwargs):
        raise RuntimeError("neo4j unavailable")

    failing_store._run = fail_run

    Neo4jMemoryStore.upsert_co_occurrence(
        failing_store,
        "pattern:b",
        "pattern:a",
        "session-1",
    )

    assert failing_store._graph_write_outbox.pending_count() == 1

    tx = _CaptureTx(rows=[[{"applied_at": None}], [{"applied": 1}], []])
    replay_store = object.__new__(Neo4jMemoryStore)
    replay_store._driver = _CaptureDriver(tx)
    replay_store._database = "neo4j"
    replay_store._graph_write_outbox = GraphWriteOutbox(outbox_path)

    processed = Neo4jMemoryStore._flush_graph_write_outbox(replay_store)

    assert processed == 1
    assert replay_store._graph_write_outbox.pending_count() == 0
    cyphers = [call[0] for call in tx.calls]
    assert any("MERGE (cu:CoOccurrenceUpdate {id: $intent_id})" in cypher for cypher in cyphers)
    assert any("MERGE (pa)-[r:CO_OCCURS_WITH]-(pb)" in cypher for cypher in cyphers)
    replay_call = next(kwargs for cypher, kwargs in tx.calls if "MERGE (pa)-[r:CO_OCCURS_WITH]-(pb)" in cypher)
    assert replay_call["a"] == "pattern:a"
    assert replay_call["b"] == "pattern:b"


def test_co_occurrence_replay_is_idempotent_by_intent_sequence():
    tx = _CaptureTx(
        rows=[
            [{"applied_at": None}],
            [{"applied": 1}],
            [],
            [{"applied_at": "2026-05-20T21:00:00+00:00"}],
        ]
    )
    replay_store = object.__new__(Neo4jMemoryStore)
    replay_store._driver = _CaptureDriver(tx)
    replay_store._database = "neo4j"

    intent = GraphWriteIntent(
        kind="co_occurrence",
        payload={
            "pattern_key_a": "pattern:a",
            "pattern_key_b": "pattern:b",
            "session_id": "session-1",
            "created_at": "2026-05-20T21:00:00+00:00",
        },
        created_at="2026-05-20T21:00:00+00:00",
        sequence=7,
    )

    Neo4jMemoryStore._apply_graph_write_intent(replay_store, intent)
    Neo4jMemoryStore._apply_graph_write_intent(replay_store, intent)

    increment_calls = [
        call for call in tx.calls if "MERGE (pa)-[r:CO_OCCURS_WITH]-(pb)" in call[0]
    ]
    assert len(increment_calls) == 1


def test_clear_all_includes_co_occurrence_update_nodes():
    calls = []
    store = object.__new__(Neo4jMemoryStore)

    def fake_run(cypher, **kwargs):
        calls.append((cypher, kwargs))
        return [{"c": 1}]

    store._run = fake_run

    counts = Neo4jMemoryStore.clear_all(store)

    assert counts["CoOccurrenceUpdate"] == 1
    assert any("MATCH (n:CoOccurrenceUpdate)" in cypher for cypher, _ in calls)


def test_clear_memory_graph_is_scoped_to_memory_labels():
    calls = []
    store = object.__new__(Neo4jMemoryStore)

    def fake_run(cypher, **kwargs):
        calls.append((cypher, kwargs))
        return [{"c": 1}]

    store._run = fake_run

    counts = Neo4jMemoryStore.clear_memory_graph(store)

    assert counts["CoOccurrenceUpdate"] == 1
    assert any("MATCH (n:Memory)" in cypher for cypher, _ in calls)
    assert any("MATCH (n:Snapshot)" in cypher for cypher, _ in calls)
    assert not any("MATCH (n:Operation)" in cypher for cypher, _ in calls)
    assert not any("MATCH (n:Dataset)" in cypher for cypher, _ in calls)


def test_preview_memory_graph_cleanup_summarizes_nodes_and_bridge_edges():
    store = object.__new__(Neo4jMemoryStore)
    calls = []

    node_counts = {
        "Memory": 4,
        "Pattern": 2,
        "Experiment": 1,
    }

    def fake_run(cypher, **kwargs):
        calls.append((cypher, kwargs))
        if "RETURN count(n) AS c" in cypher:
            label = cypher.split(":", 1)[1].split(")", 1)[0]
            return [{"c": node_counts.get(label, 0)}]
        if "total_relationships_to_delete" in cypher:
            return [{"total_relationships_to_delete": 9}]
        if "relationship_type" in cypher:
            if not kwargs.get("allowed_relationships"):
                return []
            return [
                {"relationship_type": "CITES", "c": 3},
                {"relationship_type": "DOCUMENTED_BY", "c": 2},
            ]
        if "candidate_memories" in cypher:
            return [{
                "total_memories": 4,
                "candidate_memories": 3,
                "candidate_sessions": 2,
                "oldest_memory_at": "2026-01-01T00:00:00+00:00",
                "newest_memory_at": "2026-01-04T00:00:00+00:00",
                "oldest_candidate_created_at": "2026-01-01T00:00:00+00:00",
                "newest_candidate_created_at": "2026-01-03T00:00:00+00:00",
            }]
        if "AS created_by" in cypher:
            return [
                {"created_by": "system", "c": 2},
                {"created_by": "operator", "c": 1},
            ]
        if "AS usecase" in cypher:
            return [
                {"usecase": "SITE_A", "c": 2},
                {"usecase": "SITE_B", "c": 1},
            ]
        if "AS session_id" in cypher and "memory_count" in cypher:
            return [
                {
                    "session_id": "session-legacy-1",
                    "memory_count": 2,
                    "oldest_created_at": "2026-01-01T00:00:00+00:00",
                    "newest_created_at": "2026-01-02T00:00:00+00:00",
                },
                {
                    "session_id": "session-legacy-2",
                    "memory_count": 1,
                    "oldest_created_at": "2026-01-03T00:00:00+00:00",
                    "newest_created_at": "2026-01-03T00:00:00+00:00",
                },
            ]
        raise AssertionError(f"unexpected cypher: {cypher}")

    store._run = fake_run

    preview = Neo4jMemoryStore.preview_memory_graph_cleanup(store)

    assert preview == {
        "scope": "memory_graph",
        "total_nodes_to_delete": 7,
        "total_relationships_to_delete": 9,
        "node_counts": {label: node_counts.get(label, 0) for label in MEMORY_GRAPH_LABELS},
        "bridge_relationship_counts": {},
        "legacy_candidate_summary": {
            "heuristic": "dataset_id|source_dataset_id|case_dir|operation_id|created_by!=operator|linked_experiment",
            "total_memories": 4,
            "candidate_memories": 3,
            "candidate_sessions": 2,
            "oldest_memory_at": "2026-01-01T00:00:00+00:00",
            "newest_memory_at": "2026-01-04T00:00:00+00:00",
            "oldest_candidate_created_at": "2026-01-01T00:00:00+00:00",
            "newest_candidate_created_at": "2026-01-03T00:00:00+00:00",
            "created_by_counts": {"system": 2, "operator": 1},
            "usecase_counts": {"SITE_A": 2, "SITE_B": 1},
            "top_sessions": [
                {
                    "session_id": "session-legacy-1",
                    "memory_count": 2,
                    "oldest_created_at": "2026-01-01T00:00:00+00:00",
                    "newest_created_at": "2026-01-02T00:00:00+00:00",
                },
                {
                    "session_id": "session-legacy-2",
                    "memory_count": 1,
                    "oldest_created_at": "2026-01-03T00:00:00+00:00",
                    "newest_created_at": "2026-01-03T00:00:00+00:00",
                },
            ],
        },
        "memory_labels": list(MEMORY_GRAPH_LABELS),
        "knowledge_labels_preserved": list(KNOWLEDGE_GRAPH_LABELS),
        "allowed_cross_relationships": list(ALLOWED_CROSS_GRAPH_RELATIONSHIPS),
    }
    bridge_call = next(kwargs for cypher, kwargs in calls if "relationship_type" in cypher)
    assert bridge_call["memory_labels"] == list(MEMORY_GRAPH_LABELS)
    assert bridge_call["knowledge_labels"] == list(KNOWLEDGE_GRAPH_LABELS)
    assert bridge_call["allowed_relationships"] == list(ALLOWED_CROSS_GRAPH_RELATIONSHIPS)


def test_clear_legacy_candidate_memories_deletes_candidates_and_orphans_only():
    store = object.__new__(Neo4jMemoryStore)
    calls = []

    def fake_run(cypher, **kwargs):
        calls.append((cypher, kwargs))
        if "RETURN DISTINCT m.session_id AS sid" in cypher:
            return [{"sid": "session-legacy-1"}, {"sid": "session-legacy-2"}]
        if "RETURN DISTINCT p.key AS pattern_key" in cypher:
            return [{"pattern_key": "pattern:a"}, {"pattern_key": "pattern:b"}]
        if "RETURN DISTINCT ma.id AS machine_id" in cypher:
            return [{"machine_id": "machine-a"}]
        if "RETURN DISTINCT t.id AS tool_id" in cypher:
            return [{"tool_id": "tool-a"}]
        if "RETURN DISTINCT e.run_id AS run_id" in cypher:
            return [{"run_id": "run-legacy-1"}]
        if "DETACH DELETE f" in cypher:
            return [{"c": 4}]
        if "DETACH DELETE t RETURN count(t) AS c" in cypher and "memory_id = m.id" in cypher:
            return [{"c": 3}]
        if "DETACH DELETE m RETURN count(m) AS c" in cypher:
            return [{"c": 5}]
        if "DETACH DELETE cu" in cypher:
            return [{"c": 2}]
        if "DETACH DELETE s RETURN count(s) AS c" in cypher:
            return [{"c": 2}]
        if "MATCH (e:Experiment) WHERE e.run_id IN $run_ids" in cypher:
            return [{"c": 1}]
        if "MATCH (ma:Machine) WHERE ma.id IN $machine_ids" in cypher:
            return [{"c": 1}]
        if "MATCH (t:Tool) WHERE t.id IN $tool_ids" in cypher:
            return [{"c": 1}]
        if "MATCH (p:Pattern) WHERE p.key IN $pattern_keys" in cypher:
            return [{"c": 2}]
        raise AssertionError(f"unexpected cypher: {cypher}")

    store._run = fake_run

    counts = Neo4jMemoryStore.clear_legacy_candidate_memories(store)

    assert counts == {
        "Feedback": 4,
        "Trace": 3,
        "Memory": 5,
        "CoOccurrenceUpdate": 2,
        "Session": 2,
        "Experiment": 1,
        "Machine": 1,
        "Tool": 1,
        "Pattern": 2,
    }
    session_query = next(kwargs for cypher, kwargs in calls if "DETACH DELETE cu" in cypher)
    assert session_query["sids"] == ["session-legacy-1", "session-legacy-2"]
    pattern_query = next(kwargs for cypher, kwargs in calls if "MATCH (p:Pattern) WHERE p.key IN $pattern_keys" in cypher)
    assert pattern_query["pattern_keys"] == ["pattern:a", "pattern:b"]
    assert any("dataset_id" in cypher and "created_by" in cypher for cypher, _ in calls if "MATCH (m:Memory)" in cypher)


def test_graph_stats_reports_subgraph_integrity_summary():
    store = object.__new__(Neo4jMemoryStore)
    calls = []

    node_counts = {
        "Memory": 5,
        "Pattern": 3,
    }
    rel_counts = {
        "HAS_PATTERN": 4,
        "CITES": 2,
    }

    def fake_run(cypher, **kwargs):
        calls.append((cypher, kwargs))
        if "RETURN count(n) AS c" in cypher:
            label = cypher.split(":", 1)[1].split(")", 1)[0]
            return [{"c": node_counts.get(label, 0)}]
        if "RETURN count(r) AS c" in cypher:
            rel = cypher.split(":", 1)[1].split("]", 1)[0]
            return [{"c": rel_counts.get(rel, 0)}]
        if "mixed_label_nodes" in cypher:
            return [{"mixed_label_nodes": 1}]
        if "disallowed_cross_graph_edges" in cypher:
            return [{
                "disallowed_cross_graph_edges": 2,
                "disallowed_relationship_types": ["ON_MACHINE", "USED_TOOL"],
            }]
        raise AssertionError(f"unexpected cypher: {cypher}")

    store._run = fake_run

    stats = Neo4jMemoryStore.graph_stats(store)

    assert stats["node_counts"]["Memory"] == 5
    assert stats["relationship_counts"]["CITES"] == 2
    assert stats["subgraph_integrity"] == {
        "healthy": False,
        "mixed_label_nodes": 1,
        "disallowed_cross_graph_edges": 2,
        "disallowed_relationship_types": ["ON_MACHINE", "USED_TOOL"],
        "memory_labels": list(MEMORY_GRAPH_LABELS),
        "knowledge_labels": list(KNOWLEDGE_GRAPH_LABELS),
        "allowed_cross_relationships": list(ALLOWED_CROSS_GRAPH_RELATIONSHIPS),
    }
    integrity_calls = [
        kwargs
        for cypher, kwargs in calls
        if "mixed_label_nodes" in cypher or "disallowed_cross_graph_edges" in cypher
    ]
    assert integrity_calls[0]["memory_labels"] == list(MEMORY_GRAPH_LABELS)
    assert integrity_calls[0]["knowledge_labels"] == list(KNOWLEDGE_GRAPH_LABELS)
    assert integrity_calls[0]["allowed_relationships"] == list(ALLOWED_CROSS_GRAPH_RELATIONSHIPS)


def test_graph_stats_marks_subgraph_integrity_healthy_when_no_violations():
    store = object.__new__(Neo4jMemoryStore)

    def fake_run(cypher, **kwargs):
        if "mixed_label_nodes" in cypher:
            return [{"mixed_label_nodes": 0}]
        if "disallowed_cross_graph_edges" in cypher:
            return [{"disallowed_cross_graph_edges": 0, "disallowed_relationship_types": []}]
        if "RETURN count(n) AS c" in cypher or "RETURN count(r) AS c" in cypher:
            return [{"c": 0}]
        raise AssertionError(f"unexpected cypher: {cypher}")

    store._run = fake_run

    stats = Neo4jMemoryStore.graph_stats(store)

    assert stats["subgraph_integrity"]["healthy"] is True
    assert stats["subgraph_integrity"]["mixed_label_nodes"] == 0
    assert stats["subgraph_integrity"]["disallowed_cross_graph_edges"] == 0


def test_get_runtime_identity_snapshot_reads_operation_and_dataset_nodes():
    store = object.__new__(Neo4jMemoryStore)
    calls = []

    def fake_run(cypher, **kwargs):
        calls.append((cypher, kwargs))
        if "MATCH (op:Operation" in cypher:
            return [{"props": {"id": "dataset-1::Case-1::OP-1", "dataset_id": "dataset-1"}}]
        if "MATCH (ds:Dataset" in cypher:
            return [{"props": {"id": "dataset-1", "source_dataset_id": "site_a_line2"}}]
        return []

    store._run = fake_run

    snapshot = Neo4jMemoryStore.get_runtime_identity_snapshot(
        store,
        operation_node_id="dataset-1::Case-1::OP-1",
        dataset_id="dataset-1",
    )

    assert snapshot["operation_node"]["dataset_id"] == "dataset-1"
    assert snapshot["dataset_node"]["source_dataset_id"] == "site_a_line2"
    assert any("MATCH (op:Operation" in cypher for cypher, _ in calls)
    assert any("MATCH (ds:Dataset" in cypher for cypher, _ in calls)


def test_delete_experiment_cleans_up_co_occurrence_update_nodes():
    calls = []
    store = object.__new__(Neo4jMemoryStore)

    def fake_run(cypher, **kwargs):
        calls.append((cypher, kwargs))
        if "RETURN s.id AS sid" in cypher:
            return [{"sid": "session-1"}]
        if "MATCH (cu:CoOccurrenceUpdate)" in cypher:
            return [{"c": 2}]
        return [{"c": 0}]

    store._run = fake_run

    counts = Neo4jMemoryStore.delete_experiment(store, "exp-1")

    assert counts["CoOccurrenceUpdate"] == 2
    cleanup_call = next(kwargs for cypher, kwargs in calls if "MATCH (cu:CoOccurrenceUpdate)" in cypher)
    assert cleanup_call["sids"] == ["session-1"]


def test_graph_outbox_status_helpers_report_pending_intents(tmp_path):
    outbox_path = tmp_path / "neo4j_graph_outbox.jsonl"
    outbox = GraphWriteOutbox(outbox_path)
    outbox.append(
        GraphWriteIntent(
            kind="trace",
            payload={"trace_id": "t-1", "trace_type": "score", "created_at": "2026-05-20T21:00:00+00:00"},
            created_at="2026-05-20T21:00:00+00:00",
        )
    )

    store = object.__new__(Neo4jMemoryStore)
    store._graph_write_outbox = outbox

    assert Neo4jMemoryStore.graph_outbox_enabled(store) is True
    assert Neo4jMemoryStore.graph_outbox_pending_count(store) == 1
    pending = Neo4jMemoryStore.list_pending_graph_writes(store, limit=1)
    assert pending[0]["kind"] == "trace"
    assert pending[0]["sequence"] == 1
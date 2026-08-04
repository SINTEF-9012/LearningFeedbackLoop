"""orchestrator.list_memories must use the store's scoped list_by_session when
present, not the fragile list_all(1000)+python-filter path (ISS-24).

On a large graph, list_all(limit=1000) returns only the newest 1000 rows, so a
session whose memories fall outside that window is silently returned as empty.
list_by_session runs a session-scoped query and is correct regardless of size.
"""
from backend.agents.memory.orchestrator import MemoryEventOrchestrator, OrchestratorConfig
from backend.agents.core.schemas import Memory


class _StoreWithScopedList:
    """Mimics Neo4jMemoryStore: has list_by_session + list_all but NO .list."""
    def __init__(self):
        self.by_session_called_with = None
        self.list_all_called = False

    def list_by_session(self, session_id, limit=100):
        self.by_session_called_with = (session_id, limit)
        return [Memory(id="m1", session_id=session_id, time_range=(0.0, 1.0),
                       annotation_text="x", pattern_keys=[], metadata={})]

    def list_all(self, limit=1000, **kwargs):
        # Simulate the fragile path: session's memory NOT in the newest window.
        self.list_all_called = True
        return [Memory(id="other", session_id="other-session", time_range=(0.0, 1.0),
                       annotation_text="y", pattern_keys=[], metadata={})]


def _orch(tmp_path, store):
    orch = MemoryEventOrchestrator(config=OrchestratorConfig(
        use_classical_models=False, enable_harmonic_scorer=False, dispatch_alerts=False,
        priors_path=str(tmp_path / "p.json"), model_confidence_path=str(tmp_path / "mc.json"),
    ))
    orch.store = store
    return orch


def test_session_query_uses_scoped_list(tmp_path):
    store = _StoreWithScopedList()
    orch = _orch(tmp_path, store)
    mems = orch.list_memories(session_id="sess-42")
    assert store.by_session_called_with == ("sess-42", 500)
    assert not store.list_all_called  # did NOT fall back to the fragile path
    assert len(mems) == 1 and mems[0].session_id == "sess-42"


def test_unscoped_query_still_uses_list_all(tmp_path):
    store = _StoreWithScopedList()
    orch = _orch(tmp_path, store)
    orch.list_memories(session_id=None)  # no session -> list_all
    assert store.list_all_called
    assert store.by_session_called_with is None

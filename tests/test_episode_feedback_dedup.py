"""Episode-level learning dedup (plan 1.4).

When several windows of one episode are adjudicated with the same episode_id,
the learning update (durable prior counts) must apply once, not once per window —
while every window still records its ground-truth label.
"""
import pytest

from backend.agents.core.schemas import Memory, PatternKey, PatternType
from backend.agents.memory.feedback import (
    FeedbackAction,
    MemoryFeedbackHandler,
    MemoryFeedbackRequest,
)
from backend.agents.memory.scorer import SignificanceScorer
from backend.agents.storage.store import MemoryStore


def _make(tmp_path, prefix, n=3):
    store = MemoryStore(db_path=str(tmp_path / f"{prefix}.db"), enable_ann=False, enable_embeddings=False)
    for i in range(n):
        store.create(Memory(
            id=f"m{i}", session_id="s", time_range=(float(i), i + 1.0),
            annotation_text="ep window",
            pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:ep")],
            metadata={},
        ))
    scorer = SignificanceScorer(priors_path=str(tmp_path / f"{prefix}_p.json"), feedback_store=store)
    handler = MemoryFeedbackHandler(memory_store=store, significance_scorer=scorer)
    return store, handler


def _dismiss_weight(store):
    _, d = store.get_feedback_counts(pattern_key="CUSTOM:ep")
    return d


async def _dismiss(handler, mid, episode_id=None):
    await handler.process_feedback(
        mid, MemoryFeedbackRequest(action=FeedbackAction.DISMISS, episode_id=episode_id),
    )


@pytest.mark.asyncio
async def test_same_episode_counts_once(tmp_path):
    # Baseline: a single dismiss establishes the per-event weight.
    base_store, base_handler = _make(tmp_path, "base")
    await _dismiss(base_handler, "m0")
    unit = _dismiss_weight(base_store)
    assert unit > 0

    # Three windows of ONE episode → still just one unit of learning.
    store, handler = _make(tmp_path, "ep")
    for i in range(3):
        await _dismiss(handler, f"m{i}", episode_id="sig::t0")
    assert _dismiss_weight(store) == pytest.approx(unit)
    # …but every window keeps its label.
    for i in range(3):
        assert store.get(f"m{i}").metadata.get("user_dismissed") is True


@pytest.mark.asyncio
async def test_no_episode_id_counts_every_window(tmp_path):
    base_store, base_handler = _make(tmp_path, "base")
    await _dismiss(base_handler, "m0")
    unit = _dismiss_weight(base_store)

    store, handler = _make(tmp_path, "nd")
    for i in range(3):
        await _dismiss(handler, f"m{i}")
    assert _dismiss_weight(store) == pytest.approx(3 * unit)


@pytest.mark.asyncio
async def test_distinct_episodes_each_count(tmp_path):
    base_store, base_handler = _make(tmp_path, "base")
    await _dismiss(base_handler, "m0")
    unit = _dismiss_weight(base_store)

    store, handler = _make(tmp_path, "two")
    await _dismiss(handler, "m0", episode_id="sig::t0")
    await _dismiss(handler, "m1", episode_id="sig::t9")
    assert _dismiss_weight(store) == pytest.approx(2 * unit)


def test_dispatcher_emits_stable_episode_id():
    """The alert recurrence snapshot carries a stable episode_id (signature +
    first_seen) so the operator UI can echo it back on feedback (plan 1.4)."""
    from backend.agents.memory.dispatcher import AlertDispatcher

    d = AlertDispatcher()
    a = d._touch_signature_lifecycle("sess", "sig:chatter", suppressed=False)
    b = d._touch_signature_lifecycle("sess", "sig:chatter", suppressed=False)
    assert a["episode_id"] and a["episode_id"] == b["episode_id"]
    assert b["occurrences"] == 2  # same episode, two occurrences
    # A different signature is a different episode.
    c = d._touch_signature_lifecycle("sess", "sig:breakage", suppressed=False)
    assert c["episode_id"] != a["episode_id"]
    # No signature → no episode.
    assert d._touch_signature_lifecycle("sess", None, suppressed=False) is None

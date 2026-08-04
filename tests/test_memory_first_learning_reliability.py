import json
import asyncio
from unittest.mock import ANY, Mock

import pytest

from backend.agents.config import MemorySystemConfig
from backend.agents.core.context import CuttingContext, OperatingRegime
from backend.agents.core.schemas import Memory, PatternKey, PatternType
from backend.agents.memory.feature_stream_bridge import start_memory_processor, stop_memory_processor
from backend.agents.memory.init import initialize_memory_system, shutdown_memory_system, get_store
from backend.agents.memory.feedback import MemoryFeedbackHandler, MemoryFeedbackRequest, FeedbackAction
from backend.agents.memory.scorer import FEEDBACK_WEIGHTS, SignificanceConfig, SignificanceScorer
from backend.agents.patterns.discovery import PatternDiscovery
from backend.agents.storage.store import MemoryStore
from backend.events import publish_feature


@pytest.mark.asyncio
async def test_feedback_persists_metadata_in_sqlite(tmp_path):
    db_path = str(tmp_path / "memories.db")
    store = MemoryStore(db_path=db_path, enable_ann=False, enable_embeddings=False)

    mem = Memory(
        id="m1",
        session_id="s1",
        time_range=(0.0, 1.0),
        annotation_text="test",
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:test")],
        metadata={},
    )
    store.create(mem)

    scorer = SignificanceScorer(priors_path=str(tmp_path / "priors.json"))
    handler = MemoryFeedbackHandler(memory_store=store, significance_scorer=scorer)

    resp = await handler.process_feedback(
        "m1",
        MemoryFeedbackRequest(action=FeedbackAction.CONFIRM, user_id="tester"),
    )
    assert resp.success is True

    updated = store.get("m1")
    assert updated is not None
    assert updated.metadata.get("user_confirmed") is True
    assert updated.metadata.get("confirmed_by") == "tester"
    assert "confirmed_at" in updated.metadata


def test_pattern_priors_persist_across_restart(tmp_path):
    priors_path = str(tmp_path / "pattern_priors.json")

    scorer1 = SignificanceScorer(priors_path=priors_path)
    scorer1.update_pattern_prior("CUSTOM:test", was_significant=True)

    scorer2 = SignificanceScorer(priors_path=priors_path)
    assert scorer2._pattern_priors.get("CUSTOM:test", 0.5) > 0.5


def test_store_snapshot_priors_override_file_cache(tmp_path):
    class SnapshotStore:
        def get_pattern_priors_snapshot(self):
            return {"CUSTOM:test": 0.82}

    priors_path = tmp_path / "pattern_priors.json"
    priors_path.write_text(json.dumps({
        "pattern_priors": {"CUSTOM:test": 0.11},
        "feedback_counts": {"CUSTOM:test": {"confirm": 0, "dismiss": 3}},
    }))

    scorer = SignificanceScorer(
        priors_path=str(priors_path),
        feedback_store=SnapshotStore(),
    )

    assert scorer._pattern_priors.get("CUSTOM:test") == pytest.approx(0.82, rel=1e-6)


def test_store_feedback_history_priors_override_file_cache(tmp_path):
    store = MemoryStore(
        db_path=str(tmp_path / "memories.db"),
        enable_ann=False,
        enable_embeddings=False,
    )
    store.add_feedback_event(
        memory_id=None,
        action="confirm",
        user_id="tester",
        pattern_keys=["CUSTOM:test"],
    )

    priors_path = tmp_path / "pattern_priors.json"
    priors_path.write_text(json.dumps({
        "pattern_priors": {"CUSTOM:test": 0.12},
        "feedback_counts": {"CUSTOM:test": {"confirm": 0, "dismiss": 4}},
    }))

    scorer = SignificanceScorer(
        priors_path=str(priors_path),
        feedback_store=store,
    )

    assert scorer._pattern_priors.get("CUSTOM:test", 0.5) > 0.5


def test_store_backed_scoring_is_unchanged_when_priors_file_is_absent(tmp_path):
    store = MemoryStore(
        db_path=str(tmp_path / "memories.db"),
        enable_ann=False,
        enable_embeddings=False,
    )
    for _ in range(4):
        store.add_feedback_event(
            memory_id=None,
            action="confirm",
            user_id="tester",
            pattern_keys=["CUSTOM:test"],
        )

    priors_path = tmp_path / "pattern_priors.json"
    priors_path.write_text(json.dumps({
        "pattern_priors": {"CUSTOM:test": 0.02},
        "feedback_counts": {"CUSTOM:test": {"confirm": 0, "dismiss": 9}},
        "prior_source": "conflicting_test_fixture",
    }))

    scorer_with_file = SignificanceScorer(
        config=SignificanceConfig(prior_evidence_damping_k=0.0),
        priors_path=str(priors_path),
        feedback_store=store,
    )
    scorer_without_file = SignificanceScorer(
        config=SignificanceConfig(prior_evidence_damping_k=0.0),
        priors_path=str(tmp_path / "missing_pattern_priors.json"),
        feedback_store=store,
    )

    pattern = PatternKey(pattern_type=PatternType.CUSTOM, key="CUSTOM:test")
    prior_with_file = scorer_with_file.get_pattern_prior("CUSTOM:test")
    prior_without_file = scorer_without_file.get_pattern_prior("CUSTOM:test")
    result_with_file = scorer_with_file.score([pattern])
    result_without_file = scorer_without_file.score([pattern])

    assert prior_with_file == pytest.approx(prior_without_file, rel=1e-6)
    assert result_with_file.score == pytest.approx(result_without_file.score, rel=1e-6)
    assert result_with_file.action == result_without_file.action


def test_bootstrap_seeded_priors_are_ignored_by_default(tmp_path):
    priors_path = tmp_path / "pattern_priors.json"
    priors_path.write_text(json.dumps({
        "bootstrap_seeded": True,
        "prior_source": "repo_seed",
        "pattern_priors": {"CUSTOM:test": 0.9},
        "feedback_counts": {"CUSTOM:test": {"confirm": 3, "dismiss": 0}},
    }))

    scorer = SignificanceScorer(
        config=SignificanceConfig(bootstrap_pattern_priors=False),
        priors_path=str(priors_path),
    )

    assert scorer._pattern_priors == {}
    assert scorer.get_pattern_prior("CUSTOM:test") == pytest.approx(0.5, rel=1e-6)


def test_bootstrap_seeded_priors_load_when_enabled(tmp_path):
    priors_path = tmp_path / "pattern_priors.json"
    priors_path.write_text(json.dumps({
        "bootstrap_seeded": True,
        "prior_source": "repo_seed",
        "pattern_priors": {"CUSTOM:test": 0.9},
        "feedback_counts": {"CUSTOM:test": {"confirm": 3, "dismiss": 0}},
    }))

    scorer = SignificanceScorer(
        config=SignificanceConfig(bootstrap_pattern_priors=True),
        priors_path=str(priors_path),
    )

    assert scorer._pattern_priors.get("CUSTOM:test") == pytest.approx(0.9, rel=1e-6)
    assert scorer.get_pattern_prior("CUSTOM:test") > 0.5


@pytest.mark.asyncio
async def test_feedback_updates_learning_hooks_and_buffers(tmp_path):
    db_path = str(tmp_path / "memories.db")
    store = MemoryStore(db_path=db_path, enable_ann=False, enable_embeddings=False)

    mem = Memory(
        id="m_learn",
        session_id="s_learn",
        time_range=(0.0, 1.0),
        annotation_text="test",
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="SPINDLE_POWER_SURGE")],
        metadata={
            "raw_metrics": {
                "power_spindle_mean": 8.0,
                "power_spindle_max": 9.0,
                "power_spindle_std": 1.0,
            },
            "external_signals": {"anomaly_detector_score": 0.88},
            "triggered_rules": ["classical_alert", "pattern_match"],
            "significance_score": 0.72,
            "significance_action": "alert",
            "cutting_context": {
                "tool_type": "end_mill",
                "workpiece_material": "steel",
                "machine_type": "CNC_5axis",
            },
        },
    )
    store.create(mem)

    scorer = SignificanceScorer(priors_path=str(tmp_path / "priors.json"))
    scorer.record_rule_feedback = Mock()
    scorer.update_weight_profile_from_feedback = Mock()
    scorer.record_model_feedback = Mock()
    scorer.record_rl_feedback = Mock()
    scorer.record_feedback_for_adaptive_thresholds = Mock()

    handler = MemoryFeedbackHandler(memory_store=store, significance_scorer=scorer)
    handler.pattern_discovery = Mock()
    handler.retrainer = Mock()

    resp = await handler.process_feedback(
        "m_learn",
        MemoryFeedbackRequest(action=FeedbackAction.CONFIRM, user_id="tester"),
    )

    assert resp.success is True
    scorer.record_rule_feedback.assert_called_once_with(["classical_alert", "pattern_match"], True)
    scorer.update_weight_profile_from_feedback.assert_called_once()
    scorer.record_model_feedback.assert_called_once_with(
        triggered_rules=["classical_alert", "pattern_match"],
        was_confirmed=True,
        external_signals={"anomaly_detector_score": 0.88},
        cutting_context=ANY,  # reconstructed from metadata; enables scoped trust (1.1)
    )
    scorer.record_rl_feedback.assert_called_once()
    scorer.record_feedback_for_adaptive_thresholds.assert_called_once_with(
        0.72,
        "alert",
        True,
        weight=FEEDBACK_WEIGHTS["confirm_explicit"],
    )
    handler.pattern_discovery.analyse_confirmed_event.assert_called_once()
    handler.retrainer.record_feedback.assert_called_once()


@pytest.mark.asyncio
async def test_dismiss_feedback_learns_suppression_and_stats_aliases(tmp_path):
    db_path = str(tmp_path / "memories.db")
    store = MemoryStore(db_path=db_path, enable_ann=False, enable_embeddings=False)

    mem = Memory(
        id="m_dismiss",
        session_id="s_dismiss",
        time_range=(0.0, 1.0),
        annotation_text="test",
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="amp:ch0:loud")],
        metadata={
            "raw_metrics": {"power_spindle_mean": 3.0},
            "triggered_rules": ["pattern_match"],
            "significance_score": 0.25,
            "significance_action": "ignore",
        },
    )
    store.create(mem)

    scorer = SignificanceScorer(priors_path=str(tmp_path / "priors.json"))
    scorer.record_rule_feedback = Mock()
    scorer.update_weight_profile_from_feedback = Mock()
    scorer.record_model_feedback = Mock()
    scorer.record_rl_feedback = Mock()
    scorer.record_feedback_for_adaptive_thresholds = Mock()

    handler = MemoryFeedbackHandler(memory_store=store, significance_scorer=scorer)
    handler.pattern_discovery = Mock()
    handler.retrainer = Mock()

    resp = await handler.process_feedback(
        "m_dismiss",
        MemoryFeedbackRequest(action=FeedbackAction.DISMISS, user_id="tester"),
    )

    assert resp.success is True
    handler.pattern_discovery.analyse_dismissed_event.assert_called_once()
    stats = handler.get_feedback_stats("m_dismiss")
    assert stats["dismisses"] == 1
    assert stats["dismiss_count"] == 1


def test_confirmed_discovery_skips_when_curated_pattern_already_explains_event(tmp_path):
    discovery = PatternDiscovery(data_dir=tmp_path)

    for _ in range(25):
        discovery.update_baseline({
            "power_spindle_mean": 1.0,
            "power_spindle_max": 1.0,
            "power_spindle_std": 1.0,
            "feed_rate_mean": 1.0,
        })

    novel = discovery.analyse_confirmed_event(
        {
            "power_spindle_mean": 10.0,
            "power_spindle_max": 10.0,
            "power_spindle_std": 10.0,
            "feed_rate_mean": 10.0,
        },
        existing_pattern_keys=["SPINDLE_POWER_SURGE"],
    )

    assert novel == []
    assert discovery.get_patterns() == {}


def test_confirmed_discovery_still_runs_for_generic_non_curated_patterns(tmp_path):
    discovery = PatternDiscovery(data_dir=tmp_path)

    for _ in range(25):
        discovery.update_baseline({
            "power_spindle_mean": 1.0,
            "power_spindle_max": 1.0,
            "power_spindle_std": 1.0,
            "feed_rate_mean": 1.0,
        })

    novel = discovery.analyse_confirmed_event(
        {
            "power_spindle_mean": 10.0,
            "power_spindle_max": 10.0,
            "power_spindle_std": 10.0,
            "feed_rate_mean": 10.0,
        },
        existing_pattern_keys=["amp:ch0:loud"],
    )

    assert len(novel) == 1
    assert list(discovery.get_patterns().keys())[0].startswith("discovered:")


def test_discovery_partitions_and_matches_same_signature_by_context(tmp_path):
    discovery = PatternDiscovery(data_dir=tmp_path, min_confirmations=1)
    endmill_context = CuttingContext(
        machine_type="cnc",
        tool_type="endmill",
        workpiece_material="al",
        operating_regime=OperatingRegime.ROUGHING,
    )
    drill_context = CuttingContext(
        machine_type="cnc",
        tool_type="drill",
        workpiece_material="al",
        operating_regime=OperatingRegime.ROUGHING,
    )

    for _ in range(25):
        discovery.update_baseline(
            {
                "power_spindle_mean": 1.0,
                "power_spindle_max": 1.0,
                "power_spindle_std": 1.0,
                "feed_rate_mean": 1.0,
            },
            cutting_context=endmill_context,
        )
        discovery.update_baseline(
            {
                "power_spindle_mean": 1.0,
                "power_spindle_max": 1.0,
                "power_spindle_std": 1.0,
                "feed_rate_mean": 1.0,
            },
            cutting_context=drill_context,
        )

    features = {
        "power_spindle_mean": 10.0,
        "power_spindle_max": 10.0,
        "power_spindle_std": 10.0,
        "feed_rate_mean": 10.0,
    }

    endmill_patterns = discovery.analyse_confirmed_event(
        features,
        cutting_context=endmill_context,
    )
    drill_patterns = discovery.analyse_confirmed_event(
        features,
        cutting_context=drill_context,
    )

    assert len(endmill_patterns) == 1
    assert len(drill_patterns) == 1
    assert endmill_patterns[0].key != drill_patterns[0].key
    assert len(discovery.get_patterns()) == 2
    assert discovery.match_event(features, cutting_context=endmill_context) == [endmill_patterns[0].key]
    assert discovery.match_event(features, cutting_context=drill_context) == [drill_patterns[0].key]


def test_discovery_uses_context_specific_baselines_for_same_absolute_signal(tmp_path):
    discovery = PatternDiscovery(data_dir=tmp_path, min_confirmations=1)
    endmill_context = CuttingContext(
        machine_type="cnc",
        tool_type="endmill",
        workpiece_material="al",
        operating_regime=OperatingRegime.ROUGHING,
    )
    drill_context = CuttingContext(
        machine_type="cnc",
        tool_type="drill",
        workpiece_material="al",
        operating_regime=OperatingRegime.ROUGHING,
    )
    low_baseline = {
        "power_spindle_mean": 1.0,
        "power_spindle_max": 1.0,
        "power_spindle_std": 1.0,
        "feed_rate_mean": 1.0,
    }
    high_baseline = {
        "power_spindle_mean": 10.0,
        "power_spindle_max": 10.0,
        "power_spindle_std": 10.0,
        "feed_rate_mean": 10.0,
    }

    for _ in range(25):
        discovery.update_baseline(low_baseline, cutting_context=endmill_context)
        discovery.update_baseline(high_baseline, cutting_context=drill_context)

    endmill_patterns = discovery.analyse_confirmed_event(
        high_baseline,
        cutting_context=endmill_context,
    )
    drill_patterns = discovery.analyse_confirmed_event(
        high_baseline,
        cutting_context=drill_context,
    )

    assert len(endmill_patterns) == 1
    assert drill_patterns == []


@pytest.mark.asyncio
async def test_bridge_consumes_feature_and_stores_memory(tmp_path):
    # Use in-memory DB for speed; disable heavy optional indices.
    config = MemorySystemConfig(
        storage_backend="sqlite",
        db_path=":memory:",
        enable_ann=False,
        enable_embeddings=False,
        generate_explanations=False,
        dispatch_alerts=False,
        use_classical_models=False,
    )

    initialize_memory_system(config=config, force=True)
    try:
        await start_memory_processor()

        # Give the background task a chance to subscribe before publishing.
        await asyncio.sleep(0.05)

        # Publish a feature event that will be significant due to classical alert signal.
        payload = {
            "type": "time",
            "session_id": "bridge_session",
            "position": 1024,
            "window_size": 1024,
            "external_signals": {"breakage_prediction": 1.0},
            # No metrics/patterns required; default provider will still attach minimal patterns.
            "frame": {"fs": 1000.0, "t": 1.024, "i": 1024, "A": [0.0, 1.0, 0.0]},
        }
        await publish_feature("bridge_session", payload)

        store = get_store()
        assert store is not None

        # Wait for background processor to store at least one memory.
        deadline = asyncio.get_running_loop().time() + 3.0
        while asyncio.get_running_loop().time() < deadline:
            if store.list(session_id="bridge_session", limit=10):
                break
            await asyncio.sleep(0.05)

        memories = store.list(session_id="bridge_session", limit=10)
        assert len(memories) >= 1
    finally:
        await stop_memory_processor()
        shutdown_memory_system()

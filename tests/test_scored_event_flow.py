from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.core.schemas import PatternKey, PatternType, TimeRange
from backend.agents.memory.dispatcher import AlertDispatcher, RateLimitConfig
from backend.agents.memory.orchestrator import MemoryEvent, MemoryEventOrchestrator, OrchestratorConfig
from backend.agents.memory.scorer import SignificanceAction


@pytest.mark.asyncio
async def test_dispatcher_broadcast_scored_event_accepts_event_without_memory():
    dispatcher = AlertDispatcher()
    queue = dispatcher.subscribe(session_id="session-1")
    event = MemoryEvent(
        session_id="session-1",
        time_range=TimeRange(i0=0, i1=7, t0=0.0, t1=7.0, fs=1.0),
        patterns=[PatternKey(pattern_type=PatternType.ANOMALY, key="ANOMALY_HIGH:0.1")],
    )
    significance = SimpleNamespace(
        score=0.2,
        action=SignificanceAction.IGNORE,
        reasons=["Routine observation"],
        prior_boost=0.0,
        pattern_priors={},
    )

    try:
        await dispatcher.broadcast_scored_event(
            significance=significance,
            event=event,
            metrics_summary={"anomaly_detector_score": 0.2},
        )
        alert = await asyncio.wait_for(queue.get(), timeout=1.0)
    finally:
        dispatcher.unsubscribe(queue, session_id="session-1")

    payload = alert.to_scored_dict()
    assert payload["type"] == "scored_event"
    assert payload["event_id"] == "scored:session-1:7"
    assert payload["session_id"] == "session-1"
    assert payload["patterns"] == ["ANOMALY_HIGH:0.1"]
    assert payload["time_range"] == {"i0": 0, "i1": 7, "t0": 0.0, "t1": 7.0, "fs": 1.0}
    assert payload["doc_links"] == []
    assert json.loads(alert.to_json())["type"] == "scored_event"


@pytest.mark.asyncio
async def test_dispatcher_fallback_summary_rewrites_live_reason_to_observation_language():
    dispatcher = AlertDispatcher()
    queue = dispatcher.subscribe(session_id="session-obs")
    event = MemoryEvent(
        session_id="session-obs",
        time_range=TimeRange(i0=0, i1=5, t0=0.0, t1=5.0, fs=1.0),
        patterns=[PatternKey(pattern_type=PatternType.CUSTOM, key="signature:modulated_tooth_passing_vibration")],
    )
    significance = SimpleNamespace(
        score=0.83,
        action=SignificanceAction.ALERT,
        reasons=["Chatter severity: 0.83"],
        prior_boost=0.0,
        pattern_priors={},
    )

    try:
        await dispatcher.broadcast_scored_event(significance=significance, event=event)
        alert = await asyncio.wait_for(queue.get(), timeout=1.0)
    finally:
        dispatcher.unsubscribe(queue, session_id="session-obs")

    payload = alert.to_scored_dict()
    assert payload["category"] == "Vibration Modulation"
    assert "Vibration modulation severity: 0.83" in payload["summary"]
    assert "Chatter" not in payload["summary"]


@pytest.mark.asyncio
async def test_dispatcher_demo_summary_avoids_fault_nouns_for_legacy_breakage_keys():
    dispatcher = AlertDispatcher()
    queue = dispatcher.subscribe(session_id="session-breakage")
    event = MemoryEvent(
        session_id="session-breakage",
        time_range=TimeRange(i0=0, i1=5, t0=0.0, t1=5.0, fs=1.0),
        patterns=[PatternKey(pattern_type=PatternType.CUSTOM, key="fault:tool_breakage")],
    )
    significance = SimpleNamespace(
        score=0.91,
        action=SignificanceAction.ALERT,
        reasons=["Significant pattern: fault:tool_breakage"],
        prior_boost=0.0,
        pattern_priors={},
    )

    try:
        await dispatcher.broadcast_scored_event(significance=significance, event=event)
        alert = await asyncio.wait_for(queue.get(), timeout=1.0)
    finally:
        dispatcher.unsubscribe(queue, session_id="session-breakage")

    payload = alert.to_scored_dict()
    summary = (payload.get("summary") or "").lower()
    assert payload["category"] == "High-Frequency Burst"
    assert "high-frequency burst with periodicity loss" in summary
    assert "breakage" not in summary
    assert "fracture" not in summary


@pytest.mark.asyncio
async def test_dispatcher_payload_includes_similar_history_when_provided():
    dispatcher = AlertDispatcher()
    queue = dispatcher.subscribe(session_id="session-history")
    memory = SimpleNamespace(
        id="mem-history",
        session_id="session-history",
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="signature:spindle_shift_phase_change")],
        time_range=TimeRange(i0=0, i1=5, t0=0.0, t1=5.0, fs=1.0),
    )
    significance = SimpleNamespace(
        score=0.88,
        action=SignificanceAction.ALERT,
        reasons=["Vibration modulation severity: 0.88"],
        prior_boost=0.0,
        pattern_priors={},
    )

    try:
        await dispatcher.dispatch(
            memory=memory,
            significance=significance,
            similar_memories=[],
            similar_history=[
                {
                    "id": "mem-old",
                    "label": "tool_break",
                    "shared_pattern_keys": ["signature:spindle_shift_phase_change"],
                    "shared_pattern_details": [
                        {
                            "key": "signature:spindle_shift_phase_change",
                            "candidate_strength": 0.74,
                        }
                    ],
                    "feedback": {"confirm_count": 1, "dismiss_count": 0, "last_action": "confirm"},
                }
            ],
        )
        alert = await asyncio.wait_for(queue.get(), timeout=1.0)
    finally:
        dispatcher.unsubscribe(queue, session_id="session-history")

    payload = alert.to_dict()
    assert payload["similar_history"][0]["label"] == "tool_break"
    assert payload["similar_history"][0]["feedback"]["last_action"] == "confirm"


@pytest.mark.asyncio
async def test_dispatcher_payload_includes_doc_links_when_available():
    dispatcher = AlertDispatcher()
    dispatcher._propose_doc_links = AsyncMock(return_value=[
        {
            "id": "doc-1",
            "citation": "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1",
            "page": 330,
            "score": 0.9995,
            "query_used": "Regenerative chatter with amplitude modulation and harmonic energy. Modulation Amplitude growth Harmonic energy",
            "pattern_key": "fault:chatter",
            "evidence_entities": [{"id": "e-1", "name": "Chatter", "type": "Symptom"}],
        }
    ])
    queue = dispatcher.subscribe(session_id="session-doc-links")
    memory = SimpleNamespace(
        id="mem-doc-links",
        session_id="session-doc-links",
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="fault:chatter")],
        time_range=TimeRange(i0=0, i1=5, t0=0.0, t1=5.0, fs=1.0),
        metadata={"source": "SITE_A"},
        channels=["Vibration_Severity_X", "Power_Spindle"],
        machine_uri="MACHINE_A1",
    )
    significance = SimpleNamespace(
        score=0.88,
        action=SignificanceAction.ALERT,
        reasons=["Vibration modulation severity: 0.88"],
        prior_boost=0.0,
        pattern_priors={},
    )

    try:
        await dispatcher.dispatch(
            memory=memory,
            significance=significance,
            similar_memories=[],
            cutting_context={"machine_id": "MACHINE_A1", "tool_type": "end mill"},
        )
        alert = await asyncio.wait_for(queue.get(), timeout=1.0)
    finally:
        dispatcher.unsubscribe(queue, session_id="session-doc-links")

    payload = alert.to_dict()
    assert payload["doc_links"][0]["citation"] == "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1"
    assert payload["doc_links"][0]["evidence_entities"][0]["name"] == "Chatter"
    dispatcher._propose_doc_links.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatcher_fallback_summary_quotes_historical_label_as_history():
    dispatcher = AlertDispatcher()
    queue = dispatcher.subscribe(session_id="session-history-summary")
    memory = SimpleNamespace(
        id="mem-history-summary",
        session_id="session-history-summary",
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="signature:spindle_shift_phase_change")],
        time_range=TimeRange(i0=0, i1=5, t0=0.0, t1=5.0, fs=1.0),
    )
    significance = SimpleNamespace(
        score=0.88,
        action=SignificanceAction.ALERT,
        reasons=["Spindle-order shift likelihood: 0.88"],
        prior_boost=0.0,
        pattern_priors={},
    )

    try:
        await dispatcher.dispatch(
            memory=memory,
            significance=significance,
            similar_memories=[],
            similar_history=[
                {
                    "id": "mem-old",
                    "label": "tool_break",
                    "shared_pattern_keys": ["signature:spindle_shift_phase_change"],
                    "feedback": {"confirm_count": 1, "dismiss_count": 0, "last_action": "confirm"},
                }
            ],
        )
        alert = await asyncio.wait_for(queue.get(), timeout=1.0)
    finally:
        dispatcher.unsubscribe(queue, session_id="session-history-summary")

    summary = str(alert.to_dict().get("summary") or "")
    assert "historical label \"tool break\"" in summary
    assert "Spindle-order shift likelihood: 0.88" in summary


@pytest.mark.asyncio
async def test_orchestrator_dispatch_passes_similar_history_from_store():
    orchestrator = MemoryEventOrchestrator(
        config=OrchestratorConfig(
            use_classical_models=False,
            enable_harmonic_scorer=False,
            dispatch_alerts=True,
            generate_explanations=False,
        )
    )
    event = MemoryEvent(
        session_id="session-history-2",
        time_range=TimeRange(i0=0, i1=2, t0=0.0, t1=2.0, fs=1.0),
        patterns=[PatternKey(pattern_type=PatternType.CUSTOM, key="signature:spindle_shift_phase_change")],
        metadata={},
    )
    significance = SimpleNamespace(
        score=0.92,
        action=SignificanceAction.ALERT,
        is_significant=True,
        triggered_rules=["ALERT"],
        reasons=["Spindle-order shift likelihood: 0.92"],
        prior_boost=0.0,
        pattern_priors={},
        score_trace=[],
        to_dict=lambda: {"score": 0.92, "action": "alert", "is_significant": True},
    )

    orchestrator.scorer.score = MagicMock(return_value=significance)
    orchestrator.retriever.retrieve = MagicMock(return_value=[])
    orchestrator._create_memory = MagicMock(
        return_value=SimpleNamespace(
            id="mem-h2",
            session_id="session-history-2",
            pattern_keys=event.patterns,
            time_range=event.time_range,
        )
    )
    orchestrator._store_memory = AsyncMock(return_value="mem-h2")
    orchestrator._add_trace = MagicMock()
    orchestrator._build_metrics_for_alert = MagicMock(return_value={})
    orchestrator.store.get_similar_with_resolution = MagicMock(return_value=[
        {
            "id": "mem-old",
            "label": "tool_break",
            "shared_pattern_keys": ["signature:spindle_shift_phase_change"],
            "shared_pattern_details": [{"key": "signature:spindle_shift_phase_change", "candidate_strength": 0.74}],
            "feedback": {"confirm_count": 1, "dismiss_count": 0, "last_action": "confirm", "last_comment": "changed tool"},
        }
    ])
    orchestrator.alert_dispatcher.dispatch = AsyncMock(return_value=True)
    orchestrator.alert_dispatcher.broadcast_scored_event = AsyncMock()

    result = await orchestrator.process_event(event)

    assert result.processed is True
    kwargs = orchestrator.alert_dispatcher.dispatch.await_args.kwargs
    assert kwargs["similar_history"][0]["label"] == "tool_break"
    assert kwargs["similar_history"][0]["feedback"]["last_comment"] == "changed tool"


@pytest.mark.asyncio
async def test_orchestrator_precomputes_doc_links_once_for_dispatch_and_persistence():
    orchestrator = MemoryEventOrchestrator(
        config=OrchestratorConfig(
            use_classical_models=False,
            enable_harmonic_scorer=False,
            dispatch_alerts=True,
            generate_explanations=False,
        )
    )
    event = MemoryEvent(
        session_id="session-doc-persist",
        time_range=TimeRange(i0=0, i1=2, t0=0.0, t1=2.0, fs=1.0),
        patterns=[PatternKey(pattern_type=PatternType.CUSTOM, key="fault:chatter")],
        metadata={"source": "SITE_A"},
    )
    significance = SimpleNamespace(
        score=0.91,
        action=SignificanceAction.ALERT,
        is_significant=True,
        triggered_rules=["ALERT"],
        reasons=["Vibration modulation severity: 0.91"],
        prior_boost=0.0,
        pattern_priors={},
        score_trace=[],
        to_dict=lambda: {"score": 0.91, "action": "alert", "is_significant": True},
    )
    doc_links = [
        {
            "id": "doc-1",
            "citation": "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1",
            "page": 330,
            "score": 0.9995,
            "query_used": "regenerative chatter harmonic vibration tooth passing",
            "pattern_key": "fault:chatter",
        }
    ]
    memory = SimpleNamespace(
        id="mem-doc-persist",
        session_id="session-doc-persist",
        pattern_keys=event.patterns,
        time_range=event.time_range,
        metadata={"source": "SITE_A"},
        channels=["Vibration_Severity_X"],
        machine_uri="MACHINE_A1",
    )

    orchestrator.scorer.score = MagicMock(return_value=significance)
    orchestrator.retriever.retrieve = MagicMock(return_value=[])
    orchestrator._create_memory = MagicMock(return_value=memory)
    orchestrator._store_memory = AsyncMock(return_value="mem-doc-persist")
    orchestrator._add_trace = MagicMock()
    orchestrator._build_metrics_for_alert = MagicMock(return_value={})
    orchestrator.store.get_similar_with_resolution = MagicMock(return_value=[])
    orchestrator.store.persist_doc_links = MagicMock(return_value=1)
    orchestrator.alert_dispatcher.propose_doc_links_for_memory = AsyncMock(return_value=doc_links)
    orchestrator.alert_dispatcher.dispatch = AsyncMock(return_value=True)
    orchestrator.alert_dispatcher.broadcast_scored_event = AsyncMock()

    result = await orchestrator.process_event(event)

    assert result.processed is True
    orchestrator.alert_dispatcher.propose_doc_links_for_memory.assert_awaited_once_with(
        memory=memory,
        cutting_context=None,
    )
    kwargs = orchestrator.alert_dispatcher.dispatch.await_args.kwargs
    assert kwargs["doc_links"] == doc_links
    orchestrator.store.persist_doc_links.assert_called_once_with(
        memory_id="mem-doc-persist",
        pattern_keys=["fault:chatter"],
        doc_links=doc_links,
    )


@pytest.mark.asyncio
async def test_orchestrator_persists_doc_links_even_when_dispatch_is_suppressed():
    orchestrator = MemoryEventOrchestrator(
        config=OrchestratorConfig(
            use_classical_models=False,
            enable_harmonic_scorer=False,
            dispatch_alerts=True,
            generate_explanations=False,
        )
    )
    event = MemoryEvent(
        session_id="session-doc-persist-muted",
        time_range=TimeRange(i0=0, i1=2, t0=0.0, t1=2.0, fs=1.0),
        patterns=[PatternKey(pattern_type=PatternType.CUSTOM, key="fault:chatter")],
        metadata={"source": "SITE_A"},
    )
    significance = SimpleNamespace(
        score=0.91,
        action=SignificanceAction.ALERT,
        is_significant=True,
        triggered_rules=["ALERT"],
        reasons=["Vibration modulation severity: 0.91"],
        prior_boost=0.0,
        pattern_priors={},
        score_trace=[],
        to_dict=lambda: {"score": 0.91, "action": "alert", "is_significant": True},
    )
    doc_links = [
        {
            "id": "doc-1",
            "citation": "SITE_A / chatter.pdf / p.330 / machine=MACHINE_A1",
            "page": 330,
            "score": 0.9995,
            "query_used": "regenerative chatter harmonic vibration tooth passing",
            "pattern_key": "fault:chatter",
        }
    ]
    memory = SimpleNamespace(
        id="mem-doc-persist-muted",
        session_id="session-doc-persist-muted",
        pattern_keys=event.patterns,
        time_range=event.time_range,
        metadata={"source": "SITE_A"},
        channels=["Vibration_Severity_X"],
        machine_uri="MACHINE_A1",
    )

    orchestrator.scorer.score = MagicMock(return_value=significance)
    orchestrator.retriever.retrieve = MagicMock(return_value=[])
    orchestrator._create_memory = MagicMock(return_value=memory)
    orchestrator._store_memory = AsyncMock(return_value="mem-doc-persist-muted")
    orchestrator._add_trace = MagicMock()
    orchestrator._build_metrics_for_alert = MagicMock(return_value={})
    orchestrator.store.get_similar_with_resolution = MagicMock(return_value=[])
    orchestrator.store.persist_doc_links = MagicMock(return_value=1)
    orchestrator.alert_dispatcher.propose_doc_links_for_memory = AsyncMock(return_value=doc_links)
    orchestrator.alert_dispatcher.dispatch = AsyncMock(return_value=False)
    orchestrator.alert_dispatcher.broadcast_scored_event = AsyncMock()

    result = await orchestrator.process_event(event)

    assert result.processed is True
    orchestrator.store.persist_doc_links.assert_called_once_with(
        memory_id="mem-doc-persist-muted",
        pattern_keys=["fault:chatter"],
        doc_links=doc_links,
    )


@pytest.mark.asyncio
async def test_dispatcher_marks_repeated_signature_as_recurring_within_window():
    dispatcher = AlertDispatcher(
        rate_config=RateLimitConfig(
            min_interval_seconds=0.0,
            max_alerts_per_minute=100,
            cooldown_on_dismiss=0.0,
            signature_recurrence_seconds=10.0,
            signature_suppress_seconds=0.0,
        )
    )
    queue = dispatcher.subscribe(session_id="session-recurring")
    significance = SimpleNamespace(
        score=0.84,
        action=SignificanceAction.ALERT,
        reasons=["Vibration modulation severity: 0.84"],
        prior_boost=0.0,
        pattern_priors={},
    )

    memory_a = SimpleNamespace(
        id="mem-a",
        session_id="session-recurring",
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="signature:modulated_tooth_passing_vibration")],
        time_range=TimeRange(i0=0, i1=2, t0=0.0, t1=2.0, fs=1.0),
    )
    memory_b = SimpleNamespace(
        id="mem-b",
        session_id="session-recurring",
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="signature:modulated_tooth_passing_vibration")],
        time_range=TimeRange(i0=3, i1=5, t0=3.0, t1=5.0, fs=1.0),
    )

    try:
        await dispatcher.dispatch(memory=memory_a, significance=significance, similar_memories=[])
        await dispatcher.dispatch(memory=memory_b, significance=significance, similar_memories=[])
        first = await asyncio.wait_for(queue.get(), timeout=1.0)
        second = await asyncio.wait_for(queue.get(), timeout=1.0)
    finally:
        dispatcher.unsubscribe(queue, session_id="session-recurring")

    assert first.to_dict()["persistence_label"] == "candidate"
    assert second.to_dict()["persistence_label"] == "recurring"


@pytest.mark.asyncio
async def test_dispatcher_suppresses_repeated_patternless_model_alerts():
    dispatcher = AlertDispatcher(
        rate_config=RateLimitConfig(
            min_interval_seconds=0.0,
            max_alerts_per_minute=100,
            cooldown_on_dismiss=0.0,
            signature_suppress_seconds=30.0,
            signature_score_change_threshold=0.10,
        )
    )
    queue = dispatcher.subscribe(session_id="session-patternless")
    significance = SimpleNamespace(
        score=0.60,
        action=SignificanceAction.ALERT,
        reasons=[
            "Anomaly detector score: 0.83",
            "Model confidence: 0.74",
            "Harmonic context score: 0.55",
        ],
        prior_boost=0.0,
        pattern_priors={},
    )
    memory_a = SimpleNamespace(
        id="mem-patternless-a",
        session_id="session-patternless",
        pattern_keys=[],
        time_range=TimeRange(i0=0, i1=2, t0=0.0, t1=2.0, fs=1.0),
    )
    memory_b = SimpleNamespace(
        id="mem-patternless-b",
        session_id="session-patternless",
        pattern_keys=[],
        time_range=TimeRange(i0=2, i1=4, t0=2.0, t1=4.0, fs=1.0),
    )

    try:
        first_result = await dispatcher.dispatch(memory=memory_a, significance=significance, similar_memories=[])
        second_result = await dispatcher.dispatch(memory=memory_b, significance=significance, similar_memories=[])
        first = await asyncio.wait_for(queue.get(), timeout=1.0)
    finally:
        dispatcher.unsubscribe(queue, session_id="session-patternless")

    assert first_result is True
    assert second_result is False
    assert first.to_dict()["event_id"] == "mem-patternless-a"


@pytest.mark.asyncio
async def test_dispatcher_invokes_backend_pause_handler_after_alert_dispatch():
    dispatcher = AlertDispatcher()
    queue = dispatcher.subscribe(session_id="session-pause-on-alert")
    paused: list[tuple[str, dict]] = []
    dispatcher.set_session_pause_handler(lambda session_id, alert: paused.append((session_id, dict(alert))))

    memory = SimpleNamespace(
        id="mem-pause",
        session_id="session-pause-on-alert",
        pattern_keys=[PatternKey(pattern_type=PatternType.CUSTOM, key="signature:modulated_tooth_passing_vibration")],
        time_range=TimeRange(i0=0, i1=5, t0=0.0, t1=5.0, fs=1.0),
    )
    significance = SimpleNamespace(
        score=0.84,
        action=SignificanceAction.ALERT,
        reasons=["Vibration modulation severity: 0.84"],
        prior_boost=0.0,
        pattern_priors={},
    )

    try:
        await dispatcher.dispatch(memory=memory, significance=significance, similar_memories=[])
        await asyncio.wait_for(queue.get(), timeout=1.0)
    finally:
        dispatcher.unsubscribe(queue, session_id="session-pause-on-alert")

    assert paused == [
        (
            "session-pause-on-alert",
            {
                "event_id": "mem-pause",
                "severity": "WARNING",
                "score": 0.84,
                "action": "alert",
            },
        )
    ]


@pytest.mark.asyncio
async def test_dispatcher_builds_indicator_summary_from_signature_metadata():
    dispatcher = AlertDispatcher()
    queue = dispatcher.subscribe(session_id="session-indicators")
    memory = SimpleNamespace(
        id="mem-indicators",
        session_id="session-indicators",
        pattern_keys=[
            PatternKey(
                pattern_type=PatternType.CUSTOM,
                key="spectral:modulated_vibration",
                confidence=0.62,
                source_metric="channel_crest_factors",
                additional={
                    "confidence": 0.62,
                    "reason": "crest factor indicates modulation onset",
                    "source_metric": "channel_crest_factors",
                },
            ),
            PatternKey(
                pattern_type=PatternType.CUSTOM,
                key="amp:loud",
                confidence=0.60,
                source_metric="channel_rms",
                additional={
                    "confidence": 0.60,
                    "reason": "RMS amplitude exceeds nominal band",
                    "source_metric": "channel_rms",
                },
            ),
            PatternKey(
                pattern_type=PatternType.CUSTOM,
                key="signature:modulated_tooth_passing_vibration",
                confidence=0.50,
                additional={
                    "indicators_present": 2,
                    "indicators_required": 4,
                    "supporting_patterns": ["spectral:modulated_vibration", "amp:loud"],
                    "emitted": True,
                },
            ),
        ],
        time_range=TimeRange(i0=0, i1=5, t0=0.0, t1=5.0, fs=1.0),
    )
    significance = SimpleNamespace(
        score=0.84,
        action=SignificanceAction.ALERT,
        reasons=["Vibration modulation severity: 0.84"],
        prior_boost=0.0,
        pattern_priors={},
    )

    try:
        await dispatcher.dispatch(memory=memory, significance=significance, similar_memories=[])
        alert = await asyncio.wait_for(queue.get(), timeout=1.0)
    finally:
        dispatcher.unsubscribe(queue, session_id="session-indicators")

    payload = alert.to_dict()
    assert payload["primary_observation_label"] == "Modulated tooth-passing vibration"
    assert payload["indicators_present"] == 2
    assert payload["indicators_required"] == 4
    assert payload["indicator_details"][0]["label"] == "Modulated vibration"
    assert payload["summary"].startswith("Possible Modulated tooth-passing vibration — 2/4 indicators present")


@pytest.mark.asyncio
async def test_dispatcher_broadcast_scored_events_mark_signature_recurrence():
    dispatcher = AlertDispatcher(
        rate_config=RateLimitConfig(
            min_interval_seconds=0.0,
            max_alerts_per_minute=100,
            cooldown_on_dismiss=0.0,
            signature_recurrence_seconds=10.0,
        )
    )
    queue = dispatcher.subscribe(session_id="session-score-recurring")
    significance = SimpleNamespace(
        score=0.41,
        action=SignificanceAction.STORE,
        reasons=["Routine observation"],
        prior_boost=0.0,
        pattern_priors={},
    )

    event_a = MemoryEvent(
        session_id="session-score-recurring",
        time_range=TimeRange(i0=0, i1=5, t0=0.0, t1=5.0, fs=1.0),
        patterns=[PatternKey(pattern_type=PatternType.CUSTOM, key="signature:modulated_tooth_passing_vibration")],
    )
    event_b = MemoryEvent(
        session_id="session-score-recurring",
        time_range=TimeRange(i0=5, i1=10, t0=5.0, t1=10.0, fs=1.0),
        patterns=[PatternKey(pattern_type=PatternType.CUSTOM, key="signature:modulated_tooth_passing_vibration")],
    )

    try:
        await dispatcher.broadcast_scored_event(significance=significance, event=event_a)
        await dispatcher.broadcast_scored_event(significance=significance, event=event_b)
        first = await asyncio.wait_for(queue.get(), timeout=1.0)
        second = await asyncio.wait_for(queue.get(), timeout=1.0)
    finally:
        dispatcher.unsubscribe(queue, session_id="session-score-recurring")

    assert first.to_dict()["persistence_label"] == "candidate"
    assert second.to_dict()["persistence_label"] == "recurring"


@pytest.mark.asyncio
async def test_orchestrator_broadcasts_ignore_action_scores():
    orchestrator = MemoryEventOrchestrator(
        config=OrchestratorConfig(
            use_classical_models=False,
            enable_harmonic_scorer=False,
            dispatch_alerts=True,
        )
    )
    event = MemoryEvent(
        session_id="session-2",
        time_range=TimeRange(i0=0, i1=2, t0=0.0, t1=2.0, fs=1.0),
        external_signals={"anomaly_detector_score": 0.2},
        metadata={},
    )
    significance = SimpleNamespace(
        score=0.1,
        action=SignificanceAction.IGNORE,
        is_significant=False,
        triggered_rules=[],
        reasons=["ignore"],
        prior_boost=0.0,
        pattern_priors={},
        to_dict=lambda: {"score": 0.1, "action": "ignore"},
    )

    orchestrator.scorer.score = MagicMock(return_value=significance)
    orchestrator.alert_dispatcher.broadcast_scored_event = AsyncMock()
    orchestrator._add_trace = MagicMock()

    result = await orchestrator.process_event(event)

    assert result.processed is True
    assert result.significant is False
    orchestrator.alert_dispatcher.broadcast_scored_event.assert_awaited_once()
    kwargs = orchestrator.alert_dispatcher.broadcast_scored_event.await_args.kwargs
    assert kwargs["event"] is event
    assert kwargs["memory"] is None


@pytest.mark.asyncio
async def test_orchestrator_publishes_scored_learning_for_stored_event(monkeypatch):
    published: list[dict] = []

    async def fake_publish_scored_learning(**kwargs):
        published.append(kwargs)

    monkeypatch.setattr(
        "backend.agents.memory.orchestrator.publish_scored_learning",
        fake_publish_scored_learning,
    )

    orchestrator = MemoryEventOrchestrator(
        config=OrchestratorConfig(
            use_classical_models=False,
            enable_harmonic_scorer=False,
            dispatch_alerts=False,
        )
    )
    event = MemoryEvent(
        session_id="session-4",
        time_range=TimeRange(i0=0, i1=4, t0=0.0, t1=4.0, fs=1.0),
        patterns=[PatternKey(pattern_type=PatternType.ANOMALY, key="CHATTER")],
        external_signals={"anomaly_detector_score": 0.77},
        metadata={},
    )
    significance = SimpleNamespace(
        score=0.77,
        action=SignificanceAction.STORE,
        is_significant=True,
        triggered_rules=["ANOMALY_HIGH"],
        reasons=["store"],
        prior_boost=0.0,
        pattern_priors={"CHATTER": 0.6},
        score_trace=[],
        to_dict=lambda: {"score": 0.77, "action": "store", "is_significant": True},
    )

    orchestrator.scorer.score = MagicMock(return_value=significance)
    orchestrator._create_memory = MagicMock(return_value=SimpleNamespace(id="mem-4", session_id="session-4"))
    orchestrator._store_memory = AsyncMock(return_value="mem-4")
    orchestrator._add_trace = MagicMock()

    result = await orchestrator.process_event(event)

    assert result.processed is True
    assert result.memory_id == "mem-4"
    assert len(published) == 1
    assert published[0]["session_id"] == "session-4"
    assert published[0]["memory_id"] == "mem-4"
    assert published[0]["patterns"] == event.patterns
    assert published[0]["alert_dispatched"] is False


@pytest.mark.asyncio
async def test_background_explanation_publishes_insight_learning(monkeypatch):
    published: list[dict] = []

    async def fake_publish_insight_learning(**kwargs):
        published.append(kwargs)

    monkeypatch.setattr(
        "backend.agents.memory.orchestrator.publish_insight_learning",
        fake_publish_insight_learning,
    )

    orchestrator = MemoryEventOrchestrator(
        config=OrchestratorConfig(
            use_classical_models=False,
            enable_harmonic_scorer=False,
            dispatch_alerts=False,
        )
    )
    orchestrator._persist_explanation_on_memory = MagicMock()
    orchestrator.alert_dispatcher.broadcast_explanation_update = AsyncMock()
    orchestrator.explainer.explain_grounded_async = AsyncMock(return_value=("Detailed explanation", "llm"))
    orchestrator.explainer.explain_significance_for_alert_async = AsyncMock(
        return_value=("Short alert", "llm", "Ease off feed and inspect the tool")
    )

    event = MemoryEvent(
        session_id="session-5",
        time_range=TimeRange(i0=0, i1=2, t0=0.0, t1=2.0, fs=1.0),
        metadata={},
    )

    await orchestrator._generate_explanation_background(
        memory=SimpleNamespace(session_id="session-5"),
        memory_id="mem-5",
        event=event,
        significance=SimpleNamespace(score=0.9, action=SignificanceAction.ALERT),
        similar_memories=[],
        expl_ctx=object(),
    )

    assert len(published) == 1
    assert published[0]["session_id"] == "session-5"
    assert published[0]["memory_id"] == "mem-5"
    assert published[0]["explanation"] == "Detailed explanation"
    assert published[0]["alert_line"] == "Short alert"
    # Two-tier recommendation (T0.2): the immediate action is threaded into the
    # explanation_update broadcast so the UI can show it separately.
    assert orchestrator.alert_dispatcher.broadcast_explanation_update.call_args.kwargs["recommendation"] == "Ease off feed and inspect the tool"


@pytest.mark.asyncio
async def test_orchestrator_respects_event_level_explanation_override():
    orchestrator = MemoryEventOrchestrator(
        config=OrchestratorConfig(
            use_classical_models=False,
            enable_harmonic_scorer=False,
            dispatch_alerts=True,
            generate_explanations=True,
        )
    )
    event = MemoryEvent(
        session_id="session-6",
        time_range=TimeRange(i0=0, i1=2, t0=0.0, t1=2.0, fs=1.0),
        patterns=[PatternKey(pattern_type=PatternType.ANOMALY, key="CHATTER")],
        metadata={"generate_explanations_override": False},
    )
    significance = SimpleNamespace(
        score=0.91,
        action=SignificanceAction.ALERT,
        is_significant=True,
        triggered_rules=["ALERT"],
        reasons=["alert"],
        prior_boost=0.0,
        pattern_priors={"CHATTER": 0.6},
        score_trace=[],
        to_dict=lambda: {"score": 0.91, "action": "alert", "is_significant": True},
    )

    orchestrator.scorer.score = MagicMock(return_value=significance)
    orchestrator._create_memory = MagicMock(return_value=SimpleNamespace(id="mem-6", session_id="session-6"))
    orchestrator._store_memory = AsyncMock(return_value="mem-6")
    orchestrator._add_trace = MagicMock()
    orchestrator._build_explanation_context = MagicMock()
    orchestrator.alert_dispatcher.dispatch = AsyncMock(return_value=True)
    orchestrator.alert_dispatcher.broadcast_scored_event = AsyncMock()

    result = await orchestrator.process_event(event)

    assert result.processed is True
    assert result.alert_dispatched is True
    orchestrator._build_explanation_context.assert_not_called()
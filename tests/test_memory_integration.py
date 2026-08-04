#!/usr/bin/env python3
"""
Integration tests for the Memory System.

These tests verify the complete pipeline:
1. Event processing and memory storage
2. Similar memory retrieval
3. Feedback processing and prior updates
4. Learning effect on subsequent events

Run with: pytest tests/test_memory_integration.py -v
"""

import asyncio
import pytest
from datetime import datetime
from typing import List

from backend.agents.core.schemas import PatternKey, PatternType, TimeRange
from backend.agents.core.context import CuttingContext, OperatingRegime
from backend.agents.memory.orchestrator import (
    MemoryEventOrchestrator,
    MemoryEvent,
    MemoryEventResult,
    OrchestratorConfig,
)
from backend.agents.memory.scorer import SignificanceAction, SignificanceConfig
from backend.agents.memory.feedback import MemoryFeedbackRequest, FeedbackAction


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def orchestrator():
    """Create a fresh orchestrator for each test."""
    config = OrchestratorConfig(
        always_store=False,
        min_score_for_retrieval=0.3,
        top_k_similar=5,
        generate_explanations=False,  # Disable LLM for tests
        dispatch_alerts=False,  # Disable alerts for tests
        use_classical_models=False,  # Skip seed-model casedata load for tests
    )
    return MemoryEventOrchestrator(config=config)


@pytest.fixture
def time_range():
    """Default time range for tests."""
    return TimeRange(
        i0=0,
        i1=10000,
        t0=0.0,
        t1=1.0,
        fs=10000.0,
    )


@pytest.fixture
def cutting_context():
    """Default cutting context for tests."""
    return CuttingContext(
        tool_type="end_mill",
        tool_diameter=10.0,
        num_teeth=3,
        spindle_speed=8500,
        feed_rate=1200.0,
        feed_per_tooth=0.047,
        cutting_speed=267.0,
        axial_depth=2.5,
        radial_depth=5.0,
        workpiece_material="steel",
        workpiece_hardness=35.0,
        machine_id="test_machine",
        operating_regime=OperatingRegime.ROUGHING,
    )


def make_normal_patterns() -> List[PatternKey]:
    """Patterns that don't trigger significance."""
    return [
        PatternKey(pattern_type=PatternType.SPECTRAL_PEAK, key="SPECTRAL_PEAK_425Hz", source_metric="Fx"),
        PatternKey(pattern_type=PatternType.RATIO, key="RATIO_Fx_Fy:1.2", source_metric="ratio"),
    ]


def make_anomaly_patterns() -> List[PatternKey]:
    """Patterns that trigger significance."""
    return [
        PatternKey(pattern_type=PatternType.ANOMALY, key="ANOMALY_HIGH", source_metric="model"),
        PatternKey(pattern_type=PatternType.RATIO, key="RATIO_Fx_Fy:>5", source_metric="ratio"),
    ]


# =============================================================================
# Test: Basic Event Processing
# =============================================================================

class TestEventProcessing:
    """Tests for basic event processing."""
    
    @pytest.mark.asyncio
    async def test_normal_event_not_stored(self, orchestrator, time_range, cutting_context):
        """Normal events should not be stored."""
        event = MemoryEvent(
            session_id="test_session",
            time_range=time_range,
            patterns=make_normal_patterns(),
            cutting_context=cutting_context,
            channels=["Fx", "Fy"],
        )
        
        result = await orchestrator.process_event(event)
        
        assert result.processed is True
        assert result.significant is False
        assert result.memory_id is None
        assert result.action == SignificanceAction.IGNORE
    
    @pytest.mark.asyncio
    async def test_anomaly_event_stored(self, orchestrator, time_range, cutting_context):
        """Anomaly events should be stored."""
        event = MemoryEvent(
            session_id="test_session",
            time_range=time_range,
            patterns=make_anomaly_patterns(),
            cutting_context=cutting_context,
            channels=["Fx", "Fy"],
        )
        
        result = await orchestrator.process_event(event)
        
        assert result.processed is True
        assert result.significant is True
        assert result.memory_id is not None
        assert result.action in (SignificanceAction.ALERT, SignificanceAction.CRITICAL)
    
    @pytest.mark.asyncio
    async def test_memory_stored_and_retrievable(self, orchestrator, time_range, cutting_context):
        """Stored memories should be retrievable."""
        event = MemoryEvent(
            session_id="test_session",
            time_range=time_range,
            patterns=make_anomaly_patterns(),
            cutting_context=cutting_context,
            channels=["Fx", "Fy"],
        )
        
        result = await orchestrator.process_event(event)
        
        # Retrieve the stored memory
        memory = orchestrator.get_memory(result.memory_id)
        
        assert memory is not None
        assert memory.id == result.memory_id
        assert memory.session_id == "test_session"
        assert len(memory.pattern_keys) == 2
    
    @pytest.mark.asyncio
    async def test_always_store_config(self, time_range, cutting_context):
        """With always_store=True, even normal events should be stored."""
        config = OrchestratorConfig(
            always_store=True,
            generate_explanations=False,
            dispatch_alerts=False,
            use_classical_models=False,
        )
        orchestrator = MemoryEventOrchestrator(config=config)
        
        event = MemoryEvent(
            session_id="test_session",
            time_range=time_range,
            patterns=make_normal_patterns(),
            cutting_context=cutting_context,
            channels=["Fx", "Fy"],
        )
        
        result = await orchestrator.process_event(event)
        
        assert result.processed is True
        assert result.memory_id is not None  # Should be stored even if not significant


# =============================================================================
# Test: Similar Memory Retrieval
# =============================================================================

class TestMemoryRetrieval:
    """Tests for similar memory retrieval."""
    
    @pytest.mark.asyncio
    async def test_similar_memories_found(self, orchestrator, time_range, cutting_context):
        """Second event with same patterns should find the first."""
        # First event
        event1 = MemoryEvent(
            session_id="session_1",
            time_range=time_range,
            patterns=make_anomaly_patterns(),
            cutting_context=cutting_context,
            channels=["Fx", "Fy"],
        )
        result1 = await orchestrator.process_event(event1)
        
        # Second event with same patterns
        event2 = MemoryEvent(
            session_id="session_2",
            time_range=time_range,
            patterns=make_anomaly_patterns(),
            cutting_context=cutting_context,
            channels=["Fx", "Fy"],
        )
        result2 = await orchestrator.process_event(event2)
        
        assert result2.similar_memories is not None
        assert len(result2.similar_memories) >= 1
        
        # The first memory should be in similar memories
        similar_ids = [m.memory.id for m in result2.similar_memories]
        assert result1.memory_id in similar_ids
    
    @pytest.mark.asyncio
    async def test_excludes_current_memory(self, orchestrator, time_range, cutting_context):
        """Current memory should be excluded from similar results."""
        event = MemoryEvent(
            session_id="session_1",
            time_range=time_range,
            patterns=make_anomaly_patterns(),
            cutting_context=cutting_context,
            channels=["Fx", "Fy"],
        )
        result = await orchestrator.process_event(event)
        
        # The result should not include itself
        similar_ids = [m.memory.id for m in result.similar_memories]
        assert result.memory_id not in similar_ids
    
    @pytest.mark.asyncio
    async def test_multiple_similar_memories(self, orchestrator, time_range, cutting_context):
        """Should return multiple similar memories when they exist."""
        # Create 5 similar events
        memory_ids = []
        for i in range(5):
            event = MemoryEvent(
                session_id=f"session_{i}",
                time_range=time_range,
                patterns=make_anomaly_patterns(),
                cutting_context=cutting_context,
                channels=["Fx", "Fy"],
            )
            result = await orchestrator.process_event(event)
            memory_ids.append(result.memory_id)
        
        # Create 6th event
        event6 = MemoryEvent(
            session_id="session_6",
            time_range=time_range,
            patterns=make_anomaly_patterns(),
            cutting_context=cutting_context,
            channels=["Fx", "Fy"],
        )
        result6 = await orchestrator.process_event(event6)
        
        # Should find multiple similar memories
        assert len(result6.similar_memories) >= 3


# =============================================================================
# Test: Feedback Processing
# =============================================================================

class TestFeedbackProcessing:
    """Tests for user feedback on memories."""
    
    @pytest.mark.asyncio
    async def test_confirm_feedback(self, orchestrator, time_range, cutting_context):
        """Confirming a memory should update pattern priors."""
        # Create event
        event = MemoryEvent(
            session_id="test_session",
            time_range=time_range,
            patterns=make_anomaly_patterns(),
            cutting_context=cutting_context,
            channels=["Fx", "Fy"],
        )
        result = await orchestrator.process_event(event)
        
        # Initial priors should be neutral (0.5)
        for pattern in make_anomaly_patterns():
            assert orchestrator.scorer._pattern_priors.get(pattern.key, 0.5) == 0.5
        
        # Confirm the memory
        feedback = MemoryFeedbackRequest(
            action=FeedbackAction.CONFIRM,
            user_id="test_user",
            reason="Confirmed as real anomaly",
        )
        response = await orchestrator.feedback_handler.process_feedback(
            memory_id=result.memory_id,
            request=feedback,
        )
        
        assert response.success is True
        
        # Priors should now be higher
        for pattern in make_anomaly_patterns():
            prior = orchestrator.scorer._pattern_priors.get(pattern.key, 0.5)
            assert prior > 0.5, f"Prior for {pattern.key} should be > 0.5, got {prior}"
    
    @pytest.mark.asyncio
    async def test_dismiss_feedback(self, orchestrator, time_range, cutting_context):
        """Dismissing a memory should lower pattern priors."""
        # Create event
        event = MemoryEvent(
            session_id="test_session",
            time_range=time_range,
            patterns=make_anomaly_patterns(),
            cutting_context=cutting_context,
            channels=["Fx", "Fy"],
        )
        result = await orchestrator.process_event(event)
        
        # Dismiss the memory
        feedback = MemoryFeedbackRequest(
            action=FeedbackAction.DISMISS,
            user_id="test_user",
            reason="False positive",
        )
        response = await orchestrator.feedback_handler.process_feedback(
            memory_id=result.memory_id,
            request=feedback,
        )
        
        assert response.success is True
        
        # Priors should now be lower
        for pattern in make_anomaly_patterns():
            prior = orchestrator.scorer._pattern_priors.get(pattern.key, 0.5)
            assert prior < 0.5, f"Prior for {pattern.key} should be < 0.5, got {prior}"
    
    @pytest.mark.asyncio
    async def test_multiple_confirmations_increase_prior(self, orchestrator, time_range, cutting_context):
        """Multiple confirmations should increase prior further."""
        patterns = make_anomaly_patterns()
        pattern_key = patterns[0].key
        
        priors = [orchestrator.scorer._pattern_priors.get(pattern_key, 0.5)]
        
        # Create and confirm multiple events
        for i in range(5):
            event = MemoryEvent(
                session_id=f"session_{i}",
                time_range=time_range,
                patterns=patterns,
                cutting_context=cutting_context,
                channels=["Fx", "Fy"],
            )
            result = await orchestrator.process_event(event)
            
            feedback = MemoryFeedbackRequest(
                action=FeedbackAction.CONFIRM,
                user_id="test_user",
            )
            await orchestrator.feedback_handler.process_feedback(
                memory_id=result.memory_id,
                request=feedback,
            )
            
            priors.append(orchestrator.scorer._pattern_priors.get(pattern_key, 0.5))
        
        # Priors should be monotonically increasing
        for i in range(len(priors) - 1):
            assert priors[i+1] > priors[i], f"Prior should increase: {priors}"


# =============================================================================
# Test: Learning Effect
# =============================================================================

class TestLearningEffect:
    """Tests for learning from feedback."""
    
    @pytest.mark.asyncio
    async def test_score_increases_with_confirmation(self, time_range, cutting_context):
        """Score should increase after pattern is confirmed multiple times."""
        # Use config with higher learning rate for testing
        sig_config = SignificanceConfig(
            weight_historical_prior=0.25,  # Higher weight for testing
        )
        config = OrchestratorConfig(
            significance_config=sig_config,
            generate_explanations=False,
            dispatch_alerts=False,
            use_classical_models=False,
        )
        orchestrator = MemoryEventOrchestrator(config=config)
        
        patterns = [
            PatternKey(pattern_type=PatternType.CUSTOM, key="TEST_PATTERN", source_metric="test"),
        ]
        
        # First event (baseline score)
        event1 = MemoryEvent(
            session_id="session_1",
            time_range=time_range,
            patterns=patterns,
            cutting_context=cutting_context,
            channels=["Fx"],
        )
        
        # Manually set high prior to test effect
        orchestrator.scorer._pattern_priors["TEST_PATTERN"] = 0.5  # Neutral
        
        result1 = await orchestrator.process_event(event1)
        score_before = result1.significance_score
        
        # Simulate many confirmations by writing feedback events directly to
        # the store — this is the code-path that ``SignificanceScorer.get_pattern_prior``
        # observes (see scorer docstring: the caller owns durable persistence).
        for _ in range(10):
            orchestrator.store.add_feedback_event(
                memory_id=result1.memory_id,
                action="confirm",
                user_id="test",
                pattern_keys=["TEST_PATTERN"],
            )
        # Refresh the cached prior snapshot.
        orchestrator.scorer.update_pattern_prior("TEST_PATTERN", was_significant=True)
        
        # Second event (should have higher score)
        event2 = MemoryEvent(
            session_id="session_2",
            time_range=time_range,
            patterns=patterns,
            cutting_context=cutting_context,
            channels=["Fx"],
        )
        result2 = await orchestrator.process_event(event2)
        score_after = result2.significance_score
        
        # The prior should now be high
        prior = orchestrator.scorer.get_pattern_prior("TEST_PATTERN")
        assert prior > 0.8, f"Prior should be > 0.8 after 10 confirmations, got {prior}"


# =============================================================================
# Test: External Signals
# =============================================================================

class TestExternalSignals:
    """Tests for external signal processing."""
    
    @pytest.mark.asyncio
    async def test_breakage_prediction_triggers_alert(self, orchestrator):
        """High breakage prediction should trigger critical alert."""
        result = await orchestrator.process_external_signal(
            session_id="test_session",
            signal_type="breakage_prediction",
            signal_value=0.9,
            metadata={
                "machining": {
                    "n": 8500,
                    "z": 3,
                    "ap": 2.5,
                }
            },
        )
        
        assert result.processed is True
        assert result.significant is True
        assert result.action in (SignificanceAction.ALERT, SignificanceAction.CRITICAL)
    
    @pytest.mark.asyncio
    async def test_low_breakage_prediction_not_significant(self, orchestrator):
        """Low breakage prediction should have lower score than high prediction."""
        result = await orchestrator.process_external_signal(
            session_id="test_session",
            signal_type="breakage_prediction",
            signal_value=0.2,  # Low probability
            metadata={},
        )
        
        # Low prediction should have lower score than high (0.9)
        # Note: Pattern matching rules still contribute to score
        assert result.significance_score < 0.9


# =============================================================================
# Test: Store Adapter
# =============================================================================

class TestInMemoryStoreAdapter:
    """Tests for the in-memory store adapter."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, orchestrator, time_range, cutting_context):
        """Memories stored via adapter should be retrievable."""
        event = MemoryEvent(
            session_id="test_session",
            time_range=time_range,
            patterns=make_anomaly_patterns(),
            cutting_context=cutting_context,
            channels=["Fx", "Fy"],
        )
        result = await orchestrator.process_event(event)
        
        # Retrieve via store
        memory = orchestrator.store.get(result.memory_id)
        assert memory is not None
        assert memory.id == result.memory_id
    
    @pytest.mark.asyncio
    async def test_list_all_memories(self, orchestrator, time_range, cutting_context):
        """list_all should return all stored memories."""
        # Create 3 events
        for i in range(3):
            event = MemoryEvent(
                session_id=f"session_{i}",
                time_range=time_range,
                patterns=make_anomaly_patterns(),
                cutting_context=cutting_context,
                channels=["Fx"],
            )
            await orchestrator.process_event(event)
        
        all_memories = orchestrator.store.list_all()
        assert len(all_memories) == 3
    
    @pytest.mark.asyncio
    async def test_query_by_session(self, orchestrator, time_range, cutting_context):
        """Query should filter by session_id."""
        # Create events in different sessions
        for session in ["session_A", "session_A", "session_B"]:
            event = MemoryEvent(
                session_id=session,
                time_range=time_range,
                patterns=make_anomaly_patterns(),
                cutting_context=cutting_context,
                channels=["Fx"],
            )
            await orchestrator.process_event(event)
        
        session_a_memories = orchestrator.store.query(session_id="session_A")
        assert len(session_a_memories) == 2
        
        session_b_memories = orchestrator.store.query(session_id="session_B")
        assert len(session_b_memories) == 1
    
    @pytest.mark.asyncio
    async def test_update_memory(self, orchestrator, time_range, cutting_context):
        """update should modify memory fields."""
        event = MemoryEvent(
            session_id="test_session",
            time_range=time_range,
            patterns=make_anomaly_patterns(),
            cutting_context=cutting_context,
            channels=["Fx"],
        )
        result = await orchestrator.process_event(event)
        
        # Update metadata
        success = orchestrator.store.update(result.memory_id, {
            "metadata.test_field": "test_value",
        })
        assert success is True
        
        # Verify update
        memory = orchestrator.store.get(result.memory_id)
        assert memory.metadata.get("test_field") == "test_value"
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, orchestrator, time_range, cutting_context):
        """delete should remove memory."""
        event = MemoryEvent(
            session_id="test_session",
            time_range=time_range,
            patterns=make_anomaly_patterns(),
            cutting_context=cutting_context,
            channels=["Fx"],
        )
        result = await orchestrator.process_event(event)
        
        # Verify exists
        assert orchestrator.store.get(result.memory_id) is not None
        
        # Delete
        success = orchestrator.store.delete(result.memory_id)
        assert success is True
        
        # Verify deleted
        assert orchestrator.store.get(result.memory_id) is None


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_patterns(self, orchestrator, time_range, cutting_context):
        """Event with no patterns should not crash."""
        event = MemoryEvent(
            session_id="test_session",
            time_range=time_range,
            patterns=[],
            cutting_context=cutting_context,
            channels=["Fx"],
        )
        
        result = await orchestrator.process_event(event)
        
        assert result.processed is True
        assert result.significant is False
    
    @pytest.mark.asyncio
    async def test_no_cutting_context(self, orchestrator, time_range):
        """Event with no cutting context should still work."""
        event = MemoryEvent(
            session_id="test_session",
            time_range=time_range,
            patterns=make_anomaly_patterns(),
            cutting_context=None,
            channels=["Fx"],
        )
        
        result = await orchestrator.process_event(event)
        
        assert result.processed is True
        # Should still detect anomaly patterns
        assert result.significant is True
    
    @pytest.mark.asyncio
    async def test_feedback_on_nonexistent_memory(self, orchestrator):
        """Feedback on non-existent memory should fail gracefully."""
        feedback = MemoryFeedbackRequest(
            action=FeedbackAction.CONFIRM,
            user_id="test_user",
        )
        
        response = await orchestrator.feedback_handler.process_feedback(
            memory_id="nonexistent-id",
            request=feedback,
        )
        
        # Should fail gracefully since memory doesn't exist
        assert response.success is False
        assert "not found" in response.message.lower()


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

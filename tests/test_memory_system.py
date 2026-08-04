"""
Test script for the LLM Memory System prototype.

# ===========================================================================
# DRAFT/PROTOTYPE - Tag: [PROTOTYPE_LLM_MEMORY_V1]
# This script demonstrates the memory system functionality.
# Run with: python -m pytest tests/test_memory_system.py -v
# ===========================================================================
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from backend.agents.core.schemas import PatternKey, PatternType, TimeRange, NumericMetrics
from backend.agents.core.context import CuttingContext, OperatingRegime, extract_context_from_metadata
from backend.agents.memory.scorer import (
    SignificanceScorer,
    SignificanceConfig,
    SignificanceAction,
    _parse_ratio_bucket_lower_bound,
)
from backend.agents.memory.orchestrator import (
    MemoryEvent,
    MemoryEventOrchestrator,
    OrchestratorConfig,
)
from backend.agents.processing.dataset_loader import WindowData
from backend.agents.memory.feedback import (
    MemoryFeedbackHandler,
    MemoryFeedbackRequest,
    FeedbackAction,
)
from backend.agents.memory.retriever import MemoryRetriever, PatternMatcher
from backend.agents.core.metrics import WindowMetrics


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_patterns():
    """Sample pattern keys for testing."""
    return [
        PatternKey(pattern_type=PatternType.RATIO, key="RATIO_Fx_Fy:>5"),
        PatternKey(pattern_type=PatternType.SPECTRAL_PEAK, key="SPECTRAL_PEAK_512Hz:high"),
    ]


@pytest.fixture
def sample_context():
    """Sample cutting context for testing."""
    return CuttingContext(
        tool_type="end_mill",
        tool_diameter=10.0,
        num_teeth=4,
        spindle_speed=8000.0,
        cutting_speed=250.0,
        axial_depth=2.5,
        workpiece_material="steel",
    )


@pytest.fixture
def sample_time_range():
    """Sample time range for testing."""
    return TimeRange(
        i0=1000,
        i1=2000,
        t0=1.0,
        t1=2.0,
        fs=1000.0,
    )


@pytest.fixture
def sample_metrics():
    """Sample window metrics for testing."""
    metrics = WindowMetrics()
    metrics.channel_means = [0.1, 0.2, 0.15]
    metrics.channel_stds = [0.05, 0.08, 0.06]
    metrics.channel_rms = [0.12, 0.25, 0.18]
    metrics.channel_peaks = [0.5, 0.8, 0.6]
    metrics.dominant_frequencies = [512.0, 256.0, 384.0]
    metrics.spectral_centroids = [400.0, 350.0, 420.0]
    return metrics


# ============================================================================
# SignificanceScorer Tests
# ============================================================================

class TestSignificanceScorer:
    """Tests for the significance scorer."""
    
    def test_score_with_anomaly_pattern(self, sample_patterns):
        """Test that ANOMALY patterns trigger significance."""
        scorer = SignificanceScorer()
        
        anomaly_pattern = PatternKey(
            pattern_type=PatternType.ANOMALY,
            key="ANOMALY_HIGH:0.9"
        )
        
        result = scorer.score(patterns=[anomaly_pattern])
        
        assert result.is_significant
        assert result.score > 0.5
        assert "pattern_match" in result.triggered_rules
    
    def test_score_with_high_force_ratio(self, sample_patterns):
        """Test that high force ratio triggers significance."""
        scorer = SignificanceScorer()
        
        result = scorer.score(patterns=sample_patterns)
        
        assert result.is_significant
        assert "RATIO_Fx_Fy:>5" in [p for p in sample_patterns if "RATIO" in p.key][0].key

    def test_score_with_subthreshold_bucketed_force_ratio(self):
        """A 2-5 bucket should stay below the default 5.0 chatter threshold."""
        scorer = SignificanceScorer()

        result = scorer.score(
            patterns=[PatternKey(pattern_type=PatternType.RATIO, key="RATIO_Fx_Fy:2-5")]
        )

        assert not result.is_significant

    def test_ratio_bucket_parser_handles_fixed_buckets(self):
        assert _parse_ratio_bucket_lower_bound("RATIO_Fx_Fy:>5") == pytest.approx(5.0)
        assert _parse_ratio_bucket_lower_bound("RATIO_Fx_Fy:2-5") == pytest.approx(2.0)
        assert _parse_ratio_bucket_lower_bound("RATIO_Fx_Fy:0.5-2") == pytest.approx(0.5)
        assert _parse_ratio_bucket_lower_bound("RATIO_Fx_Fy:<0.5") == pytest.approx(0.0)


def test_dataset_loader_emits_bucketed_ratio_patterns():
    window = WindowData(
        operation_id="OF0001",
        t_start="2026-01-01T00:00:00Z",
        t_end="2026-01-01T00:00:30Z",
        duration_s=30.0,
        n_samples=30,
        features={
            "vib_severity_x_max": 1.0,
            "vib_severity_y_max": 0.5,
            "chatter_ratio": 0.2,
            "chatter_x_count": 0,
            "power_spindle_max": 10.0,
            "op_status_mode": 3,
        },
    )

    patterns = window._derive_patterns()

    assert "RATIO_Fx_Fy:2-5" in patterns
    assert "RATIO_Fx_Fy:>2" not in patterns
    
    def test_score_with_external_signal(self):
        """Test that external signals are processed."""
        scorer = SignificanceScorer()
        
        result = scorer.score(
            patterns=[],
            external_signals={"breakage_prediction": 0.85}
        )
        
        assert result.is_significant
        assert result.score >= 0.5
        assert "classical_alert" in result.triggered_rules
    
    def test_score_updates_with_baseline(self, sample_metrics):
        """Test baseline updates and anomaly detection."""
        scorer = SignificanceScorer()
        
        # Build baseline
        for _ in range(10):
            scorer.update_baseline("test_session", sample_metrics)
        
        # Create anomalous metrics
        anomalous = WindowMetrics()
        anomalous.channel_means = [10.0, 20.0, 15.0]  # 100x normal
        anomalous.channel_stds = [5.0, 8.0, 6.0]
        anomalous.channel_rms = [12.0, 25.0, 18.0]
        anomalous.channel_peaks = [50.0, 80.0, 60.0]
        anomalous.dominant_frequencies = [512.0, 256.0, 384.0]
        anomalous.spectral_centroids = [400.0, 350.0, 420.0]
        
        result = scorer.score(
            patterns=[],
            metrics=anomalous,
            session_id="test_session"
        )
        
        assert "anomaly_deviation" in result.triggered_rules

    def test_baseline_resets_when_metric_shape_changes(self, sample_metrics):
        """Changing channel dimensionality should reset the baseline instead of crashing."""
        scorer = SignificanceScorer()

        for _ in range(6):
            scorer.update_baseline("test_session", sample_metrics)

        different_shape = WindowMetrics()
        different_shape.channel_means = [0.1]
        different_shape.channel_stds = [0.01]
        different_shape.channel_rms = [0.12]
        different_shape.channel_peaks = [0.2]
        different_shape.dominant_frequencies = [128.0]
        different_shape.spectral_centroids = [96.0]

        result = scorer.score(
            patterns=[],
            metrics=different_shape,
            session_id="test_session",
        )

        assert result is not None
        baseline = scorer._session_baselines["test_session"]
        assert baseline._shape == different_shape.to_vector().shape
        assert len(baseline._buffer) == 1
    
    def test_prior_update(self):
        """Test pattern prior updates from feedback."""
        scorer = SignificanceScorer()
        
        pattern_key = "TEST_PATTERN:value"
        
        # Initial prior should be 0.5 (neutral)
        assert scorer._pattern_priors.get(pattern_key, 0.5) == 0.5
        
        # Update with positive feedback
        scorer.update_pattern_prior(pattern_key, was_significant=True)
        assert scorer._pattern_priors[pattern_key] > 0.5
        
        # Update with negative feedback
        for _ in range(5):
            scorer.update_pattern_prior(pattern_key, was_significant=False)
        assert scorer._pattern_priors[pattern_key] < 0.5


# ============================================================================
# CuttingContext Tests
# ============================================================================

class TestCuttingContext:
    """Tests for cutting context."""
    
    def test_tooth_passing_frequency(self, sample_context):
        """Test tooth passing frequency calculation."""
        assert sample_context.tooth_passing_freq is not None
        # f = (n * z) / 60 = (8000 * 4) / 60 = 533.33 Hz
        assert abs(sample_context.tooth_passing_freq - 533.33) < 1.0
    
    def test_spindle_frequency(self, sample_context):
        """Test spindle frequency calculation."""
        assert sample_context.spindle_freq is not None
        # f = n / 60 = 8000 / 60 = 133.33 Hz
        assert abs(sample_context.spindle_freq - 133.33) < 1.0
    
    def test_regime_classification(self, sample_context):
        """Test operating regime classification."""
        # axial_depth = 2.5 -> ROUGHING
        assert sample_context.classify_regime() == OperatingRegime.ROUGHING
        
        # Finishing context
        finishing = CuttingContext(axial_depth=0.2)
        assert finishing.classify_regime() == OperatingRegime.FINISHING
    
    def test_context_matching(self, sample_context):
        """Test context similarity matching."""
        # Similar context
        similar = CuttingContext(
            tool_type="end_mill",
            spindle_speed=7800.0,  # Within 15%
            axial_depth=2.3,  # Within 30%
            workpiece_material="steel",
        )
        
        is_match, score = sample_context.matches(similar)
        assert is_match
        assert score > 0.7
        
        # Different context
        different = CuttingContext(
            tool_type="drill",
            spindle_speed=2000.0,
            axial_depth=10.0,
            workpiece_material="aluminum",
        )
        
        is_match, score = sample_context.matches(different)
        assert score < 0.5

    def test_extract_context_from_casedata_metadata(self):
        """Simulated casedata metadata should populate a real cutting context."""
        ctx = extract_context_from_metadata({
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
            "casedata": {
                "root": "/tmp/casedata",
                "tool_type": "end_mill",
                "workpiece_material": "steel",
                "spindle_speed": 628.0,
                "feed_rate": 1034.0,
                "operating_regime": "semi_finishing",
                "tool_id": "T7",
                "extra": {"temperature_head": 22.9},
            },
        })

        assert ctx.tool_type == "end_mill"
        assert ctx.workpiece_material == "steel"
        assert ctx.spindle_speed == pytest.approx(628.0)
        assert ctx.feed_rate == pytest.approx(1034.0)
        assert ctx.tool_id == "T7"
        assert ctx.operating_regime == OperatingRegime.SEMI_FINISHING
        assert ctx.extra["root"] == "/tmp/casedata"
        assert ctx.extra["sample_frequency"] == pytest.approx(1.0)
        assert ctx.extra["source"] == "simulated_casedata"
        assert ctx.extra["temperature_head"] == pytest.approx(22.9)


# ============================================================================
# PatternMatcher Tests
# ============================================================================

class TestPatternMatcher:
    """Tests for pattern matching utilities."""
    
    def test_exact_match(self):
        """Test exact pattern matching."""
        assert PatternMatcher.exact_match("RATIO_Fx_Fy:>5", "RATIO_Fx_Fy:>5")
        assert PatternMatcher.exact_match("ratio_fx_fy:>5", "RATIO_Fx_Fy:>5")  # Case insensitive
        assert not PatternMatcher.exact_match("RATIO_Fx_Fy:>5", "RATIO_Fx_Fy:2-5")
    
    def test_family_match(self):
        """Test pattern family matching."""
        assert PatternMatcher.family_match("RATIO_Fx_Fy:>5", "RATIO_Fx_Fy:2-5")
        assert not PatternMatcher.family_match("RATIO_Fx_Fy:>5", "RATIO_Fz_Fy:>5")
    
    def test_type_match(self):
        """Test pattern type matching."""
        assert PatternMatcher.type_match("RATIO_Fx_Fy:>5", "RATIO_Fz_My:>3")
        assert not PatternMatcher.type_match("RATIO_Fx_Fy:>5", "SPECTRAL_PEAK_512Hz")
    
    def test_pattern_similarity_score(self):
        """Test pattern similarity scoring."""
        query = ["RATIO_Fx_Fy:>5", "SPECTRAL_PEAK_512Hz:high"]
        memory = ["RATIO_Fx_Fy:>5", "ANOMALY_HIGH"]  # One exact match
        
        score, matched = PatternMatcher.score_pattern_similarity(query, memory)
        
        assert score > 0
        assert "RATIO_Fx_Fy:>5" in matched


# ============================================================================
# MemoryEventOrchestrator Tests
# ============================================================================

class TestMemoryEventOrchestrator:
    """Tests for the memory orchestrator."""
    
    @pytest.mark.asyncio
    async def test_process_significant_event(
        self, sample_patterns, sample_context, sample_time_range, sample_metrics
    ):
        """Test processing a significant event."""
        config = OrchestratorConfig(
            generate_explanations=False,  # Skip LLM for testing
            dispatch_alerts=False,  # Skip alerts for testing
            use_classical_models=False,  # Skip slow seed model training in tests
        )
        orchestrator = MemoryEventOrchestrator(config=config)
        
        event = MemoryEvent(
            session_id="test_session",
            time_range=sample_time_range,
            patterns=sample_patterns,
            metrics=sample_metrics,
            cutting_context=sample_context,
        )
        
        result = await orchestrator.process_event(event)
        
        assert result.processed
        assert result.significant
        assert result.memory_id is not None
        assert result.significance_score > 0
    
    @pytest.mark.asyncio
    async def test_process_insignificant_event(self, sample_time_range):
        """Test processing an insignificant event."""
        config = OrchestratorConfig(
            generate_explanations=False,
            dispatch_alerts=False,
            use_classical_models=False,  # Skip slow seed model training in tests
        )
        orchestrator = MemoryEventOrchestrator(config=config)
        
        # Empty patterns, no external signals
        event = MemoryEvent(
            session_id="test_session",
            time_range=sample_time_range,
            patterns=[],
        )
        
        result = await orchestrator.process_event(event)
        
        assert result.processed
        assert not result.significant
        assert result.memory_id is None
    
    @pytest.mark.asyncio
    async def test_process_external_signal(self):
        """Test processing external model signals."""
        config = OrchestratorConfig(
            generate_explanations=False,
            dispatch_alerts=False,
            use_classical_models=False,  # Skip slow seed model training in tests
        )
        orchestrator = MemoryEventOrchestrator(config=config)
        
        result = await orchestrator.process_external_signal(
            session_id="test_session",
            signal_type="breakage_prediction",
            signal_value=0.9,
        )
        
        assert result.processed
        assert result.significant
        assert result.memory_id is not None


# ============================================================================
# MemoryFeedbackHandler Tests
# ============================================================================

class TestMemoryFeedbackHandler:
    """Tests for the feedback handler."""
    
    @pytest.mark.asyncio
    async def test_confirm_feedback(self):
        """Test confirmation feedback."""
        handler = MemoryFeedbackHandler()
        
        request = MemoryFeedbackRequest(
            action=FeedbackAction.CONFIRM,
            user_id="test_user",
        )
        
        response = await handler.process_feedback("memory_123", request)
        
        assert response.success
        assert response.action == FeedbackAction.CONFIRM
        
        # Check stats
        stats = handler.get_feedback_stats("memory_123")
        assert stats["confirms"] == 1
    
    @pytest.mark.asyncio
    async def test_dismiss_feedback(self):
        """Test dismissal feedback."""
        handler = MemoryFeedbackHandler()
        
        request = MemoryFeedbackRequest(
            action=FeedbackAction.DISMISS,
            user_id="test_user",
            reason="Normal for this material",
        )
        
        response = await handler.process_feedback("memory_123", request)
        
        assert response.success
        assert response.action == FeedbackAction.DISMISS
        
        # Check stats
        stats = handler.get_feedback_stats("memory_123")
        assert stats["dismisses"] == 1
    
    @pytest.mark.asyncio
    async def test_comment_feedback(self):
        """Test comment feedback."""
        handler = MemoryFeedbackHandler()
        
        request = MemoryFeedbackRequest(
            action=FeedbackAction.COMMENT,
            user_id="test_user",
            comment="This looks like chatter onset",
        )
        
        response = await handler.process_feedback("memory_123", request)
        
        assert response.success
        assert "annotation_text" in response.updated_fields
    
    @pytest.mark.asyncio
    async def test_feedback_history(self):
        """Test feedback history tracking."""
        handler = MemoryFeedbackHandler()
        
        # Add multiple feedback
        await handler.process_feedback("memory_123", MemoryFeedbackRequest(
            action=FeedbackAction.CONFIRM, user_id="user1"
        ))
        await handler.process_feedback("memory_123", MemoryFeedbackRequest(
            action=FeedbackAction.COMMENT, user_id="user2", comment="Noted"
        ))
        
        history = handler.get_feedback_history("memory_123")
        assert len(history) == 2


# ============================================================================
# Integration Test
# ============================================================================

class TestIntegration:
    """Integration tests for the full memory system flow."""
    
    @pytest.mark.asyncio
    async def test_full_flow(
        self, sample_patterns, sample_context, sample_time_range, sample_metrics
    ):
        """Test the complete flow: event -> significance -> store -> feedback."""
        config = OrchestratorConfig(
            generate_explanations=False,
            dispatch_alerts=False,
            use_classical_models=False,
        )
        orchestrator = MemoryEventOrchestrator(config=config)
        
        # 1. Process event
        event = MemoryEvent(
            session_id="integration_test",
            time_range=sample_time_range,
            patterns=sample_patterns,
            metrics=sample_metrics,
            cutting_context=sample_context,
        )
        
        result = await orchestrator.process_event(event)
        assert result.significant
        memory_id = result.memory_id
        
        # 2. Get memory
        memory = orchestrator.get_memory(memory_id)
        assert memory is not None
        assert memory.session_id == "integration_test"
        
        # 3. Add feedback
        feedback_request = MemoryFeedbackRequest(
            action=FeedbackAction.CONFIRM,
            user_id="test_operator",
        )
        feedback_response = await orchestrator.feedback_handler.process_feedback(
            memory_id, feedback_request
        )
        assert feedback_response.success
        
        # 4. Check pattern prior was updated
        # (SignificanceScorer should have updated prior for patterns)
        for pattern in sample_patterns:
            prior = orchestrator.scorer._pattern_priors.get(pattern.key, 0.5)
            # After confirmation, prior should increase
            assert prior >= 0.5
        
        # 5. List memories
        memories = orchestrator.list_memories(session_id="integration_test")
        assert len(memories) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

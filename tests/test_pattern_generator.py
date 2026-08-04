"""
Unit tests for the pattern generator module.

Tests cover:
- Frequency classification
- Amplitude classification  
- Temporal pattern classification
- Cross-channel pattern classification
- Summary pattern generation
- Adaptive threshold learning
"""

import pytest
import numpy as np

from backend.agents.core.context import CuttingContext
from backend.agents.patterns.generator import (
    PatternGenerator,
    PatternThresholds,
    AdaptivePatternGenerator,
    generate_patterns,
)
from backend.agents.patterns.signatures import WORKPIECE_SLIP_SIGNATURE
from backend.agents.patterns.signatures import (
    CHATTER_SIGNATURE,
    CHIP_ADHESION_SIGNATURE,
    TOOL_BREAKAGE_SIGNATURE,
)
from backend.agents.core.metrics import WindowMetrics


@pytest.fixture
def default_generator():
    return PatternGenerator()


@pytest.fixture
def custom_thresholds():
    return PatternThresholds(
        freq_low_max=100.0,
        freq_mid_max=1000.0,
        amp_quiet_max=0.05,
        amp_normal_max=0.3,
    )


@pytest.fixture
def low_freq_metrics():
    """Metrics for a low-frequency signal."""
    return WindowMetrics(
        dominant_frequencies=[25.0],
        dominant_amplitudes=[1.0],
        channel_rms=[0.2],
        channel_crest_factors=[2.0],
        channel_kurtosis=[0.0],
        spectral_bandwidths=[5.0],
        spectral_centroids=[30.0],
        snr_estimate=40.0,
        total_energy=100.0,
    )


@pytest.fixture
def high_freq_loud_metrics():
    """Metrics for a high-frequency loud signal."""
    return WindowMetrics(
        dominant_frequencies=[800.0],
        dominant_amplitudes=[5.0],
        channel_rms=[0.8],
        channel_crest_factors=[3.0],
        channel_kurtosis=[0.5],
        spectral_bandwidths=[200.0],
        spectral_centroids=[750.0],
        snr_estimate=35.0,
        total_energy=500.0,
    )


@pytest.fixture
def multi_channel_metrics():
    """Metrics for multi-channel data."""
    return WindowMetrics(
        dominant_frequencies=[100.0, 100.0],
        dominant_amplitudes=[1.0, 1.0],
        channel_rms=[0.3, 0.3],
        channel_crest_factors=[1.5, 1.5],
        channel_kurtosis=[0.0, 0.0],
        channel_skewness=[0.0, 0.0],
        spectral_bandwidths=[50.0, 50.0],
        spectral_centroids=[110.0, 110.0],
        spectral_rolloffs=[200.0, 200.0],
        cross_correlations=[0.9],  # High correlation
        phase_differences=[0.1],   # Nearly in-phase
        coherence_values=[0.95],   # High coherence
        snr_estimate=30.0,
        total_energy=200.0,
    )


class TestPatternGeneratorBasic:
    """Basic pattern generation tests."""
    
    def test_create_generator(self):
        """Should create a generator with defaults."""
        gen = PatternGenerator()
        assert gen is not None
        assert gen.thresholds is not None
    
    def test_generate_returns_list(self, default_generator, low_freq_metrics):
        """Should return a list of pattern strings."""
        patterns = default_generator.generate(low_freq_metrics)
        
        assert isinstance(patterns, list)
        assert all(isinstance(p, str) for p in patterns)
    
    def test_patterns_have_category_format(self, default_generator, low_freq_metrics):
        """Patterns should have category:value format."""
        patterns = default_generator.generate(low_freq_metrics)
        
        for p in patterns:
            assert ":" in p, f"Pattern '{p}' missing category separator"


class TestFrequencyClassification:
    """Tests for frequency pattern classification."""
    
    def test_low_frequency(self, default_generator, low_freq_metrics):
        """Should classify as low frequency."""
        patterns = default_generator.generate(low_freq_metrics)
        
        assert any("freq" in p and "low" in p for p in patterns)
    
    def test_high_frequency(self, default_generator, high_freq_loud_metrics):
        """Should classify as high frequency."""
        patterns = default_generator.generate(high_freq_loud_metrics)
        
        assert any("freq" in p and "high" in p for p in patterns)
    
    def test_mid_frequency(self, default_generator):
        """Should classify as mid frequency."""
        metrics = WindowMetrics(
            dominant_frequencies=[200.0],
            channel_rms=[0.2],
            snr_estimate=30.0,
        )
        
        patterns = default_generator.generate(metrics)
        
        assert any("freq" in p and "mid" in p for p in patterns)


class TestAmplitudeClassification:
    """Tests for amplitude pattern classification."""
    
    def test_quiet_amplitude(self, default_generator):
        """Should classify as quiet."""
        metrics = WindowMetrics(
            dominant_frequencies=[100.0],
            channel_rms=[0.05],
            snr_estimate=30.0,
        )
        
        patterns = default_generator.generate(metrics)
        
        assert any("amp" in p and "quiet" in p for p in patterns)
    
    def test_loud_amplitude(self, default_generator, high_freq_loud_metrics):
        """Should classify as loud."""
        patterns = default_generator.generate(high_freq_loud_metrics)
        
        assert any("amp" in p and "loud" in p for p in patterns)
    
    def test_normal_amplitude(self, default_generator, low_freq_metrics):
        """Should classify as normal amplitude."""
        patterns = default_generator.generate(low_freq_metrics)
        
        assert any("amp" in p and "normal" in p for p in patterns)

    def test_observation_metadata_exposes_strength_reason_and_metric(self, default_generator, high_freq_loud_metrics):
        metadata = default_generator.get_observation_metadata(high_freq_loud_metrics)

        loud = metadata["amp:loud"]

        assert 0.0 < loud["confidence"] <= 1.0
        assert loud["source_metric"] == "channel_rms"
        assert "RMS amplitude" in loud["reason"]

    def test_loud_amplitude_strength_increases_with_rms(self, default_generator):
        lower = WindowMetrics(
            dominant_frequencies=[100.0],
            channel_rms=[0.6],
            snr_estimate=30.0,
        )
        higher = WindowMetrics(
            dominant_frequencies=[100.0],
            channel_rms=[1.0],
            snr_estimate=30.0,
        )

        lower_conf = default_generator.get_observation_metadata(lower)["amp:loud"]["confidence"]
        higher_conf = default_generator.get_observation_metadata(higher)["amp:loud"]["confidence"]

        assert higher_conf > lower_conf


class TestTemporalClassification:
    """Tests for temporal pattern classification."""
    
    def test_sustained_signal(self, default_generator):
        """Should classify low crest factor as sustained."""
        metrics = WindowMetrics(
            dominant_frequencies=[100.0],
            channel_rms=[0.3],
            channel_crest_factors=[2.0],  # Low crest factor
            channel_kurtosis=[0.0],
            snr_estimate=30.0,
        )
        
        patterns = default_generator.generate(metrics)
        
        assert any("temporal" in p and "sustained" in p for p in patterns)
    
    def test_impulsive_signal(self, default_generator):
        """Should classify high crest factor as impulsive."""
        metrics = WindowMetrics(
            dominant_frequencies=[100.0],
            channel_rms=[0.3],
            channel_crest_factors=[8.0],  # High crest factor
            channel_kurtosis=[5.0],
            snr_estimate=30.0,
        )
        
        patterns = default_generator.generate(metrics)
        
        assert any("temporal" in p and "impulsive" in p for p in patterns)

    def test_workpiece_slip_uses_signature_namespace(self, default_generator):
        metrics = WindowMetrics(
            dominant_frequencies=[100.0],
            channel_rms=[0.8],
            channel_crest_factors=[2.0],
            phase_differences=[np.radians(20.0)],
            snr_estimate=30.0,
        )
        context = CuttingContext(spindle_speed=6000.0)

        patterns = default_generator.generate(metrics, context=context)

        assert WORKPIECE_SLIP_SIGNATURE in patterns
        assert not any(p.startswith("hypothesis:") for p in patterns)
        assert "fault:workpiece_slip" not in patterns

    def test_workpiece_slip_abstains_without_spindle_context(self, default_generator):
        metrics = WindowMetrics(
            dominant_frequencies=[100.0],
            channel_rms=[0.8],
            channel_crest_factors=[2.0],
            phase_differences=[np.radians(20.0)],
            snr_estimate=30.0,
        )

        patterns = default_generator.generate(metrics)

        assert WORKPIECE_SLIP_SIGNATURE not in patterns

    def test_signature_metadata_exposes_non_binary_confidence(self, default_generator):
        metrics = WindowMetrics(
            dominant_frequencies=[100.0],
            channel_rms=[0.8],
            channel_crest_factors=[5.0],
            phase_differences=[0.0],
            snr_estimate=30.0,
        )
        context = CuttingContext(spindle_speed=6000.0)

        metadata = default_generator.get_signature_metadata(metrics, context=context)
        slip = metadata[WORKPIECE_SLIP_SIGNATURE]

        assert slip["emitted"] is True
        assert slip["indicators_present"] == 2
        assert slip["indicators_required"] == 4
        assert slip["confidence"] == pytest.approx(0.5, rel=1e-6)
        assert "spectral:spindle_freq_shift" in slip["supporting_patterns"]

    def test_chatter_abstains_without_tooth_passing_context(self, default_generator):
        metrics = WindowMetrics(
            dominant_frequencies=[120.0],
            channel_rms=[0.8],
            channel_crest_factors=[5.0],
            cross_correlations=[0.9],
            snr_estimate=30.0,
        )

        patterns = default_generator.generate(metrics)
        metadata = default_generator.get_signature_metadata(metrics)

        assert CHATTER_SIGNATURE not in patterns
        assert metadata[CHATTER_SIGNATURE]["abstain_reason"] == "missing tooth-passing context"

    def test_chatter_requires_tooth_passing_support_to_emit(self, default_generator):
        metrics = WindowMetrics(
            dominant_frequencies=[175.0],
            channel_rms=[0.8],
            channel_crest_factors=[5.0],
            cross_correlations=[0.9],
            snr_estimate=30.0,
        )
        context = CuttingContext(spindle_speed=3000.0, num_teeth=2)

        patterns = default_generator.generate(metrics, context=context)
        metadata = default_generator.get_signature_metadata(metrics, context=context)

        assert CHATTER_SIGNATURE not in patterns
        assert metadata[CHATTER_SIGNATURE]["abstain_reason"] == "missing tooth-passing spectral support"

    def test_tool_breakage_requires_high_frequency_support(self, default_generator):
        metrics = WindowMetrics(
            dominant_frequencies=[120.0],
            channel_rms=[0.7],
            channel_crest_factors=[9.0],
            channel_kurtosis=[7.0],
            spectral_bandwidths=[90.0],
            spectral_centroids=[120.0],
            snr_estimate=30.0,
        )

        patterns = default_generator.generate(metrics)
        metadata = default_generator.get_signature_metadata(metrics)

        assert TOOL_BREAKAGE_SIGNATURE not in patterns
        assert metadata[TOOL_BREAKAGE_SIGNATURE]["abstain_reason"] == "missing high-frequency spectral support"

    def test_chip_adhesion_requires_tooth_passing_context(self, default_generator):
        metrics = WindowMetrics(
            dominant_frequencies=[200.0],
            dominant_amplitudes=[1.0, 2.0, 4.0],
            channel_rms=[0.6],
            channel_kurtosis=[4.0],
            phase_differences=[0.8],
            snr_estimate=30.0,
        )

        patterns = default_generator.generate(metrics)
        metadata = default_generator.get_signature_metadata(metrics)

        assert CHIP_ADHESION_SIGNATURE not in patterns
        assert metadata[CHIP_ADHESION_SIGNATURE]["abstain_reason"] == "missing tooth-passing context"

    def test_pattern_metadata_exposes_signature_support_pattern_details(self, default_generator):
        metrics = WindowMetrics(
            dominant_frequencies=[100.0, 200.0],
            dominant_amplitudes=[1.0, 1.6, 2.8],
            channel_rms=[0.8],
            channel_crest_factors=[9.0],
            channel_kurtosis=[7.0],
            spectral_bandwidths=[80.0],
            spectral_centroids=[900.0],
            phase_differences=[np.radians(45.0)],
            cross_correlations=[0.9],
            snr_estimate=30.0,
        )
        context = CuttingContext(spindle_speed=6000.0, num_teeth=2)

        metadata = default_generator.get_pattern_metadata(metrics, context=context)

        assert metadata["spectral:hf_burst"]["source_metric"] == "spectral_centroids"
        assert metadata["temporal:phase_shift"]["source_metric"] == "phase_differences"
        assert metadata["spectral:tp_harmonic_1x"]["source_metric"] == "dominant_frequencies"


class TestSpectralShapeClassification:
    """Tests for spectral shape classification."""
    
    def test_narrowband(self, default_generator):
        """Should classify narrow bandwidth as narrowband."""
        metrics = WindowMetrics(
            dominant_frequencies=[100.0],
            channel_rms=[0.3],
            spectral_bandwidths=[5.0],
            spectral_centroids=[100.0],
            snr_estimate=30.0,
        )
        
        patterns = default_generator.generate(metrics)
        
        assert any("spectral" in p and "narrowband" in p for p in patterns)
    
    def test_wideband(self, default_generator):
        """Should classify wide bandwidth as wideband."""
        metrics = WindowMetrics(
            dominant_frequencies=[100.0],
            channel_rms=[0.3],
            spectral_bandwidths=[50.0],
            spectral_centroids=[100.0],
            snr_estimate=30.0,
        )
        
        patterns = default_generator.generate(metrics)
        
        assert any("spectral" in p and "wideband" in p for p in patterns)


class TestCrossChannelClassification:
    """Tests for cross-channel pattern classification."""
    
    def test_high_correlation(self, default_generator, multi_channel_metrics):
        """Should classify high correlation."""
        patterns = default_generator.generate(multi_channel_metrics)
        
        assert any("correlation" in p and "high" in p for p in patterns)
    
    def test_high_coherence(self, default_generator, multi_channel_metrics):
        """Should classify high coherence."""
        patterns = default_generator.generate(multi_channel_metrics)
        
        assert any("coherence" in p and "high" in p for p in patterns)
    
    def test_in_phase(self, default_generator, multi_channel_metrics):
        """Should classify as in-phase."""
        patterns = default_generator.generate(multi_channel_metrics)
        
        assert any("phase" in p and "in-phase" in p for p in patterns)
    
    def test_low_correlation(self, default_generator):
        """Should classify low correlation."""
        metrics = WindowMetrics(
            dominant_frequencies=[100.0, 100.0],
            channel_rms=[0.3, 0.3],
            cross_correlations=[0.1],
            phase_differences=[1.0],
            coherence_values=[0.2],
            snr_estimate=30.0,
        )
        
        patterns = default_generator.generate(metrics)
        
        assert any("correlation" in p and "low" in p for p in patterns)


class TestSNRClassification:
    """Tests for SNR classification."""
    
    def test_high_snr(self, default_generator, low_freq_metrics):
        """Should classify high SNR."""
        patterns = default_generator.generate(low_freq_metrics)
        
        assert any("snr" in p and "high" in p for p in patterns)
    
    def test_low_snr(self, default_generator):
        """Should classify low SNR."""
        metrics = WindowMetrics(
            dominant_frequencies=[100.0],
            channel_rms=[0.3],
            snr_estimate=5.0,  # Low SNR
        )
        
        patterns = default_generator.generate(metrics)
        
        assert any("snr" in p and "low" in p for p in patterns)


class TestSummaryPatterns:
    """Tests for summary pattern generation."""
    
    def test_summary_patterns_shorter(self, default_generator, multi_channel_metrics):
        """Summary patterns should be fewer than full patterns."""
        full = default_generator.generate(multi_channel_metrics)
        summary = default_generator.generate_summary_patterns(multi_channel_metrics)
        
        assert len(summary) <= len(full)
    
    def test_summary_patterns_unique_categories(self, default_generator, multi_channel_metrics):
        """Summary patterns should have unique categories."""
        summary = default_generator.generate_summary_patterns(multi_channel_metrics)
        
        categories = [p.split(":")[0] for p in summary]
        assert len(categories) == len(set(categories))


class TestCustomThresholds:
    """Tests for custom threshold configuration."""
    
    def test_custom_frequency_threshold(self, custom_thresholds):
        """Should use custom frequency thresholds."""
        gen = PatternGenerator(thresholds=custom_thresholds)
        
        # 50 Hz should now be low (default threshold is 50)
        metrics = WindowMetrics(
            dominant_frequencies=[50.0],
            channel_rms=[0.2],
            snr_estimate=30.0,
        )
        
        patterns = gen.generate(metrics)
        
        # With custom threshold freq_low_max=100, 50Hz is still low
        assert any("freq" in p and "low" in p for p in patterns)
    
    def test_custom_amplitude_threshold(self, custom_thresholds):
        """Should use custom amplitude thresholds."""
        gen = PatternGenerator(thresholds=custom_thresholds)
        
        # 0.04 RMS should be quiet (custom threshold is 0.05)
        metrics = WindowMetrics(
            dominant_frequencies=[100.0],
            channel_rms=[0.04],
            snr_estimate=30.0,
        )
        
        patterns = gen.generate(metrics)
        
        assert any("amp" in p and "quiet" in p for p in patterns)


class TestAdaptiveGenerator:
    """Tests for adaptive pattern generator."""
    
    def test_create_adaptive_generator(self):
        """Should create adaptive generator."""
        gen = AdaptivePatternGenerator()
        assert gen is not None
    
    def test_update_statistics(self):
        """Should update statistics from metrics."""
        gen = AdaptivePatternGenerator()
        
        for _ in range(5):
            metrics = WindowMetrics(
                dominant_frequencies=[100.0],
                channel_rms=[0.3],
                snr_estimate=30.0,
            )
            gen.update_statistics(metrics)
        
        assert gen._n_samples == 5
        assert len(gen._freq_stats) == 5


class TestConvenienceFunction:
    """Tests for convenience function."""
    
    def test_generate_patterns_full(self, low_freq_metrics):
        """Should generate full patterns via convenience function."""
        patterns = generate_patterns(low_freq_metrics)
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
    
    def test_generate_patterns_summary(self, low_freq_metrics):
        """Should generate summary patterns via convenience function."""
        patterns = generate_patterns(low_freq_metrics, summary_only=True)
        
        assert isinstance(patterns, list)


class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_metrics(self, default_generator):
        """Should handle empty metrics."""
        metrics = WindowMetrics()
        
        patterns = default_generator.generate(metrics)
        
        # Should still return a list (possibly with SNR/energy patterns)
        assert isinstance(patterns, list)
    
    def test_single_channel_no_cross_channel(self, default_generator, low_freq_metrics):
        """Single channel should not have cross-channel patterns."""
        patterns = default_generator.generate(low_freq_metrics)
        
        # Should not have correlation/coherence/phase patterns
        assert not any("correlation:pair" in p for p in patterns)
        assert not any("coherence:pair" in p for p in patterns)
    
    def test_zero_centroid_handling(self, default_generator):
        """Should handle zero spectral centroid."""
        metrics = WindowMetrics(
            dominant_frequencies=[100.0],
            channel_rms=[0.3],
            spectral_bandwidths=[10.0],
            spectral_centroids=[0.0],  # Zero centroid
            snr_estimate=30.0,
        )
        
        # Should not raise
        patterns = default_generator.generate(metrics)
        assert isinstance(patterns, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

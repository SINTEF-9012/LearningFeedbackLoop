"""
Unit tests for the metrics computer module.

Tests cover:
- Time-domain feature extraction
- Spectral feature extraction
- Cross-channel metrics
- Fixed-dimension output
"""

import pytest
import numpy as np
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from agents.metrics_computer import (
    MetricsComputer,
    FixedDimensionMetricsComputer,
    WindowMetrics,
    compute_window_metrics,
    compute_feature_vector,
)


@pytest.fixture
def sample_rate():
    return 1000.0


@pytest.fixture
def simple_sine(sample_rate):
    """Generate a simple sine wave."""
    t = np.arange(0, 1.0, 1.0/sample_rate)
    return np.sin(2 * np.pi * 50 * t)


@pytest.fixture
def multi_channel_data(sample_rate):
    """Generate multi-channel test data."""
    t = np.arange(0, 1.0, 1.0/sample_rate)
    ch1 = np.sin(2 * np.pi * 50 * t)
    ch2 = np.sin(2 * np.pi * 50 * t + np.pi/4)  # Phase shifted
    ch3 = 2.0 * np.sin(2 * np.pi * 100 * t)     # Different freq
    return np.vstack([ch1, ch2, ch3])


@pytest.fixture
def metrics_computer(sample_rate):
    return MetricsComputer(sample_rate=sample_rate)


class TestWindowMetrics:
    """Tests for WindowMetrics dataclass."""
    
    def test_to_vector(self):
        """Should convert to flat vector."""
        metrics = WindowMetrics(
            channel_means=[1.0, 2.0],
            channel_stds=[0.1, 0.2],
            channel_rms=[1.1, 2.1],
            channel_peaks=[1.5, 2.5],
            channel_crest_factors=[1.3, 1.2],
            channel_kurtosis=[0.5, 0.6],
            channel_skewness=[0.1, 0.2],
            dominant_frequencies=[50.0, 100.0],
            dominant_amplitudes=[0.5, 1.0],
            spectral_centroids=[55.0, 110.0],
            spectral_bandwidths=[10.0, 20.0],
            spectral_rolloffs=[100.0, 200.0],
            total_energy=10.0,
            snr_estimate=30.0,
        )
        
        vec = metrics.to_vector()
        
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert len(vec) > 0
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = WindowMetrics(
            channel_means=[1.0],
            total_energy=5.0,
        )
        
        d = metrics.to_dict()
        
        assert isinstance(d, dict)
        assert "channel_means" in d
        assert "total_energy" in d
        assert d["total_energy"] == 5.0
    
    def test_from_dict(self):
        """Should reconstruct from dictionary."""
        original = WindowMetrics(
            channel_means=[1.0, 2.0],
            channel_stds=[0.1, 0.2],
            total_energy=10.0,
        )
        
        d = original.to_dict()
        reconstructed = WindowMetrics.from_dict(d)
        
        assert reconstructed.channel_means == original.channel_means
        assert reconstructed.total_energy == original.total_energy


class TestMetricsComputerTimeDomain:
    """Tests for time-domain feature extraction."""
    
    def test_mean_computation(self, metrics_computer):
        """Should compute correct mean."""
        signal = np.ones(1000) * 5.0
        metrics = metrics_computer.compute(signal)
        
        assert len(metrics.channel_means) == 1
        assert metrics.channel_means[0] == pytest.approx(5.0)
    
    def test_std_computation(self, metrics_computer):
        """Should compute correct standard deviation."""
        signal = np.concatenate([np.zeros(500), np.ones(500)])
        metrics = metrics_computer.compute(signal)
        
        # Std of [0,0,...,1,1,...] should be ~0.5
        assert metrics.channel_stds[0] == pytest.approx(0.5, rel=0.01)
    
    def test_rms_computation(self, metrics_computer, simple_sine):
        """Should compute correct RMS for sine wave."""
        metrics = metrics_computer.compute(simple_sine)
        
        # RMS of unit sine is 1/sqrt(2)
        expected_rms = 1.0 / np.sqrt(2)
        assert metrics.channel_rms[0] == pytest.approx(expected_rms, rel=0.01)
    
    def test_peak_computation(self, metrics_computer, simple_sine):
        """Should compute correct peak."""
        metrics = metrics_computer.compute(simple_sine)
        
        assert metrics.channel_peaks[0] == pytest.approx(1.0, rel=0.01)
    
    def test_crest_factor(self, metrics_computer, simple_sine):
        """Should compute correct crest factor for sine."""
        metrics = metrics_computer.compute(simple_sine)
        
        # Crest factor of sine is sqrt(2)
        expected_crest = np.sqrt(2)
        assert metrics.channel_crest_factors[0] == pytest.approx(expected_crest, rel=0.02)


class TestMetricsComputerSpectral:
    """Tests for spectral feature extraction."""
    
    def test_dominant_frequency(self, metrics_computer):
        """Should find correct dominant frequency."""
        fs = 1000.0
        f0 = 100.0
        t = np.arange(0, 1.0, 1.0/fs)
        signal = np.sin(2 * np.pi * f0 * t)
        
        metrics = metrics_computer.compute(signal, sample_rate=fs)
        
        # Should be close to f0
        assert metrics.dominant_frequencies[0] == pytest.approx(f0, rel=0.05)
    
    def test_spectral_centroid_for_pure_tone(self, metrics_computer):
        """Spectral centroid should be near dominant frequency for pure tone."""
        fs = 1000.0
        f0 = 150.0
        t = np.arange(0, 1.0, 1.0/fs)
        signal = np.sin(2 * np.pi * f0 * t)
        
        metrics = metrics_computer.compute(signal, sample_rate=fs)
        
        # Centroid should be close to f0 for pure tone
        assert metrics.spectral_centroids[0] == pytest.approx(f0, rel=0.1)
    
    def test_narrowband_signal(self, metrics_computer):
        """Narrowband signal should have small bandwidth."""
        fs = 1000.0
        f0 = 100.0
        t = np.arange(0, 1.0, 1.0/fs)
        signal = np.sin(2 * np.pi * f0 * t)
        
        metrics = metrics_computer.compute(signal, sample_rate=fs)
        
        # Bandwidth should be small relative to centroid
        relative_bw = metrics.spectral_bandwidths[0] / metrics.spectral_centroids[0]
        assert relative_bw < 0.3


class TestMetricsComputerCrossChannel:
    """Tests for cross-channel metrics."""
    
    def test_identical_channels_high_correlation(self, metrics_computer):
        """Identical channels should have correlation ~1."""
        signal = np.random.randn(1000)
        data = np.vstack([signal, signal])
        
        metrics = metrics_computer.compute(data)
        
        assert len(metrics.cross_correlations) == 1
        assert metrics.cross_correlations[0] == pytest.approx(1.0, rel=0.01)
    
    def test_inverted_channels_negative_correlation(self, metrics_computer):
        """Inverted channels should have correlation ~-1."""
        signal = np.random.randn(1000)
        data = np.vstack([signal, -signal])
        
        metrics = metrics_computer.compute(data)
        
        assert metrics.cross_correlations[0] == pytest.approx(-1.0, rel=0.01)
    
    def test_uncorrelated_channels(self, metrics_computer):
        """Independent channels should have low correlation."""
        np.random.seed(42)
        data = np.random.randn(2, 1000)
        
        metrics = metrics_computer.compute(data)
        
        # Should be close to 0
        assert abs(metrics.cross_correlations[0]) < 0.1
    
    def test_phase_difference_in_phase(self, metrics_computer):
        """In-phase signals should have ~0 phase difference."""
        fs = 1000.0
        f0 = 50.0
        t = np.arange(0, 1.0, 1.0/fs)
        ch1 = np.sin(2 * np.pi * f0 * t)
        ch2 = np.sin(2 * np.pi * f0 * t)
        data = np.vstack([ch1, ch2])
        
        metrics = metrics_computer.compute(data, sample_rate=fs)
        
        assert metrics.phase_differences[0] == pytest.approx(0.0, abs=0.1)
    
    def test_phase_difference_quadrature(self, metrics_computer):
        """Quadrature signals should have ~pi/2 phase difference."""
        fs = 1000.0
        f0 = 50.0
        t = np.arange(0, 1.0, 1.0/fs)
        ch1 = np.sin(2 * np.pi * f0 * t)
        ch2 = np.cos(2 * np.pi * f0 * t)  # 90 degree lead
        data = np.vstack([ch1, ch2])
        
        metrics = metrics_computer.compute(data, sample_rate=fs)
        
        # Should be close to pi/2 (may be negative depending on order)
        assert abs(abs(metrics.phase_differences[0]) - np.pi/2) < 0.2


class TestFixedDimensionMetrics:
    """Tests for fixed-dimension output."""
    
    def test_consistent_dimension_single_channel(self):
        """Should produce consistent dimension for single channel."""
        computer = FixedDimensionMetricsComputer(max_channels=4, sample_rate=1000.0)
        
        signal1 = np.random.randn(1000)
        signal2 = np.random.randn(500)
        
        vec1 = computer.compute_fixed(signal1)
        vec2 = computer.compute_fixed(signal2)
        
        assert len(vec1) == len(vec2)
        assert len(vec1) == computer.fixed_dimension
    
    def test_consistent_dimension_varying_channels(self):
        """Should pad to consistent dimension for varying channel counts."""
        computer = FixedDimensionMetricsComputer(max_channels=4, sample_rate=1000.0)
        
        data1 = np.random.randn(1, 1000)  # 1 channel
        data2 = np.random.randn(2, 1000)  # 2 channels
        data3 = np.random.randn(4, 1000)  # 4 channels
        
        vec1 = computer.compute_fixed(data1)
        vec2 = computer.compute_fixed(data2)
        vec3 = computer.compute_fixed(data3)
        
        assert len(vec1) == len(vec2) == len(vec3)
    
    def test_truncation_for_excess_channels(self):
        """Should truncate if more channels than max."""
        computer = FixedDimensionMetricsComputer(max_channels=2, sample_rate=1000.0)
        
        data = np.random.randn(5, 1000)  # 5 channels
        
        vec = computer.compute_fixed(data)
        
        assert len(vec) == computer.fixed_dimension


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_compute_window_metrics(self):
        """Should compute metrics via convenience function."""
        signal = np.sin(2 * np.pi * 50 * np.arange(0, 1.0, 0.001))
        
        metrics = compute_window_metrics(signal, sample_rate=1000.0)
        
        assert isinstance(metrics, WindowMetrics)
        assert len(metrics.channel_means) > 0
    
    def test_compute_feature_vector(self):
        """Should return fixed-dimension vector via convenience function."""
        signal = np.random.randn(1000)
        
        vec = compute_feature_vector(signal, sample_rate=1000.0, max_channels=4)
        
        assert isinstance(vec, np.ndarray)
        assert len(vec) > 0


class TestEdgeCases:
    """Edge case tests."""
    
    def test_constant_signal(self, metrics_computer):
        """Should handle constant signal."""
        signal = np.ones(1000) * 3.0
        
        metrics = metrics_computer.compute(signal)
        
        assert metrics.channel_means[0] == pytest.approx(3.0)
        assert metrics.channel_stds[0] == pytest.approx(0.0, abs=1e-10)
    
    def test_very_short_signal(self, metrics_computer):
        """Should handle very short signals."""
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        
        metrics = metrics_computer.compute(signal)
        
        assert isinstance(metrics, WindowMetrics)
    
    def test_zero_signal(self, metrics_computer):
        """Should handle all-zero signal."""
        signal = np.zeros(1000)
        
        metrics = metrics_computer.compute(signal)
        
        assert metrics.channel_means[0] == 0.0
        assert metrics.channel_rms[0] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

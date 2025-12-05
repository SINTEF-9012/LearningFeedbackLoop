"""
Metrics Computer - Extract numeric features from time-series windows.

This module computes feature vectors from raw time-series data for:
1. Numeric vector storage in ANN index
2. Input to pattern key generation
3. Anomaly/similarity detection

Features are grouped into:
- Time-domain statistics (mean, std, rms, peak, crest factor, etc.)
- Spectral features (dominant frequency, spectral centroid, bandwidth, etc.)
- Cross-channel metrics (coherence, phase difference, correlation)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class WindowMetrics:
    """Computed metrics for a single time window."""
    
    # Time-domain features per channel
    channel_means: List[float] = field(default_factory=list)
    channel_stds: List[float] = field(default_factory=list)
    channel_rms: List[float] = field(default_factory=list)
    channel_peaks: List[float] = field(default_factory=list)
    channel_crest_factors: List[float] = field(default_factory=list)
    channel_kurtosis: List[float] = field(default_factory=list)
    channel_skewness: List[float] = field(default_factory=list)
    
    # Spectral features per channel
    dominant_frequencies: List[float] = field(default_factory=list)
    dominant_amplitudes: List[float] = field(default_factory=list)
    spectral_centroids: List[float] = field(default_factory=list)
    spectral_bandwidths: List[float] = field(default_factory=list)
    spectral_rolloffs: List[float] = field(default_factory=list)
    
    # Cross-channel metrics (if multiple channels)
    cross_correlations: List[float] = field(default_factory=list)  # pairwise
    phase_differences: List[float] = field(default_factory=list)   # at dominant freq
    coherence_values: List[float] = field(default_factory=list)    # pairwise coherence
    
    # Global/aggregate features
    total_energy: float = 0.0
    snr_estimate: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Flatten all metrics into a single feature vector."""
        components = [
            self.channel_means,
            self.channel_stds,
            self.channel_rms,
            self.channel_peaks,
            self.channel_crest_factors,
            self.channel_kurtosis,
            self.channel_skewness,
            self.dominant_frequencies,
            self.dominant_amplitudes,
            self.spectral_centroids,
            self.spectral_bandwidths,
            self.spectral_rolloffs,
            self.cross_correlations,
            self.phase_differences,
            self.coherence_values,
            [self.total_energy, self.snr_estimate]
        ]
        # Flatten all components
        flat = []
        for c in components:
            if isinstance(c, list):
                flat.extend(c)
            else:
                flat.append(c)
        return np.array(flat, dtype=np.float32)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "channel_means": self.channel_means,
            "channel_stds": self.channel_stds,
            "channel_rms": self.channel_rms,
            "channel_peaks": self.channel_peaks,
            "channel_crest_factors": self.channel_crest_factors,
            "channel_kurtosis": self.channel_kurtosis,
            "channel_skewness": self.channel_skewness,
            "dominant_frequencies": self.dominant_frequencies,
            "dominant_amplitudes": self.dominant_amplitudes,
            "spectral_centroids": self.spectral_centroids,
            "spectral_bandwidths": self.spectral_bandwidths,
            "spectral_rolloffs": self.spectral_rolloffs,
            "cross_correlations": self.cross_correlations,
            "phase_differences": self.phase_differences,
            "coherence_values": self.coherence_values,
            "total_energy": self.total_energy,
            "snr_estimate": self.snr_estimate
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "WindowMetrics":
        """Reconstruct from dictionary."""
        return cls(**d)


class MetricsComputer:
    """
    Compute feature metrics from time-series windows.
    
    This class extracts a comprehensive set of features from multi-channel
    time-series data for use in similarity search and pattern detection.
    """
    
    def __init__(
        self,
        sample_rate: float = 1000.0,
        n_fft: Optional[int] = None,
        spectral_rolloff_percent: float = 0.85
    ):
        """
        Initialize the metrics computer.
        
        Args:
            sample_rate: Sampling rate in Hz
            n_fft: FFT size (defaults to next power of 2 of window length)
            spectral_rolloff_percent: Percentile for spectral rolloff
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.spectral_rolloff_percent = spectral_rolloff_percent
    
    def compute(
        self,
        data: np.ndarray,
        sample_rate: Optional[float] = None
    ) -> WindowMetrics:
        """
        Compute all metrics for a data window.
        
        Args:
            data: Array of shape (n_samples,) for single channel
                  or (n_channels, n_samples) for multi-channel
            sample_rate: Override the default sample rate
        
        Returns:
            WindowMetrics with all computed features
        """
        sr = sample_rate or self.sample_rate
        
        # Ensure 2D: (n_channels, n_samples)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_channels, n_samples = data.shape
        
        # Initialize metrics
        metrics = WindowMetrics()
        
        # Time-domain features per channel
        for ch in range(n_channels):
            signal = data[ch]
            
            # Basic statistics
            mean = float(np.mean(signal))
            std = float(np.std(signal))
            rms = float(np.sqrt(np.mean(signal ** 2)))
            peak = float(np.max(np.abs(signal)))
            crest = peak / rms if rms > 1e-10 else 0.0
            
            # Higher-order statistics
            if std > 1e-10:
                normalized = (signal - mean) / std
                kurt = float(np.mean(normalized ** 4) - 3)  # excess kurtosis
                skew = float(np.mean(normalized ** 3))
            else:
                kurt = 0.0
                skew = 0.0
            
            metrics.channel_means.append(mean)
            metrics.channel_stds.append(std)
            metrics.channel_rms.append(rms)
            metrics.channel_peaks.append(peak)
            metrics.channel_crest_factors.append(crest)
            metrics.channel_kurtosis.append(kurt)
            metrics.channel_skewness.append(skew)
        
        # Spectral features per channel
        n_fft = self.n_fft or int(2 ** np.ceil(np.log2(n_samples)))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        
        for ch in range(n_channels):
            signal = data[ch]
            
            # Compute FFT magnitude spectrum
            spectrum = np.abs(np.fft.rfft(signal, n=n_fft))
            power = spectrum ** 2
            
            # Dominant frequency
            peak_idx = int(np.argmax(spectrum[1:])) + 1  # skip DC
            dom_freq = float(freqs[peak_idx])
            dom_amp = float(spectrum[peak_idx])
            
            # Spectral centroid (weighted average of frequencies)
            total_power = np.sum(power) + 1e-10
            centroid = float(np.sum(freqs * power) / total_power)
            
            # Spectral bandwidth (weighted std of frequencies)
            bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
            
            # Spectral rolloff (frequency below which X% of energy is concentrated)
            cumsum = np.cumsum(power)
            rolloff_threshold = self.spectral_rolloff_percent * cumsum[-1]
            rolloff_idx = int(np.searchsorted(cumsum, rolloff_threshold))
            rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
            
            metrics.dominant_frequencies.append(dom_freq)
            metrics.dominant_amplitudes.append(dom_amp)
            metrics.spectral_centroids.append(centroid)
            metrics.spectral_bandwidths.append(bandwidth)
            metrics.spectral_rolloffs.append(rolloff)
        
        # Cross-channel metrics
        if n_channels >= 2:
            self._compute_cross_channel_metrics(data, sr, n_fft, freqs, metrics)
        
        # Global features
        metrics.total_energy = float(np.sum(data ** 2))
        
        # Simple SNR estimate (ratio of signal variance to noise floor estimate)
        # Use median absolute deviation as noise estimate
        all_signal = data.flatten()
        noise_estimate = 1.4826 * np.median(np.abs(all_signal - np.median(all_signal)))
        signal_power = np.var(all_signal)
        if noise_estimate > 1e-10:
            metrics.snr_estimate = float(10 * np.log10(signal_power / (noise_estimate ** 2)))
        else:
            metrics.snr_estimate = 60.0  # Clamp to high SNR
        
        return metrics
    
    def _compute_cross_channel_metrics(
        self,
        data: np.ndarray,
        sr: float,
        n_fft: int,
        freqs: np.ndarray,
        metrics: WindowMetrics
    ) -> None:
        """Compute pairwise cross-channel metrics."""
        n_channels = data.shape[0]
        
        # Compute FFT for all channels
        spectra = []
        for ch in range(n_channels):
            spectra.append(np.fft.rfft(data[ch], n=n_fft))
        
        # Pairwise metrics
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Cross-correlation at lag 0
                corr = float(np.corrcoef(data[i], data[j])[0, 1])
                if np.isnan(corr):
                    corr = 0.0
                metrics.cross_correlations.append(corr)
                
                # Phase difference at dominant frequency
                spec_i = spectra[i]
                spec_j = spectra[j]
                
                # Find dominant frequency index (largest combined magnitude)
                combined_mag = np.abs(spec_i) + np.abs(spec_j)
                dom_idx = int(np.argmax(combined_mag[1:])) + 1
                
                phase_i = np.angle(spec_i[dom_idx])
                phase_j = np.angle(spec_j[dom_idx])
                phase_diff = float(phase_j - phase_i)
                # Wrap to [-pi, pi]
                phase_diff = float(np.arctan2(np.sin(phase_diff), np.cos(phase_diff)))
                metrics.phase_differences.append(phase_diff)
                
                # Coherence estimate (simplified - magnitude squared coherence at dominant freq)
                csd = spec_i[dom_idx] * np.conj(spec_j[dom_idx])
                psd_i = np.abs(spec_i[dom_idx]) ** 2
                psd_j = np.abs(spec_j[dom_idx]) ** 2
                if psd_i > 1e-10 and psd_j > 1e-10:
                    coherence = float(np.abs(csd) ** 2 / (psd_i * psd_j))
                else:
                    coherence = 0.0
                metrics.coherence_values.append(coherence)
    
    def compute_batch(
        self,
        windows: List[np.ndarray],
        sample_rate: Optional[float] = None
    ) -> List[WindowMetrics]:
        """
        Compute metrics for multiple windows.
        
        Args:
            windows: List of data arrays
            sample_rate: Override sample rate
        
        Returns:
            List of WindowMetrics
        """
        return [self.compute(w, sample_rate) for w in windows]
    
    def get_feature_dimension(self, n_channels: int) -> int:
        """
        Get the dimension of the feature vector for given number of channels.
        
        Args:
            n_channels: Number of channels in the data
        
        Returns:
            Dimension of the feature vector
        """
        # Per-channel features: 7 time-domain + 5 spectral = 12 per channel
        per_channel = 12 * n_channels
        
        # Cross-channel features (pairwise): 3 per pair
        n_pairs = n_channels * (n_channels - 1) // 2
        cross_channel = 3 * n_pairs
        
        # Global features: 2
        global_features = 2
        
        return per_channel + cross_channel + global_features


class FixedDimensionMetricsComputer(MetricsComputer):
    """
    Metrics computer that produces fixed-dimension vectors.
    
    Useful when the number of channels may vary but vectors must have
    consistent dimensions for ANN indexing.
    """
    
    def __init__(
        self,
        max_channels: int = 4,
        sample_rate: float = 1000.0,
        n_fft: Optional[int] = None,
        spectral_rolloff_percent: float = 0.85
    ):
        """
        Initialize with maximum number of channels.
        
        Args:
            max_channels: Maximum number of channels to support
            sample_rate: Sampling rate in Hz
            n_fft: FFT size
            spectral_rolloff_percent: Percentile for spectral rolloff
        """
        super().__init__(sample_rate, n_fft, spectral_rolloff_percent)
        self.max_channels = max_channels
        self._fixed_dim = self.get_feature_dimension(max_channels)
    
    def compute_fixed(
        self,
        data: np.ndarray,
        sample_rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute metrics and return fixed-dimension vector.
        
        If fewer channels than max_channels, pads with zeros.
        If more channels than max_channels, truncates.
        
        Args:
            data: Input data array
            sample_rate: Override sample rate
        
        Returns:
            Fixed-dimension feature vector
        """
        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_channels = data.shape[0]
        
        # Truncate if needed
        if n_channels > self.max_channels:
            data = data[:self.max_channels]
            n_channels = self.max_channels
        
        # Compute metrics
        metrics = self.compute(data, sample_rate)
        vec = metrics.to_vector()
        
        # Pad if needed
        if n_channels < self.max_channels:
            current_dim = len(vec)
            padded = np.zeros(self._fixed_dim, dtype=np.float32)
            padded[:current_dim] = vec
            return padded
        
        return vec
    
    @property
    def fixed_dimension(self) -> int:
        """Return the fixed output dimension."""
        return self._fixed_dim


# Convenience functions

def compute_window_metrics(
    data: np.ndarray,
    sample_rate: float = 1000.0
) -> WindowMetrics:
    """
    Convenience function to compute metrics for a single window.
    
    Args:
        data: Time-series data (1D or 2D)
        sample_rate: Sampling rate in Hz
    
    Returns:
        WindowMetrics object
    """
    computer = MetricsComputer(sample_rate=sample_rate)
    return computer.compute(data)


def compute_feature_vector(
    data: np.ndarray,
    sample_rate: float = 1000.0,
    max_channels: int = 4
) -> np.ndarray:
    """
    Convenience function to get fixed-dimension feature vector.
    
    Args:
        data: Time-series data
        sample_rate: Sampling rate in Hz
        max_channels: Maximum channels for consistent dimension
    
    Returns:
        Feature vector as numpy array
    """
    computer = FixedDimensionMetricsComputer(
        max_channels=max_channels,
        sample_rate=sample_rate
    )
    return computer.compute_fixed(data)

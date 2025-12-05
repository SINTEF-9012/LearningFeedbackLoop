"""
Pattern Generator - Generate symbolic pattern keys from computed metrics.

Pattern keys are short descriptive labels that capture the qualitative
characteristics of a signal window. They are used for:
1. Fast filtering before expensive vector similarity search
2. Human-readable indexing of memory records
3. Rule-based retrieval when exact patterns are known

Pattern key format: "category:value" or "category:subcategory:value"
Examples:
- "freq:low", "freq:mid", "freq:high"
- "amp:quiet", "amp:normal", "amp:loud"
- "stability:stable", "stability:transient"
- "correlation:high", "correlation:low"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import logging

from .metrics_computer import WindowMetrics

logger = logging.getLogger(__name__)


@dataclass
class PatternThresholds:
    """
    Configurable thresholds for pattern classification.
    
    All thresholds are inclusive on the lower bound, exclusive on upper.
    """
    
    # Frequency bands (Hz)
    freq_low_max: float = 50.0
    freq_mid_max: float = 500.0
    # Above freq_mid_max is "high"
    
    # Amplitude levels (normalized or raw depending on use)
    amp_quiet_max: float = 0.1
    amp_normal_max: float = 0.5
    # Above amp_normal_max is "loud"
    
    # Crest factor thresholds
    crest_impulsive_min: float = 6.0  # highly impulsive
    crest_transient_min: float = 4.0  # moderately impulsive
    # Below crest_transient_min is "sustained"
    
    # Cross-correlation thresholds
    corr_high_min: float = 0.8
    corr_moderate_min: float = 0.5
    # Below corr_moderate_min is "low"
    
    # Coherence thresholds
    coherence_high_min: float = 0.9
    coherence_moderate_min: float = 0.6
    
    # Spectral bandwidth (relative to centroid)
    bandwidth_narrow_max: float = 0.1
    bandwidth_moderate_max: float = 0.3
    # Above bandwidth_moderate_max is "wide"
    
    # SNR thresholds (dB)
    snr_high_min: float = 30.0
    snr_moderate_min: float = 15.0
    # Below snr_moderate_min is "low"
    
    # Kurtosis thresholds (for detecting outliers/impulses)
    kurtosis_high_min: float = 3.0  # heavy tails
    kurtosis_low_max: float = -1.0  # light tails
    # Between is "normal"


class PatternGenerator:
    """
    Generate symbolic pattern keys from computed metrics.
    
    Pattern keys are categorical labels that summarize the qualitative
    characteristics of a signal. They enable fast filtering and
    human-readable indexing.
    """
    
    def __init__(
        self,
        thresholds: Optional[PatternThresholds] = None,
        include_channel_index: bool = True,
        max_patterns_per_category: int = 3
    ):
        """
        Initialize the pattern generator.
        
        Args:
            thresholds: Classification thresholds (uses defaults if None)
            include_channel_index: Whether to include channel index in keys
            max_patterns_per_category: Limit patterns per category
        """
        self.thresholds = thresholds or PatternThresholds()
        self.include_channel_index = include_channel_index
        self.max_patterns_per_category = max_patterns_per_category
    
    def generate(self, metrics: WindowMetrics) -> List[str]:
        """
        Generate pattern keys from metrics.
        
        Args:
            metrics: Computed window metrics
        
        Returns:
            List of pattern key strings
        """
        patterns: List[str] = []
        
        # Frequency patterns
        patterns.extend(self._classify_frequencies(metrics))
        
        # Amplitude patterns
        patterns.extend(self._classify_amplitudes(metrics))
        
        # Temporal patterns (crest factor, kurtosis)
        patterns.extend(self._classify_temporal(metrics))
        
        # Spectral shape patterns
        patterns.extend(self._classify_spectral_shape(metrics))
        
        # Cross-channel patterns
        patterns.extend(self._classify_cross_channel(metrics))
        
        # Global patterns
        patterns.extend(self._classify_global(metrics))
        
        return patterns
    
    def _classify_frequencies(self, metrics: WindowMetrics) -> List[str]:
        """Classify dominant frequencies into bands."""
        patterns = []
        th = self.thresholds
        
        for i, freq in enumerate(metrics.dominant_frequencies):
            if freq < th.freq_low_max:
                label = "low"
            elif freq < th.freq_mid_max:
                label = "mid"
            else:
                label = "high"
            
            if self.include_channel_index and len(metrics.dominant_frequencies) > 1:
                patterns.append(f"freq:ch{i}:{label}")
            else:
                patterns.append(f"freq:{label}")
        
        # Add overall frequency pattern if all channels agree
        if len(set(patterns)) == 1 and patterns:
            patterns = [patterns[0].replace(f":ch0:", ":")]
        
        return self._limit_patterns(patterns, "freq")
    
    def _classify_amplitudes(self, metrics: WindowMetrics) -> List[str]:
        """Classify amplitude levels."""
        patterns = []
        th = self.thresholds
        
        for i, rms in enumerate(metrics.channel_rms):
            if rms < th.amp_quiet_max:
                label = "quiet"
            elif rms < th.amp_normal_max:
                label = "normal"
            else:
                label = "loud"
            
            if self.include_channel_index and len(metrics.channel_rms) > 1:
                patterns.append(f"amp:ch{i}:{label}")
            else:
                patterns.append(f"amp:{label}")
        
        return self._limit_patterns(patterns, "amp")
    
    def _classify_temporal(self, metrics: WindowMetrics) -> List[str]:
        """Classify temporal characteristics."""
        patterns = []
        th = self.thresholds
        
        # Crest factor classification
        for i, crest in enumerate(metrics.channel_crest_factors):
            if crest >= th.crest_impulsive_min:
                label = "impulsive"
            elif crest >= th.crest_transient_min:
                label = "transient"
            else:
                label = "sustained"
            
            prefix = f"temporal:ch{i}:" if self.include_channel_index and len(metrics.channel_crest_factors) > 1 else "temporal:"
            patterns.append(f"{prefix}{label}")
        
        # Kurtosis classification
        for i, kurt in enumerate(metrics.channel_kurtosis):
            if kurt >= th.kurtosis_high_min:
                label = "heavy-tails"
            elif kurt <= th.kurtosis_low_max:
                label = "light-tails"
            else:
                label = "normal-tails"
            
            prefix = f"kurtosis:ch{i}:" if self.include_channel_index and len(metrics.channel_kurtosis) > 1 else "kurtosis:"
            patterns.append(f"{prefix}{label}")
        
        return self._limit_patterns(patterns, "temporal")
    
    def _classify_spectral_shape(self, metrics: WindowMetrics) -> List[str]:
        """Classify spectral shape characteristics."""
        patterns = []
        th = self.thresholds
        
        # Bandwidth classification
        for i, (bw, centroid) in enumerate(zip(
            metrics.spectral_bandwidths,
            metrics.spectral_centroids
        )):
            if centroid < 1e-10:
                relative_bw = 0.0
            else:
                relative_bw = bw / centroid
            
            if relative_bw < th.bandwidth_narrow_max:
                label = "narrowband"
            elif relative_bw < th.bandwidth_moderate_max:
                label = "moderate-band"
            else:
                label = "wideband"
            
            prefix = f"spectral:ch{i}:" if self.include_channel_index and len(metrics.spectral_bandwidths) > 1 else "spectral:"
            patterns.append(f"{prefix}{label}")
        
        return self._limit_patterns(patterns, "spectral")
    
    def _classify_cross_channel(self, metrics: WindowMetrics) -> List[str]:
        """Classify cross-channel relationships."""
        patterns = []
        th = self.thresholds
        
        # Cross-correlation
        for i, corr in enumerate(metrics.cross_correlations):
            abs_corr = abs(corr)
            if abs_corr >= th.corr_high_min:
                if corr > 0:
                    label = "high-positive-corr"
                else:
                    label = "high-negative-corr"
            elif abs_corr >= th.corr_moderate_min:
                label = "moderate-corr"
            else:
                label = "low-corr"
            patterns.append(f"correlation:pair{i}:{label}")
        
        # Coherence
        for i, coh in enumerate(metrics.coherence_values):
            if coh >= th.coherence_high_min:
                label = "high-coherence"
            elif coh >= th.coherence_moderate_min:
                label = "moderate-coherence"
            else:
                label = "low-coherence"
            patterns.append(f"coherence:pair{i}:{label}")
        
        # Phase relationships
        for i, phase in enumerate(metrics.phase_differences):
            # Classify phase into quadrants
            deg = np.degrees(phase)
            if -22.5 <= deg < 22.5:
                label = "in-phase"
            elif 22.5 <= deg < 67.5:
                label = "phase-lead-45"
            elif 67.5 <= deg < 112.5:
                label = "quadrature"
            elif 112.5 <= deg < 157.5:
                label = "phase-lead-135"
            elif deg >= 157.5 or deg < -157.5:
                label = "anti-phase"
            elif -157.5 <= deg < -112.5:
                label = "phase-lag-135"
            elif -112.5 <= deg < -67.5:
                label = "neg-quadrature"
            else:  # -67.5 <= deg < -22.5
                label = "phase-lag-45"
            patterns.append(f"phase:pair{i}:{label}")
        
        return self._limit_patterns(patterns, "cross")
    
    def _classify_global(self, metrics: WindowMetrics) -> List[str]:
        """Classify global/aggregate characteristics."""
        patterns = []
        th = self.thresholds
        
        # SNR classification
        if metrics.snr_estimate >= th.snr_high_min:
            patterns.append("snr:high")
        elif metrics.snr_estimate >= th.snr_moderate_min:
            patterns.append("snr:moderate")
        else:
            patterns.append("snr:low")
        
        # Energy level (relative - would need context for absolute)
        # For now, just include the total energy pattern if available
        if metrics.total_energy > 0:
            log_energy = np.log10(metrics.total_energy + 1e-10)
            if log_energy > 0:
                patterns.append("energy:high")
            elif log_energy > -3:
                patterns.append("energy:moderate")
            else:
                patterns.append("energy:low")
        
        return patterns
    
    def _limit_patterns(self, patterns: List[str], category: str) -> List[str]:
        """Limit the number of patterns per category."""
        # Count patterns by category prefix
        by_prefix: Dict[str, List[str]] = {}
        for p in patterns:
            prefix = p.split(":")[0]
            if prefix not in by_prefix:
                by_prefix[prefix] = []
            by_prefix[prefix].append(p)
        
        # Limit each category
        result = []
        for prefix, pats in by_prefix.items():
            result.extend(pats[:self.max_patterns_per_category])
        
        return result
    
    def generate_summary_patterns(self, metrics: WindowMetrics) -> List[str]:
        """
        Generate a reduced set of summary patterns.
        
        Returns only the most salient patterns, useful for quick filtering.
        """
        patterns = []
        th = self.thresholds
        
        # Overall frequency character
        if metrics.dominant_frequencies:
            avg_freq = np.mean(metrics.dominant_frequencies)
            if avg_freq < th.freq_low_max:
                patterns.append("freq:low")
            elif avg_freq < th.freq_mid_max:
                patterns.append("freq:mid")
            else:
                patterns.append("freq:high")
        
        # Overall amplitude character
        if metrics.channel_rms:
            avg_rms = np.mean(metrics.channel_rms)
            if avg_rms < th.amp_quiet_max:
                patterns.append("amp:quiet")
            elif avg_rms < th.amp_normal_max:
                patterns.append("amp:normal")
            else:
                patterns.append("amp:loud")
        
        # Overall temporal character (use max crest factor)
        if metrics.channel_crest_factors:
            max_crest = max(metrics.channel_crest_factors)
            if max_crest >= th.crest_impulsive_min:
                patterns.append("temporal:impulsive")
            elif max_crest >= th.crest_transient_min:
                patterns.append("temporal:transient")
            else:
                patterns.append("temporal:sustained")
        
        # Overall spectral character
        if metrics.spectral_bandwidths and metrics.spectral_centroids:
            avg_bw = np.mean(metrics.spectral_bandwidths)
            avg_cent = np.mean(metrics.spectral_centroids)
            if avg_cent > 1e-10:
                rel_bw = avg_bw / avg_cent
                if rel_bw < th.bandwidth_narrow_max:
                    patterns.append("spectral:narrowband")
                elif rel_bw < th.bandwidth_moderate_max:
                    patterns.append("spectral:moderate-band")
                else:
                    patterns.append("spectral:wideband")
        
        # Overall correlation (if multi-channel)
        if metrics.cross_correlations:
            avg_corr = np.mean([abs(c) for c in metrics.cross_correlations])
            if avg_corr >= th.corr_high_min:
                patterns.append("correlation:high")
            elif avg_corr >= th.corr_moderate_min:
                patterns.append("correlation:moderate")
            else:
                patterns.append("correlation:low")
        
        # SNR
        if metrics.snr_estimate >= th.snr_high_min:
            patterns.append("snr:high")
        elif metrics.snr_estimate >= th.snr_moderate_min:
            patterns.append("snr:moderate")
        else:
            patterns.append("snr:low")
        
        return patterns


class AdaptivePatternGenerator(PatternGenerator):
    """
    Pattern generator that learns thresholds from data.
    
    Useful when signal characteristics vary across sessions or datasets.
    """
    
    def __init__(
        self,
        initial_thresholds: Optional[PatternThresholds] = None,
        adaptation_rate: float = 0.1,
        include_channel_index: bool = True
    ):
        """
        Initialize adaptive generator.
        
        Args:
            initial_thresholds: Starting thresholds
            adaptation_rate: Learning rate for threshold adaptation
            include_channel_index: Whether to include channel index in keys
        """
        super().__init__(initial_thresholds, include_channel_index)
        self.adaptation_rate = adaptation_rate
        
        # Running statistics for adaptation
        self._freq_stats: List[float] = []
        self._amp_stats: List[float] = []
        self._n_samples = 0
    
    def update_statistics(self, metrics: WindowMetrics) -> None:
        """
        Update running statistics from observed metrics.
        
        Call this as new data arrives to adapt thresholds.
        """
        # Accumulate frequency stats
        self._freq_stats.extend(metrics.dominant_frequencies)
        
        # Accumulate amplitude stats
        self._amp_stats.extend(metrics.channel_rms)
        
        self._n_samples += 1
        
        # Periodically update thresholds
        if self._n_samples % 100 == 0:
            self._adapt_thresholds()
    
    def _adapt_thresholds(self) -> None:
        """Adapt thresholds based on observed statistics."""
        if len(self._freq_stats) < 10:
            return
        
        freq_arr = np.array(self._freq_stats)
        amp_arr = np.array(self._amp_stats)
        
        alpha = self.adaptation_rate
        
        # Adapt frequency thresholds using percentiles
        self.thresholds.freq_low_max = (
            (1 - alpha) * self.thresholds.freq_low_max +
            alpha * np.percentile(freq_arr, 33)
        )
        self.thresholds.freq_mid_max = (
            (1 - alpha) * self.thresholds.freq_mid_max +
            alpha * np.percentile(freq_arr, 67)
        )
        
        # Adapt amplitude thresholds
        self.thresholds.amp_quiet_max = (
            (1 - alpha) * self.thresholds.amp_quiet_max +
            alpha * np.percentile(amp_arr, 25)
        )
        self.thresholds.amp_normal_max = (
            (1 - alpha) * self.thresholds.amp_normal_max +
            alpha * np.percentile(amp_arr, 75)
        )
        
        # Trim statistics to prevent memory growth
        max_stats = 1000
        if len(self._freq_stats) > max_stats:
            self._freq_stats = self._freq_stats[-max_stats:]
        if len(self._amp_stats) > max_stats:
            self._amp_stats = self._amp_stats[-max_stats:]
        
        logger.debug(
            f"Adapted thresholds: freq_low={self.thresholds.freq_low_max:.1f}, "
            f"freq_mid={self.thresholds.freq_mid_max:.1f}, "
            f"amp_quiet={self.thresholds.amp_quiet_max:.3f}, "
            f"amp_normal={self.thresholds.amp_normal_max:.3f}"
        )


# Convenience function

def generate_patterns(
    metrics: WindowMetrics,
    summary_only: bool = False
) -> List[str]:
    """
    Convenience function to generate pattern keys from metrics.
    
    Args:
        metrics: Computed window metrics
        summary_only: If True, return only summary patterns
    
    Returns:
        List of pattern key strings
    """
    generator = PatternGenerator()
    if summary_only:
        return generator.generate_summary_patterns(metrics)
    return generator.generate(metrics)

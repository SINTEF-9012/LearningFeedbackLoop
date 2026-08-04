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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import logging

from ..core.metrics import WindowMetrics
from ..core.context import CuttingContext
from ..domain_config import DomainConfig, get_active_domain
from .signatures import signature_key_for_fault_name

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

    # --- Fault-specific thresholds ---

    # Tool breakage: sudden high-frequency burst, loss of periodicity
    breakage_hf_energy_ratio_min: float = 0.4     # fraction of energy above 500 Hz
    breakage_crest_factor_min: float = 8.0         # impulsive burst threshold
    breakage_kurtosis_min: float = 6.0             # heavy-tailed impulse

    # Chatter: modulated vibration, increased amplitude
    chatter_modulation_depth_min: float = 0.3      # amplitude modulation index
    chatter_amplitude_growth_min: float = 2.0      # ratio of current RMS to baseline

    # Chip adhesion / built-up edge: irregular tooth-passing pattern
    chip_adhesion_tp_var_max: float = 0.25         # normalised variance at tooth-passing freq
    chip_adhesion_harmonic_irregularity: float = 0.3  # normalised std of harmonic amplitudes

    # Workpiece slip / clamping: shift at spindle frequency
    slip_spindle_amp_change_min: float = 0.5       # fractional amplitude jump at 1× spindle
    slip_phase_shift_min_deg: float = 30.0         # minimum phase shift (degrees) to flag


@dataclass
class HypothesisAssessment:
    key: str
    supporting_patterns: List[str] = field(default_factory=list)
    indicators_present: int = 0
    indicators_required: int = 1
    min_indicators_to_emit: int = 1
    emitted: bool = False
    abstain_reason: Optional[str] = None

    @property
    def confidence(self) -> float:
        if self.indicators_required <= 0:
            return 0.0
        return max(0.0, min(1.0, self.indicators_present / self.indicators_required))

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "confidence": round(self.confidence, 4),
            "indicators_present": self.indicators_present,
            "indicators_required": self.indicators_required,
            "min_indicators_to_emit": self.min_indicators_to_emit,
            "supporting_patterns": list(self.supporting_patterns),
            "emitted": self.emitted,
            "abstain_reason": self.abstain_reason,
        }


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
        max_patterns_per_category: int = 3,
        domain: Optional[DomainConfig] = None,
    ):
        """
        Initialize the pattern generator.
        
        Args:
            thresholds: Classification thresholds (uses defaults if None)
            include_channel_index: Whether to include channel index in keys
            max_patterns_per_category: Limit patterns per category
            domain: Optional domain config (auto-detected if None)
        """
        self.thresholds = thresholds or PatternThresholds()
        self.include_channel_index = include_channel_index
        self.max_patterns_per_category = max_patterns_per_category
        self._domain = domain

        # Fault classifier registry: fault_name -> method
        # Subclasses or callers can register custom classifiers.
        self._fault_classifiers: Dict[
            str,
            Callable[[WindowMetrics, Optional[CuttingContext]], List[str]],
        ] = {
            "tool_breakage": self._classify_tool_breakage,
            "chatter": self._classify_chatter,
            "chip_adhesion": self._classify_chip_adhesion,
            "workpiece_slip": self._classify_workpiece_slip,
        }
        self._fault_assessors: Dict[
            str,
            Callable[[WindowMetrics, Optional[CuttingContext]], HypothesisAssessment],
        ] = {
            "tool_breakage": self._assess_tool_breakage,
            "chatter": self._assess_chatter,
            "chip_adhesion": self._assess_chip_adhesion,
            "workpiece_slip": self._assess_workpiece_slip,
        }

    # ── Fault classifier registry API ─────────────────────────────────

    def register_fault_classifier(
        self,
        name: str,
        classifier: Callable[[WindowMetrics, Optional[CuttingContext]], List[str]],
    ) -> None:
        """Register (or replace) a fault-specific classifier."""
        self._fault_classifiers[name] = classifier

    def unregister_fault_classifier(self, name: str) -> None:
        """Remove a fault classifier by name."""
        self._fault_classifiers.pop(name, None)

    @property
    def domain(self) -> DomainConfig:
        if self._domain is None:
            self._domain = get_active_domain()
        return self._domain

    def _active_fault_names(self) -> Set[str]:
        return {ft.name for ft in self.domain.fault_types}

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def _score_above(cls, value: float, threshold: float, scale: Optional[float] = None) -> float:
        denom = scale if scale is not None and scale > 1e-9 else max(abs(threshold), 1.0)
        return cls._clamp01((value - threshold) / denom)

    @classmethod
    def _score_below(cls, value: float, threshold: float, scale: Optional[float] = None) -> float:
        denom = scale if scale is not None and scale > 1e-9 else max(abs(threshold), 1.0)
        return cls._clamp01((threshold - value) / denom)

    @classmethod
    def _score_band(cls, value: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return 1.0
        midpoint = (lower + upper) / 2.0
        half_width = max((upper - lower) / 2.0, 1e-9)
        return cls._clamp01(1.0 - (abs(value - midpoint) / half_width))

    @staticmethod
    def _channel_value(values: List[float], token: Optional[str]) -> Optional[float]:
        if not values:
            return None
        if token and token.startswith("ch"):
            try:
                index = int(token[2:])
            except ValueError:
                return None
            if 0 <= index < len(values):
                return float(values[index])
            return None
        return float(np.mean(values))

    @staticmethod
    def _pair_value(values: List[float], token: Optional[str]) -> Optional[float]:
        if not values:
            return None
        if token and token.startswith("pair"):
            try:
                index = int(token[4:])
            except ValueError:
                return None
            if 0 <= index < len(values):
                return float(values[index])
            return None
        return float(np.mean(values))

    @staticmethod
    def _phase_distance_deg(value_deg: float, target_deg: float) -> float:
        diff = (value_deg - target_deg + 180.0) % 360.0 - 180.0
        return abs(diff)

    def _observation_metadata_for_pattern(
        self,
        key: str,
        metrics: WindowMetrics,
    ) -> Optional[Dict[str, Any]]:
        parts = key.split(":")
        if len(parts) < 2:
            return None

        category = parts[0]
        token = parts[1] if len(parts) >= 3 else None
        label = parts[-1]
        th = self.thresholds

        if category == "freq":
            freq = self._channel_value(metrics.dominant_frequencies, token)
            if freq is None:
                return None
            if label == "low":
                confidence = self._score_below(freq, th.freq_low_max)
                reason = (
                    f"dominant frequency {freq:.1f} Hz is below the low-band threshold "
                    f"{th.freq_low_max:.1f} Hz"
                )
            elif label == "mid":
                confidence = self._score_band(freq, th.freq_low_max, th.freq_mid_max)
                reason = (
                    f"dominant frequency {freq:.1f} Hz sits in the mid band "
                    f"[{th.freq_low_max:.1f}, {th.freq_mid_max:.1f}) Hz"
                )
            elif label == "high":
                confidence = self._score_above(freq, th.freq_mid_max)
                reason = (
                    f"dominant frequency {freq:.1f} Hz exceeds the high-band threshold "
                    f"{th.freq_mid_max:.1f} Hz"
                )
            else:
                return None
            return {
                "confidence": round(confidence, 4),
                "reason": reason,
                "source_metric": "dominant_frequencies",
                "metric_value": round(freq, 4),
            }

        if category == "amp":
            rms = self._channel_value(metrics.channel_rms, token)
            if rms is None:
                return None
            if label == "quiet":
                confidence = self._score_below(rms, th.amp_quiet_max)
                reason = (
                    f"RMS amplitude {rms:.4f} stays below the quiet threshold "
                    f"{th.amp_quiet_max:.4f}"
                )
            elif label == "normal":
                confidence = self._score_band(rms, th.amp_quiet_max, th.amp_normal_max)
                reason = (
                    f"RMS amplitude {rms:.4f} lies in the normal band "
                    f"[{th.amp_quiet_max:.4f}, {th.amp_normal_max:.4f})"
                )
            elif label == "loud":
                confidence = self._score_above(rms, th.amp_normal_max)
                reason = (
                    f"RMS amplitude {rms:.4f} exceeds the loud threshold "
                    f"{th.amp_normal_max:.4f}"
                )
            else:
                return None
            return {
                "confidence": round(confidence, 4),
                "reason": reason,
                "source_metric": "channel_rms",
                "metric_value": round(rms, 4),
            }

        if category == "temporal":
            crest = self._channel_value(metrics.channel_crest_factors, token)
            if crest is None:
                return None
            if label == "sustained":
                confidence = self._score_below(crest, th.crest_transient_min)
                reason = (
                    f"crest factor {crest:.3f} stays below the transient threshold "
                    f"{th.crest_transient_min:.3f}"
                )
            elif label == "transient":
                confidence = self._score_band(crest, th.crest_transient_min, th.crest_impulsive_min)
                reason = (
                    f"crest factor {crest:.3f} lies between the transient and impulsive "
                    f"thresholds [{th.crest_transient_min:.3f}, {th.crest_impulsive_min:.3f})"
                )
            elif label == "impulsive":
                confidence = self._score_above(crest, th.crest_impulsive_min)
                reason = (
                    f"crest factor {crest:.3f} exceeds the impulsive threshold "
                    f"{th.crest_impulsive_min:.3f}"
                )
            else:
                return None
            return {
                "confidence": round(confidence, 4),
                "reason": reason,
                "source_metric": "channel_crest_factors",
                "metric_value": round(crest, 4),
            }

        if category == "kurtosis":
            kurt = self._channel_value(metrics.channel_kurtosis, token)
            if kurt is None:
                return None
            if label == "heavy-tails":
                confidence = self._score_above(kurt, th.kurtosis_high_min)
                reason = (
                    f"kurtosis {kurt:.3f} exceeds the heavy-tail threshold "
                    f"{th.kurtosis_high_min:.3f}"
                )
            elif label == "light-tails":
                confidence = self._score_below(kurt, th.kurtosis_low_max)
                reason = (
                    f"kurtosis {kurt:.3f} is below the light-tail threshold "
                    f"{th.kurtosis_low_max:.3f}"
                )
            elif label == "normal-tails":
                confidence = self._score_band(kurt, th.kurtosis_low_max, th.kurtosis_high_min)
                reason = (
                    f"kurtosis {kurt:.3f} sits between the light-tail and heavy-tail "
                    f"thresholds [{th.kurtosis_low_max:.3f}, {th.kurtosis_high_min:.3f})"
                )
            else:
                return None
            return {
                "confidence": round(confidence, 4),
                "reason": reason,
                "source_metric": "channel_kurtosis",
                "metric_value": round(kurt, 4),
            }

        if category == "spectral":
            if label not in {"narrowband", "moderate-band", "wideband"}:
                return None
            centroid = self._channel_value(metrics.spectral_centroids, token)
            bandwidth = self._channel_value(metrics.spectral_bandwidths, token)
            if centroid is None or bandwidth is None:
                return None
            relative_bw = 0.0 if abs(centroid) < 1e-10 else bandwidth / centroid
            if label == "narrowband":
                confidence = self._score_below(relative_bw, th.bandwidth_narrow_max)
                reason = (
                    f"relative bandwidth {relative_bw:.3f} stays below the narrowband threshold "
                    f"{th.bandwidth_narrow_max:.3f}"
                )
            elif label == "moderate-band":
                confidence = self._score_band(
                    relative_bw,
                    th.bandwidth_narrow_max,
                    th.bandwidth_moderate_max,
                )
                reason = (
                    f"relative bandwidth {relative_bw:.3f} lies in the moderate band "
                    f"[{th.bandwidth_narrow_max:.3f}, {th.bandwidth_moderate_max:.3f})"
                )
            else:
                confidence = self._score_above(relative_bw, th.bandwidth_moderate_max)
                reason = (
                    f"relative bandwidth {relative_bw:.3f} exceeds the wideband threshold "
                    f"{th.bandwidth_moderate_max:.3f}"
                )
            return {
                "confidence": round(confidence, 4),
                "reason": reason,
                "source_metric": "spectral_bandwidths",
                "metric_value": round(relative_bw, 4),
            }

        if category == "correlation":
            corr = self._pair_value(metrics.cross_correlations, token)
            if corr is None:
                return None
            abs_corr = abs(corr)
            if label in {"high-positive-corr", "high-negative-corr"}:
                confidence = self._score_above(abs_corr, th.corr_high_min, scale=max(1.0 - th.corr_high_min, 1e-9))
                polarity = "positive" if corr >= 0 else "negative"
                reason = (
                    f"pairwise correlation {corr:.3f} shows strong {polarity} coupling above "
                    f"{th.corr_high_min:.3f}"
                )
            elif label == "moderate-corr":
                confidence = self._score_band(abs_corr, th.corr_moderate_min, th.corr_high_min)
                reason = (
                    f"pairwise correlation magnitude {abs_corr:.3f} lies in the moderate band "
                    f"[{th.corr_moderate_min:.3f}, {th.corr_high_min:.3f})"
                )
            elif label == "low-corr":
                confidence = self._score_below(abs_corr, th.corr_moderate_min)
                reason = (
                    f"pairwise correlation magnitude {abs_corr:.3f} stays below the moderate "
                    f"threshold {th.corr_moderate_min:.3f}"
                )
            else:
                return None
            return {
                "confidence": round(confidence, 4),
                "reason": reason,
                "source_metric": "cross_correlations",
                "metric_value": round(corr, 4),
            }

        if category == "coherence":
            coherence = self._pair_value(metrics.coherence_values, token)
            if coherence is None:
                return None
            if label == "high-coherence":
                confidence = self._score_above(
                    coherence,
                    th.coherence_high_min,
                    scale=max(1.0 - th.coherence_high_min, 1e-9),
                )
                reason = (
                    f"coherence {coherence:.3f} exceeds the high-coherence threshold "
                    f"{th.coherence_high_min:.3f}"
                )
            elif label == "moderate-coherence":
                confidence = self._score_band(coherence, th.coherence_moderate_min, th.coherence_high_min)
                reason = (
                    f"coherence {coherence:.3f} lies in the moderate band "
                    f"[{th.coherence_moderate_min:.3f}, {th.coherence_high_min:.3f})"
                )
            elif label == "low-coherence":
                confidence = self._score_below(coherence, th.coherence_moderate_min)
                reason = (
                    f"coherence {coherence:.3f} stays below the moderate threshold "
                    f"{th.coherence_moderate_min:.3f}"
                )
            else:
                return None
            return {
                "confidence": round(confidence, 4),
                "reason": reason,
                "source_metric": "coherence_values",
                "metric_value": round(coherence, 4),
            }

        if category == "phase":
            phase = self._pair_value(metrics.phase_differences, token)
            if phase is None:
                return None
            phase_deg = float(np.degrees(phase))
            targets = {
                "in-phase": 0.0,
                "phase-lead-45": 45.0,
                "quadrature": 90.0,
                "phase-lead-135": 135.0,
                "anti-phase": 180.0,
                "phase-lag-135": -135.0,
                "neg-quadrature": -90.0,
                "phase-lag-45": -45.0,
            }
            target = targets.get(label)
            if target is None:
                return None
            confidence = self._clamp01(1.0 - (self._phase_distance_deg(phase_deg, target) / 22.5))
            return {
                "confidence": round(confidence, 4),
                "reason": f"phase difference {phase_deg:.1f}° is closest to the {label} archetype ({target:.1f}°)",
                "source_metric": "phase_differences",
                "metric_value": round(phase_deg, 4),
            }

        if category == "snr":
            snr = float(metrics.snr_estimate)
            if label == "high":
                confidence = self._score_above(snr, th.snr_high_min, scale=max(10.0, 1.0))
                reason = (
                    f"SNR {snr:.2f} dB exceeds the high-SNR threshold {th.snr_high_min:.2f} dB"
                )
            elif label == "moderate":
                confidence = self._score_band(snr, th.snr_moderate_min, th.snr_high_min)
                reason = (
                    f"SNR {snr:.2f} dB lies in the moderate band "
                    f"[{th.snr_moderate_min:.2f}, {th.snr_high_min:.2f}) dB"
                )
            elif label == "low":
                confidence = self._score_below(snr, th.snr_moderate_min)
                reason = (
                    f"SNR {snr:.2f} dB stays below the moderate threshold {th.snr_moderate_min:.2f} dB"
                )
            else:
                return None
            return {
                "confidence": round(confidence, 4),
                "reason": reason,
                "source_metric": "snr_estimate",
                "metric_value": round(snr, 4),
            }

        if category == "energy":
            log_energy = float(np.log10(metrics.total_energy + 1e-10))
            if label == "high":
                confidence = self._score_above(log_energy, 0.0, scale=3.0)
                reason = f"log10(total energy) {log_energy:.3f} is above the high-energy split at 0.0"
            elif label == "moderate":
                confidence = self._score_band(log_energy, -3.0, 0.0)
                reason = f"log10(total energy) {log_energy:.3f} lies in the moderate band [-3.0, 0.0)"
            elif label == "low":
                confidence = self._score_below(log_energy, -3.0, scale=3.0)
                reason = f"log10(total energy) {log_energy:.3f} is below the low-energy split at -3.0"
            else:
                return None
            return {
                "confidence": round(confidence, 4),
                "reason": reason,
                "source_metric": "total_energy",
                "metric_value": round(log_energy, 4),
            }

        return None

    def get_observation_metadata(self, metrics: WindowMetrics) -> Dict[str, Dict[str, Any]]:
        metadata: Dict[str, Dict[str, Any]] = {}
        for key in self._classify_frequencies(metrics):
            entry = self._observation_metadata_for_pattern(key, metrics)
            if entry is not None:
                metadata[key] = entry
        for key in self._classify_amplitudes(metrics):
            entry = self._observation_metadata_for_pattern(key, metrics)
            if entry is not None:
                metadata[key] = entry
        for key in self._classify_temporal(metrics):
            entry = self._observation_metadata_for_pattern(key, metrics)
            if entry is not None:
                metadata[key] = entry
        for key in self._classify_spectral_shape(metrics):
            entry = self._observation_metadata_for_pattern(key, metrics)
            if entry is not None:
                metadata[key] = entry
        for key in self._classify_cross_channel(metrics):
            entry = self._observation_metadata_for_pattern(key, metrics)
            if entry is not None:
                metadata[key] = entry
        for key in self._classify_global(metrics):
            entry = self._observation_metadata_for_pattern(key, metrics)
            if entry is not None:
                metadata[key] = entry
        return metadata

    def get_pattern_metadata(
        self,
        metrics: WindowMetrics,
        context: Optional[CuttingContext] = None,
    ) -> Dict[str, Dict[str, Any]]:
        metadata = self.get_observation_metadata(metrics)
        support_keys: Set[str] = set()
        active_fault_names = self._active_fault_names()
        signature_metadata: Dict[str, Dict[str, Any]] = {}

        for fault_name, assessor in self._fault_assessors.items():
            if active_fault_names and fault_name not in active_fault_names:
                continue
            assessment = assessor(metrics, context)
            signature_metadata[assessment.key] = assessment.to_metadata()
            support_keys.update(assessment.supporting_patterns)

        for key in sorted(support_keys):
            if key in metadata:
                continue
            entry = self._supporting_pattern_metadata_for_pattern(key, metrics, context)
            if entry is not None:
                metadata[key] = entry

        metadata.update(signature_metadata)
        return metadata

    def get_signature_metadata(
        self,
        metrics: WindowMetrics,
        context: Optional[CuttingContext] = None,
    ) -> Dict[str, Dict[str, Any]]:
        metadata: Dict[str, Dict[str, Any]] = {}
        active_fault_names = self._active_fault_names()

        for fault_name, assessor in self._fault_assessors.items():
            if active_fault_names and fault_name not in active_fault_names:
                continue
            assessment = assessor(metrics, context)
            metadata[assessment.key] = assessment.to_metadata()

        return metadata

    def get_hypothesis_metadata(
        self,
        metrics: WindowMetrics,
        context: Optional[CuttingContext] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Legacy wrapper retained during the hypothesis -> signature migration."""
        return self.get_signature_metadata(metrics, context)

    def _supporting_pattern_metadata_for_pattern(
        self,
        key: str,
        metrics: WindowMetrics,
        context: Optional[CuttingContext] = None,
    ) -> Optional[Dict[str, Any]]:
        th = self.thresholds

        if key == "spectral:hf_burst":
            if not metrics.spectral_centroids:
                return None
            avg_centroid = float(np.mean(metrics.spectral_centroids))
            confidence = self._score_above(avg_centroid, th.freq_mid_max, scale=max(th.freq_mid_max, 1.0))
            return {
                "confidence": round(confidence, 4),
                "reason": (
                    f"average spectral centroid {avg_centroid:.1f} Hz exceeds the high-frequency threshold "
                    f"{th.freq_mid_max:.1f} Hz"
                ),
                "source_metric": "spectral_centroids",
                "metric_value": round(avg_centroid, 4),
            }

        if key == "spectral:modulated_vibration":
            if not metrics.channel_crest_factors:
                return None
            max_crest = max(float(value) for value in metrics.channel_crest_factors)
            confidence = self._score_above(
                max_crest,
                th.crest_transient_min,
                scale=max(th.crest_impulsive_min - th.crest_transient_min, 1.0),
            )
            return {
                "confidence": round(confidence, 4),
                "reason": (
                    f"crest factor {max_crest:.3f} exceeds the modulation onset threshold "
                    f"{th.crest_transient_min:.3f}"
                ),
                "source_metric": "channel_crest_factors",
                "metric_value": round(max_crest, 4),
            }

        if key == "spectral:irregular_tooth_passing":
            if not metrics.dominant_amplitudes or len(metrics.dominant_amplitudes) < 2:
                return None
            amps = np.array(metrics.dominant_amplitudes, dtype=float)
            amp_mean = float(np.mean(amps))
            if amp_mean <= 1e-10:
                return None
            amp_cv = float(np.std(amps) / amp_mean)
            confidence = self._score_above(
                amp_cv,
                th.chip_adhesion_harmonic_irregularity,
                scale=max(th.chip_adhesion_harmonic_irregularity, 0.1),
            )
            return {
                "confidence": round(confidence, 4),
                "reason": (
                    f"harmonic amplitude variation {amp_cv:.3f} exceeds the tooth-passing irregularity threshold "
                    f"{th.chip_adhesion_harmonic_irregularity:.3f}"
                ),
                "source_metric": "dominant_amplitudes",
                "metric_value": round(amp_cv, 4),
            }

        if key == "spectral:spindle_freq_shift":
            if not (context and context.spindle_freq and metrics.dominant_frequencies):
                return None
            target = float(context.spindle_freq)
            closest = min(
                (float(freq) for freq in metrics.dominant_frequencies),
                key=lambda freq: abs(freq - target),
            )
            rel_error = abs(closest - target) / (target + 1e-9)
            confidence = self._clamp01(1.0 - (rel_error / 0.10))
            return {
                "confidence": round(confidence, 4),
                "reason": (
                    f"dominant frequency {closest:.2f} Hz aligns with the spindle-order reference "
                    f"{target:.2f} Hz"
                ),
                "source_metric": "dominant_frequencies",
                "metric_value": round(closest, 4),
            }

        if key.startswith("spectral:tp_harmonic_") and key.endswith("x"):
            if not (context and context.tooth_passing_freq and metrics.dominant_frequencies):
                return None
            harmonic_token = key.split("_")[-1]
            try:
                harmonic = float(harmonic_token[:-1])
            except ValueError:
                return None
            target = float(context.tooth_passing_freq) * harmonic
            closest = min(
                (float(freq) for freq in metrics.dominant_frequencies),
                key=lambda freq: abs(freq - target),
            )
            rel_error = abs(closest - target) / (target + 1e-9)
            confidence = self._clamp01(1.0 - (rel_error / 0.10))
            return {
                "confidence": round(confidence, 4),
                "reason": (
                    f"dominant frequency {closest:.2f} Hz aligns with tooth-passing harmonic {harmonic:.0f}\u00d7 "
                    f"at {target:.2f} Hz"
                ),
                "source_metric": "dominant_frequencies",
                "metric_value": round(closest, 4),
            }

        if key == "temporal:periodicity_loss":
            if not metrics.spectral_bandwidths or not metrics.spectral_centroids:
                return None
            avg_bw = float(np.mean(metrics.spectral_bandwidths))
            avg_cent = float(np.mean(metrics.spectral_centroids))
            if avg_cent <= 1e-10:
                return None
            rel_bw = avg_bw / avg_cent
            confidence = self._score_above(rel_bw, 0.5, scale=0.5)
            return {
                "confidence": round(confidence, 4),
                "reason": f"relative bandwidth {rel_bw:.3f} indicates loss of periodic structure",
                "source_metric": "spectral_bandwidths",
                "metric_value": round(rel_bw, 4),
            }

        if key == "temporal:impulsive_burst":
            if not metrics.channel_crest_factors:
                return None
            max_crest = max(float(value) for value in metrics.channel_crest_factors)
            confidence = self._score_above(
                max_crest,
                th.breakage_crest_factor_min,
                scale=max(th.breakage_crest_factor_min, 1.0),
            )
            return {
                "confidence": round(confidence, 4),
                "reason": (
                    f"crest factor {max_crest:.3f} exceeds the impulsive-burst threshold "
                    f"{th.breakage_crest_factor_min:.3f}"
                ),
                "source_metric": "channel_crest_factors",
                "metric_value": round(max_crest, 4),
            }

        if key == "temporal:phase_shift":
            if not metrics.phase_differences:
                return None
            max_phase = max(abs(float(np.degrees(phase))) for phase in metrics.phase_differences)
            confidence = self._score_above(
                max_phase,
                th.slip_phase_shift_min_deg,
                scale=max(th.slip_phase_shift_min_deg, 1.0),
            )
            return {
                "confidence": round(confidence, 4),
                "reason": (
                    f"phase shift {max_phase:.1f}\u00b0 exceeds the spindle-shift threshold "
                    f"{th.slip_phase_shift_min_deg:.1f}\u00b0"
                ),
                "source_metric": "phase_differences",
                "metric_value": round(max_phase, 4),
            }

        if key == "amp:increasing":
            if not metrics.channel_rms:
                return None
            max_rms = max(float(value) for value in metrics.channel_rms)
            confidence = self._score_above(
                max_rms,
                th.amp_normal_max,
                scale=max(th.amp_normal_max, 0.1),
            )
            return {
                "confidence": round(confidence, 4),
                "reason": (
                    f"RMS amplitude {max_rms:.4f} exceeds the nominal operating band upper bound "
                    f"{th.amp_normal_max:.4f}"
                ),
                "source_metric": "channel_rms",
                "metric_value": round(max_rms, 4),
            }

        return None
    
    def generate(
        self,
        metrics: WindowMetrics,
        context: Optional[CuttingContext] = None,
    ) -> List[str]:
        """
        Generate pattern keys from metrics.
        
        Args:
            metrics: Computed window metrics
            context: Optional cutting context (enables fault-specific patterns
                     that depend on spindle speed / tooth-passing frequency)
        
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

        # ----- Fault-specific classifiers (domain-driven) -----
        # Only run classifiers whose fault type is registered in the active
        # domain.  If the domain has no fault types (generic profile), the
        # generic classifiers above still produce useful patterns.
        active_fault_names = self._active_fault_names()

        for fault_name, classifier_fn in self._fault_classifiers.items():
            # Skip classifiers that don't belong to the active domain
            if active_fault_names and fault_name not in active_fault_names:
                continue
            try:
                patterns.extend(classifier_fn(metrics, context))
            except Exception:
                logger.debug("Fault classifier '%s' raised; skipping", fault_name)
        
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
    
    # ------------------------------------------------------------------
    # Fault-specific classifiers
    # ------------------------------------------------------------------

    def _classify_tool_breakage(
        self, metrics: WindowMetrics, context: Optional[CuttingContext] = None,
    ) -> List[str]:
        assessment = self._assess_tool_breakage(metrics, context)
        if assessment.abstain_reason:
            logger.debug("Abstaining %s due to %s", assessment.key, assessment.abstain_reason)
        patterns = list(assessment.supporting_patterns)
        if assessment.emitted:
            patterns.append(assessment.key)
        return patterns

    def _classify_chatter(
        self, metrics: WindowMetrics, context: Optional[CuttingContext] = None,
    ) -> List[str]:
        assessment = self._assess_chatter(metrics, context)
        if assessment.abstain_reason:
            logger.debug("Abstaining %s due to %s", assessment.key, assessment.abstain_reason)
        patterns = list(assessment.supporting_patterns)
        if assessment.emitted:
            patterns.append(assessment.key)
        return patterns

    def _classify_chip_adhesion(
        self, metrics: WindowMetrics, context: Optional[CuttingContext] = None,
    ) -> List[str]:
        assessment = self._assess_chip_adhesion(metrics, context)
        if assessment.abstain_reason:
            logger.debug("Abstaining %s due to %s", assessment.key, assessment.abstain_reason)
        patterns = list(assessment.supporting_patterns)
        if assessment.emitted:
            patterns.append(assessment.key)
        return patterns

    def _classify_workpiece_slip(
        self, metrics: WindowMetrics, context: Optional[CuttingContext] = None,
    ) -> List[str]:
        assessment = self._assess_workpiece_slip(metrics, context)
        if assessment.abstain_reason:
            logger.debug("Abstaining %s due to %s", assessment.key, assessment.abstain_reason)

        patterns = list(assessment.supporting_patterns)
        if assessment.emitted:
            patterns.append(assessment.key)
        return patterns

    def _assess_tool_breakage(
        self, metrics: WindowMetrics, context: Optional[CuttingContext] = None,
    ) -> HypothesisAssessment:
        supporting_patterns: List[str] = []
        th = self.thresholds
        indicators_present = 0
        hf_supported = False
        corroborated = False
        abstain_reason: Optional[str] = None

        if metrics.spectral_centroids:
            avg_centroid = float(np.mean(metrics.spectral_centroids))
            if avg_centroid > th.freq_mid_max:
                indicators_present += 1
                supporting_patterns.append("spectral:hf_burst")
                hf_supported = True

        if metrics.channel_crest_factors:
            max_crest = max(metrics.channel_crest_factors)
            if max_crest >= th.breakage_crest_factor_min:
                indicators_present += 1
                supporting_patterns.append("temporal:impulsive_burst")
                corroborated = True

        if metrics.channel_kurtosis:
            max_kurt = max(metrics.channel_kurtosis)
            if max_kurt >= th.breakage_kurtosis_min:
                indicators_present += 1
                supporting_patterns.append("kurtosis:heavy-tails")

        if metrics.spectral_bandwidths and metrics.spectral_centroids:
            avg_bw = float(np.mean(metrics.spectral_bandwidths))
            avg_cent = float(np.mean(metrics.spectral_centroids))
            if avg_cent > 1e-10:
                rel_bw = avg_bw / avg_cent
                if rel_bw > 0.5:
                    indicators_present += 1
                    supporting_patterns.append("temporal:periodicity_loss")
                    corroborated = True

        if not hf_supported:
            abstain_reason = "missing high-frequency spectral support"
        elif not corroborated:
            abstain_reason = "missing impulsive or periodicity-loss corroboration"

        return HypothesisAssessment(
            key=signature_key_for_fault_name("tool_breakage"),
            supporting_patterns=supporting_patterns,
            indicators_present=indicators_present,
            indicators_required=4,
            min_indicators_to_emit=2,
            emitted=hf_supported and corroborated and indicators_present >= 2,
            abstain_reason=abstain_reason,
        )

    def _assess_chatter(
        self, metrics: WindowMetrics, context: Optional[CuttingContext] = None,
    ) -> HypothesisAssessment:
        supporting_patterns: List[str] = []
        th = self.thresholds
        indicators_present = 0
        tooth_passing_supported = False
        abstain_reason: Optional[str] = None

        if not (context and context.tooth_passing_freq):
            return HypothesisAssessment(
                key=signature_key_for_fault_name("chatter"),
                supporting_patterns=supporting_patterns,
                indicators_present=0,
                indicators_required=4,
                min_indicators_to_emit=2,
                emitted=False,
                abstain_reason="missing tooth-passing context",
            )

        if metrics.channel_crest_factors:
            max_crest = max(metrics.channel_crest_factors)
            if max_crest >= th.crest_transient_min:
                indicators_present += 1
                supporting_patterns.append("spectral:modulated_vibration")

        if metrics.cross_correlations:
            max_corr = max(abs(c) for c in metrics.cross_correlations)
            if max_corr >= th.corr_high_min:
                indicators_present += 1

        if metrics.channel_rms:
            max_rms = max(metrics.channel_rms)
            if max_rms >= th.amp_normal_max:
                indicators_present += 1
                supporting_patterns.append("amp:loud")

        if metrics.dominant_frequencies:
            tp_freq = context.tooth_passing_freq
            for dom_f in metrics.dominant_frequencies:
                for harmonic in range(1, 4):
                    target = tp_freq * harmonic
                    if abs(dom_f - target) / (target + 1e-9) < 0.10:
                        indicators_present += 1
                        supporting_patterns.append(f"spectral:tp_harmonic_{harmonic}x")
                        tooth_passing_supported = True
                        break
                else:
                    continue
                break

        if not tooth_passing_supported:
            abstain_reason = "missing tooth-passing spectral support"

        return HypothesisAssessment(
            key=signature_key_for_fault_name("chatter"),
            supporting_patterns=supporting_patterns,
            indicators_present=indicators_present,
            indicators_required=4,
            min_indicators_to_emit=2,
            emitted=tooth_passing_supported and indicators_present >= 2,
            abstain_reason=abstain_reason,
        )

    def _assess_chip_adhesion(
        self, metrics: WindowMetrics, context: Optional[CuttingContext] = None,
    ) -> HypothesisAssessment:
        supporting_patterns: List[str] = []
        th = self.thresholds
        indicators_present = 0
        tooth_passing_supported = False
        irregular_supported = False
        abstain_reason: Optional[str] = None

        if not (context and context.tooth_passing_freq):
            return HypothesisAssessment(
                key=signature_key_for_fault_name("chip_adhesion"),
                supporting_patterns=supporting_patterns,
                indicators_present=0,
                indicators_required=4,
                min_indicators_to_emit=2,
                emitted=False,
                abstain_reason="missing tooth-passing context",
            )

        if metrics.dominant_amplitudes and len(metrics.dominant_amplitudes) >= 2:
            amps = np.array(metrics.dominant_amplitudes)
            amp_mean = float(np.mean(amps))
            if amp_mean > 1e-10:
                amp_cv = float(np.std(amps) / amp_mean)
                if amp_cv > th.chip_adhesion_harmonic_irregularity:
                    indicators_present += 1
                    supporting_patterns.append("spectral:irregular_tooth_passing")
                    irregular_supported = True

        if metrics.channel_kurtosis:
            max_kurt = max(metrics.channel_kurtosis)
            if th.kurtosis_high_min <= max_kurt < th.breakage_kurtosis_min:
                indicators_present += 1
                supporting_patterns.append("kurtosis:heavy-tails")

        if metrics.phase_differences and len(metrics.phase_differences) >= 1:
            phase_std = float(np.std(metrics.phase_differences))
            if phase_std > 0.5:
                indicators_present += 1
                supporting_patterns.append("temporal:phase_shift")

        if metrics.dominant_frequencies:
            tp_freq = context.tooth_passing_freq
            near_tp = [
                f for f in metrics.dominant_frequencies
                if abs(f - tp_freq) / (tp_freq + 1e-9) < 0.15
            ]
            if near_tp:
                indicators_present += 1
                tooth_passing_supported = True
                supporting_patterns.append("spectral:tp_harmonic_1x")

        if not tooth_passing_supported:
            abstain_reason = "missing tooth-passing spectral support"
        elif not irregular_supported:
            abstain_reason = "missing irregular tooth-passing corroboration"

        return HypothesisAssessment(
            key=signature_key_for_fault_name("chip_adhesion"),
            supporting_patterns=supporting_patterns,
            indicators_present=indicators_present,
            indicators_required=4,
            min_indicators_to_emit=2,
            emitted=tooth_passing_supported and irregular_supported and indicators_present >= 2,
            abstain_reason=abstain_reason,
        )

    def _assess_workpiece_slip(
        self, metrics: WindowMetrics, context: Optional[CuttingContext] = None,
    ) -> HypothesisAssessment:
        supporting_patterns: List[str] = []
        th = self.thresholds
        indicators_present = 0
        spindle_order_supported = False
        abstain_reason: Optional[str] = None

        if not (context and context.spindle_freq):
            abstain_reason = "missing spindle frequency context"
            return HypothesisAssessment(
                key=signature_key_for_fault_name("workpiece_slip"),
                supporting_patterns=supporting_patterns,
                indicators_present=0,
                indicators_required=4,
                min_indicators_to_emit=2,
                emitted=False,
                abstain_reason=abstain_reason,
            )

        if metrics.dominant_frequencies:
            sf = context.spindle_freq
            for dom_f in metrics.dominant_frequencies:
                if abs(dom_f - sf) / (sf + 1e-9) < 0.10:
                    indicators_present += 1
                    spindle_order_supported = True
                    supporting_patterns.append("spectral:spindle_freq_shift")
                    break

        if metrics.phase_differences:
            for phase in metrics.phase_differences:
                if abs(np.degrees(phase)) > th.slip_phase_shift_min_deg:
                    indicators_present += 1
                    supporting_patterns.append("temporal:phase_shift")
                    break

        if metrics.channel_rms:
            max_rms = max(metrics.channel_rms)
            if max_rms >= th.amp_normal_max:
                indicators_present += 1
                supporting_patterns.append("amp:loud")

        if metrics.channel_crest_factors:
            min_crest = min(metrics.channel_crest_factors)
            if min_crest < th.crest_transient_min:
                indicators_present += 1

        if not spindle_order_supported:
            abstain_reason = "missing spindle-order spectral support"

        return HypothesisAssessment(
            key=signature_key_for_fault_name("workpiece_slip"),
            supporting_patterns=supporting_patterns,
            indicators_present=indicators_present,
            indicators_required=4,
            min_indicators_to_emit=2,
            emitted=spindle_order_supported and indicators_present >= 2,
            abstain_reason=abstain_reason,
        )

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

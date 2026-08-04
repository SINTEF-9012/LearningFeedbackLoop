"""
Memory Event Bridge - Connect feature stream to memory system.

# ===========================================================================
# [PROTOTYPE_LLM_MEMORY_V1] - Simple bridge implementation
# NOTE: This event generator will likely be replaced. Keeping minimal.
# ===========================================================================

This module subscribes to the feature bus and triggers memory processing
when significant patterns are detected.

Usage:
    from backend.agents.memory.feature_stream_bridge import start_memory_processor, stop_memory_processor
    
    # Start processing (call after memory system is initialized)
    await start_memory_processor()
    
    # Stop when shutting down
    await stop_memory_processor()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Protocol
from datetime import datetime

import numpy as np

from backend.events import subscribe_features, bus
from backend.ingestion.schema import FrameEnvelope, envelope_to_dict

from ..core.schemas import PatternKey, PatternType, TimeRange
from ..core.batch_context import extract_batch_context
from ..core.context import CuttingContext, extract_context_from_metadata
from ..core.metrics import WindowMetrics, MetricsComputer
from ..patterns.generator import PatternGenerator
from ..patterns.registry import detect_patterns
from ..patterns.signatures import signature_key_for_fault_name
from .cycle_tracker import get_cycle_tracker
from .orchestrator import MemoryEvent, MemoryEventResult, get_orchestrator
from .init import is_initialized

logger = logging.getLogger(__name__)

_FRAME_TIMING_KEYS = {"t", "i", "fs", "t0", "t1", "i0", "i1", "ts_unix", "timestamp"}
_LIVE_CONTEXT_CHANNELS = {
    "spindle_speed": ("Spindle_Speed_Actual", "Spindle_Speed_Commanded", "spindle_speed", "spindle"),
    "feed_rate": ("Feed_Rate_Actual", "Feed_Rate_Commanded", "feed_rate", "feed"),
    "operating_regime": ("Operation_Mode", "operating_regime", "regime"),
}
_ANONYMOUS_CHANNEL_RE = re.compile(r"ch\d+", re.IGNORECASE)
_RATIO_NOISE_FLOOR = 1e-6


def _sanitize_pattern_channel_name(name: str) -> str:
    """Normalise channel names for stable pattern keys."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")
    return cleaned or "unknown"


def _build_ratio_pattern(channel_a: str, peak_a: float, channel_b: str, peak_b: float) -> Optional[str]:
    """Return a bucketed ratio pattern when both channels are meaningful and audible."""
    if _ANONYMOUS_CHANNEL_RE.fullmatch(channel_a) or _ANONYMOUS_CHANNEL_RE.fullmatch(channel_b):
        return None
    if abs(peak_a) < _RATIO_NOISE_FLOOR or abs(peak_b) < _RATIO_NOISE_FLOOR:
        return None

    ratio = abs(peak_a) / max(abs(peak_b), _RATIO_NOISE_FLOOR)
    if ratio < 0.5:
        bucket = "<0.5"
    elif ratio < 2.0:
        bucket = "0.5-2"
    elif ratio < 5.0:
        bucket = "2-5"
    else:
        bucket = ">5"

    return (
        f"RATIO_{_sanitize_pattern_channel_name(channel_a)}_"
        f"{_sanitize_pattern_channel_name(channel_b)}:{bucket}"
    )

# Lazy import to avoid hard dependency — SINDIT is optional.
_sindit_provider = None  # Optional[SinditContextProvider]


def set_sindit_provider(provider: Any) -> None:
    """Inject a ``SinditContextProvider`` for live machine-context enrichment.

    Call once at startup *after* initialising the SINDIT client.  When set,
    :func:`create_memory_event_from_feature` will auto-fetch live cutting
    parameters from the SINDIT digital-twin API.
    """
    global _sindit_provider
    _sindit_provider = provider
    logger.info("SINDIT context provider registered with memory bridge")


# ============================================================================
# Global State
# ============================================================================

_processor_task: Optional[asyncio.Task] = None
_running: bool = False
_pattern_generator: Optional[PatternGenerator] = None


# ==========================================================================
# Provider hook (bridge-side)
# ==========================================================================


class FeatureExtractor(Protocol):
    """Optional bridge-side extractor to provide metrics/patterns for events."""

    def extract(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dict that may include metrics/patterns/external_signals."""


class DefaultFeatureExtractor:
    """Enriched extractor for scaffolding + tests.

    Computes time-domain features (RMS, crest, spikes) **and** spectral
    features relevant to the four machining fault types:
      - Tool breakage:    high-frequency energy ratio, impulsive burst
      - Chatter:          modulation depth, amplitude growth, TP harmonics
      - Chip adhesion:    harmonic amplitude irregularity
      - Workpiece slip:   amplitude/phase at spindle frequency
    """

    _RESERVED_KEYS = {"type", "session_id", "position", "frame", "payload", "i_center", "t_center"}

    def extract(self, event: Dict[str, Any]) -> Dict[str, Any]:
        if "metrics" in event or "patterns" in event:
            return {}

        frame = event.get("frame")
        if not isinstance(frame, dict):
            return {}

        # Extract channel samples from frame (ignore timing/position keys)
        channel_keys = [
            k for k in frame.keys()
            if k not in _FRAME_TIMING_KEYS
        ]
        if not channel_keys:
            return {}

        # Raw tag_sample payloads are single instants, not analysis windows.
        # Generating spectral and ratio patterns from them creates the generic
        # ch0/ch1 alerts seen in the casedata demo logs.
        if event.get("kind") == "tag_sample":
            max_samples = 0
            for ch in channel_keys:
                values = frame.get(ch)
                if isinstance(values, (list, tuple, np.ndarray)):
                    max_samples = max(max_samples, int(np.atleast_1d(values).size))
                elif isinstance(values, (int, float, np.number)):
                    max_samples = max(max_samples, 1)
            if max_samples < 4:
                return {}

        fs = float(event.get("fs", frame.get("fs", 10000.0)))

        channel_means: List[float] = []
        channel_stds: List[float] = []
        channel_rms: List[float] = []
        channel_peaks: List[float] = []
        channel_crest: List[float] = []
        total_energy = 0.0

        # Spectral accumulators
        channel_dominant_freqs: List[float] = []
        channel_dominant_amps: List[float] = []
        channel_spectral_centroids: List[float] = []
        channel_spectral_bws: List[float] = []
        channel_kurtosis: List[float] = []

        # Heuristic patterns that match the scorer's existing rule substrings.
        derived_patterns: List[str] = []
        spike_count_total = 0
        hf_energy_ratio = 0.0  # fraction of energy > 500 Hz

        arrays: Dict[str, np.ndarray] = {}

        for ch in channel_keys:
            values = frame.get(ch)
            if values is None:
                arr = np.asarray([], dtype=float)
            else:
                arr = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
            arrays[ch] = arr

            if arr.size == 0:
                channel_means.append(0.0)
                channel_stds.append(0.0)
                channel_rms.append(0.0)
                channel_peaks.append(0.0)
                channel_crest.append(0.0)
                channel_kurtosis.append(0.0)
                channel_dominant_freqs.append(0.0)
                channel_dominant_amps.append(0.0)
                channel_spectral_centroids.append(0.0)
                channel_spectral_bws.append(0.0)
                continue

            # Spectral features need at least a handful of samples; for scalar
            # or very short payloads (e.g. demo-injected events carrying a
            # single value per channel) skip the FFT and emit time-domain
            # features only.
            if arr.size < 4:
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                rms = float(np.sqrt(np.mean(arr ** 2)))
                peak = float(np.max(np.abs(arr)))
                crest = peak / (rms + 1e-12)
                channel_means.append(mean)
                channel_stds.append(std)
                channel_rms.append(rms)
                channel_peaks.append(peak)
                channel_crest.append(crest)
                channel_kurtosis.append(0.0)
                channel_dominant_freqs.append(0.0)
                channel_dominant_amps.append(0.0)
                channel_spectral_centroids.append(0.0)
                channel_spectral_bws.append(0.0)
                total_energy += float(np.sum(arr ** 2))
                continue

            mean = float(np.mean(arr))
            std = float(np.std(arr))
            rms = float(np.sqrt(np.mean(arr ** 2)))
            peak = float(np.max(np.abs(arr)))
            energy = float(np.sum(arr ** 2))
            crest = peak / (rms + 1e-12)

            # Kurtosis (excess)
            if std > 1e-12:
                normalised = (arr - mean) / std
                kurt = float(np.mean(normalised ** 4) - 3.0)
            else:
                kurt = 0.0

            # Spike detector: samples > mean + 3*std
            if std > 1e-12:
                spike_count_total += int(np.sum(np.abs(arr - mean) > (3.0 * std)))

            # ---- Spectral features ----
            n_fft = int(2 ** np.ceil(np.log2(max(arr.size, 64))))
            spec = np.abs(np.fft.rfft(arr, n=n_fft))
            freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
            power = spec ** 2

            # Dominant frequency (skip DC)
            peak_idx = int(np.argmax(spec[1:])) + 1
            dom_freq = float(freqs[peak_idx])
            dom_amp = float(spec[peak_idx])

            # Spectral centroid & bandwidth
            total_pow = float(np.sum(power) + 1e-10)
            centroid = float(np.sum(freqs * power) / total_pow)
            bw = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_pow))

            # High-frequency energy ratio — Nyquist-aware cutoff
            nyquist = fs / 2.0
            hf_cutoff = min(500.0, nyquist * 0.5)  # scale with Nyquist; max 500 Hz
            hf_mask = freqs > hf_cutoff
            if np.any(hf_mask):
                hf_energy_ratio = max(hf_energy_ratio, float(np.sum(power[hf_mask]) / total_pow))

            channel_means.append(mean)
            channel_stds.append(std)
            channel_rms.append(rms)
            channel_peaks.append(peak)
            channel_crest.append(float(crest))
            channel_kurtosis.append(kurt)
            channel_dominant_freqs.append(dom_freq)
            channel_dominant_amps.append(dom_amp)
            channel_spectral_centroids.append(centroid)
            channel_spectral_bws.append(bw)
            total_energy += energy

        # ================================================================
        # Pattern heuristics (generic)
        # ================================================================
        crest_max = max(channel_crest) if channel_crest else 0.0
        if crest_max >= 6.0:
            derived_patterns.append("ANOMALY_HIGH")

        if spike_count_total > 10:
            derived_patterns.append("SPIKE_RATE:>10")

        if len(channel_peaks) >= 2:
            ratio_pattern = _build_ratio_pattern(
                channel_keys[0],
                channel_peaks[0],
                channel_keys[1],
                channel_peaks[1],
            )
            if ratio_pattern:
                derived_patterns.append(ratio_pattern)

        # ================================================================
        # Fault-specific pattern heuristics
        # ================================================================

        # --- Tool breakage: HF burst + impulsive + loss of periodicity ---
        breakage_hits = 0
        if hf_energy_ratio > 0.4:
            derived_patterns.append("spectral:hf_burst")
            breakage_hits += 1
        if crest_max >= 8.0:
            derived_patterns.append("temporal:impulsive_burst")
            breakage_hits += 1
        kurt_max = max(channel_kurtosis) if channel_kurtosis else 0.0
        if kurt_max >= 6.0:
            breakage_hits += 1
        avg_centroid = float(np.mean(channel_spectral_centroids)) if channel_spectral_centroids else 0.0
        avg_bw = float(np.mean(channel_spectral_bws)) if channel_spectral_bws else 0.0
        if avg_centroid > 1e-10 and (avg_bw / avg_centroid) > 0.5:
            derived_patterns.append("temporal:periodicity_loss")
            breakage_hits += 1
        if breakage_hits >= 2:
            derived_patterns.append(signature_key_for_fault_name("tool_breakage"))

        # --- Chatter: modulated vibration + increased amplitude ---
        chatter_hits = 0
        if crest_max >= 4.0:
            derived_patterns.append("spectral:modulated_vibration")
            chatter_hits += 1
        rms_max = max(channel_rms) if channel_rms else 0.0
        if rms_max >= 0.5:
            derived_patterns.append("amp:increasing")
            chatter_hits += 1
        if chatter_hits >= 2:
            derived_patterns.append(signature_key_for_fault_name("chatter"))

        # --- Chip adhesion / BUE: irregular harmonic amplitudes ---
        if channel_dominant_amps and len(channel_dominant_amps) >= 2:
            amps_arr = np.array(channel_dominant_amps)
            amp_mean = float(np.mean(amps_arr))
            if amp_mean > 1e-10:
                amp_cv = float(np.std(amps_arr) / amp_mean)
                if amp_cv > 0.3:
                    derived_patterns.append("spectral:irregular_tooth_passing")
                    if 3.0 <= kurt_max < 6.0:
                        derived_patterns.append(signature_key_for_fault_name("chip_adhesion"))

        # --- Workpiece slip: low crest + phase offset + high amplitude ---
        #     (limited without CuttingContext; fuller detection in PatternGenerator)
        crest_min = min(channel_crest) if channel_crest else 0.0
        if crest_min < 4.0 and rms_max >= 0.5:
            derived_patterns.append("spectral:spindle_freq_shift")

        # ================================================================
        # Assemble output
        # ================================================================
        metrics = {
            "channel_means": channel_means,
            "channel_stds": channel_stds,
            "channel_rms": channel_rms,
            "channel_peaks": channel_peaks,
            "channel_crest_factors": channel_crest,
            "channel_kurtosis": channel_kurtosis,
            "dominant_frequencies": channel_dominant_freqs,
            "dominant_amplitudes": channel_dominant_amps,
            "spectral_centroids": channel_spectral_centroids,
            "spectral_bandwidths": channel_spectral_bws,
            "total_energy": float(total_energy),
            "hf_energy_ratio": hf_energy_ratio,
        }

        out: Dict[str, Any] = {"metrics": metrics}
        if derived_patterns:
            out["patterns"] = derived_patterns
        return out


def _merge_nested_metadata(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nested payload metadata into cached session metadata."""
    merged = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_metadata(merged[key], value)
        else:
            merged[key] = value
    return merged


_MACHINE_PROFILES_PATH = os.environ.get(
    "MACHINE_PROFILES_PATH",
    str(Path(__file__).resolve().parents[3] / "data" / "machine_profiles.json"),
)
# Fields the static machine profile may supply (only filled when missing).
_MACHINE_PROFILE_FIELDS = ("workpiece_material", "operating_regime", "feed_rate")
_machine_profiles_cache: Optional[Dict[str, Dict[str, Any]]] = None


def _load_machine_profiles() -> Dict[str, Dict[str, Any]]:
    """Load per-machine nominal twin profiles (cached). Never raises."""
    global _machine_profiles_cache
    if _machine_profiles_cache is not None:
        return _machine_profiles_cache
    profiles: Dict[str, Dict[str, Any]] = {}
    try:
        with open(_MACHINE_PROFILES_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        entries = raw.get("profiles") if isinstance(raw, dict) else None
        if isinstance(entries, dict):
            for key, val in entries.items():
                if isinstance(key, str) and isinstance(val, dict):
                    profiles[key.strip().lower()] = val
    except FileNotFoundError:
        pass
    except Exception:
        logger.debug("failed to load machine profiles from %s", _MACHINE_PROFILES_PATH, exc_info=True)
    _machine_profiles_cache = profiles
    return profiles


def _apply_machine_profile(cutting_context: "CuttingContext") -> "CuttingContext":
    """Fill missing material/regime/feed from the machine's static twin profile.

    Matches the profile whose key appears (case-insensitive) in the context's
    ``machine_id``. Only fields that are currently ``None`` are filled, and the
    filled field names are recorded in ``extra['nominal_context_fields']`` so
    downstream surfaces can tell nominal profile values from live readings.
    """
    profiles = _load_machine_profiles()
    if not profiles:
        return cutting_context
    machine_id = (cutting_context.machine_id or "").strip().lower()
    if not machine_id:
        return cutting_context
    profile = next((v for k, v in profiles.items() if k and k in machine_id), None)
    if not profile:
        return cutting_context

    ctx_dict = cutting_context.model_dump()
    extra = dict(ctx_dict.get("extra") or {})
    filled = list(extra.get("nominal_context_fields") or [])
    changed = False
    for field in _MACHINE_PROFILE_FIELDS:
        if ctx_dict.get(field) is None and profile.get(field) not in (None, ""):
            ctx_dict[field] = profile[field]
            if field not in filled:
                filled.append(field)
            changed = True
    if not changed:
        return cutting_context
    extra["nominal_context_fields"] = filled
    extra["context_profile_source"] = "machine_twin_profile"
    ctx_dict["extra"] = extra
    try:
        return CuttingContext(**ctx_dict)
    except Exception:
        return cutting_context


def _extract_live_casedata_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Lift live casedata values from the current payload into CuttingContext fields."""
    context: Dict[str, Any] = {}
    sources: List[Dict[str, Any]] = []
    for candidate in (payload.get("metadata"), payload.get("signals"), payload.get("frame")):
        if isinstance(candidate, dict):
            sources.append(candidate)

    for context_key, aliases in _LIVE_CONTEXT_CHANNELS.items():
        for source in sources:
            for alias in aliases:
                value = source.get(alias)
                if value is None:
                    continue
                if isinstance(value, str):
                    value = value.strip()
                    if not value:
                        continue
                    context[context_key] = value
                    break
                if isinstance(value, (int, float, np.number)):
                    numeric = float(value)
                    if np.isfinite(numeric):
                        context[context_key] = numeric
                        break
            if context_key in context:
                break

    return context


def _merge_session_metadata(existing: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Combine cached session metadata with payload metadata and live casedata context."""
    merged = dict(existing)

    payload_meta = payload.get("metadata")
    if isinstance(payload_meta, dict) and payload_meta:
        merged = _merge_nested_metadata(merged, payload_meta)

    live_context = _extract_live_casedata_context(payload)
    if live_context:
        casedata = dict(merged.get("casedata") or {})
        cutting_context = dict(casedata.get("cutting_context") or {})
        cutting_context.update(live_context)
        casedata["cutting_context"] = cutting_context
        merged["casedata"] = casedata

    if "source" not in merged and payload.get("source"):
        merged["source"] = payload["source"]

    # Resolve machine / tool / dataset identity from the machine_id and live signals
    # (dataset_id, machine_family, machine_uri, and tool context). Best-effort: leaves
    # the metadata unchanged when nothing can be resolved (e.g. no machine_id).
    try:
        from ..sindit.runtime_context import resolve_runtime_metadata
        merged = resolve_runtime_metadata(merged, payload)
    except Exception:
        logger.debug("runtime metadata resolution skipped", exc_info=True)

    return merged


def _coerce_feature_payload(payload: Any) -> Dict[str, Any]:
    """Normalize bus payloads to the legacy dict shape expected by the bridge."""

    if isinstance(payload, FrameEnvelope):
        data = envelope_to_dict(payload)
        signals = data.get("signals")
        if isinstance(signals, dict) and "frame" not in data:
            frame: Dict[str, Any] = {
                "t": float(data.get("position", 0)),
                "i": int(data.get("position", 0)),
                "fs": float(data.get("fs", 1.0)),
            }
            frame.update(signals)
            data["frame"] = frame
        return data
    if isinstance(payload, dict):
        return dict(payload)
    logger.debug("Skipping unsupported feature payload type: %s", type(payload).__name__)
    return {}


_feature_extractor: FeatureExtractor = DefaultFeatureExtractor()


def set_feature_extractor(extractor: FeatureExtractor) -> None:
    """Inject a custom extractor (for richer metrics/patterns)."""
    global _feature_extractor
    _feature_extractor = extractor


def _augment_payload_with_provider(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pass-through if precomputed fields exist; otherwise compute via provider."""
    if "metrics" in payload or "patterns" in payload:
        return payload

    try:
        provided = _feature_extractor.extract(payload)
    except Exception:
        logger.warning("FeatureExtractor failed; continuing without extracted fields", exc_info=True)
        return payload

    if not isinstance(provided, dict):
        return payload

    merged = dict(payload)
    if "metrics" not in merged and provided.get("metrics") is not None:
        merged["metrics"] = provided["metrics"]
    if "patterns" not in merged and provided.get("patterns"):
        merged["patterns"] = provided["patterns"]
    if provided.get("external_signals"):
        merged["external_signals"] = provided["external_signals"]
    return merged


def _merge_stoppage_signals(event: MemoryEvent, pred: Any, gap_s: float) -> None:
    """Expose stoppage predictor output in both flat and nested forms."""

    event.external_signals["stoppage_probability"] = float(pred.probability)
    event.external_signals["stoppage_label"] = str(pred.label)
    event.external_signals["stoppage_eta_s"] = float(gap_s)
    event.external_signals["stoppage_predictor"] = {
        "probability": pred.probability,
        "label": pred.label,
        "is_stop_predicted": pred.is_stop_predicted,
        "gap_s": gap_s,
    }


def _should_apply_stoppage_predictor(session_meta: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    """Return whether the stoppage predictor should run for the current payload."""

    payload_meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    for source in (session_meta.get("source"), payload_meta.get("source"), payload.get("source")):
        if not isinstance(source, str):
            continue
        normalized = source.strip().lower()
        if not normalized or normalized == "unknown":
            continue
        return normalized != "simulated_casedata"
    return True


# ============================================================================
# Feature to MemoryEvent Conversion
# ============================================================================

def _extract_time_range(payload: Dict[str, Any], session_meta: Dict[str, Any]) -> TimeRange:
    """Extract TimeRange from feature payload."""
    fs = session_meta.get("fs", session_meta.get("sample_frequency", None))
    if fs is None:
        fs = 1000.0
        logger.warning(
            "No sample rate in session metadata — falling back to %.0f Hz. "
            "Set 'fs' or 'sample_frequency' in session metadata for correct timing.",
            fs,
        )
    else:
        fs = float(fs)

    # Derive window_size: prefer window_seconds × fs, fall back to explicit samples
    window_seconds = payload.get("window_seconds")
    if window_seconds is not None:
        window_size = int(float(window_seconds) * fs)
    else:
        window_size = payload.get("window_size", int(fs * 2))  # default ~2 s
    position = payload.get("position", 0)
    
    i0 = max(0, position - window_size)
    i1 = position
    t0 = i0 / fs
    t1 = i1 / fs
    
    return TimeRange(i0=i0, i1=i1, t0=t0, t1=t1, fs=fs)


def _convert_patterns(
    pattern_strings: List[Any],
    pattern_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[PatternKey]:
    """Convert pattern strings to PatternKey objects."""
    patterns = []
    for p in pattern_strings:
        if isinstance(p, PatternKey):
            patterns.append(p)
            continue
        if isinstance(p, dict):
            try:
                patterns.append(PatternKey(**p))
                continue
            except Exception:
                pass

        raw_key = str(p).strip()
        if not raw_key:
            continue

        # Parse pattern string format: "category:value" or "category:subcategory:value"
        parts = raw_key.split(":")
        if len(parts) >= 2:
            category = parts[0].upper()
            key = raw_key
            
            # Map category to PatternType
            if category in ("FREQ", "SPECTRAL", "PSD"):
                ptype = PatternType.SPECTRAL_PEAK
            elif category in ("RATIO", "AMP"):
                ptype = PatternType.RATIO
            elif category in ("ANOMALY", "OUTLIER"):
                ptype = PatternType.ANOMALY
            elif category in ("CLUSTER",):
                ptype = PatternType.CLUSTER
            elif category in ("FAULT",):
                ptype = PatternType.FAULT
            elif category in ("HYPOTHESIS", "SIGNATURE"):
                ptype = PatternType.CUSTOM
            elif category in ("TEMPORAL",):
                ptype = PatternType.SPIKE_RATE
            else:
                ptype = PatternType.CUSTOM
            
            # Extract fault_type for FAULT patterns
            fault_type = None
            if category in ("FAULT", "HYPOTHESIS") and len(parts) >= 2:
                fault_type = parts[1]

            metadata = (pattern_metadata or {}).get(key, {})
            confidence = float(metadata.get("confidence", 1.0)) if metadata else 1.0
            
            patterns.append(PatternKey(
                pattern_type=ptype,
                key=key,
                fault_type=fault_type,
                confidence=confidence,
                source_metric=metadata.get("source_metric") if metadata else None,
                additional=metadata or None,
            ))
        else:
            metadata = (pattern_metadata or {}).get(raw_key, {})
            confidence = float(metadata.get("confidence", 1.0)) if metadata else 1.0
            patterns.append(PatternKey(
                pattern_type=PatternType.CUSTOM,
                key=raw_key,
                confidence=confidence,
                source_metric=metadata.get("source_metric") if metadata else None,
                additional=metadata or None,
            ))
    
    return patterns


def create_memory_event_from_feature(
    session_id: str,
    payload: Dict[str, Any],
    session_meta: Dict[str, Any],
) -> MemoryEvent:
    """
    Convert a feature bus payload to a MemoryEvent.
    
    If a ``SinditContextProvider`` has been set (via :func:`set_sindit_provider`),
    missing cutting-context fields are automatically populated from the SINDIT
    digital-twin API.  When SINDIT is unavailable the system falls back to
    whatever metadata is already in *session_meta*.
    
    Args:
        session_id: Session identifier
        payload: Feature payload from bus
        session_meta: Session metadata (for context extraction)
    
    Returns:
        MemoryEvent ready for processing
    """
    global _pattern_generator
    
    if _pattern_generator is None:
        _pattern_generator = PatternGenerator()
    
    # Extract time range
    time_range = _extract_time_range(payload, session_meta)
    
    # Extract cutting context from metadata
    cutting_context = None
    if session_meta:
        cutting_context = extract_context_from_metadata(session_meta)

    batch = extract_batch_context(payload, payload.get("metadata"), session_meta)
    
    # --- SINDIT enrichment (non-blocking, best-effort) -----------------
    # Two stages: first the machine asset, then (if a tool IRI is known) the tool's
    # master properties (diameter, teeth, ...).
    if cutting_context is not None and _sindit_provider is not None:
        try:
            import asyncio
            ctx_dict = cutting_context.model_dump()
            # Determine the asset IRI from session metadata (if available)
            asset_iri = session_meta.get("sindit_asset_iri") or session_meta.get("machine_iri")
            extra = ctx_dict.get("extra") if isinstance(ctx_dict.get("extra"), dict) else {}
            tool_iri = extra.get("sindit_tool_iri") if extra else None
            provider = _sindit_provider

            async def _enrich_machine_then_tool():
                out = await provider.enrich_context(ctx_dict, asset_iri=asset_iri)
                if tool_iri and hasattr(provider, "enrich_tool_properties"):
                    out = await provider.enrich_tool_properties(out, tool_iri=tool_iri)
                return out

            try:
                asyncio.get_running_loop()
                loop_running = True
            except RuntimeError:
                loop_running = False  # no loop running in this thread (robust vs. get_event_loop)
            if loop_running:
                # Run off the live event loop so we don't block it.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    enriched = pool.submit(
                        asyncio.run, _enrich_machine_then_tool(),
                    ).result(timeout=6.0)
            else:
                enriched = asyncio.run(_enrich_machine_then_tool())
            # Rebuild CuttingContext from the enriched dict
            cutting_context = CuttingContext(**enriched)
        except Exception:
            logger.debug("SINDIT enrichment skipped (unavailable or timeout)", exc_info=True)

    # --- Machine twin-profile fallback (static, best-effort) -----------
    # The live twin streams spindle/machine-state but not the workpiece
    # material / nominal feed / regime for these machines. Fill only the
    # fields still missing from the machine's static profile so the process
    # overview is not left blank on live sessions. Live/SINDIT values always
    # win — this never overwrites a populated field.
    if cutting_context is not None:
        try:
            cutting_context = _apply_machine_profile(cutting_context)
        except Exception:
            logger.debug("machine-profile fallback skipped", exc_info=True)
    # -------------------------------------------------------------------
    
    # Get metrics from payload (if present)
    metrics = None
    if "metrics" in payload:
        # Filter to only kwargs that WindowMetrics accepts. Upstream
        # extractors (e.g. DefaultFeatureExtractor) emit diagnostic keys
        # like ``hf_energy_ratio`` that are not dataclass fields; dropping
        # them here keeps the bridge robust to extractor evolution.
        import dataclasses as _dc
        accepted = {f.name for f in _dc.fields(WindowMetrics)}
        raw_metrics = payload.get("metrics") or {}
        clean_metrics = {k: v for k, v in raw_metrics.items() if k in accepted}
        metrics = WindowMetrics(**clean_metrics)

    raw_metrics = None
    if isinstance(payload.get("raw_metrics"), dict):
        raw_metrics = {
            str(k): float(v)
            for k, v in payload["raw_metrics"].items()
            if isinstance(v, (int, float))
        }
    
    # Generate patterns from metrics
    patterns = []
    if metrics:
        pattern_strings = _pattern_generator.generate(metrics, context=cutting_context)
        pattern_metadata = _pattern_generator.get_pattern_metadata(metrics, context=cutting_context)
        patterns = _convert_patterns(pattern_strings, pattern_metadata=pattern_metadata)
    elif raw_metrics:
        detected = detect_patterns(raw_metrics, include_details=True)
        pattern_strings = [
            str(key).strip()
            for key in (detected.get("fired") or [])
            if str(key).strip()
        ]
        pattern_metadata = {
            str(item.get("name") or "").strip(): {
                key: value
                for key, value in item.items()
                if key != "name" and value is not None
            }
            for item in (detected.get("details") or [])
            if str(item.get("name") or "").strip()
        }
        patterns = _convert_patterns(pattern_strings, pattern_metadata=pattern_metadata)
    
    # Add any patterns already in payload
    if "patterns" in payload:
        patterns.extend(_convert_patterns(payload["patterns"]))
    
    # Extract external signals
    external_signals = {}
    if "anomaly_score" in payload:
        external_signals["online_agent"] = {"anomaly_score": payload["anomaly_score"]}
    if "prediction" in payload:
        external_signals["prediction_model"] = payload["prediction"]
    if "external_signals" in payload and isinstance(payload["external_signals"], dict):
        external_signals.update(payload["external_signals"])
    if "harmonic_thresholds" in payload and isinstance(payload["harmonic_thresholds"], dict):
        thresholds = payload["harmonic_thresholds"]
        context_threshold = thresholds.get("context")
        if isinstance(context_threshold, (int, float)):
            external_signals.setdefault("harmonic_context_threshold", float(context_threshold))
        pair_threshold = thresholds.get("pair")
        if isinstance(pair_threshold, (int, float)):
            external_signals.setdefault("harmonic_pair_threshold", float(pair_threshold))
    
    # Extract channels
    channels = list(payload.get("slice", {}).keys()) if "slice" in payload else []
    if not channels and isinstance(payload.get("signals"), dict):
        channels = list(payload["signals"].keys())
    if not channels and isinstance(payload.get("frame"), dict):
        channels = [k for k in payload["frame"].keys() if k not in _FRAME_TIMING_KEYS]
    
    return MemoryEvent(
        session_id=session_id,
        time_range=time_range,
        patterns=patterns,
        metrics=metrics,
        cutting_context=cutting_context,
        external_signals=external_signals,
        channels=channels,
        raw_metrics=raw_metrics,
        batch=batch,
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
    )


# ============================================================================
# Feature Bus Processor
# ============================================================================

async def _process_features():
    """Background task that processes features from the bus."""
    global _running
    
    logger.info("Starting memory feature processor")
    queue = subscribe_features()  # Subscribe to global features channel
    
    # Simple session metadata cache (in production, get from session store)
    session_meta_cache: Dict[str, Dict[str, Any]] = {}
    
    while _running:
        try:
            # Wait for next feature with timeout
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            payload = _coerce_feature_payload(payload)
            if not payload:
                continue
            
            session_id = payload.get("session_id")
            if not session_id:
                continue

            # Avoid mutating the bus payload object that other subscribers might read.
            payload = _augment_payload_with_provider(payload)
            
            # Get session metadata (simplified - in production, fetch from session store)
            session_meta = _merge_session_metadata(session_meta_cache.get(session_id, {}), payload)
            payload_fs = payload.get("fs")
            if payload_fs is None and isinstance(payload.get("frame"), dict):
                payload_fs = payload["frame"].get("fs")
            if payload_fs is None and isinstance(payload.get("metadata"), dict):
                payload_fs = payload["metadata"].get("sample_frequency") or payload["metadata"].get("fs")

            if payload_fs is not None:
                try:
                    fs_value = float(payload_fs)
                    session_meta["fs"] = fs_value
                    session_meta.setdefault("sample_frequency", fs_value)
                except Exception:
                    logger.debug("Invalid payload fs for session %s: %r", session_id, payload_fs)
            elif "fs" not in session_meta:
                session_meta["fs"] = 10000.0  # Default
            session_meta_cache[session_id] = session_meta
            
            # Create memory event
            event = create_memory_event_from_feature(session_id, payload, session_meta)

            if is_initialized():
                cycle_end = get_cycle_tracker().observe(
                    session_id=session_id,
                    metadata=session_meta,
                    ts=float(event.time_range.t1),
                )
                if cycle_end is not None:
                    orchestrator = get_orchestrator()
                    if orchestrator is not None:
                        asyncio.create_task(orchestrator.attach_passive_cycle_outcome(cycle_end))

            # --- Stoppage prediction hook ---
            try:
                from ..processing.stoppage_predictor import get_predictor
                if _should_apply_stoppage_predictor(session_meta, payload):
                    predictor = get_predictor()
                    # Build features dict from payload metrics/frame
                    stop_features = {}
                    if "metrics" in payload and isinstance(payload["metrics"], dict):
                        stop_features.update(payload["metrics"])
                    if "frame" in payload and isinstance(payload["frame"], dict):
                        for k, v in payload["frame"].items():
                            if isinstance(v, (int, float)):
                                stop_features[k] = v
                    if stop_features:
                        pred = predictor.predict(stop_features)
                        if pred.is_stop_predicted:
                            event.patterns.extend(pred.pattern_keys)
                        _merge_stoppage_signals(event, pred, predictor.prediction_gap_s)
            except Exception:
                pass  # Predictor not available or features incompatible
            
            # Skip if no patterns detected (nothing significant)
            if not event.patterns and not event.external_signals:
                continue
            
            # Process through orchestrator
            if is_initialized():
                orchestrator = get_orchestrator()
                result = await orchestrator.process_event(event)
                
                if result.significant:
                    logger.info(
                        "Significant event detected: session=%s score=%.2f action=%s",
                        session_id, result.significance_score, result.action.value
                    )
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Error processing feature: %s", e, exc_info=True)
    
    logger.info("Memory feature processor stopped")


async def start_memory_processor():
    """Start the background memory processor task."""
    global _processor_task, _running
    
    if _processor_task and not _processor_task.done():
        logger.warning("Memory processor already running")
        return
    
    if not is_initialized():
        logger.warning("Memory system not initialized, skipping processor start")
        return
    
    _running = True
    _processor_task = asyncio.create_task(_process_features())
    logger.info("Memory processor started")


async def stop_memory_processor():
    """Stop the background memory processor task."""
    global _processor_task, _running
    
    _running = False
    
    if _processor_task:
        _processor_task.cancel()
        try:
            await _processor_task
        except asyncio.CancelledError:
            pass
        _processor_task = None
    
    logger.info("Memory processor stopped")

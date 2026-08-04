# inference_streamer.py
"""
Continuous windowed inference streamer.

Modelled after fft_streamer.py — hooks into the session's position tracking,
waits for enough new samples, computes features from a sliding window, runs
the SeedModel (Isolation Forest + LOF), and broadcasts per-model anomaly
scores to inference WS subscribers.

The output is a time-series of anomaly scores suitable for real-time plotting
in the Inference tab.
"""

from __future__ import annotations

import asyncio
from collections import deque
import logging
import time
from pathlib import Path
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .agents.model_confidence import current_model_confidence
from .agents.domain_config import (
    ChannelRole,
    DomainConfig,
    FaultTypeConfig,
    get_active_domain,
)
from .events import publish_feature
from .metadata_utils import get_sample_frequency

logger = logging.getLogger(__name__)


ROLE_BINDING_ORDER = (
    ChannelRole.PRIMARY_POWER,
    ChannelRole.SECONDARY_POWER,
    ChannelRole.PRIMARY_VIBRATION,
    ChannelRole.CHATTER_AMPLITUDE,
    ChannelRole.ACTIVE_POWER,
    ChannelRole.SPINDLE_SPEED,
    ChannelRole.FEED_RATE,
)


def _resolve_domain_bindings(
    domain: DomainConfig,
    available_channels: List[str],
) -> Dict[str, str]:
    """Resolve semantic roles to actual channels for the current window/session."""
    return {
        str(role): domain.resolve_channel(role, available_channels)
        for role in ROLE_BINDING_ORDER
    }


def _metadata_num_teeth(metadata: Dict[str, Any]) -> Optional[float]:
    """Resolve flute count from session metadata when available."""
    candidates: list[Any] = []

    cutting_context = metadata.get("cutting_context")
    if isinstance(cutting_context, dict):
        candidates.append(cutting_context.get("num_teeth"))

    casedata = metadata.get("casedata")
    if isinstance(casedata, dict):
        nested_context = casedata.get("cutting_context")
        if isinstance(nested_context, dict):
            candidates.append(nested_context.get("num_teeth"))

    for value in candidates:
        if isinstance(value, (int, float)):
            numeric = float(value)
        elif isinstance(value, str):
            try:
                numeric = float(value.strip())
            except ValueError:
                continue
        else:
            continue
        if np.isfinite(numeric) and numeric > 0:
            return numeric
    return None


def _merge_cutting_context(target: Dict[str, Any], candidate: Any) -> None:
    if not isinstance(candidate, dict):
        return

    for key in (
        "machine_id",
        "tool_id",
        "tool_type",
        "tool_diameter",
        "num_teeth",
        "tool_length",
        "tool_material",
        "spindle_speed",
        "feed_rate",
    ):
        value = candidate.get(key)
        if value not in (None, ""):
            target[key] = value

    extra = candidate.get("extra")
    if isinstance(extra, dict):
        target_extra = target.setdefault("extra", {})
        for key in ("machine_family", "tool_number", "sindit_tool_iri", "sindit_asset_iri"):
            value = extra.get(key)
            if value not in (None, ""):
                target_extra[key] = value


def _metadata_cutting_context(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    merged: Dict[str, Any] = {}
    _merge_cutting_context(merged, metadata.get("cutting_context"))

    casedata = metadata.get("casedata")
    if isinstance(casedata, dict):
        _merge_cutting_context(merged, casedata.get("cutting_context"))

    return merged or None


def _runtime_observation_context(
    metadata: Dict[str, Any],
    feature_dict: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    cutting_context = _metadata_cutting_context(metadata)
    if cutting_context is None:
        return None

    spindle_speed = feature_dict.get("spindle_speed_mean")
    if isinstance(spindle_speed, (int, float, np.floating, np.integer)) and np.isfinite(float(spindle_speed)):
        cutting_context["spindle_speed"] = float(spindle_speed)

    feed_rate = feature_dict.get("feed_rate_mean")
    if isinstance(feed_rate, (int, float, np.floating, np.integer)) and np.isfinite(float(feed_rate)):
        cutting_context["feed_rate"] = float(feed_rate)

    return cutting_context


def _session_sample_labels(session: Dict[str, Any]) -> Optional[List[str]]:
    labels = session.get("sample_labels")
    if not isinstance(labels, list) or not labels:
        return None

    normalized: List[str] = []
    for value in labels:
        if value is None:
            normalized.append("unknown")
            continue
        label = str(value).strip()
        normalized.append(label or "unknown")
    return normalized


def _ground_truth_snapshot(sample_labels: Optional[List[str]], *, win_end: int) -> Optional[Dict[str, Any]]:
    if not sample_labels:
        return None

    sample_index = min(max(int(win_end) - 1, 0), len(sample_labels) - 1)
    label = str(sample_labels[sample_index]).strip() or "unknown"
    return {
        "ground_truth_label": label,
        "ground_truth_index": sample_index,
    }


def _harmonic_feature_config(metadata: Optional[Dict[str, Any]] = None):
    for _, config_factory in _harmonic_scorer_candidates(metadata):
        try:
            return config_factory()
        except Exception:
            logger.debug("Failed to create harmonic feature config candidate", exc_info=True)
    return None


def _harmonic_feature_labels(config: Any) -> list[str]:
    from .agents.processing.harmonic_runtime import harmonic_feature_labels

    return harmonic_feature_labels(config)


def _latest_finite_window_value(values: Any) -> Optional[float]:
    if values is None:
        return None

    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return None

    for raw_value in arr[::-1]:
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            return numeric
    return None


def _harmonic_values_have_signal(values: Any, atol: float = 1e-9) -> bool:
    if values is None:
        return False

    try:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return False

    if arr.size == 0:
        return False

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False

    return bool(np.any(np.abs(finite) > atol))


def _validated_harmonic_score(
    out: Optional[Dict[str, Any]],
    *,
    kind: str,
) -> tuple[Optional[float], Optional[str]]:
    if out is None:
        return None, None

    labels = out.get("labels") or []
    values = out.get("values") or []
    score = out.get("score")

    if score is None:
        if labels or values:
            return None, "warming_up"
        return None, "no_pair_columns" if kind == "pair" else "no_harmonic_columns"

    try:
        numeric = float(score)
    except (TypeError, ValueError):
        return None, "invalid_score"

    if not np.isfinite(numeric):
        return None, "nan_logit"

    if not _harmonic_values_have_signal(values):
        return None, "zero_input"

    return numeric, None


def _augment_harmonic_runtime_features(
    feature_dict: Dict[str, float],
    window: Dict[str, np.ndarray],
    harmonic_feature_cfg: Any | None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Add harmonic source columns from the live window when the scorer needs them."""
    runtime_features = dict(feature_dict)
    if harmonic_feature_cfg is None:
        return runtime_features

    cutting_context = _metadata_cutting_context(metadata or {}) if metadata else None

    def _coerce_numeric(value: Any) -> Optional[float]:
        if isinstance(value, (int, float, np.floating, np.integer)):
            numeric = float(value)
        elif isinstance(value, str):
            try:
                numeric = float(value.strip())
            except ValueError:
                return None
        else:
            return None
        return numeric if np.isfinite(numeric) else None

    def _metadata_numeric(source_column: str) -> Optional[float]:
        if not isinstance(cutting_context, dict):
            return None
        candidates = [cutting_context.get(source_column)]
        if source_column.endswith("_mean"):
            candidates.append(cutting_context.get(source_column[:-5]))
        for candidate in candidates:
            numeric = _coerce_numeric(candidate)
            if numeric is not None:
                return numeric
        return None

    context_sources = (
        {
            key: value.get("source_column", key)
            for key, value in getattr(harmonic_feature_cfg, "context_param_stats", {}).items()
        }
        if getattr(harmonic_feature_cfg, "context_param_stats", None)
        else dict(getattr(harmonic_feature_cfg, "context_param_sources", {}) or {})
    )
    for source_column in context_sources.values():
        if not source_column or source_column in runtime_features:
            continue
        numeric = _latest_finite_window_value(window.get(source_column))
        if numeric is None:
            numeric = _metadata_numeric(str(source_column))
        if numeric is not None:
            runtime_features[str(source_column)] = numeric

    requested_sources = {str(source_column) for source_column in context_sources.values() if source_column}
    if "feed_per_tooth" in requested_sources and "feed_per_tooth" not in runtime_features:
        feed_rate = _coerce_numeric(runtime_features.get("feed_rate_mean"))
        if feed_rate is None:
            feed_rate = _coerce_numeric(runtime_features.get("feed_rate"))
        spindle_speed = _coerce_numeric(runtime_features.get("spindle_speed_mean"))
        if spindle_speed is None:
            spindle_speed = _coerce_numeric(runtime_features.get("spindle_speed"))
        num_teeth = _coerce_numeric(runtime_features.get("num_teeth"))
        if feed_rate is not None and spindle_speed is not None and num_teeth is not None and spindle_speed > 0 and num_teeth > 0:
            runtime_features["feed_per_tooth"] = feed_rate / (num_teeth * spindle_speed)

    harmonic_mode = str(getattr(harmonic_feature_cfg, "harmonic_mode", "") or "").strip().lower()
    if harmonic_mode != "pre_extracted":
        return runtime_features

    try:
        scorer_kind = str(getattr(harmonic_feature_cfg, "scorer_kind", "context") or "context").strip().lower()
        feature_source = str(
            getattr(harmonic_feature_cfg, "pre_extracted_feature_source", "harmonic_columns") or "harmonic_columns"
        ).strip().lower()
        if scorer_kind == "pair" or feature_source == "peak_bins":
            from .agents.processing.harmonic_peak_pairs import discover_peak_pair_columns

            pair_specs = discover_peak_pair_columns(
                list(window.keys()),
                frequency_patterns=list(getattr(harmonic_feature_cfg, "pair_frequency_column_patterns", []) or []),
                amplitude_patterns=list(getattr(harmonic_feature_cfg, "pair_amplitude_column_patterns", []) or []),
                k_peaks=int(getattr(harmonic_feature_cfg, "k_peaks", 5)),
            )
            required_columns = [
                column_name
                for spec in pair_specs
                for column_name in (spec.frequency_col, spec.amplitude_col)
            ]
        else:
            from .agents.processing.harmonic_features import select_harmonic_columns

            harmonic_columns = list(getattr(harmonic_feature_cfg, "harmonic_columns", []) or [])
            if not harmonic_columns:
                harmonic_columns = select_harmonic_columns(
                    list(window.keys()),
                    list(getattr(harmonic_feature_cfg, "harmonic_column_patterns", []) or []),
                )
                if harmonic_columns:
                    harmonic_feature_cfg.harmonic_columns = harmonic_columns
            required_columns = harmonic_columns

        for column_name in required_columns:
            if column_name in runtime_features:
                continue
            numeric = _latest_finite_window_value(window.get(column_name))
            if numeric is not None:
                runtime_features[str(column_name)] = numeric
    except Exception:
        logger.debug("Failed to augment harmonic runtime features", exc_info=True)

    return runtime_features


# ── Feature mapping ─────────────────────────────────────────────────────────
# Maps our CNC session channel names to the SeedModel's 17-feature vector.
# The SeedModel was trained on CaseData which has different column names,
# so we need to bridge our live channels → feature dict.

def _features_from_channels(
    window: Dict[str, np.ndarray],
    fs: float,
    domain: Optional[DomainConfig] = None,
    num_teeth: Optional[float] = None,
) -> Dict[str, float]:
    """Compute the SeedModel features from a multi-channel window.

    Uses the active :class:`DomainConfig` to resolve channel names from
    semantic roles.  If a channel is unavailable the array defaults to
    empty, so feature extraction degrades gracefully.  This makes the
    function work with *any* sensor layout — not just the original 7 CNC
    channels.

    Returns a dict containing the original 17 statistical features **plus**
    physics-based fault features derived from the domain's fault-type
    definitions.
    """
    if domain is None:
        domain = get_active_domain(list(window.keys()))

    available_channels = list(window.keys())
    resolved_bindings = _resolve_domain_bindings(domain, available_channels)

    def _safe(arr: np.ndarray, fn, default: float = 0.0) -> float:
        if len(arr) == 0:
            return default
        v = fn(arr)
        return float(v) if np.isfinite(v) else default

    def _named_channel(name: str) -> np.ndarray:
        if name and name in window:
            return np.asarray(window[name], dtype=np.float64)
        return np.array([], dtype=np.float64)

    def _ch(role: str) -> np.ndarray:
        """Resolve a channel role to an array (empty if missing)."""
        return _named_channel(resolved_bindings.get(str(role), ""))

    sp = _ch(ChannelRole.PRIMARY_POWER)
    xp = _ch(ChannelRole.SECONDARY_POWER)
    vib = _ch(ChannelRole.PRIMARY_VIBRATION)
    chatter = _ch(ChannelRole.CHATTER_AMPLITUDE)
    active = _ch(ChannelRole.ACTIVE_POWER)
    speed = _ch(ChannelRole.SPINDLE_SPEED)
    feed = _ch(ChannelRole.FEED_RATE)
    vib_y = _named_channel("Vibration_Severity_Y")
    power_z = _named_channel("Power_Z")
    chatter_on_x = _named_channel("Chatter_Detection_OnOff_X")
    chatter_on_y = _named_channel("Chatter_Detection_OnOff_Y")

    if len(vib_y) == 0:
        vib_y = chatter

    # Fallback: if no channels resolved from roles, try to pick the first
    # numeric channel in the window so extraction still produces *something*.
    if all(len(a) == 0 for a in [sp, xp, vib, chatter, active, speed, feed]):
        for _name, _arr in window.items():
            arr = np.asarray(_arr, dtype=np.float64)
            if arr.ndim == 1 and len(arr) > 0:
                # Map first 3 generic channels into sp, vib, active slots
                if len(sp) == 0:
                    sp = arr
                elif len(vib) == 0:
                    vib = arr
                elif len(active) == 0:
                    active = arr
                else:
                    break

    # Chatter ratio: fraction of samples where chatter amplitude exceeds a
    # threshold relative to vibration.
    vib_rms = float(np.sqrt(np.mean(vib ** 2))) if len(vib) > 0 else 1e-9
    if len(chatter_on_x) > 0 or len(chatter_on_y) > 0:
        chatter_flags: List[np.ndarray] = []
        if len(chatter_on_x) > 0:
            chatter_flags.append(chatter_on_x > 0)
        if len(chatter_on_y) > 0:
            chatter_flags.append(chatter_on_y > 0)
        chatter_ratio = float(np.mean(np.concatenate(chatter_flags))) if chatter_flags else 0.0
    else:
        chatter_thresh = max(0.3, vib_rms * 0.5)
        chatter_ratio = float(np.mean(chatter > chatter_thresh)) if len(chatter) > 0 else 0.0

    # ── Original 17 features ────────────────────────────────────────────
    features = {
        "power_spindle_mean": _safe(sp, np.mean),
        "power_spindle_max": _safe(sp, np.max),
        "power_spindle_std": _safe(sp, np.std),
        "power_y_mean": _safe(xp, np.mean),
        "power_y_max": _safe(xp, np.max),
        "power_z_mean": _safe(power_z, np.mean),
        "vib_severity_x_mean": _safe(vib, np.mean),
        "vib_severity_x_max": _safe(vib, np.max),
        "vib_severity_y_mean": _safe(vib_y, np.mean),
        "vib_severity_y_max": _safe(vib_y, np.max),
        "chatter_ratio": chatter_ratio,
        "power_active_mean": _safe(active, np.mean),
        "power_active_std": _safe(active, np.std),
        "power_factor_mean": 0.0,
        "spindle_speed_mean": _safe(speed, np.mean),
        "feed_rate_mean": _safe(feed, np.mean),
        "temp_head_mean": 0.0,
    }

    # ── Physics-based fault features ────────────────────────────────────

    # Use vibration channel as primary signal for spectral analysis
    sig = vib if len(vib) > 0 else chatter
    if len(sig) == 0:
        sig = np.zeros(1)

    n_fft = int(2 ** np.ceil(np.log2(max(len(sig), 64))))
    spec = np.abs(np.fft.rfft(sig, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    power = spec ** 2
    total_pow = float(np.sum(power) + 1e-10)

    # ---- Tool breakage features ----

    # High-frequency energy ratio — Nyquist-aware cutoff
    nyquist = fs / 2.0
    hf_cutoff = min(500.0, nyquist * 0.5)  # scale with Nyquist; max 500 Hz
    hf_mask = freqs > hf_cutoff
    hf_energy_ratio = float(np.sum(power[hf_mask]) / total_pow) if np.any(hf_mask) else 0.0

    # Impulse crest factor (max peak / RMS across vibration channels)
    crest_values = []
    for arr in [vib, chatter]:
        if len(arr) > 0:
            rms = float(np.sqrt(np.mean(arr ** 2)))
            peak = float(np.max(np.abs(arr)))
            crest_values.append(peak / (rms + 1e-12))
    impulse_crest = max(crest_values) if crest_values else 0.0

    # Max excess kurtosis
    kurt_values = []
    for arr in [vib, chatter]:
        if len(arr) > 0:
            std = float(np.std(arr))
            if std > 1e-12:
                norm = (arr - np.mean(arr)) / std
                kurt_values.append(float(np.mean(norm ** 4) - 3.0))
    kurtosis_max = max(kurt_values) if kurt_values else 0.0

    # Periodicity strength (autocorrelation at expected tooth period)
    # Use spindle speed and flute count to derive the tooth-pass period.
    spindle_rpm = features["spindle_speed_mean"]
    tooth_count = 1.0
    if isinstance(num_teeth, (int, float)) and np.isfinite(float(num_teeth)) and float(num_teeth) > 0:
        tooth_count = float(num_teeth)
    tooth_pass_freq = (spindle_rpm * tooth_count / 60.0) if spindle_rpm > 0 else 0.0
    periodicity = 0.0
    if tooth_pass_freq > 0 and len(sig) > 10:
        period_samples = int(round(fs / tooth_pass_freq))
        if 2 <= period_samples < len(sig) // 2:
            sig_centered = sig - np.mean(sig)
            autocorr_full = np.correlate(sig_centered, sig_centered, mode="full")
            autocorr = autocorr_full[len(sig_centered) - 1:]  # positive lags
            ac0 = autocorr[0] + 1e-12
            periodicity = float(autocorr[period_samples] / ac0)

    # ---- Chatter features ----

    # Modulation depth: crest factor of the envelope (Hilbert)
    modulation_depth = 0.0
    if len(sig) > 10:
        try:
            # Correct analytic signal: DC and Nyquist at 1×, positive freqs at 2×
            _N = len(sig)
            _h = np.zeros(_N)
            _h[0] = 1.0                       # DC at 1×
            _h[1:(_N + 1) // 2] = 2.0         # positive freqs at 2×
            if _N % 2 == 0:
                _h[_N // 2] = 1.0              # Nyquist at 1×
            analytic = np.fft.ifft(np.fft.fft(sig) * _h)
            envelope = np.abs(analytic)
            env_rms = float(np.sqrt(np.mean(envelope ** 2)))
            env_peak = float(np.max(envelope))
            modulation_depth = (env_peak / (env_rms + 1e-12)) - 1.0
            modulation_depth = max(0.0, modulation_depth)
        except Exception:
            modulation_depth = 0.0

    # Vibration amplitude growth (current RMS / overall baseline estimate)
    # Approximation: ratio of window RMS to a conservative baseline
    vib_amplitude_growth = 1.0
    if vib_rms > 1e-9:
        vib_amplitude_growth = vib_rms / max(0.01, _safe(vib, np.median))

    # Tooth-passing harmonic energy
    tp_harmonic_energy = 0.0
    if tooth_pass_freq > 0:
        tp_freq = tooth_pass_freq
        for harmonic in range(1, 5):
            target = tp_freq * harmonic
            idx = int(round(target / (fs / n_fft)))
            if 0 < idx < len(spec):
                tp_harmonic_energy += float(power[idx])
        tp_harmonic_energy /= (total_pow + 1e-12)

    # ---- Chip adhesion features ----

    # Harmonic amplitude CV (coefficient of variation of spectral peak amplitudes)
    harmonic_amplitude_cv = 0.0
    harmonic_amps = []
    if tooth_pass_freq > 0:
        base_freq = tooth_pass_freq
        for h in range(1, 9):
            target = base_freq * h
            idx = int(round(target / (fs / n_fft)))
            if 0 < idx < len(spec):
                harmonic_amps.append(float(spec[idx]))
    if len(harmonic_amps) >= 2:
        ha = np.array(harmonic_amps)
        ha_mean = float(np.mean(ha))
        if ha_mean > 1e-12:
            harmonic_amplitude_cv = float(np.std(ha) / ha_mean)

    # Tooth-passing amplitude variance (normalised)
    tp_amplitude_variance = 0.0
    if harmonic_amps:
        tp_amplitude_variance = float(np.var(harmonic_amps) / (np.mean(harmonic_amps) ** 2 + 1e-12))

    # ---- Workpiece slip features ----

    # Spindle order amplitude (amplitude at 1× spindle frequency)
    spindle_order_amp = 0.0
    if spindle_rpm > 0:
        sf = spindle_rpm / 60.0
        idx_sf = int(round(sf / (fs / n_fft)))
        if 0 < idx_sf < len(spec):
            spindle_order_amp = float(spec[idx_sf])

    # Spindle phase shift: phase at 1× spindle frequency (radians)
    spindle_phase_shift = 0.0
    if spindle_rpm > 0 and len(sig) > 0:
        full_spec = np.fft.rfft(sig, n=n_fft)
        sf = spindle_rpm / 60.0
        idx_sf = int(round(sf / (fs / n_fft)))
        if 0 < idx_sf < len(full_spec):
            spindle_phase_shift = float(np.angle(full_spec[idx_sf]))

    # ── Assemble all features ───────────────────────────────────────────
    features.update({
        "hf_energy_ratio": round(hf_energy_ratio, 6),
        "impulse_crest_factor": round(impulse_crest, 4),
        "kurtosis_max": round(kurtosis_max, 4),
        "periodicity_strength": round(periodicity, 4),
        "modulation_depth": round(modulation_depth, 4),
        "vib_amplitude_growth": round(vib_amplitude_growth, 4),
        "tp_harmonic_energy": round(tp_harmonic_energy, 6),
        "harmonic_amplitude_cv": round(harmonic_amplitude_cv, 4),
        "tp_amplitude_variance": round(tp_amplitude_variance, 6),
        "spindle_order_amplitude": round(spindle_order_amp, 4),
        "spindle_phase_shift": round(spindle_phase_shift, 4),
    })

    return features


# ── Fault indicator scores ──────────────────────────────────────────────────
# Each indicator maps the physics-based features into a [0, 1] likelihood
# score for a specific machining fault type.  The mapping is a simple
# sigmoid-style rescaling — no ML model, just interpretable thresholds.

def _sigmoid(x: float, center: float, steepness: float = 10.0) -> float:
    """Logistic sigmoid centred at `center`."""
    return float(1.0 / (1.0 + np.exp(-steepness * (x - center))))


def _compute_fault_indicators(
    f: Dict[str, float],
    domain: Optional[DomainConfig] = None,
) -> Dict[str, Any]:
    """Compute per-fault-type likelihood indicators from physics features.

    **Domain-adaptive**: iterates over fault types registered in the active
    :class:`DomainConfig`.  Each fault type defines indicator features with
    sigmoid parameters and weights.  This makes the function work for CNC,
    bearing-fault, HVAC, or any other domain without code changes.

    Returns a dict like::

        {
          "tool_breakage":   { "score": 0.82, "signals": { ... } },
          ...  (one entry per registered fault type)
          "dominant_fault":  "tool_breakage" | null
        }
    """
    if domain is None:
        domain = get_active_domain()

    result: Dict[str, Any] = {}
    scores: Dict[str, float] = {}

    for ft in domain.fault_types:
        signals: Dict[str, float] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for ind in ft.indicators:
            raw_value = f.get(ind.feature_name, 0.0)
            steepness = ind.sigmoid_steepness

            if steepness < 0:
                # Inverted: high raw → low score (e.g. periodicity loss)
                sig_score = 1.0 - _sigmoid(raw_value, ind.sigmoid_center, abs(steepness))
            else:
                sig_score = _sigmoid(raw_value, ind.sigmoid_center, steepness)

            display = ind.display_name or ind.feature_name
            signals[display] = round(sig_score, 3)
            weighted_sum += ind.weight * sig_score
            total_weight += ind.weight

        fault_score = round(weighted_sum / max(total_weight, 1e-9), 3)
        scores[ft.name] = fault_score
        result[ft.name] = {"score": fault_score, "signals": signals}

    # Dominant fault (use first fault type's threshold, or 0.35 default)
    dominant_threshold = 0.35
    if domain and domain.fault_types:
        dominant_threshold = domain.fault_types[0].dominant_threshold
    if scores:
        max_fault = max(scores, key=scores.get)  # type: ignore[arg-type]
        dominant = max_fault if scores[max_fault] > dominant_threshold else None
    else:
        dominant = None

    result["dominant_fault"] = dominant
    return result


# ── Model loading ───────────────────────────────────────────────────────────

_cached_model = None

def _get_seed_model():
    """Load and cache the trained SeedModel (lazy singleton)."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    try:
        from .agents.processing.classical_models import SeedModel
        model_path = Path(__file__).resolve().parent.parent / "data" / "models" / "seed_model.pkl"
        if not model_path.exists():
            logger.warning("No trained seed model at %s — inference will return 0.5", model_path)
            return None
        model = SeedModel()
        model.load(model_path)
        logger.info("Loaded seed model from %s (%d features, trained on %d samples)",
                     model_path,
                     len(model.feature_names),
                     model.n_training_samples)
        _cached_model = model
        return model
    except Exception:
        logger.exception("Failed to load seed model")
        return None


# ── Harmonic Context scorer (optional, lazy) ────────────────────────────────

_cached_harmonic_scorers: Dict[str, Any] = {}


def clear_harmonic_scorer_cache() -> None:
    """Invalidate cached harmonic scorers so newly trained checkpoints reload."""
    _cached_harmonic_scorers.clear()


def _preferred_pair_dataset(metadata: Optional[Dict[str, Any]] = None) -> str:
    meta = metadata or {}
    requested_dataset = str(
        meta.get("harmonic_dataset")
        or meta.get("harmonic_dataset_name")
        or ""
    ).strip().lower()
    if requested_dataset in {"pair_raw", "pair_casedata", "pair_lfl"}:
        return requested_dataset
    if requested_dataset == "casedata":
        return "pair_lfl"

    casedata_meta = meta.get("casedata") if isinstance(meta.get("casedata"), dict) else {}
    has_casedata = bool(casedata_meta)
    source_hints = " ".join(
        part
        for part in (
            str(meta.get("source") or "").strip().lower(),
            str(casedata_meta.get("root") or "").strip().lower(),
            str(casedata_meta.get("case_dir") or "").strip().lower(),
            str(meta.get("machine_id") or "").strip().lower(),
        )
        if part
    )
    if has_casedata or "casedata" in source_hints or "site_b" in source_hints or "site_c" in source_hints:
        return "pair_lfl"
    return "pair_raw"


def _harmonic_scorer_candidates(metadata: Optional[Dict[str, Any]] = None):
    from .agents.processing.harmonic_config import (
        HarmonicContextConfig,
        casedata_peak_context_preset,
        casedata_stoppage_preset,
        pair_casedata_preset,
        pair_lfl_preset,
        pair_raw_preset,
        raw_accelerometer_preset,
        stoppage_1hz_preset,
        site_a_line2_breakage_preset,
    )

    meta = metadata or {}
    source = str(meta.get("source") or "").strip().lower()
    casedata_meta = meta.get("casedata") if isinstance(meta.get("casedata"), dict) else {}
    has_casedata = bool(casedata_meta)
    source_hints = " ".join(
        part
        for part in (
            source,
            str(casedata_meta.get("root") or "").strip().lower(),
            str(casedata_meta.get("case_dir") or "").strip().lower(),
            str(meta.get("machine_id") or "").strip().lower(),
        )
        if part
    )
    requested_kind = str(
        meta.get("harmonic_scorer_kind")
        or meta.get("harmonic_model_kind")
        or meta.get("harmonic_model")
        or ""
    ).strip().lower()
    requested_dataset = str(
        meta.get("harmonic_dataset")
        or meta.get("harmonic_dataset_name")
        or ""
    ).strip().lower()
    if requested_kind == "context" and requested_dataset == "pair_raw":
        requested_dataset = ""
    pair_requested = (
        requested_kind == "pair"
        or (
            requested_kind != "context"
            and (requested_dataset == "pair_raw" or "pair_raw" in source_hints)
        )
    )

    explicit_dataset_factories = {
        "casedata_peaks": casedata_peak_context_preset,
        "casedata": casedata_stoppage_preset,
        "stoppage_1hz": stoppage_1hz_preset,
        "site_a_line2": site_a_line2_breakage_preset,
        "raw_accelerometer": raw_accelerometer_preset,
        "pair_casedata": pair_casedata_preset,
        "pair_lfl": pair_lfl_preset,
        "pair_raw": pair_raw_preset,
    }
    if requested_kind == "pair":
        pair_dataset = _preferred_pair_dataset(meta)
        return [(pair_dataset, explicit_dataset_factories[pair_dataset])]
    if requested_dataset in explicit_dataset_factories:
        return [(requested_dataset, explicit_dataset_factories[requested_dataset])]
    if pair_requested:
        pair_dataset = _preferred_pair_dataset(meta)
        return [(pair_dataset, explicit_dataset_factories[pair_dataset])]

    ordered = [
        ("casedata_peaks", casedata_peak_context_preset),
        ("casedata", casedata_stoppage_preset),
        ("stoppage_1hz", stoppage_1hz_preset),
        ("site_a_line2", site_a_line2_breakage_preset),
        ("raw_accelerometer", raw_accelerometer_preset),
        ("default", HarmonicContextConfig),
    ]
    if "site_a_line2" in source_hints:
        ordered = [ordered[3], ordered[0], ordered[1], ordered[2], ordered[4], ordered[5]]
    elif "site_a" in source_hints:
        ordered = [ordered[0], ordered[1], ordered[2], ordered[3], ordered[4], ordered[5]]
    elif "olddata" in source_hints or "stoppage" in source_hints:
        ordered = [ordered[2], ordered[0], ordered[1], ordered[3], ordered[4], ordered[5]]
    elif has_casedata or "casedata" in source_hints or "site_b" in source_hints:
        ordered = [ordered[0], ordered[1], ordered[2], ordered[3], ordered[4], ordered[5]]
    elif "raw" in source_hints or "accelerometer" in source_hints:
        ordered = [ordered[4], ordered[0], ordered[1], ordered[2], ordered[3], ordered[5]]

    return ordered


def _get_harmonic_scorer(metadata: Optional[Dict[str, Any]] = None):
    """Load and cache the HarmonicContextScorer (lazy, optional).

    Returns None if torch is not installed or no trained model exists.
    Prefers dataset-specific preset checkpoints before falling back to the
    generic default config.
    """
    try:
        from .agents.processing.harmonic_runtime import ensure_harmonic_scorer, harmonic_torch_available

        candidate_keys: list[str] = []
        for cache_key, config_factory in _harmonic_scorer_candidates(metadata):
            candidate_keys.append(cache_key)

            cached = _cached_harmonic_scorers.get(cache_key)
            if cached is not None:
                return cached

            config = config_factory()
            if not harmonic_torch_available(config):
                continue

            scorer = ensure_harmonic_scorer(config)
            if scorer is None:
                continue

            _cached_harmonic_scorers[cache_key] = scorer
            logger.info(
                "Loaded harmonic scorer (candidate=%s kind=%s dataset=%s, n_harm=%d, n_params=%d)",
                cache_key,
                getattr(scorer.config, "scorer_kind", "context"),
                scorer.config.dataset_name,
                scorer.config.n_harm_features,
                scorer.config.n_params,
            )
            return scorer

        logger.debug(
            "Harmonic scorer: no trained model found for candidates=%s",
            candidate_keys,
        )
        return None
    except Exception:
        logger.debug("Harmonic scorer load failed", exc_info=True)
        return None


def _get_harmonic_scorer_of_kind(
    kind: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Load a harmonic scorer of an explicit kind ('context' or 'pair').

    Used to load a secondary scorer alongside the auto-selected primary so the
    UI can show both context and pair model outputs per window.
    """
    kind_norm = (kind or "").strip().lower()
    if kind_norm not in {"context", "pair"}:
        return None
    meta_override: Dict[str, Any] = dict(metadata or {})
    meta_override["harmonic_scorer_kind"] = kind_norm
    if kind_norm == "pair":
        meta_override["harmonic_dataset"] = _preferred_pair_dataset(meta_override)
    else:
        meta_override.pop("harmonic_dataset", None)
    return _get_harmonic_scorer(meta_override)


def _compute_harmonic_window(
    *,
    scorer: Any,
    cfg: Any,
    feature_dict: Dict[str, float],
    win_data: Dict[str, np.ndarray],
    fs: float,
    row_history: Optional[deque],
) -> Dict[str, Any]:
    """Run one harmonic scorer for the current window.

    Returns a dict with ``score``, ``weights``, ``labels``, ``values``,
    ``row_history`` (possibly newly allocated), and ``kind``. Empty/None
    fields are returned when the scorer cannot produce output.
    """
    result: Dict[str, Any] = {
        "score": None,
        "weights": [],
        "labels": [],
        "values": [],
        "decision_threshold": float(getattr(cfg, "decision_threshold", 0.5) or 0.5) if cfg is not None else 0.5,
        "row_history": row_history,
        "kind": str(getattr(cfg, "scorer_kind", "context") or "context").strip().lower(),
    }
    if cfg is None:
        return result
    try:
        from .agents.processing.harmonic_features import (
            compute_harmonics,
            extract_channels_from_window,
            extract_context_params,
            extract_harmonic_matrix_from_df,
            extract_peak_binned_harmonic_matrix_from_df,
            resolve_spindle_speed_source_column,
            runtime_context_normalize,
            runtime_context_param_stats,
            select_harmonic_columns,
        )
        from .agents.processing.harmonic_peak_pairs import (
            build_pair_feature_labels,
            discover_peak_pair_columns,
            extract_peak_pairs_from_df,
        )

        scorer_available = scorer is not None and scorer.is_available()
        scorer_kind = result["kind"]

        ctx_vec = None
        if scorer_available:
            runtime_stats = runtime_context_param_stats(cfg)
            normalize_context = runtime_context_normalize(cfg)
            ctx_vec = extract_context_params(
                feature_dict,
                cfg.context_param_keys,
                {k: v.get("source_column", k) for k, v in cfg.context_param_stats.items()}
                if cfg.context_param_stats else cfg.context_param_sources,
                runtime_stats,
                normalize=normalize_context,
            )

        if cfg.harmonic_mode == "pre_extracted":
            import pandas as _pd
            _row = _pd.DataFrame([feature_dict])
            if scorer_kind == "pair":
                pair_specs = discover_peak_pair_columns(
                    list(_row.columns),
                    frequency_patterns=list(getattr(cfg, "pair_frequency_column_patterns", []) or []),
                    amplitude_patterns=list(getattr(cfg, "pair_amplitude_column_patterns", []) or []),
                    k_peaks=int(getattr(cfg, "k_peaks", 5)),
                )
                if pair_specs:
                    pair_labels = list(getattr(cfg, "harmonic_columns", []) or [])
                    if not pair_labels:
                        pair_labels = build_pair_feature_labels(pair_specs)
                        cfg.harmonic_columns = pair_labels
                    spindle_speed_col = resolve_spindle_speed_source_column(cfg)
                    p_mat = extract_peak_pairs_from_df(
                        _row,
                        pair_specs,
                        spindle_speed_col=spindle_speed_col,
                        k_peaks=int(getattr(cfg, "k_peaks", 5)),
                        f_max_rel=float(getattr(cfg, "f_max_rel", 12.0)),
                    )
                    if p_mat.shape[0] > 0 and p_mat.shape[1] > 0:
                        result["values"] = np.asarray(
                            p_mat[-1, :, :, 1], dtype=np.float32
                        ).reshape(-1).tolist()
                        result["labels"] = pair_labels
                        if scorer_available:
                            maxlen = max(1, int(getattr(cfg, "cnn_window", 1)))
                            hist = row_history
                            if hist is None or hist.maxlen != maxlen:
                                hist = deque(maxlen=maxlen)
                            for row in np.asarray(p_mat, dtype=np.float32):
                                hist.append(np.asarray(row, dtype=np.float32))
                            result["row_history"] = hist
                            if len(hist) > 0:
                                p_window = np.asarray(list(hist), dtype=np.float32)
                                h_result = scorer.score(p_window, ctx_vec)
                                result["score"] = h_result["harmonic_context_score"]
                                result["weights"] = h_result.get("context_weights", [])
                                result["labels"] = h_result.get("feature_labels", []) or result["labels"]
                                result["values"] = h_result.get("harmonic_values", []) or result["values"]
                                result["decision_threshold"] = float(
                                    h_result.get("decision_threshold", result["decision_threshold"]) or result["decision_threshold"]
                                )
            else:
                feature_source = str(
                    getattr(cfg, "pre_extracted_feature_source", "harmonic_columns") or "harmonic_columns"
                ).strip().lower()
                h_mat = np.empty((0, 0), dtype=np.float32)
                if feature_source == "peak_bins":
                    spindle_speed_col = resolve_spindle_speed_source_column(cfg)
                    h_mat, peak_labels = extract_peak_binned_harmonic_matrix_from_df(
                        _row,
                        frequency_patterns=list(getattr(cfg, "pair_frequency_column_patterns", []) or []),
                        amplitude_patterns=list(getattr(cfg, "pair_amplitude_column_patterns", []) or []),
                        spindle_speed_col=spindle_speed_col,
                        harmonic_bins=list(getattr(cfg, "peak_harmonic_bins", []) or []),
                        k_peaks=int(getattr(cfg, "k_peaks", 5)),
                        f_max_rel=float(getattr(cfg, "f_max_rel", 12.0)),
                        tolerance=float(getattr(cfg, "peak_bin_tolerance", 0.35)),
                    )
                    if peak_labels:
                        cfg.harmonic_columns = peak_labels
                # Plain harmonic-column extraction: the default path, and the fallback
                # when peak-bin extraction yielded nothing (e.g. pre-extracted amplitude
                # columns with no peak structure, or a no-scorer simulated session).
                if h_mat.shape[0] == 0 or h_mat.shape[1] == 0:
                    harmonic_columns = list(getattr(cfg, "harmonic_columns", []) or [])
                    if not harmonic_columns:
                        harmonic_columns = select_harmonic_columns(
                            list(_row.columns),
                            list(getattr(cfg, "harmonic_column_patterns", []) or []),
                        )
                        if harmonic_columns:
                            cfg.harmonic_columns = harmonic_columns
                    h_mat = extract_harmonic_matrix_from_df(_row, harmonic_columns)
                if h_mat.shape[0] > 0 and h_mat.shape[1] > 0:
                    result["values"] = np.asarray(h_mat[-1], dtype=np.float32).tolist()
                    result["labels"] = _harmonic_feature_labels(cfg)
                    if scorer_available:
                        maxlen = max(1, int(getattr(cfg, "cnn_window", 1)))
                        hist = row_history
                        if hist is None or hist.maxlen != maxlen:
                            hist = deque(maxlen=maxlen)
                        for row in np.asarray(h_mat, dtype=np.float32):
                            hist.append(np.asarray(row, dtype=np.float32))
                        result["row_history"] = hist
                        if len(hist) >= maxlen:
                            h_window = np.asarray(list(hist), dtype=np.float32)
                            h_result = scorer.score(h_window, ctx_vec)
                            if h_result.get("model_source") != "harmonic_context_insufficient_window":
                                result["score"] = h_result["harmonic_context_score"]
                                result["weights"] = h_result.get("context_weights", [])
                                result["labels"] = h_result.get("feature_labels", []) or result["labels"]
                                result["values"] = h_result.get("harmonic_values", []) or result["values"]
                                result["decision_threshold"] = float(
                                    h_result.get("decision_threshold", result["decision_threshold"]) or result["decision_threshold"]
                                )
        else:
            spindle_rpm = feature_dict.get("spindle_speed_mean", 0)
            fg = spindle_rpm / 60.0 if spindle_rpm > 0 else 0
            if fg > 0:
                ch_data = extract_channels_from_window(win_data, cfg)
                if ch_data.shape[0] > 0:
                    if scorer_available:
                        h_result = scorer.score_from_raw(
                            ch_data, ctx_vec, fg, sample_rate=fs,
                        )
                        result["score"] = h_result["harmonic_context_score"]
                        result["weights"] = h_result.get("context_weights", [])
                        result["labels"] = h_result.get("feature_labels", [])
                        result["values"] = h_result.get("harmonic_values", [])
                        result["decision_threshold"] = float(
                            h_result.get("decision_threshold", result["decision_threshold"]) or result["decision_threshold"]
                        )
                    elif scorer_kind != "pair":
                        raw_harmonics = compute_harmonics(
                            ch_data,
                            fg,
                            harm_mults=getattr(cfg, "harmonic_multipliers", None),
                            fft_win=int(getattr(cfg, "fft_window", 4096)),
                            fft_step=int(getattr(cfg, "fft_step", 1024)),
                            sample_rate=fs,
                        )
                        if raw_harmonics.shape[0] > 0 and raw_harmonics.shape[1] > 0:
                            cfg.n_harm_features = int(raw_harmonics.shape[1])
                            result["values"] = np.asarray(raw_harmonics[-1], dtype=np.float32).tolist()
                            result["labels"] = _harmonic_feature_labels(cfg)
    except Exception as hc_err:
        logger.debug("Harmonic context scoring error: %s", hc_err)
    return result


# ── Safe queue put ──────────────────────────────────────────────────────────

async def _safe_put(q: asyncio.Queue, item: Any, timeout: float = 1.0):
    try:
        await asyncio.wait_for(q.put(item), timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        pass


def _model_confidence(model: Any, confidence_path: Optional[str | Path] = None) -> float:
    if model is None or not getattr(model, "is_trained", False):
        return 0.0
    return float(current_model_confidence(confidence_path))


def _seed_model_schema_diagnostics(model: Any, feature_dict: Dict[str, float]) -> Dict[str, Any]:
    """Describe whether the loaded seed model matches the runtime feature schema."""
    try:
        from .agents.processing.classical_models import FEATURE_NAMES
    except Exception:
        FEATURE_NAMES = []

    runtime_feature_order = list(FEATURE_NAMES)
    model_feature_order = list(getattr(model, "feature_names", []) or [])
    live_feature_keys = list(feature_dict.keys())

    if not model_feature_order:
        return {
            "aligned": True,
            "reason": "model_feature_names_unavailable",
            "runtime_feature_order": runtime_feature_order,
            "model_feature_order": model_feature_order,
            "live_feature_keys": live_feature_keys,
            "model_only": [],
            "runtime_only": [],
        }

    model_only = [name for name in model_feature_order if name not in runtime_feature_order]
    runtime_only = [name for name in runtime_feature_order if name not in model_feature_order]
    return {
        "aligned": model_feature_order == runtime_feature_order,
        "reason": "aligned" if model_feature_order == runtime_feature_order else "feature_order_mismatch",
        "runtime_feature_order": runtime_feature_order,
        "model_feature_order": model_feature_order,
        "live_feature_keys": live_feature_keys,
        "model_only": model_only,
        "runtime_only": runtime_only,
    }


def _memory_metadata_snapshot(metadata: Dict[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}

    source = metadata.get("source")
    if isinstance(source, str) and source:
        snapshot["source"] = source

    for key in ("machine_id", "machine_family", "dataset_id", "source_dataset_id"):
        value = metadata.get(key)
        if isinstance(value, (str, int, float, bool)) and value not in ("", None):
            snapshot[key] = value

    harmonic_runtime: Dict[str, Any] = {}
    harmonic_scorer_kind = metadata.get("harmonic_scorer_kind")
    if isinstance(harmonic_scorer_kind, str) and harmonic_scorer_kind:
        harmonic_runtime["scorer_kind"] = harmonic_scorer_kind
    harmonic_dataset = metadata.get("harmonic_dataset") or metadata.get("harmonic_dataset_name")
    if isinstance(harmonic_dataset, str) and harmonic_dataset:
        harmonic_runtime["dataset"] = harmonic_dataset
    if harmonic_runtime:
        snapshot["harmonic_runtime"] = harmonic_runtime

    sample_frequency = metadata.get("sample_frequency")
    if isinstance(sample_frequency, (int, float)):
        snapshot["sample_frequency"] = float(sample_frequency)

    machine_uri = metadata.get("machine_uri")
    if isinstance(machine_uri, str) and machine_uri:
        snapshot["machine_uri"] = machine_uri

    machine_iri = metadata.get("machine_iri")
    if isinstance(machine_iri, str) and machine_iri:
        snapshot["machine_iri"] = machine_iri

    sindit_asset_iri = metadata.get("sindit_asset_iri")
    if isinstance(sindit_asset_iri, str) and sindit_asset_iri:
        snapshot["sindit_asset_iri"] = sindit_asset_iri

    casedata = metadata.get("casedata")
    if isinstance(casedata, dict):
        safe_casedata: Dict[str, Any] = {}
        for key in ("operation_id", "tool_id", "root", "case_dir", "dataset_id"):
            value = casedata.get(key)
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                safe_casedata[key] = value
            elif isinstance(value, Path):
                safe_casedata[key] = str(value)
            else:
                safe_casedata[key] = str(value)
        cutting_context = casedata.get("cutting_context")
        if isinstance(cutting_context, dict):
            safe_cutting_context: Dict[str, Any] = {}
            for key in (
                "machine_id",
                "tool_id",
                "tool_type",
                "tool_diameter",
                "num_teeth",
                "tool_length",
                "tool_material",
                "spindle_speed",
                "feed_rate",
            ):
                value = cutting_context.get(key)
                if isinstance(value, (str, int, float, bool)) and value not in ("", None):
                    safe_cutting_context[key] = value
            extra = cutting_context.get("extra")
            if isinstance(extra, dict):
                safe_extra: Dict[str, Any] = {}
                for key in ("machine_family", "tool_number", "sindit_tool_iri", "sindit_asset_iri"):
                    value = extra.get(key)
                    if isinstance(value, (str, int, float, bool)) and value not in ("", None):
                        safe_extra[key] = value
                if safe_extra:
                    safe_cutting_context["extra"] = safe_extra
            if safe_cutting_context:
                safe_casedata["cutting_context"] = safe_cutting_context
        if safe_casedata:
            snapshot["casedata"] = safe_casedata

    return snapshot


def _window_time_payload(
    session: Dict[str, Any],
    *,
    win_start: int,
    win_end: int,
    fs: float,
) -> Dict[str, Any]:
    end_index = max(win_start, win_end - 1)
    center_index = max(win_start, min(end_index, (win_start + end_index) // 2))

    payload = {
        "t": round(end_index / fs, 4),
        "t0": round(win_start / fs, 4),
        "t1": round(end_index / fs, 4),
        "t_center": round((win_start + win_end) / 2.0 / fs, 4),
    }

    time_axis_unix = session.get("time_axis_unix")
    if not isinstance(time_axis_unix, list) or not time_axis_unix:
        return payload

    last_idx = len(time_axis_unix) - 1
    indices = {
        "t0": max(0, min(win_start, last_idx)),
        "t1": max(0, min(end_index, last_idx)),
        "t_center": max(0, min(center_index, last_idx)),
    }
    resolved: Dict[str, float] = {}
    for key, index in indices.items():
        try:
            numeric = float(time_axis_unix[index])
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            resolved[key] = round(numeric, 4)

    if "t1" in resolved:
        payload["t"] = resolved["t1"]
    payload.update(resolved)
    return payload


def _build_memory_feature_payload(
    *,
    session_id: str,
    metadata: Dict[str, Any],
    fs: float,
    win_start: int,
    win_end: int,
    window_seconds: float,
    feature_dict: Dict[str, float],
    ensemble_score: Optional[float],
    z_anomaly: float,
    harmonic_score_val: Optional[float],
    harmonic_labels: Optional[List[str]],
    harmonic_values: Optional[List[float]],
    harmonic_weights: Optional[List[float]],
    ground_truth: Optional[Dict[str, Any]],
    fault_indicators: Dict[str, Any],
    model_confidence: float,
    model_source: Optional[str],
) -> Dict[str, Any]:
    external_signals: Dict[str, Any] = {
        "model_confidence": round(model_confidence, 4),
        "z_score": round(z_anomaly, 4),
        "fault_indicators": fault_indicators,
    }
    if ensemble_score is not None:
        external_signals["anomaly_detector_score"] = round(float(ensemble_score), 4)
    if model_source:
        external_signals["model_source"] = model_source
    if harmonic_score_val is not None:
        external_signals["harmonic_context_score"] = round(float(harmonic_score_val), 4)
        external_signals["harmonic_context_source"] = "harmonic_context_v1"

    payload_metadata = _memory_metadata_snapshot(metadata)
    if isinstance(ground_truth, dict):
        ground_truth_label = ground_truth.get("ground_truth_label")
        if isinstance(ground_truth_label, str) and ground_truth_label:
            payload_metadata["ground_truth_label"] = ground_truth_label
        ground_truth_index = ground_truth.get("ground_truth_index")
        if isinstance(ground_truth_index, (int, np.integer)) and not isinstance(ground_truth_index, bool):
            payload_metadata["ground_truth_index"] = int(ground_truth_index)

    harmonic_context: Dict[str, Any] = {}
    if isinstance(external_signals.get("harmonic_context_source"), str):
        harmonic_context["source"] = external_signals["harmonic_context_source"]
    if harmonic_labels:
        safe_labels = [str(label) for label in harmonic_labels if label is not None]
        if safe_labels:
            harmonic_context["feature_labels"] = safe_labels
    if harmonic_values:
        safe_values = [
            float(value)
            for value in harmonic_values
            if isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value)
        ]
        if safe_values:
            harmonic_context["feature_values"] = safe_values
    if harmonic_weights:
        safe_weights = [
            float(value)
            for value in harmonic_weights
            if isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value)
        ]
        if safe_weights:
            harmonic_context["context_weights"] = safe_weights
    if harmonic_context:
        payload_metadata["harmonic_context"] = harmonic_context

    return {
        "type": "inference_window",
        "session_id": session_id,
        "source": metadata.get("source"),
        "metadata": payload_metadata,
        "fs": fs,
        "position": int(win_end),
        "window_seconds": round(window_seconds, 4),
        "window": [int(win_start), int(win_end)],
        "raw_metrics": {
            str(key): float(value)
            for key, value in feature_dict.items()
            if isinstance(value, (int, float, np.floating, np.integer))
            and np.isfinite(value)
        },
        "external_signals": external_signals,
    }


# ── Main inference streaming task ───────────────────────────────────────────

async def inference_stream_task(session: Dict[str, Any]):
    """Background task that continuously scores sliding windows.

    Hooks into the playback position (like fft_stream_task) and emits
    per-model anomaly scores every `window_samples` new samples.
    """
    s = session
    sid = s.get("session_id", "?")
    metadata = s.get("metadata", {})
    source = str(metadata.get("source") or "unknown")
    logger.info(
        "[inference_stream_task] starting sid=%s source=%s",
        sid,
        source,
    )

    data: Dict[str, np.ndarray] = s["data"]
    sample_labels = _session_sample_labels(s)

    # Sampling frequency
    fs = get_sample_frequency(metadata)
    num_teeth = _metadata_num_teeth(metadata)
    logger.info(
        "[inference_stream_task] startup sid=%s source=%s fs=%.1f",
        sid,
        source,
        fs,
    )

    configured_channels = s["config"].get("channels")
    channels = configured_channels or list(data.keys())
    domain: Optional[DomainConfig] = None
    resolved_bindings: Dict[str, str] = {}

    # Inference config
    inf_cfg = s.get("inference_config", {})

    # Load models early so inference defaults can inherit live model semantics.
    model = _get_seed_model()
    model_name = "SeedModel (IF+LOF)"
    harmonic_scorer_getter = _get_harmonic_scorer

    seed_window_seconds = None
    if model is not None:
        try:
            configured_window = getattr(getattr(model, "config", None), "window_seconds", None)
            if configured_window is not None:
                seed_window_seconds = float(configured_window)
        except Exception:
            seed_window_seconds = None

    # --- Window alignment with live SeedModel / training pipeline ---
    # Prefer an explicit inference_config.window_seconds. If it is omitted,
    # inherit the loaded SeedModel training window so live feature extraction
    # defaults to the same duration as the model it feeds. Legacy
    # window_samples remains a compatibility fallback.
    if "window_seconds" in inf_cfg:
        window_seconds = float(inf_cfg["window_seconds"])
        window_samples = int(window_seconds * fs)
        logger.info(
            "[inference_stream_task] window_seconds=%.1f  → window_samples=%d (fs=%.0f)",
            window_seconds, window_samples, fs,
        )
    elif "window_samples" in inf_cfg:
        window_samples = int(inf_cfg["window_samples"])
        window_seconds = window_samples / fs
        logger.warning(
            "[inference_stream_task] window_seconds not set — using legacy "
            "window_samples=%d (%.2f s at fs=%.0f).  "
            "Set inference_config.window_seconds to avoid train/serve skew.",
            window_samples, window_seconds, fs,
        )
    elif seed_window_seconds is not None and seed_window_seconds > 0:
        window_seconds = seed_window_seconds
        window_samples = int(window_seconds * fs)
        logger.info(
            "[inference_stream_task] window_seconds not set — inheriting SeedModel "
            "training window %.1f s (%d samples at fs=%.0f)",
            window_seconds, window_samples, fs,
        )
    else:
        window_samples = 500
        window_seconds = window_samples / fs
        logger.warning(
            "[inference_stream_task] no explicit window config and SeedModel window unavailable; "
            "using legacy window_samples=%d (%.2f s at fs=%.0f)",
            window_samples, window_seconds, fs,
        )

    stride_samples = int(inf_cfg.get("stride_samples", window_samples // 2))  # 50% overlap
    inherit_speed = bool(inf_cfg.get("inherit_speed", True))

    speed = float(s["config"].get("speed", 1.0)) if inherit_speed else 1.0
    speed = max(speed, 1e-9)

    try:
        import inspect

        if len(inspect.signature(harmonic_scorer_getter).parameters) == 0:
            harmonic_scorer = harmonic_scorer_getter()
        else:
            harmonic_scorer = harmonic_scorer_getter(metadata)
    except (TypeError, ValueError):
        harmonic_scorer = harmonic_scorer_getter(metadata)
    harmonic_row_history = None
    if harmonic_scorer is not None and harmonic_scorer.is_available():
        harmonic_row_history = deque(
            maxlen=max(1, int(getattr(harmonic_scorer.config, "cnn_window", 1)))
        )
    harmonic_feature_cfg = getattr(harmonic_scorer, "config", None) or _harmonic_feature_config(metadata)

    # Dual-model setup: also load the *other* harmonic kind so the UI can show
    # context AND pair model outputs in the same window.
    primary_kind = str(
        getattr(harmonic_feature_cfg, "scorer_kind", "context") or "context"
    ).strip().lower()
    secondary_kind = "pair" if primary_kind != "pair" else "context"
    try:
        harmonic_secondary_scorer = _get_harmonic_scorer_of_kind(secondary_kind, metadata)
    except Exception:
        harmonic_secondary_scorer = None
        logger.debug("Secondary harmonic scorer load failed", exc_info=True)
    harmonic_secondary_cfg = getattr(harmonic_secondary_scorer, "config", None)
    if harmonic_secondary_cfg is None:
        try:
            meta_alt = dict(metadata or {})
            meta_alt["harmonic_scorer_kind"] = secondary_kind
            if secondary_kind == "pair":
                meta_alt["harmonic_dataset"] = _preferred_pair_dataset(meta_alt)
            else:
                meta_alt.pop("harmonic_dataset", None)
            harmonic_secondary_cfg = _harmonic_feature_config(meta_alt)
        except Exception:
            harmonic_secondary_cfg = None
    harmonic_secondary_row_history = None
    if harmonic_secondary_scorer is not None and harmonic_secondary_scorer.is_available():
        harmonic_secondary_row_history = deque(
            maxlen=max(1, int(getattr(harmonic_secondary_scorer.config, "cnn_window", 1)))
        )

    try:
        from .agents.sindit.tool_audit import record_tool_observation as _record_tool_observation
    except Exception:
        _record_tool_observation = None

    # Scheduling
    start_wall = time.perf_counter()
    frames_sent = 0
    resume_resync_needed = False
    # If playback starts from a seeked position, align the first scored window
    # to the current playback head instead of replaying inference from row 0.
    initial_position = max(0, int(s.get("position", 0) or 0))
    last_scored_end = max(window_samples - stride_samples, initial_position - stride_samples)
    feature_schema_checked = False
    model_schema_aligned = True

    logger.info("[inference_stream_task] config: window=%d stride=%d fs=%.1f model=%s",
                window_samples, stride_samples, fs,
                "loaded" if model and model.is_trained else "untrained/missing")

    try:
        while s.get("running_inference", False):
            if s.get("paused", False):
                resume_resync_needed = True
                await asyncio.sleep(0.05)
                continue

            if resume_resync_needed:
                start_wall = time.perf_counter()
                frames_sent = 0
                resume_resync_needed = False

            channels = configured_channels or list(data.keys())
            if not channels:
                if not s.get("running", False):
                    break
                await asyncio.sleep(0.05)
                continue

            try:
                n_max = min(len(data.get(ch, [])) for ch in channels)
            except Exception:
                if not s.get("running", False):
                    break
                await asyncio.sleep(0.05)
                continue

            if domain is None:
                domain = get_active_domain(channel_names=channels)
                logger.info("[inference_stream_task] domain profile: %s", domain.display_name or domain.name)
                resolved_bindings = _resolve_domain_bindings(domain, channels)
                resolved_summary = ", ".join(
                    f"{role.split('.')[-1]}={name}"
                    for role, name in resolved_bindings.items()
                    if name
                )
                missing_roles = [
                    role.split('.')[-1]
                    for role, name in resolved_bindings.items()
                    if not name
                ]
                logger.info(
                    "[inference_stream_task] domain bindings: %s%s",
                    resolved_summary or "none",
                    f" missing={missing_roles}" if missing_roles else "",
                )

            # Current playback position
            pos = int(s.get("position", 0))
            i1 = min(pos, n_max)

            # Not enough data for first window yet
            if i1 < window_samples:
                if not s.get("running", False) and i1 >= n_max:
                    break
                await asyncio.sleep(0.05)
                continue

            # Wait until we have at least stride_samples of new data
            next_end = last_scored_end + stride_samples
            if next_end > i1:
                if not s.get("running", False) and i1 >= n_max:
                    break
                await asyncio.sleep(0.05)  # yield meaningfully to avoid starving the event loop
                continue

            # Window bounds
            win_end = min(next_end, n_max)
            win_start = max(0, win_end - window_samples)

            # Extract window data per channel
            win_data: Dict[str, np.ndarray] = {}
            ok = True
            for ch in channels:
                arr = np.asarray(data[ch])
                if win_end > len(arr):
                    ok = False
                    break
                win_data[ch] = arr[win_start:win_end]
            if not ok:
                await asyncio.sleep(0.005)
                continue

            # Compute features
            feature_dict = _features_from_channels(
                win_data,
                fs,
                domain=domain,
                num_teeth=num_teeth,
            )
            harmonic_feature_dict = _augment_harmonic_runtime_features(
                feature_dict,
                win_data,
                harmonic_feature_cfg,
                metadata,
            )

            if _record_tool_observation is not None:
                try:
                    cutting_context = _runtime_observation_context(metadata, feature_dict)
                    if cutting_context is not None:
                        _record_tool_observation(sid, cutting_context)
                except Exception:
                    logger.debug("Tool observation recording from inference stream skipped", exc_info=True)

            if not feature_schema_checked:
                logger.info(
                    "[inference_stream_task] first frame sid=%s feature_keys=%s",
                    sid,
                    sorted(feature_dict.keys()),
                )
                if model and model.is_trained:
                    schema_diag = _seed_model_schema_diagnostics(model, feature_dict)
                    model_schema_aligned = bool(schema_diag["aligned"])
                    if model_schema_aligned:
                        logger.info(
                            "[inference_stream_task] seed model schema aligned sid=%s n_model_features=%d",
                            sid,
                            len(schema_diag["model_feature_order"]),
                        )
                    else:
                        logger.warning(
                            "[inference_stream_task] seed model schema mismatch sid=%s model_only=%s runtime_only=%s model_order=%s runtime_order=%s; skipping model scoring",
                            sid,
                            schema_diag["model_only"],
                            schema_diag["runtime_only"],
                            schema_diag["model_feature_order"],
                            schema_diag["runtime_feature_order"],
                        )
                feature_schema_checked = True

            # Run models

            # -- Model 1: SeedModel ensemble (IF + LOF) --
            if model and model.is_trained and model_schema_aligned:
                # Use public API — avoids coupling to private model internals
                scores = model.score_detailed_dict(feature_dict)
                ensemble_score = scores["ensemble"]
                if_score = scores["isolation_forest"]
                lof_score = scores["lof"]
            else:
                ensemble_score = 0.5
                if_score = 0.5
                lof_score = 0.5

            # -- Simple statistical anomaly score (Z-score based) --
            # Compute a basic z-score magnitude from key channels
            z_scores = []
            z_channels = [
                name
                for role_name, name in resolved_bindings.items()
                if role_name in {
                    str(ChannelRole.PRIMARY_POWER),
                    str(ChannelRole.PRIMARY_VIBRATION),
                    str(ChannelRole.CHATTER_AMPLITUDE),
                    str(ChannelRole.ACTIVE_POWER),
                }
                and name
            ]
            if not z_channels:
                z_channels = [ch_name for ch_name in domain.z_score_channels if ch_name in data]
            if not z_channels:
                z_channels = channels[:3]
            for ch_name in z_channels:
                arr = np.asarray(data.get(ch_name, []), dtype=np.float64)
                if len(arr) < window_samples:
                    continue
                # Use all data up to this point as baseline
                baseline = arr[:max(win_start, window_samples)]
                mu, sigma = float(np.mean(baseline)), float(np.std(baseline))
                if sigma < 1e-9:
                    continue
                win_mean = float(np.mean(win_data.get(ch_name, arr[win_start:win_end])))
                z = abs(win_mean - mu) / sigma
                z_scores.append(z)
            z_anomaly = float(1.0 / (1.0 + np.exp(-np.mean(z_scores) + 2))) if z_scores else 0.5

            # ── Harmonic scoring: run primary + secondary kinds ────
            # Guarded (ISS-57): a harmonic failure degrades this window to
            # ensemble/z-score only instead of raising into the outer handler,
            # which would kill the whole inference stream for the session
            # ("Waiting for inference data…" on the Monitoring page). The full
            # traceback is logged once per session so the root cause is visible.
            primary_out = None
            secondary_out = None
            try:
                primary_out = _compute_harmonic_window(
                    scorer=harmonic_scorer,
                    cfg=harmonic_feature_cfg,
                    feature_dict=harmonic_feature_dict,
                    win_data=win_data,
                    fs=fs,
                    row_history=harmonic_row_history,
                ) if harmonic_feature_cfg is not None else None
                if primary_out is not None:
                    harmonic_row_history = primary_out["row_history"]

                if harmonic_secondary_cfg is not None and harmonic_secondary_cfg is not harmonic_feature_cfg:
                    secondary_feature_dict = _augment_harmonic_runtime_features(
                        feature_dict,
                        win_data,
                        harmonic_secondary_cfg,
                        metadata,
                    )
                    secondary_out = _compute_harmonic_window(
                        scorer=harmonic_secondary_scorer,
                        cfg=harmonic_secondary_cfg,
                        feature_dict=secondary_feature_dict,
                        win_data=win_data,
                        fs=fs,
                        row_history=harmonic_secondary_row_history,
                    )
                    harmonic_secondary_row_history = secondary_out["row_history"]
            except Exception as _harm_err:
                if not s.get("_harmonic_error_logged"):
                    logger.exception(
                        "[inference_stream_task] harmonic scoring failed for session %s; "
                        "continuing with ensemble/z-score only: %s",
                        s.get("session_id"), _harm_err,
                    )
                    s["_harmonic_error_logged"] = True
                primary_out = None
                secondary_out = None

            # Map by kind: ``context_*`` is always the context model, ``pair_*``
            # is always the pair model. ``harmonic_*`` (no suffix) preserves
            # back-compat and tracks the primary scorer's output.
            context_out = None
            pair_out = None
            for out in (primary_out, secondary_out):
                if out is None:
                    continue
                if out["kind"] == "pair":
                    pair_out = out
                else:
                    context_out = out

            harmonic_score_val = primary_out["score"] if primary_out else None
            harmonic_weights = primary_out["weights"] if primary_out else []
            harmonic_labels = primary_out["labels"] if primary_out else []
            harmonic_values = primary_out["values"] if primary_out else []

            # ── Fault indicator scores ────────────────────────────
            # Derived from the physics-based features, each in [0, 1].
            fault_indicators = _compute_fault_indicators(feature_dict, domain=domain)

            model_confidence = _model_confidence(model) if model_schema_aligned else 0.0
            model_source = (
                "seed_model_v1"
                if model and model.is_trained and model_schema_aligned else None
            )
            ground_truth = _ground_truth_snapshot(sample_labels, win_end=win_end)

            # Build payload
            payload = {
                **_window_time_payload(s, win_start=win_start, win_end=win_end, fs=fs),
                "i_center": int((win_start + win_end) // 2),
                "window": [win_start, win_end],
                "window_seconds": round(window_seconds, 4),
                "window_entries": window_samples,
                "sample_rate_hz": fs,
                "fs": fs,
                "scores": {
                    "ensemble": round(ensemble_score, 4),
                    "isolation_forest": round(if_score, 4),
                    "lof": round(lof_score, 4),
                    "z_score": round(z_anomaly, 4),
                },
                "features": {k: round(v, 4) for k, v in feature_dict.items()},
                "fault_indicators": fault_indicators,
            }
            if isinstance(ground_truth, dict):
                payload.update(ground_truth)

            # Append harmonic scores: emit context and pair under explicit keys
            # so the UI can show both models in parallel. Legacy aliased keys
            # (``harmonic_feature_labels``/``harmonic_values``) follow the primary.
            context_score, context_status = _validated_harmonic_score(
                context_out,
                kind="context",
            )
            pair_score, pair_status = _validated_harmonic_score(
                pair_out,
                kind="pair",
            )
            primary_kind = (primary_out or {}).get("kind") or "context"
            primary_score, _primary_status = _validated_harmonic_score(
                primary_out,
                kind=str(primary_kind),
            )
            harmonic_status: Dict[str, str] = {}
            harmonic_thresholds: Dict[str, float] = {}
            if context_out is not None:
                context_threshold = context_out.get("decision_threshold")
                if isinstance(context_threshold, (int, float)) and np.isfinite(float(context_threshold)):
                    harmonic_thresholds["context"] = round(float(context_threshold), 4)
                if context_score is not None:
                    payload["scores"]["harmonic_context_score"] = round(context_score, 4)
                elif context_status is not None:
                    harmonic_status["context"] = context_status
                if context_out["weights"]:
                    payload["harmonic_context_weights"] = context_out["weights"]
                if context_out["labels"] or context_out["values"]:
                    payload["harmonic_context_feature_labels"] = context_out["labels"]
                    payload["harmonic_context_values"] = context_out["values"]
            if pair_out is not None:
                pair_threshold = pair_out.get("decision_threshold")
                if isinstance(pair_threshold, (int, float)) and np.isfinite(float(pair_threshold)):
                    harmonic_thresholds["pair"] = round(float(pair_threshold), 4)
                if pair_score is not None:
                    payload["scores"]["harmonic_pair_score"] = round(pair_score, 4)
                elif pair_status is not None:
                    harmonic_status["pair"] = pair_status
                if pair_out["weights"]:
                    payload["harmonic_pair_weights"] = pair_out["weights"]
                if pair_out["labels"] or pair_out["values"]:
                    payload["harmonic_pair_feature_labels"] = pair_out["labels"]
                    payload["harmonic_pair_values"] = pair_out["values"]
            if harmonic_thresholds:
                payload["harmonic_thresholds"] = harmonic_thresholds
            # Legacy aliases mirror the primary scorer's output.
            if primary_score is not None:
                payload["scores"].setdefault(
                    "harmonic_context_score", round(primary_score, 4)
                )
            if harmonic_weights and "harmonic_context_weights" not in payload:
                payload["harmonic_context_weights"] = harmonic_weights
            if (harmonic_labels or harmonic_values) and "harmonic_feature_labels" not in payload:
                payload["harmonic_feature_labels"] = harmonic_labels
                payload["harmonic_values"] = harmonic_values
            if harmonic_status:
                payload["harmonic_status"] = harmonic_status

            # Broadcast to inference subscribers
            subscribers = list(s.get("inference_subscribers", []))
            if subscribers:
                await asyncio.gather(
                    *(_safe_put(q, payload, 1.0) for q in subscribers),
                    return_exceptions=True,
                )

            if model_source or harmonic_score_val is not None:
                await publish_feature(
                    sid,
                    _build_memory_feature_payload(
                        session_id=sid,
                        metadata=metadata,
                        fs=fs,
                        win_start=win_start,
                        win_end=win_end,
                        window_seconds=window_seconds,
                        feature_dict=harmonic_feature_dict,
                        ensemble_score=ensemble_score if model_source else None,
                        z_anomaly=z_anomaly,
                        harmonic_score_val=harmonic_score_val,
                        harmonic_labels=harmonic_labels,
                        harmonic_values=harmonic_values,
                        harmonic_weights=harmonic_weights,
                        ground_truth=ground_truth,
                        fault_indicators=fault_indicators,
                        model_confidence=model_confidence,
                        model_source=model_source,
                    ),
                )

            last_scored_end = win_end
            frames_sent += 1

            # Drift-resistant scheduling
            target_elapsed = (frames_sent * stride_samples / fs) / speed
            delay = (start_wall + target_elapsed) - time.perf_counter()
            await asyncio.sleep(max(delay, 0))

    except Exception as e:
        logger.exception("[inference_stream_task] error: %s", e)
    finally:
        logger.info("[inference_stream_task] finished for session %s (%d frames)", sid, frames_sent)
        s["inference_task"] = None
        s["running_inference"] = False

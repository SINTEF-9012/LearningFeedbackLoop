from __future__ import annotations

from typing import Dict, Optional


TOOL_BREAKAGE_SIGNATURE = "signature:hf_burst_periodicity_loss"
CHATTER_SIGNATURE = "signature:modulated_tooth_passing_vibration"
CHIP_ADHESION_SIGNATURE = "signature:irregular_tooth_passing"
WORKPIECE_SLIP_SIGNATURE = "signature:spindle_shift_phase_change"


FAULT_NAME_TO_SIGNATURE_KEY: Dict[str, str] = {
    "tool_breakage": TOOL_BREAKAGE_SIGNATURE,
    "chatter": CHATTER_SIGNATURE,
    "chip_adhesion": CHIP_ADHESION_SIGNATURE,
    "workpiece_slip": WORKPIECE_SLIP_SIGNATURE,
}


def signature_key_for_fault_name(name: str) -> str:
    clean = str(name).strip().lower()
    return FAULT_NAME_TO_SIGNATURE_KEY.get(clean, f"signature:{clean}")


def normalize_signature_key(key: str) -> str:
    raw = str(key).strip()
    if not raw:
        return raw
    lowered = raw.lower()
    if lowered.startswith("fault:") or lowered.startswith("hypothesis:"):
        return signature_key_for_fault_name(raw.split(":", 1)[1])
    return raw


def infer_pattern_kind(key: str, pattern_type: Optional[str] = None) -> str:
    raw = str(key).strip()
    canonical = normalize_signature_key(raw)
    ptype = (pattern_type or "").strip().lower()
    upper = raw.upper()

    if canonical.startswith("signature:"):
        return "signature"
    if canonical.startswith("discovered:") or canonical.startswith("suppressed:"):
        return "discovered"
    if ptype == "anomaly" or upper.startswith("ANOMALY") or upper.startswith("OUTLIER"):
        return "model_score"
    if upper in {
        "SPINDLE_POWER_SURGE",
        "VIBRATION_REGIME_SHIFT",
        "FEED_OVERRIDE_DROP",
        "SENSOR_DECORRELATION",
        "SPINDLE_LOAD_RAMP",
        "FEED_STALL",
        "POWER_ASYMMETRY",
        "ENERGY_ACCUMULATION",
        "VARIANCE_EXPLOSION",
        "TREND_REVERSAL",
        "AUTOCORRELATION_BREAK",
    }:
        return "domain_rule"
    if canonical.startswith((
        "freq:",
        "amp:",
        "temporal:",
        "spectral:",
        "kurtosis:",
        "snr:",
        "energy:",
        "correlation:",
        "coherence:",
        "phase:",
    )) or upper.startswith("RATIO_"):
        return "generic_physics"
    return "domain_rule"
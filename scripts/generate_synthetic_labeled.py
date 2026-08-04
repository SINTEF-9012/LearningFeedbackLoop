#!/usr/bin/env python3
"""
Generate synthetic labelled training data from real CNC data distributions.

[PROTOTYPE_CLASSICAL_RL_V1]

Reads real data via DatasetLoader.extract_bulk_features() to estimate
per-feature mean/std, then generates synthetic windows with known labels:
  0 = normal
  1 = chatter
  2 = tool_wear
  3 = breakage_risk

Also generates 12 demo event JSON files (8 warmup + 4 live) in
scripts/demo_data_labeled/ with correct metrics dicts for classical model
scoring.

Usage:
    python scripts/generate_synthetic_labeled.py
    python scripts/generate_synthetic_labeled.py --max-real-windows 500
    python scripts/generate_synthetic_labeled.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# -------------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.agents.processing.dataset_loader import DatasetLoader
from backend.agents.processing.classical_models import FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

CASEDATA_ROOT = ROOT / "data" / "casedata"
OUTPUT_NPZ = ROOT / "data" / "models" / "synthetic_training.npz"
DEMO_DIR = ROOT / "scripts" / "demo_data_labeled"

# How many synthetic windows per label
COUNTS = {
    0: 500,   # normal
    1: 80,    # chatter
    2: 40,    # tool_wear
    3: 20,    # breakage_risk
    4: 30,    # chip_adhesion
    5: 25,    # workpiece_slip
}

LABEL_NAMES = {
    0: "normal",
    1: "chatter",
    2: "tool_wear",
    3: "breakage_risk",
    4: "chip_adhesion",
    5: "workpiece_slip",
}

# Noise scale for normal windows (fraction of std)
NORMAL_NOISE = 0.15

# Feature indices (must match FEATURE_NAMES order)
IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


# -------------------------------------------------------------------------
# Distribution estimation from real data
# -------------------------------------------------------------------------

def estimate_distributions(
    max_real_windows: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) vectors of shape (17,) from real data."""
    loader = DatasetLoader(CASEDATA_ROOT)
    ops = loader.list_operations()
    logger.info("Estimating distributions from %d operations", len(ops))

    all_features: list[list[float]] = []
    per_op = max(max_real_windows // max(len(ops), 1), 50)

    for op in ops:
        logger.info("  Loading %s (max %d windows)...", op.operation_id, per_op)
        feats = loader.extract_bulk_features(
            op.operation_id, window_seconds=10, stride_seconds=10, max_windows=per_op
        )
        for f in feats:
            vec = [float(f.get(name, 0.0)) for name in FEATURE_NAMES]
            all_features.append(vec)

    arr = np.array(all_features, dtype=np.float64)
    logger.info("Estimated from %d real windows, %d features", arr.shape[0], arr.shape[1])

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0) + 1e-8  # avoid zero std
    # Replace NaN (features that couldn't be computed for any window)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    return mean, std


# -------------------------------------------------------------------------
# Synthetic sample generation
# -------------------------------------------------------------------------

def generate_normal(
    n: int, mean: np.ndarray, std: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Normal operation windows — sampled from fitted distribution."""
    noise = rng.normal(0, NORMAL_NOISE, size=(n, len(mean)))
    return mean + noise * std


def generate_chatter(
    n: int, mean: np.ndarray, std: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Chatter windows — elevated vibration + chatter features."""
    base = generate_normal(n, mean, std, rng)

    # Boost chatter-related features
    base[:, IDX["chatter_ratio"]] = rng.uniform(3.0, 15.0, n)
    base[:, IDX["vib_severity_x_mean"]] = mean[IDX["vib_severity_x_mean"]] + rng.uniform(1.5, 4.0, n) * std[IDX["vib_severity_x_mean"]]
    base[:, IDX["vib_severity_x_max"]] = mean[IDX["vib_severity_x_max"]] + rng.uniform(2.0, 5.0, n) * std[IDX["vib_severity_x_max"]]
    base[:, IDX["vib_severity_y_mean"]] = mean[IDX["vib_severity_y_mean"]] + rng.uniform(1.0, 3.5, n) * std[IDX["vib_severity_y_mean"]]
    base[:, IDX["vib_severity_y_max"]] = mean[IDX["vib_severity_y_max"]] + rng.uniform(1.5, 4.5, n) * std[IDX["vib_severity_y_max"]]

    # Slight power increase from chatter-induced forces
    base[:, IDX["power_spindle_std"]] *= rng.uniform(1.2, 2.0, n)

    # Physics-based: modulated vibration, increased amplitude
    if "modulation_depth" in IDX:
        base[:, IDX["modulation_depth"]] = rng.uniform(2.0, 8.0, n)
        base[:, IDX["vib_amplitude_growth"]] = rng.uniform(1.5, 3.0, n)
        base[:, IDX["periodicity_strength"]] *= rng.uniform(0.5, 0.8, n)  # less periodic
        base[:, IDX["harmonic_amplitude_cv"]] = rng.uniform(0.6, 1.5, n)

    return base


def generate_tool_wear(
    n: int, mean: np.ndarray, std: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Tool wear windows — elevated power + degraded power factor."""
    base = generate_normal(n, mean, std, rng)

    # Power increases as tool dulls
    base[:, IDX["power_spindle_mean"]] *= rng.uniform(1.3, 1.8, n)
    base[:, IDX["power_spindle_max"]] *= rng.uniform(1.4, 2.0, n)
    base[:, IDX["power_y_mean"]] *= rng.uniform(1.2, 1.6, n)
    base[:, IDX["power_active_mean"]] *= rng.uniform(1.2, 1.7, n)
    base[:, IDX["power_active_std"]] *= rng.uniform(1.3, 2.0, n)

    # Power factor degrades
    base[:, IDX["power_factor_mean"]] *= rng.uniform(0.6, 0.85, n)

    # Temperature rises
    base[:, IDX["temp_head_mean"]] += rng.uniform(3.0, 12.0, n)

    # Physics-based: gradual degradation
    if "tp_harmonic_energy" in IDX:
        base[:, IDX["tp_harmonic_energy"]] *= rng.uniform(1.3, 2.0, n)
        base[:, IDX["harmonic_amplitude_cv"]] = rng.uniform(0.4, 1.0, n)
        base[:, IDX["kurtosis_max"]] = rng.uniform(1.0, 3.0, n)

    return base


def generate_breakage_risk(
    n: int, mean: np.ndarray, std: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Breakage risk windows — extreme power spikes + vibration surge."""
    base = generate_normal(n, mean, std, rng)

    # Extreme power spikes
    base[:, IDX["power_spindle_mean"]] *= rng.uniform(2.0, 3.5, n)
    base[:, IDX["power_spindle_max"]] *= rng.uniform(2.5, 5.0, n)
    base[:, IDX["power_spindle_std"]] *= rng.uniform(2.0, 4.0, n)
    base[:, IDX["power_y_mean"]] *= rng.uniform(1.8, 3.0, n)
    base[:, IDX["power_y_max"]] *= rng.uniform(2.0, 4.0, n)
    base[:, IDX["power_z_mean"]] *= rng.uniform(1.5, 2.5, n)

    # Vibration spike (often precedes breakage)
    base[:, IDX["vib_severity_x_max"]] = mean[IDX["vib_severity_x_max"]] + rng.uniform(3.0, 8.0, n) * std[IDX["vib_severity_x_max"]]
    base[:, IDX["vib_severity_y_max"]] = mean[IDX["vib_severity_y_max"]] + rng.uniform(3.0, 7.0, n) * std[IDX["vib_severity_y_max"]]

    # Chatter often accompanies breakage
    base[:, IDX["chatter_ratio"]] = rng.uniform(5.0, 20.0, n)

    # Feed rate drops (machine struggling)
    base[:, IDX["feed_rate_mean"]] *= rng.uniform(0.3, 0.7, n)

    # Physics-based: sudden HF burst, loss of periodicity
    if "hf_energy_ratio" in IDX:
        base[:, IDX["hf_energy_ratio"]] = rng.uniform(0.5, 0.95, n)
        base[:, IDX["impulse_crest_factor"]] = rng.uniform(6.0, 20.0, n)
        base[:, IDX["kurtosis_max"]] = rng.uniform(5.0, 15.0, n)
        base[:, IDX["periodicity_strength"]] *= rng.uniform(0.1, 0.3, n)  # loss of periodicity
        base[:, IDX["modulation_depth"]] = rng.uniform(3.0, 10.0, n)

    return base


def generate_chip_adhesion(
    n: int, mean: np.ndarray, std: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Chip adhesion / built-up edge — irregular tooth-passing pattern."""
    base = generate_normal(n, mean, std, rng)

    # Moderate power increase from adhesion friction
    base[:, IDX["power_spindle_mean"]] *= rng.uniform(1.1, 1.4, n)
    base[:, IDX["power_active_mean"]] *= rng.uniform(1.1, 1.3, n)

    # Vibration severity slightly elevated, irregular
    base[:, IDX["vib_severity_x_mean"]] *= rng.uniform(1.2, 2.0, n)
    base[:, IDX["vib_severity_y_mean"]] *= rng.uniform(1.1, 1.8, n)

    # Temperature rises from adhesion friction
    base[:, IDX["temp_head_mean"]] += rng.uniform(2.0, 8.0, n)

    # Physics-based: irregular tooth-passing pattern
    if "tp_amplitude_variance" in IDX:
        base[:, IDX["tp_amplitude_variance"]] = rng.uniform(0.5, 2.0, n)  # HIGH variance
        base[:, IDX["harmonic_amplitude_cv"]] = rng.uniform(0.8, 2.0, n)  # irregular
        base[:, IDX["tp_harmonic_energy"]] *= rng.uniform(0.6, 0.9, n)  # slightly reduced
        base[:, IDX["periodicity_strength"]] *= rng.uniform(0.4, 0.7, n)  # irregular period
        base[:, IDX["kurtosis_max"]] = rng.uniform(2.0, 6.0, n)
        base[:, IDX["impulse_crest_factor"]] = rng.uniform(3.0, 7.0, n)

    return base


def generate_workpiece_slip(
    n: int, mean: np.ndarray, std: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Workpiece slip / clamping issue — shift at spindle frequency."""
    base = generate_normal(n, mean, std, rng)

    # Vibration severity spikes from sudden slip
    base[:, IDX["vib_severity_x_max"]] *= rng.uniform(2.0, 4.0, n)
    base[:, IDX["vib_severity_y_max"]] *= rng.uniform(1.5, 3.0, n)

    # Power momentarily increases or drops
    base[:, IDX["power_spindle_std"]] *= rng.uniform(1.5, 3.0, n)

    # Feed rate may drop if controller reacts
    base[:, IDX["feed_rate_mean"]] *= rng.uniform(0.5, 0.9, n)

    # Physics-based: shift at spindle frequency
    if "spindle_order_amplitude" in IDX:
        base[:, IDX["spindle_order_amplitude"]] = rng.uniform(3.0, 10.0, n)  # high 1× spindle
        base[:, IDX["spindle_phase_shift"]] = rng.uniform(0.5, 3.0, n)  # large phase shift
        base[:, IDX["periodicity_strength"]] *= rng.uniform(0.3, 0.6, n)  # disrupted
        base[:, IDX["vib_amplitude_growth"]] = rng.uniform(2.0, 5.0, n)  # sudden growth
        base[:, IDX["impulse_crest_factor"]] = rng.uniform(4.0, 10.0, n)  # impulsive

    return base


GENERATORS = {
    0: generate_normal,
    1: generate_chatter,
    2: generate_tool_wear,
    3: generate_breakage_risk,
    4: generate_chip_adhesion,
    5: generate_workpiece_slip,
}


# -------------------------------------------------------------------------
# Demo event generation from synthetic samples
# -------------------------------------------------------------------------

PATTERN_KEYS_BY_LABEL = {
    0: ["STABLE_CUT", "POWER_NOMINAL"],
    1: ["CHATTER_DETECTED", "VIB_SEVERITY_HIGH", "ANOMALY_HIGH_VIBRATION", "FAULT_CHATTER"],
    2: ["POWER_SPIKE_SUSTAINED", "ANOMALY_HIGH_POWER", "TOOL_WEAR_RISK"],
    3: ["BREAKAGE_RISK_HIGH", "CHATTER_DETECTED", "POWER_SPIKE_EXTREME", "ANOMALY_HIGH_VIBRATION", "FAULT_TOOL_BREAKAGE"],
    4: ["FAULT_CHIP_ADHESION", "VIB_SEVERITY_HIGH", "ANOMALY_HIGH_POWER"],
    5: ["FAULT_WORKPIECE_SLIP", "VIB_SEVERITY_HIGH", "ANOMALY_HIGH_VIBRATION"],
}

CUTTING_CONTEXT = {
    "machine_type": "cnc_mill",
    "tool_type": "end_mill",
    "workpiece_material": "steel",
    "operating_regime": "roughing",
}


def make_demo_event(
    feature_vec: np.ndarray,
    label: int,
    event_idx: int,
    session_id: str = "demo-session",
) -> dict:
    """Build a demo event dict from a synthetic feature vector."""
    metrics = {
        name: round(float(feature_vec[i]), 6)
        for i, name in enumerate(FEATURE_NAMES)
    }

    # Derive external_signals from features
    external_signals: dict = {}
    if label == 3:
        external_signals["breakage_prediction"] = round(
            float(np.clip(0.5 + feature_vec[IDX["power_spindle_max"]] / 100, 0.4, 0.95)), 3
        )
    if label in (2, 3):
        # Tool wear estimate: lower = more worn
        external_signals["tool_wear_estimate"] = round(
            float(np.clip(1.0 - feature_vec[IDX["power_spindle_mean"]] / 80, 0.05, 0.9)), 3
        )
    if label == 1 and "modulation_depth" in IDX:
        external_signals["chatter_severity"] = round(
            float(np.clip(feature_vec[IDX["modulation_depth"]] / 10.0, 0.2, 0.95)), 3
        )
    if label == 5 and "spindle_phase_shift" in IDX:
        external_signals["slip_likelihood"] = round(
            float(np.clip(feature_vec[IDX["spindle_phase_shift"]] / 3.0, 0.3, 0.95)), 3
        )

    return {
        "session_id": session_id,
        "time_range": {
            "i0": event_idx * 100,
            "i1": (event_idx + 1) * 100,
            "t0": float(event_idx * 10),
            "t1": float((event_idx + 1) * 10),
            "fs": 10.0,
        },
        "pattern_keys": PATTERN_KEYS_BY_LABEL[label],
        "channels": ["Power_Spindle", "Power_Y", "Vibration_Severity_X", "Vibration_Severity_Y"],
        "cutting_context": CUTTING_CONTEXT,
        "external_signals": external_signals if external_signals else None,
        "metrics": metrics,
        "label": LABEL_NAMES[label],
        "label_id": label,
    }


# 16 events: 8 warmup (pre-seeded history) + 8 live
DEMO_SEQUENCE = [
    # -- Warmup events (pre-seeded history) --
    {"label": 0, "name": "warmup_1_normal_steel"},
    {"label": 0, "name": "warmup_2_normal_steel"},
    {"label": 1, "name": "warmup_3_chatter_steel"},          # will be confirmed
    {"label": 0, "name": "warmup_4_normal_steel"},            # will be dismissed (false alarm)
    {"label": 2, "name": "warmup_5_tool_wear_steel"},         # will be confirmed
    {"label": 1, "name": "warmup_6_chatter_steel"},           # will be confirmed
    {"label": 4, "name": "warmup_7_chip_adhesion_steel"},     # will be confirmed
    {"label": 5, "name": "warmup_8_workpiece_slip_steel"},    # will be confirmed
    # -- Live events (operator sees these) --
    {"label": 0, "name": "event_1_normal_production"},
    {"label": 1, "name": "event_2_chatter_alert"},            # prior-boosted (warm for steel)
    {"label": 0, "name": "event_3_normal_steady"},
    {"label": 4, "name": "event_4_chip_adhesion_detected"},   # irregular tooth-passing
    {"label": 2, "name": "event_5_tool_wear_detected"},       # prior-boosted
    {"label": 5, "name": "event_6_workpiece_slip"},           # spindle-freq shift
    {"label": 3, "name": "event_7_breakage_risk"},            # high score
    {"label": 0, "name": "event_8_normal_recovery"},
]


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic labelled CNC data")
    parser.add_argument(
        "--max-real-windows", type=int, default=2000,
        help="Max real windows to read for distribution estimation",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even if output files exist",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if OUTPUT_NPZ.exists() and DEMO_DIR.exists() and not args.force:
        logger.info("Output files already exist. Use --force to regenerate.")
        return

    # Estimate distributions from real data
    logger.info("=== Estimating distributions from real CNC data ===")
    mean, std = estimate_distributions(max_real_windows=args.max_real_windows)

    logger.info("Feature distributions (mean ± std):")
    for i, name in enumerate(FEATURE_NAMES):
        logger.info("  %-25s  %8.3f ± %8.3f", name, mean[i], std[i])

    # Generate synthetic samples
    logger.info("=== Generating synthetic samples ===")
    rng = np.random.default_rng(args.seed)

    X_parts = []
    y_parts = []
    for label, count in COUNTS.items():
        gen_fn = GENERATORS[label]
        samples = gen_fn(count, mean, std, rng)
        X_parts.append(samples)
        y_parts.append(np.full(count, label, dtype=np.int32))
        logger.info("  Generated %d %s samples", count, LABEL_NAMES[label])

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Clip negative values (physical features can't be negative)
    X = np.clip(X, 0.0, None)

    # Save NPZ
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_NPZ,
        X=X,
        y=y,
        feature_names=FEATURE_NAMES,
        label_names=list(LABEL_NAMES.values()),
        mean=mean,
        std=std,
    )
    logger.info("Saved %s  (X: %s, y: %s)", OUTPUT_NPZ, X.shape, y.shape)

    # Generate demo event JSON files
    logger.info("=== Generating demo event files ===")
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    for event_idx, spec in enumerate(DEMO_SEQUENCE):
        label = spec["label"]
        name = spec["name"]

        # Pick a representative sample for this label
        gen_fn = GENERATORS[label]
        sample = gen_fn(1, mean, std, rng)[0]
        sample = np.clip(sample, 0.0, None)

        event = make_demo_event(sample, label, event_idx)
        filepath = DEMO_DIR / f"{name}.json"
        with open(filepath, "w") as f:
            json.dump(event, f, indent=2)
        logger.info("  %s  (label=%s)", filepath.name, LABEL_NAMES[label])

    logger.info("=== Done ===")
    logger.info("  NPZ: %s", OUTPUT_NPZ)
    logger.info("  Demo events: %s (%d files)", DEMO_DIR, len(DEMO_SEQUENCE))
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Retrain seed model: python scripts/train_seed_model.py --labeled --force")
    logger.info("  2. Run warm-up:        python scripts/warmup_demo_history.py")
    logger.info("  3. Run demo:           python scripts/demo_ui_mock_run.py --labeled --warmup")


if __name__ == "__main__":
    main()

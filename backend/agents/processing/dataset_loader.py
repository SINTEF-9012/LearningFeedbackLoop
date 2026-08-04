"""
Dataset Loader — read CNC machining time-series from a plain CSV directory tree.

This is the generic, bring-your-own-data adapter. It makes no assumption about
vendor-specific file naming: channel groups are identified from the **CSV
header columns** (see ``KEY_COLUMNS``), with the file *stem* used only as a
secondary hint. Any dataset laid out as described below can be streamed by the
system without code changes.

Expected layout (either form works)::

    <root>/<case>/<operation>/*.csv     # grouped by case
    <root>/<operation>/*.csv            # flat

``<operation>`` directories are named ``OF<number>`` (OF = "operation folder"),
e.g. ``OF00001``; each holds one machining run for one tool. Inside an
operation directory, put one CSV per sensor group. Name them after the group
for clarity — ``axis_power.csv``, ``vibration.csv``, ``energy.csv``,
``machine_state.csv`` — though detection is driven by the column headers, so
any filename works as long as the columns match.

Recognised channel groups and their columns are declared in ``KEY_COLUMNS``
below. A CSV needs only a subset of a group's columns to be matched to it; the
group with the most overlapping columns wins.

A minimal synthetic dataset in this layout ships under ``test_data/`` so the
system is runnable immediately after cloning.

Usage:
    loader = DatasetLoader("data/my_dataset")
    ops = loader.list_operations()
    window = loader.extract_window("OF00001", "2025-01-01T00:00:00Z", "2025-01-01T00:00:30Z")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .tool_lookup import lookup as lookup_tool_spec, resolve_machine_family

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]
    logger.warning("pandas not installed — DatasetLoader unavailable")


# Key columns per channel group. These drive channel-group detection: a CSV is
# assigned to whichever group shares the most column names with it. Extend this
# mapping to teach the loader about additional sensor groups.
KEY_COLUMNS: Dict[str, List[str]] = {
    "axis_power": [
        "Operation_Status", "Power_Spindle", "Power_X1", "Power_X2",
        "Power_Y", "Power_Z",
    ],
    "vibration": [
        "Chatter_Detection_OnOff_X", "Chatter_Detection_OnOff_Y",
        "Chatter_Detection_Amplitude_X", "Chatter_Detection_Amplitude_Y",
        "Chatter_Detection_Frequency_X", "Chatter_Detection_Frequency_Y",
        "Vibration_Severity_X", "Vibration_Severity_Y",
    ],
    "energy": [
        "Power_Active", "Power_Apparent", "Power_Factor", "Power_Reactive",
    ],
    "machine_state": [
        "Feed_Rate_Actual", "Feed_Rate_Commanded", "Spindle_Speed_Actual",
        "Spindle_Speed_Commanded", "Temperature_Head", "Temperature_Room",
        "Tool_Number", "Program_Name", "Operation_Mode",
        "Power_Spindle", "Power_Active",
    ],
}

CHANNEL_GROUPS: Tuple[str, ...] = tuple(KEY_COLUMNS.keys())


def _match_channel_group(columns: set[str]) -> Optional[str]:
    best_group: Optional[str] = None
    best_score = 0
    for group_name, expected in KEY_COLUMNS.items():
        score = sum(1 for column in expected if column in columns)
        if score > best_score:
            best_group = group_name
            best_score = score
    return best_group if best_score > 0 else None


def _detect_channel_group(csv_file: Path) -> Optional[str]:
    if pd is not None:
        try:
            header = pd.read_csv(csv_file, nrows=0)
            detected = _match_channel_group(set(str(column) for column in header.columns))
            if detected is not None:
                return detected
        except Exception:
            logger.debug("Failed to inspect CSV header for %s", csv_file, exc_info=True)

    stem = csv_file.stem.lower()
    for group_name in KEY_COLUMNS:
        if group_name in stem:
            return group_name
    return None


@dataclass
class OperationInfo:
    """Summary of a machining operation found in casedata."""
    operation_id: str
    case_dir: str
    tool_id: str
    channel_files: Dict[str, Path]
    row_counts: Dict[str, int] = field(default_factory=dict)
    time_range: Optional[Tuple[str, str]] = None  # (first_ts, last_ts)


@dataclass
class WindowData:
    """Extracted time window across all sensor channels."""
    operation_id: str
    t_start: str  # ISO timestamp
    t_end: str
    duration_s: float
    n_samples: int
    case_dir: Optional[str] = None

    # Raw data per channel group
    axis_power: Optional[Any] = None  # pd.DataFrame
    vibration: Optional[Any] = None
    energy: Optional[Any] = None
    machine_state: Optional[Any] = None

    # Derived features
    features: Dict[str, float] = field(default_factory=dict)

    def compute_features(self) -> Dict[str, float]:
        """Compute summary features for this window."""
        feats: Dict[str, float] = {}

        frames_by_group = {
            "axis_power": self.axis_power,
            "vibration": self.vibration,
            "energy": self.energy,
            "machine_state": self.machine_state,
        }

        def _series(column: str, *preferred_groups: str):
            groups = preferred_groups or CHANNEL_GROUPS
            for group_name in groups:
                frame = frames_by_group.get(group_name)
                if frame is not None and column in frame.columns:
                    return frame[column]
            return None

        power_spindle = _series("Power_Spindle", "axis_power", "machine_state")
        if power_spindle is not None and len(power_spindle) > 0:
            feats["power_spindle_mean"] = float(power_spindle.mean())
            feats["power_spindle_max"] = float(power_spindle.max())
            feats["power_spindle_std"] = float(power_spindle.std())

        power_y = _series("Power_Y", "axis_power")
        if power_y is not None and len(power_y) > 0:
            feats["power_y_mean"] = float(power_y.mean())
            feats["power_y_max"] = float(power_y.max())

        power_z = _series("Power_Z", "axis_power")
        if power_z is not None and len(power_z) > 0:
            feats["power_z_mean"] = float(power_z.mean())

        op_status = _series("Operation_Status", "axis_power", "machine_state")
        if op_status is not None and len(op_status) > 0:
            feats["op_status_mode"] = float(op_status.mode().iloc[0])

        if self.vibration is not None and len(self.vibration) > 0:
            vb = self.vibration
            feats["vib_severity_x_mean"] = float(vb["Vibration_Severity_X"].mean())
            feats["vib_severity_x_max"] = float(vb["Vibration_Severity_X"].max())
            feats["vib_severity_y_mean"] = float(vb["Vibration_Severity_Y"].mean())
            feats["vib_severity_y_max"] = float(vb["Vibration_Severity_Y"].max())
            chatter_x = float((vb["Chatter_Detection_OnOff_X"] > 0).sum())
            chatter_y = float((vb["Chatter_Detection_OnOff_Y"] > 0).sum())
            feats["chatter_x_count"] = chatter_x
            feats["chatter_y_count"] = chatter_y
            feats["chatter_ratio"] = (chatter_x + chatter_y) / max(len(vb) * 2, 1)
            if chatter_x > 0:
                chatter_rows = vb[vb["Chatter_Detection_OnOff_X"] > 0]
                feats["chatter_amp_x_max"] = float(
                    chatter_rows["Chatter_Detection_Amplitude_X"].max()
                )
                feats["chatter_freq_x_dominant"] = float(
                    chatter_rows["Chatter_Detection_Frequency_X"].mode().iloc[0]
                )

        power_active = _series("Power_Active", "energy", "machine_state")
        if power_active is not None and len(power_active) > 0:
            feats["power_active_mean"] = float(power_active.mean())
            feats["power_active_std"] = float(power_active.std())

        power_factor = _series("Power_Factor", "energy", "machine_state")
        if power_factor is not None and len(power_factor) > 0:
            feats["power_factor_mean"] = float(power_factor.mean())

        spindle_speed = _series("Spindle_Speed_Actual", "machine_state")
        if spindle_speed is not None and len(spindle_speed) > 0:
            feats["spindle_speed_mean"] = float(spindle_speed.mean())

        feed_rate = _series("Feed_Rate_Actual", "machine_state")
        if feed_rate is not None and len(feed_rate) > 0:
            feats["feed_rate_mean"] = float(feed_rate.mean())

        temp_head = _series("Temperature_Head", "machine_state")
        if temp_head is not None and len(temp_head) > 0:
            feats["temp_head_mean"] = float(temp_head.mean())

        tool = _series("Tool_Number", "machine_state")
        if tool is not None and len(tool) > 0:
            mode = tool.mode()
            if len(mode) > 0:
                feats["tool_number"] = float(mode.iloc[0])

        # ================================================================
        # Physics-based fault features (11 new features)
        # Computed from vibration harmonics + spindle speed
        # ================================================================
        spindle_rpm = feats.get("spindle_speed_mean", 0.0)

        if self.vibration is not None and len(self.vibration) > 0:
            vb = self.vibration

            # --- Collect harmonic amplitudes and frequencies ---
            harmonic_amps_x: list[float] = []
            harmonic_freqs_x: list[float] = []
            harmonic_amps_y: list[float] = []
            for h in range(1, 9):
                col_ax = f"Vibration_Harmonic_{h}_X_Amplitude"
                col_fx = f"Vibration_Harmonic_{h}_X_Frequency"
                col_ay = f"Vibration_Harmonic_{h}_Y_Amplitude"
                if col_ax in vb.columns:
                    harmonic_amps_x.append(float(vb[col_ax].mean()))
                    harmonic_freqs_x.append(float(vb[col_fx].mean()) if col_fx in vb.columns else 0.0)
                if col_ay in vb.columns:
                    harmonic_amps_y.append(float(vb[col_ay].mean()))

            all_harmonic_amps = harmonic_amps_x + harmonic_amps_y
            total_harmonic_energy = sum(a ** 2 for a in all_harmonic_amps) if all_harmonic_amps else 1e-10

            # (1) hf_energy_ratio — fraction of harmonic energy above cutoff.
            # Note: harmonic frequencies are physical (from vibration sensor), not
            # sample-rate dependent.  500 Hz is a machine-domain threshold for
            # "high-frequency" vibration in CNC milling context.  If data comes
            # from a sensor with Nyquist < 500 Hz the harmonics themselves will
            # never exceed that, so the ratio naturally goes to 0.
            hf_cutoff = 500.0
            hf_energy = 0.0
            for amp, freq in zip(harmonic_amps_x, harmonic_freqs_x):
                if freq > hf_cutoff:
                    hf_energy += amp ** 2
            feats["hf_energy_ratio"] = hf_energy / (total_harmonic_energy + 1e-10)

            # (2) impulse_crest_factor — max severity peak / mean severity
            sev_x = vb["Vibration_Severity_X"] if "Vibration_Severity_X" in vb.columns else None
            sev_y = vb["Vibration_Severity_Y"] if "Vibration_Severity_Y" in vb.columns else None
            crest_vals = []
            for sev in [sev_x, sev_y]:
                if sev is not None and len(sev) > 0:
                    sev_arr = sev.values.astype(float)
                    rms = float(np.sqrt(np.mean(sev_arr ** 2)))
                    peak = float(np.max(np.abs(sev_arr)))
                    crest_vals.append(peak / (rms + 1e-12))
            feats["impulse_crest_factor"] = max(crest_vals) if crest_vals else 0.0

            # (3) kurtosis_max — max excess kurtosis across severity channels
            kurt_vals = []
            for sev in [sev_x, sev_y]:
                if sev is not None and len(sev) > 0:
                    sev_arr = sev.values.astype(float)
                    std_val = float(np.std(sev_arr))
                    if std_val > 1e-12:
                        norm = (sev_arr - np.mean(sev_arr)) / std_val
                        kurt_vals.append(float(np.mean(norm ** 4) - 3.0))
            feats["kurtosis_max"] = max(kurt_vals) if kurt_vals else 0.0

            # (4) periodicity_strength — ratio of harmonic 1 to broadband
            h1_amp = harmonic_amps_x[0] if harmonic_amps_x else 0.0
            mean_amp = float(np.mean(harmonic_amps_x)) if harmonic_amps_x else 1e-10
            feats["periodicity_strength"] = h1_amp / (mean_amp + 1e-10)

            # (5) modulation_depth — chatter amplitude envelope crest
            # Use chatter detection amplitudes as modulation envelope proxy
            chatter_amp_x = vb["Chatter_Detection_Amplitude_X"] if "Chatter_Detection_Amplitude_X" in vb.columns else None
            if chatter_amp_x is not None and len(chatter_amp_x) > 0:
                env = chatter_amp_x.values.astype(float)
                env_rms = float(np.sqrt(np.mean(env ** 2)))
                env_peak = float(np.max(np.abs(env)))
                feats["modulation_depth"] = max(0.0, (env_peak / (env_rms + 1e-12)) - 1.0)
            else:
                feats["modulation_depth"] = 0.0

            # (6) vib_amplitude_growth — max severity / median severity
            if sev_x is not None and len(sev_x) > 0:
                sev_arr = sev_x.values.astype(float)
                median_sev = float(np.median(np.abs(sev_arr)))
                rms_sev = float(np.sqrt(np.mean(sev_arr ** 2)))
                feats["vib_amplitude_growth"] = rms_sev / max(median_sev, 0.01)
            else:
                feats["vib_amplitude_growth"] = 1.0

            # (7) tp_harmonic_energy — energy at spindle-related harmonics
            if spindle_rpm > 0 and harmonic_freqs_x:
                spindle_freq = spindle_rpm / 60.0
                tp_energy = 0.0
                for amp, freq in zip(harmonic_amps_x, harmonic_freqs_x):
                    if freq > 0:
                        ratio = freq / spindle_freq
                        # Check if near any integer harmonic (1×–4×)
                        nearest_int = round(ratio)
                        if 1 <= nearest_int <= 4 and abs(ratio - nearest_int) < 0.15:
                            tp_energy += amp ** 2
                feats["tp_harmonic_energy"] = tp_energy / (total_harmonic_energy + 1e-10)
            else:
                feats["tp_harmonic_energy"] = 0.0

            # (8) harmonic_amplitude_cv — coefficient of variation across harmonics
            if len(all_harmonic_amps) >= 2:
                ha = np.array(all_harmonic_amps)
                ha_mean = float(np.mean(ha))
                feats["harmonic_amplitude_cv"] = float(np.std(ha) / (ha_mean + 1e-12))
            else:
                feats["harmonic_amplitude_cv"] = 0.0

            # (9) tp_amplitude_variance — normalised variance at harmonics
            if harmonic_amps_x and len(harmonic_amps_x) >= 2:
                ha_x = np.array(harmonic_amps_x)
                ha_mean_x = float(np.mean(ha_x))
                feats["tp_amplitude_variance"] = float(np.var(ha_x) / (ha_mean_x ** 2 + 1e-12))
            else:
                feats["tp_amplitude_variance"] = 0.0

            # (10) spindle_order_amplitude — amplitude of harmonic closest to 1× spindle freq
            if spindle_rpm > 0 and harmonic_freqs_x:
                spindle_freq = spindle_rpm / 60.0
                best_amp = 0.0
                for amp, freq in zip(harmonic_amps_x, harmonic_freqs_x):
                    if freq > 0 and abs(freq - spindle_freq) / (spindle_freq + 1e-9) < 0.15:
                        best_amp = max(best_amp, amp)
                feats["spindle_order_amplitude"] = best_amp
            else:
                feats["spindle_order_amplitude"] = 0.0

            # (11) spindle_phase_shift — phase proxy from severity temporal variation
            # Without raw waveform we approximate with temporal std / mean of severity
            if sev_x is not None and len(sev_x) > 1:
                sev_arr = sev_x.values.astype(float)
                half = len(sev_arr) // 2
                first_half_mean = float(np.mean(sev_arr[:half]))
                second_half_mean = float(np.mean(sev_arr[half:]))
                overall_mean = float(np.mean(np.abs(sev_arr)))
                feats["spindle_phase_shift"] = abs(second_half_mean - first_half_mean) / (overall_mean + 1e-12)
            else:
                feats["spindle_phase_shift"] = 0.0
        else:
            # No vibration data — fill with defaults
            for fname in [
                "hf_energy_ratio", "impulse_crest_factor", "kurtosis_max",
                "periodicity_strength", "modulation_depth", "vib_amplitude_growth",
                "tp_harmonic_energy", "harmonic_amplitude_cv", "tp_amplitude_variance",
                "spindle_order_amplitude", "spindle_phase_shift",
            ]:
                feats[fname] = 0.0

        self.features = feats
        return feats

    def to_demo_event(
        self,
        session_id: str = "demo-session-001",
        *,
        description: str = "",
        explanation: str = "",
    ) -> Dict[str, Any]:
        """Convert this window into a demo event JSON compatible with the
        existing ``/agent/memory/events`` endpoint."""
        if not self.features:
            self.compute_features()

        # Derive pattern keys from the real data
        patterns = self._derive_patterns()

        # Derive external signals (classical model outputs)
        external_signals = self._derive_external_signals()

        # Build cutting context from machine state
        cutting_context = self._derive_cutting_context()

        event: Dict[str, Any] = {}
        if description:
            event["_description"] = description
        if explanation:
            event["_explanation"] = explanation

        # Include the 17 CNC features so the classical model can score
        # the same data that generated the pattern_keys (sync fix).
        metrics = {
            k: round(v, 6) for k, v in self.features.items()
            if isinstance(v, (int, float))
        }

        event.update(
            {
                "session_id": session_id,
                "time_range": {
                    "i0": 0,
                    "i1": max(self.n_samples, 100),
                    "t0": 0.0,
                    "t1": self.duration_s,
                    "fs": float(self.n_samples / max(self.duration_s, 0.001)),
                },
                "pattern_keys": patterns,
                "channels": [
                    "Power_Spindle",
                    "Power_Y",
                    "Vibration_Severity_X",
                    "Vibration_Severity_Y",
                ],
                "cutting_context": cutting_context,
                "external_signals": external_signals if external_signals else None,
                "metrics": metrics if metrics else None,
            }
        )
        return event

    # ------------------------------------------------------------------
    # Internal derivation helpers
    # ------------------------------------------------------------------

    def _derive_patterns(self) -> List[str]:
        f = self.features
        pats: List[str] = []

        sev_x = f.get("vib_severity_x_max", 0)
        sev_y = f.get("vib_severity_y_max", 0)
        chatter_ratio = f.get("chatter_ratio", 0)
        chatter_x = f.get("chatter_x_count", 0)
        spindle_pwr = f.get("power_spindle_max", 0)

        # Chatter detection
        if chatter_x > 0:
            pats.append("CHATTER_DETECTED")
            freq = f.get("chatter_freq_x_dominant")
            if freq:
                pats.append(f"SPECTRAL_PEAK_{int(freq)}Hz")

        # Severity-based patterns
        if sev_x > 10:
            pats.append("VIBRATION_SEVERITY_CRITICAL")
            pats.append("amp:ch0:loud")
        elif sev_x > 5:
            pats.append("VIBRATION_SEVERITY_HIGH")
            pats.append("amp:ch0:loud")
        elif sev_x > 2:
            pats.append("amp:ch0:mid")
        else:
            pats.append("amp:ch0:normal")

        if sev_y > 5:
            pats.append("amp:ch1:loud")
        elif sev_y > 2:
            pats.append("amp:ch1:mid")

        # Spindle power anomaly
        if spindle_pwr > 45:
            pats.append("POWER_SPIKE_SPINDLE")
        elif spindle_pwr > 30:
            pats.append("freq:spindle:high")

        # Chatter ratio as force ratio analog. Emit fixed buckets so the
        # downstream scorer and retriever see stable ratio families.
        if chatter_ratio > 0.3:
            pats.append("RATIO_Fx_Fy:>5")
        elif chatter_ratio > 0.1:
            pats.append("RATIO_Fx_Fy:2-5")

        # Operating status
        op_status = f.get("op_status_mode", 3)
        if op_status == 0:
            pats.append("MACHINE_STOPPED")
        elif op_status == 2:
            pats.append("MACHINE_MANUAL")

        if not pats:
            pats.append("freq:ch0:mid")
            pats.append("amp:ch0:normal")

        return pats

    def _derive_external_signals(self) -> Dict[str, Any]:
        f = self.features
        signals: Dict[str, Any] = {}

        sev_x = f.get("vib_severity_x_max", 0)
        sev_y = f.get("vib_severity_y_max", 0)
        max_sev = max(sev_x, sev_y)

        # Anomaly score based on vibration severity
        if max_sev > 3:
            # Normalise: severity 3→0.5, 10→0.9, 20→1.0
            score = min(1.0, 0.5 + (max_sev - 3) * 0.05)
            signals["anomaly_detector_score"] = round(score, 3)

        # Breakage prediction based on combined high severity + high power
        spindle_pwr = f.get("power_spindle_max", 0)
        if max_sev > 8 and spindle_pwr > 30:
            pred = min(1.0, 0.4 + (max_sev - 8) * 0.05 + (spindle_pwr - 30) * 0.01)
            signals["breakage_prediction"] = round(pred, 3)

        # Tool wear estimate (inverse of severity persistence)
        chatter_ratio = f.get("chatter_ratio", 0)
        if chatter_ratio > 0.05:
            wear = max(0.05, 1.0 - chatter_ratio * 3)
            signals["tool_wear_estimate"] = round(wear, 3)

        return signals

    def _derive_cutting_context(self) -> Dict[str, Any]:
        f = self.features
        extra: Dict[str, Any] = {}
        ctx: Dict[str, Any] = {
            "tool_type": "end_mill",
            "workpiece_material": "steel",
        }

        machine_family = None
        if self.case_dir:
            ctx["machine_id"] = self.case_dir
            machine_family = resolve_machine_family(self.case_dir)
            if machine_family:
                extra["machine_family"] = machine_family

        spindle = f.get("spindle_speed_mean")
        if spindle and spindle > 0:
            ctx["spindle_speed"] = round(spindle, 0)

        feed = f.get("feed_rate_mean")
        if feed and feed > 0:
            ctx["feed_rate"] = round(feed, 0)

        temp = f.get("temp_head_mean")
        if temp:
            extra["temperature_head"] = round(temp, 1)

        tool_num = f.get("tool_number")
        if tool_num:
            tool_number = int(tool_num)
            ctx["tool_id"] = f"T{tool_number}"
            extra["tool_number"] = tool_number

            if machine_family:
                extra["sindit_tool_iri"] = f"urn:lfl:tool:{machine_family}-t{tool_number}"
                spec = lookup_tool_spec(machine_family, tool_number)
                if spec is not None:
                    if spec.tool_type:
                        ctx["tool_type"] = spec.tool_type
                    if spec.diameter_mm is not None:
                        ctx["tool_diameter"] = spec.diameter_mm
                    if spec.teeth is not None:
                        ctx["num_teeth"] = spec.teeth
                    if spec.tool_length_mm is not None:
                        ctx["tool_length"] = spec.tool_length_mm
                    if spec.tool_substrate:
                        ctx["tool_material"] = spec.tool_substrate
                    if spec.tool_id:
                        extra["master_tool_id"] = spec.tool_id

        # Guess regime from feed rate
        if feed and feed > 5000:
            ctx["operating_regime"] = "roughing"
        elif feed and feed > 1000:
            ctx["operating_regime"] = "semi_finishing"
        elif feed and feed > 0:
            ctx["operating_regime"] = "finishing"

        if extra:
            ctx["extra"] = extra

        return ctx


class DatasetLoader:
    """Load and query real CNC case data from data/casedata directory."""

    def __init__(self, casedata_root: str | Path):
        self.root = Path(casedata_root)
        if not self.root.exists():
            raise FileNotFoundError(f"casedata directory not found: {self.root}")
        self._operations: Optional[Dict[str, OperationInfo]] = None
        self._operations_by_case: Optional[Dict[Tuple[str, str], OperationInfo]] = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_cases(self) -> List[str]:
        """List available case directories."""
        if self._operations is None or self._operations_by_case is None:
            self._scan()
        return sorted({case_dir for case_dir, _operation_id in (self._operations_by_case or {}).keys()})

    def list_operations(self, case: Optional[str] = None) -> List[OperationInfo]:
        """List all operations, optionally filtered to a single case."""
        if self._operations is None or self._operations_by_case is None:
            self._scan()
        if case:
            return [
                op
                for (case_dir, _operation_id), op in sorted((self._operations_by_case or {}).items())
                if case_dir == case
            ]
        return [
            op
            for (_case_dir, _operation_id), op in sorted((self._operations_by_case or {}).items())
        ]

    def get_operation(self, operation_id: str, case: Optional[str] = None) -> OperationInfo:
        if self._operations is None or self._operations_by_case is None:
            self._scan()
        if case:
            key = (case, operation_id)
            if key not in (self._operations_by_case or {}):
                raise KeyError(f"Operation not found: {case}/{operation_id}")
            return self._operations_by_case[key]  # type: ignore[index]
        if operation_id not in (self._operations or {}):
            raise KeyError(f"Operation not found: {operation_id}")
        return self._operations[operation_id]  # type: ignore[index]

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def extract_window(
        self,
        operation_id: str,
        t_start: str,
        t_end: str,
        *,
        max_rows: int = 50000,
    ) -> WindowData:
        """Extract a time window across all channels for an operation.

        Args:
            operation_id: e.g. 'OF00001'
            t_start: ISO timestamp
            t_end: ISO timestamp
            max_rows: Safety limit to prevent loading entire files
        """
        if pd is None:
            raise RuntimeError("pandas is required for DatasetLoader")

        op = self.get_operation(operation_id)
        window = WindowData(
            operation_id=operation_id,
            case_dir=op.case_dir,
            t_start=t_start,
            t_end=t_end,
            duration_s=0.0,
            n_samples=0,
        )

        for friendly in CHANNEL_GROUPS:
            file_path = op.channel_files.get(friendly)
            if file_path is None or not file_path.exists():
                continue

            # Optimisation: read only the timestamp column first to find
            # the row range, then read only the needed rows with all cols.
            ts_col = pd.read_csv(file_path, usecols=["timestamp"])
            ts_parsed = pd.to_datetime(ts_col["timestamp"])
            mask = (ts_parsed >= t_start) & (ts_parsed <= t_end)
            indices = mask[mask].index
            if len(indices) == 0:
                continue

            skip_start = int(indices[0])
            n_rows = min(int(indices[-1]) - skip_start + 1, max_rows)

            chunk = pd.read_csv(
                file_path,
                skiprows=range(1, skip_start + 1),  # skip rows after header
                nrows=n_rows,
                parse_dates=["timestamp"],
            )

            if len(chunk) > 0:
                setattr(window, friendly, chunk)
                window.n_samples = max(window.n_samples, len(chunk))
                ts = chunk["timestamp"]
                dt = (ts.max() - ts.min()).total_seconds()
                window.duration_s = max(window.duration_s, dt)

        return window

    def extract_interesting_windows(
        self,
        operation_id: str,
        *,
        window_seconds: int = 30,
        top_n: int = 5,
    ) -> List[WindowData]:
        """Automatically find interesting windows in an operation.

        Uses vibration severity spikes and chatter detection as triggers.
        Returns the top N most interesting windows.
        """
        if pd is None:
            raise RuntimeError("pandas is required for DatasetLoader")

        op = self.get_operation(operation_id)

        # Load only the columns we need for severity analysis
        vib_path = op.channel_files.get("vibration")
        if vib_path is None or not vib_path.exists():
            return []

        vib = pd.read_csv(
            vib_path,
            usecols=["Vibration_Severity_X", "Vibration_Severity_Y", "timestamp"],
            parse_dates=["timestamp"],
        )
        vib["sev_total"] = (
            vib["Vibration_Severity_X"] + vib["Vibration_Severity_Y"]
        )

        # Find peak severity timestamps
        peaks = vib.nlargest(top_n * 3, "sev_total")

        # De-duplicate: keep peaks that are at least window_seconds apart
        selected: List[str] = []
        for _, row in peaks.iterrows():
            ts = row["timestamp"]
            too_close = False
            for sel in selected:
                diff = abs((pd.Timestamp(ts) - pd.Timestamp(sel)).total_seconds())
                if diff < window_seconds:
                    too_close = True
                    break
            if not too_close:
                selected.append(str(ts.isoformat()))
            if len(selected) >= top_n:
                break

        # Extract windows centred on each peak
        windows: List[WindowData] = []
        half = window_seconds / 2
        for ts_str in selected:
            centre = pd.Timestamp(ts_str)
            t_start = (centre - pd.Timedelta(seconds=half)).isoformat()
            t_end = (centre + pd.Timedelta(seconds=half)).isoformat()
            w = self.extract_window(operation_id, t_start, t_end)
            w.compute_features()
            windows.append(w)

        return windows

    def extract_bulk_features(
        self,
        operation_id: str,
        *,
        window_seconds: int = 10,
        stride_seconds: int = 5,
        max_windows: int = 5000,
    ) -> List[Dict[str, float]]:
        """Efficiently extract features from many sliding windows.

        Loads each CSV once and slices in-memory rather than reading per window.
        Much faster than calling extract_window() in a loop.

        Returns:
            List of feature dicts, one per window.
        """
        if pd is None:
            raise RuntimeError("pandas is required for DatasetLoader")

        op = self.get_operation(operation_id)

        # Load each channel once
        loaded: Dict[str, Any] = {}  # friendly_name → DataFrame
        timestamps_ref = None
        for friendly in CHANNEL_GROUPS:
            file_path = op.channel_files.get(friendly)
            if file_path is None or not file_path.exists():
                continue
            try:
                df = pd.read_csv(file_path, parse_dates=["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                loaded[friendly] = df
                if timestamps_ref is None or len(df) > len(timestamps_ref):
                    timestamps_ref = df["timestamp"]
            except Exception as e:
                logger.debug("Failed to load %s: %s", file_path, e)

        if timestamps_ref is None or len(timestamps_ref) < 2:
            return []

        t_min = timestamps_ref.min()
        t_max = timestamps_ref.max()
        total_s = (t_max - t_min).total_seconds()

        n_windows = int((total_s - window_seconds) / stride_seconds) + 1
        if n_windows <= 0:
            return []

        # Subsample if too many
        step = max(1, n_windows // max_windows)
        features_list: List[Dict[str, float]] = []

        for i in range(0, n_windows, step):
            ws = t_min + pd.Timedelta(seconds=i * stride_seconds)
            we = ws + pd.Timedelta(seconds=window_seconds)

            window = WindowData(
                operation_id=operation_id,
                case_dir=op.case_dir,
                t_start=ws.isoformat(),
                t_end=we.isoformat(),
                duration_s=float(window_seconds),
                n_samples=0,
            )

            for friendly, df in loaded.items():
                mask = (df["timestamp"] >= ws) & (df["timestamp"] <= we)
                chunk = df.loc[mask]
                if len(chunk) > 0:
                    setattr(window, friendly, chunk)
                    window.n_samples = max(window.n_samples, len(chunk))

            if window.n_samples > 0:
                try:
                    window.compute_features()
                    features_list.append(window.features)
                except Exception:
                    pass

            if len(features_list) >= max_windows:
                break

        return features_list

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan(self) -> None:
        self._operations = {}
        self._operations_by_case = {}
        root_entries = [
            entry
            for entry in sorted(self.root.iterdir())
            if entry.is_dir() and not entry.name.startswith(".")
        ]

        if any(entry.name.startswith("OF") for entry in root_entries):
            for op_dir in root_entries:
                if not op_dir.name.startswith("OF"):
                    continue
                case_dir, tool_id = self._derive_flat_operation_metadata(op_dir)
                self._register_operation(case_dir, tool_id, op_dir)
        else:
            for case_dir in root_entries:
                # Extract tool ID from case name (last segment after " - ")
                parts = case_dir.name.split(" - ")
                tool_id = parts[-1].strip() if len(parts) > 1 else case_dir.name

                for op_dir in sorted(case_dir.iterdir()):
                    if not op_dir.is_dir() or not op_dir.name.startswith("OF"):
                        continue
                    self._register_operation(case_dir.name, tool_id, op_dir)

        logger.info(
            "DatasetLoader: found %d operations across %d cases",
            len(self._operations_by_case),
            len({o.case_dir for o in self._operations_by_case.values()}),
        )

    def _derive_flat_operation_metadata(self, op_dir: Path) -> Tuple[str, str]:
        csv_files = sorted(op_dir.glob("*.csv"))
        if not csv_files:
            return self.root.name, self.root.name

        parts = csv_files[0].stem.split("_")
        if len(parts) < 3:
            return self.root.name, self.root.name

        case_dir = "_".join(parts[1:-1]).strip("_") or self.root.name
        tool_id = parts[-2].strip() if len(parts) >= 4 else case_dir
        return case_dir, tool_id or case_dir

    def _register_operation(self, case_dir: str, tool_id: str, op_dir: Path) -> None:
        op_id = op_dir.name
        channel_files: Dict[str, Path] = {}
        for csv_file in op_dir.glob("*.csv"):
            friendly = _detect_channel_group(csv_file)
            if friendly is None:
                continue
            if friendly in channel_files:
                logger.debug(
                    "Ignoring duplicate %s channel for %s/%s: %s",
                    friendly,
                    case_dir,
                    op_id,
                    csv_file.name,
                )
                continue
            channel_files[friendly] = csv_file

        operation = OperationInfo(
            operation_id=op_id,
            case_dir=case_dir,
            tool_id=tool_id,
            channel_files=channel_files,
        )
        self._operations[op_id] = operation
        self._operations_by_case[(case_dir, op_id)] = operation

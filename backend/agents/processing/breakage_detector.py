"""
Breakage Feature Extractor — Maps pre-stoppage features to LFL MemoryEvents.

Reads rows from data/breakage_patterns/breakage_features.csv and converts
them into MemoryEvent objects with:
  - raw_metrics (mapped to FEATURE_NAMES for the seed model)
  - pattern keys (threshold-based breakage pattern detection)
  - cutting context (tool number + operation)

This module is the bridge between the extracted breakage ground-truth dataset
and the Learning Feedback Loop pipeline.

Usage:
    from backend.agents.processing.breakage_detector import BreakageFeatureExtractor

    extractor = BreakageFeatureExtractor("data/breakage_patterns/breakage_features.csv")
    for event, meta in extractor.iter_events(session_prefix="exp"):
        result = await orchestrator.process_event(event)
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..core.schemas import PatternKey, PatternType, TimeRange
from ..core.context import CuttingContext
from ..memory.orchestrator import MemoryEvent
from .classical_models import FEATURE_NAMES, features_from_dict

logger = logging.getLogger(__name__)


# ============================================================================
# Column mapping: breakage CSV → FEATURE_NAMES (28 features)
# ============================================================================

# Direct 1:1 mappings (CSV column name → FEATURE_NAMES name)
_COL_MAP: Dict[str, str] = {
    "power_spindle_mean": "power_spindle_mean",
    "power_spindle_max": "power_spindle_max",
    "power_spindle_std": "power_spindle_std",
    "power_y_mean": "power_y_mean",
    "power_y_max": "power_y_max",
    "power_z_mean": "power_z_mean",
    "vib_severity_x_mean": "vib_severity_x_mean",
    "vib_severity_x_max": "vib_severity_x_max",
    "vib_severity_y_mean": "vib_severity_y_mean",
    "vib_severity_y_max": "vib_severity_y_max",
    "power_active_mean": "power_active_mean",
    "power_active_std": "power_active_std",
    "power_factor_mean": "power_factor_mean",
    # Renamed columns
    "spindle_actual_mean": "spindle_speed_mean",
    "feed_actual_mean": "feed_rate_mean",
    "temperature_head_mean": "temp_head_mean",
}

# Metadata columns (not features)
_META_COLS = {
    "sample_id", "label", "operation_id", "tool_number",
    "event_timestamp", "severity", "stop_type", "window_seconds",
}


# ============================================================================
# Pattern detection thresholds
# ============================================================================

@dataclass
class BreakageThresholds:
    """Thresholds for detecting breakage-specific patterns from feature windows.

    All thresholds are set based on the Cohen's d analysis from the
    extract_pre_stoppage_patterns.py output.
    """
    # BREAKAGE_POWER_SPIKE: axis power delta exceeds threshold
    power_spindle_delta_max: float = 15.0   # % of rated power
    power_y_delta_max: float = 10.0         # % of rated power

    # BREAKAGE_VIB_SHIFT: vibration severity or chatter frequency changes
    vib_severity_x_delta_max: float = 0.8   # mm/s²
    chatter_freq_x_slope_abs: float = 5.0   # Hz/s (frequency shift rate)

    # BREAKAGE_FEED_OVERRIDE_DROP: operator lowering feed override
    feed_override_delta_mean: float = -10.0  # % drop
    feed_override_min: float = 50.0          # % (below this = suspicious)

    # BREAKAGE_DECORRELATION: cross-signal correlation breakdown
    corr_spindle_power_vib_x_low: float = 0.3  # normally > 0.6


# ============================================================================
# Sample metadata
# ============================================================================

@dataclass
class BreakageSampleMeta:
    """Metadata for a single breakage sample (from CSV row)."""
    sample_id: str
    label: str          # "pre_break" or "normal"
    operation_id: str
    tool_number: str
    event_timestamp: str
    severity: str
    stop_type: str
    window_seconds: float
    is_pre_break: bool = field(init=False)

    def __post_init__(self) -> None:
        # Support both legacy and newer positive-class labels.
        normalized = (self.label or "").strip().lower()
        self.is_pre_break = normalized in {"pre_break", "pre_stoppage"}


# ============================================================================
# Main Extractor
# ============================================================================

class BreakageFeatureExtractor:
    """Extracts MemoryEvents from breakage feature CSV for LFL processing.

    Loads the CSV once, then provides iteration over samples as
    (MemoryEvent, BreakageSampleMeta) tuples.  Each event includes:
    - raw_metrics: mapped to FEATURE_NAMES for the seed model
    - patterns: threshold-based breakage pattern keys
    - cutting_context: CuttingContext with tool number
    - external_signals: all raw breakage features (for the scorer)
    """

    def __init__(
        self,
        csv_path: str | Path,
        thresholds: Optional[BreakageThresholds] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.thresholds = thresholds or BreakageThresholds()
        self._rows: List[Dict[str, str]] = []
        self._loaded = False

    def load(self) -> int:
        """Load the CSV file. Returns number of rows loaded."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Breakage features CSV not found: {self.csv_path}\n"
                "Generate it: python scripts/extract_pre_stoppage_patterns.py"
            )
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            self._rows = list(reader)
        self._loaded = True
        logger.info("Loaded %d breakage samples from %s", len(self._rows), self.csv_path)
        return len(self._rows)

    @property
    def n_samples(self) -> int:
        return len(self._rows)

    @property
    def n_pre_break(self) -> int:
        return sum(
            1
            for r in self._rows
            if str(r.get("label", "")).strip().lower() in {"pre_break", "pre_stoppage"}
        )

    @property
    def n_normal(self) -> int:
        return sum(1 for r in self._rows if r.get("label") == "normal")

    @property
    def operation_ids(self) -> List[str]:
        return sorted(set(r.get("operation_id", "") for r in self._rows))

    def rows_for_operations(self, op_ids: List[str]) -> List[Dict[str, str]]:
        """Filter rows to specific operations."""
        op_set = set(op_ids)
        return [r for r in self._rows if r.get("operation_id") in op_set]

    # ------------------------------------------------------------------
    # Feature mapping
    # ------------------------------------------------------------------

    @staticmethod
    def row_to_feature_dict(row: Dict[str, str]) -> Dict[str, float]:
        """Map a CSV row to a FEATURE_NAMES-compatible dict."""
        fd: Dict[str, float] = {}
        feat_to_csv = {feat_name: csv_col for csv_col, feat_name in _COL_MAP.items()}
        for feat_name in FEATURE_NAMES:
            csv_col = feat_to_csv.get(feat_name, feat_name)
            raw_value = row.get(csv_col)
            if (raw_value is None or raw_value == "") and csv_col != feat_name:
                raw_value = row.get(feat_name)
            fd[feat_name] = _safe_float(raw_value)

        # Compute chatter_ratio from chatter amplitudes
        chatter_x = _safe_float(row.get("chatter_amp_x_mean"))
        chatter_y = _safe_float(row.get("chatter_amp_y_mean"))
        if chatter_x > 0.0 or chatter_y > 0.0:
            fd["chatter_ratio"] = chatter_x / (chatter_y + 1e-10)
        else:
            fd["chatter_ratio"] = _safe_float(row.get("chatter_ratio"))

        # Physics features: fill with 0.0 (model handles missing gracefully)
        for name in FEATURE_NAMES:
            fd.setdefault(name, 0.0)

        return fd

    @staticmethod
    def row_to_all_features(row: Dict[str, str]) -> Dict[str, float]:
        """Extract ALL numeric features from a CSV row (for extended analysis)."""
        fd: Dict[str, float] = {}
        for key, val in row.items():
            if key in _META_COLS:
                continue
            fd[key] = _safe_float(val)
        return fd

    # ------------------------------------------------------------------
    # Pattern detection
    # ------------------------------------------------------------------

    def detect_patterns(self, row: Dict[str, str]) -> List[PatternKey]:
        """Detect breakage-specific patterns from feature thresholds.

        Returns a list of PatternKey objects for patterns that fired.
        """
        th = self.thresholds
        patterns: List[PatternKey] = []

        # BREAKAGE_POWER_SPIKE — axis power delta exceeds threshold
        # Fallback: derive delta from (max - mean) when the dedicated delta column is absent.
        if "power_spindle_delta_max" in row:
            pwr_sp_delta = _safe_float(row.get("power_spindle_delta_max"))
        else:
            pwr_sp_delta = max(0.0, _safe_float(row.get("power_spindle_max")) - _safe_float(row.get("power_spindle_mean")))
        if "power_y_delta_max" in row:
            pwr_y_delta = _safe_float(row.get("power_y_delta_max"))
        else:
            pwr_y_delta = max(0.0, _safe_float(row.get("power_y_max")) - _safe_float(row.get("power_y_mean")))
        if pwr_sp_delta > th.power_spindle_delta_max or pwr_y_delta > th.power_y_delta_max:
            patterns.append(PatternKey(
                pattern_type=PatternType.FAULT,
                key="BREAKAGE_POWER_SPIKE",
                fault_type="tool_breakage",
                source_metric="power_delta_max",
            ))

        # BREAKAGE_VIB_SHIFT — vibration severity or frequency shift
        # Fallback: derive from (max - mean) when the dedicated delta column is absent.
        if "vib_severity_x_delta_max" in row:
            vib_delta = _safe_float(row.get("vib_severity_x_delta_max"))
        else:
            vib_delta = max(0.0, _safe_float(row.get("vib_severity_x_max")) - _safe_float(row.get("vib_severity_x_mean")))
        chatter_slope = abs(_safe_float(row.get("chatter_freq_x_slope")))
        if vib_delta > th.vib_severity_x_delta_max or chatter_slope > th.chatter_freq_x_slope_abs:
            patterns.append(PatternKey(
                pattern_type=PatternType.FAULT,
                key="BREAKAGE_VIB_SHIFT",
                fault_type="tool_breakage",
                source_metric="vib_severity_x_delta",
            ))

        # BREAKAGE_FEED_OVERRIDE_DROP — operator lowering feed override
        fo_delta_mean = _safe_float(row.get("feed_override_delta_mean"))
        fo_min = _safe_float(row.get("feed_override_min"))
        if fo_delta_mean < th.feed_override_delta_mean or (fo_min > 0 and fo_min < th.feed_override_min):
            patterns.append(PatternKey(
                pattern_type=PatternType.FAULT,
                key="BREAKAGE_FEED_OVERRIDE_DROP",
                fault_type="tool_breakage",
                source_metric="feed_override",
            ))

        # BREAKAGE_DECORRELATION — spindle-power-vibration correlation breakdown
        corr_val = _safe_float(row.get("corr_spindle_power_vib_x"))
        if 0.0 < abs(corr_val) < th.corr_spindle_power_vib_x_low:
            patterns.append(PatternKey(
                pattern_type=PatternType.FAULT,
                key="BREAKAGE_DECORRELATION",
                fault_type="tool_breakage",
                source_metric="corr_spindle_power_vib_x",
            ))

        # Always include base fault pattern if any breakage pattern fired
        if patterns:
            patterns.insert(0, PatternKey(
                pattern_type=PatternType.FAULT,
                key="fault:tool_breakage",
                fault_type="tool_breakage",
            ))

        return patterns

    # ------------------------------------------------------------------
    # MemoryEvent construction
    # ------------------------------------------------------------------

    def row_to_event(
        self,
        row: Dict[str, str],
        session_id: str,
    ) -> Tuple[MemoryEvent, BreakageSampleMeta]:
        """Convert a CSV row to a (MemoryEvent, BreakageSampleMeta) tuple."""
        meta = BreakageSampleMeta(
            sample_id=row.get("sample_id", ""),
            label=row.get("label", ""),
            operation_id=row.get("operation_id", ""),
            tool_number=row.get("tool_number", ""),
            event_timestamp=row.get("event_timestamp", ""),
            severity=row.get("severity", ""),
            stop_type=row.get("stop_type", ""),
            window_seconds=_safe_float(row.get("window_seconds", "60")),
        )

        # Map to FEATURE_NAMES for seed model scoring
        feature_dict = self.row_to_feature_dict(row)

        # Detect breakage-specific patterns
        patterns = self.detect_patterns(row)

        # Build cutting context, preferring any explicit tool-master-enriched
        # columns in the CSV over the legacy tool-number-only fallback.
        extra: Dict[str, Any] = {}
        for key in ("operation_id", "session", "machine_family", "sindit_tool_iri"):
            value = _safe_text(row.get(key))
            if value is not None:
                extra[key] = value

        spindle_speed = _optional_positive_float(row.get("spindle_speed_mean"))
        if spindle_speed is None:
            spindle_speed = _optional_positive_float(row.get("spindle_actual_mean"))

        feed_rate = _optional_positive_float(row.get("feed_rate_mean"))
        if feed_rate is None:
            feed_rate = _optional_positive_float(row.get("feed_actual_mean"))

        tool_id = _safe_text(row.get("tool_id"))
        if tool_id is None and meta.tool_number:
            tool_id = f"T{meta.tool_number}"

        tool_type = _safe_text(row.get("tool_type"))
        if tool_type is None and meta.tool_number:
            tool_type = f"T{meta.tool_number}"

        cutting_context = CuttingContext(
            machine_type="CNC_5axis",
            machine_id=_safe_text(row.get("machine_id")),
            tool_id=tool_id,
            tool_type=tool_type,
            tool_diameter=_optional_positive_float(row.get("tool_diameter")),
            num_teeth=_optional_positive_int(row.get("num_teeth")),
            tool_length=_optional_positive_float(row.get("tool_length")),
            tool_material=_safe_text(row.get("tool_material")),
            spindle_speed=spindle_speed,
            feed_rate=feed_rate,
            extra=extra,
        )

        # Time range (synthetic, 60s window at 1 Hz)
        window_s = int(meta.window_seconds) if meta.window_seconds else 60
        time_range = TimeRange(
            i0=0, i1=window_s,
            t0=0.0, t1=float(window_s),
            fs=1.0,
        )

        event = MemoryEvent(
            session_id=session_id,
            time_range=time_range,
            patterns=patterns,
            cutting_context=cutting_context,
            external_signals={},  # populated by orchestrator's anomaly detector
            raw_metrics=feature_dict,
            channels=["TYZBPS", "BXCZ3M", "7DTZHE", "92SQBY"],
        )

        return event, meta

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_events(
        self,
        session_prefix: str = "breakage_exp",
        operations: Optional[List[str]] = None,
    ) -> Iterator[Tuple[MemoryEvent, BreakageSampleMeta]]:
        """Iterate over all samples as (MemoryEvent, BreakageSampleMeta).

        Yields events in CSV order (chronological within each operation).
        """
        if not self._loaded:
            self.load()

        rows = self._rows
        if operations:
            op_set = set(operations)
            rows = [r for r in rows if r.get("operation_id") in op_set]

        for row in rows:
            op_id = row.get("operation_id", "unknown")
            session_id = f"{session_prefix}_{op_id}"
            yield self.row_to_event(row, session_id)

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the loaded dataset."""
        if not self._loaded:
            self.load()

        ops: Dict[str, Dict[str, int]] = {}
        for row in self._rows:
            op = row.get("operation_id", "?")
            raw_label = str(row.get("label", "?")).strip().lower()
            label = "pre_break" if raw_label in {"pre_break", "pre_stoppage"} else raw_label
            if op not in ops:
                ops[op] = {"pre_break": 0, "normal": 0}
            ops[op][label] = ops[op].get(label, 0) + 1

        return {
            "total_samples": len(self._rows),
            "pre_break": self.n_pre_break,
            "normal": self.n_normal,
            "operations": ops,
        }


# ============================================================================
# Helpers
# ============================================================================

def _safe_float(val: Any) -> float:
    """Convert a value to float, returning 0.0 on failure."""
    if val is None or val == "":
        return 0.0
    try:
        v = float(val)
        return v if math.isfinite(v) else 0.0
    except (ValueError, TypeError):
        return 0.0


def _safe_text(val: Any) -> str | None:
    if val is None:
        return None
    text = str(val).strip()
    return text or None


def _optional_positive_float(val: Any) -> float | None:
    numeric = _safe_float(val)
    return numeric if numeric > 0.0 else None


def _optional_positive_int(val: Any) -> int | None:
    numeric = _optional_positive_float(val)
    if numeric is None:
        return None
    return int(round(numeric))

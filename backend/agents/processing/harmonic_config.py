"""
Harmonic Context-Weighted CNN — Configuration.

Domain-agnostic configuration for the harmonic context-weighted CNN approach.
Adapted from the HarmonicBreakNet architecture (classical/lfl) to work across
datasets (casedata → stoppage, Site_a_line2 → breakage, future domains).

The core idea: a learned W matrix maps available cutting/context parameters
to per-harmonic-feature weights, producing context-conditioned anomaly scoring.
The approach is configurable to any combination of input channels, context
parameters, harmonic multipliers, and window sizes.

Tag: [HARMONIC_CONTEXT_V1]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def resolve_training_model_save_path(
    base_path: Any,
    *,
    model_save_path: Optional[str] = None,
    checkpoint_suffix: Optional[str] = None,
    random_seed: Optional[int] = None,
    replace_checkpoint: bool = False,
) -> str:
    """Resolve the checkpoint path for a training run.

    Explicit ``model_save_path`` wins. Otherwise, when ``replace_checkpoint`` is
    false and either ``checkpoint_suffix`` or an explicit ``random_seed`` is
    provided, derive an experiment path from the canonical checkpoint path.
    """

    explicit_path = str(model_save_path or "").strip()
    if explicit_path:
        return explicit_path

    resolved_base = str(base_path or "").strip()
    if not resolved_base:
        return ""

    if replace_checkpoint:
        return resolved_base

    raw_suffix = str(checkpoint_suffix or "").strip()
    if not raw_suffix and random_seed is not None:
        raw_suffix = f"seed{int(random_seed)}"

    safe_suffix = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_suffix).strip("-._")
    if not safe_suffix:
        return resolved_base

    path = Path(resolved_base)
    suffixes = "".join(path.suffixes)
    stem = path.name[:-len(suffixes)] if suffixes else path.name
    return str(path.with_name(f"{stem}.{safe_suffix}{suffixes}"))


@dataclass
class HarmonicContextConfig:
    """Domain-agnostic configuration for the harmonic context-weighted CNN.

    All parameters are configurable per-dataset/domain.  Use the preset
    factory methods for common configurations, or ``from_dict()`` for
    arbitrary domains.

    Architecture Parameters
    -----------------------
    The CNN receives ``(B, cnn_window, n_channels * n_harmonics)`` harmonic
    features and ``(B, n_params)`` context parameters.  The learned W matrix
    of shape ``(n_harm_features, n_params)`` projects context params into
    per-feature weights, collapsing harmonics into a 1-D signal before the
    Conv1d blocks.

    Harmonic Modes
    --------------
    - **raw_fft**: Compute harmonics from raw high-frequency signals via
      sliding FFT (uses ``fft_window``, ``fft_step``, ``harmonic_multipliers``).
    - **pre_extracted**: Use pre-computed harmonic columns already present in
      the data (e.g., CMS-provided ``Vibration_Harmonic_1-8``).  Bypass FFT.

    This matters because casedata/Site_a_line2 CSVs are typically at 1 Hz and
    contain pre-computed harmonics, while raw accelerometer data at 50 kHz
    requires FFT extraction.
    """

    # ── Target / label configuration ──────────────────────────────────
    target_label: str = "label"  # Column name containing ground-truth labels
    positive_labels: List[str] = field(
        default_factory=lambda: ["pre_stoppage", "pre_break"]
    )

    # ── Input channel configuration ───────────────────────────────────
    # Semantic role names (resolved via DomainConfig) or column patterns
    input_channel_roles: List[str] = field(
        default_factory=lambda: ["primary_vibration"]
    )
    # Fallback: regex patterns to select columns from raw DataFrames
    input_column_patterns: List[str] = field(default_factory=list)
    # Exact column names (highest priority, used when patterns don't apply)
    input_columns: List[str] = field(default_factory=list)

    # ── Context parameter configuration ───────────────────────────────
    # Keys used as the conditioning vector fed to the W matrix.
    # Only parameters actually AVAILABLE in the dataset should be listed.
    context_param_keys: List[str] = field(
        default_factory=lambda: ["spindle_speed", "feed_rate"]
    )
    # Maps param key → source column name (for data loading)
    context_param_sources: Dict[str, str] = field(default_factory=dict)

    # Normalisation stats for context params (z-score normalisation).
    # Populated during training, persisted with model weights.
    # {param_key: {"mean": float, "std": float}}
    context_param_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Scorer family to instantiate for this config.
    scorer_kind: str = "context"  # "context" or "pair"
    # Pair-model architecture family. ``legacy_v1`` is the currently shipped
    # DeepSets + temporal CNN model; ``lfl_v2`` matches the original LFL
    # parameter-conditioned per-pair encoder.
    model_kind: str = "legacy_v1"

    # ── Harmonic extraction configuration ─────────────────────────────
    harmonic_mode: str = "pre_extracted"  # "raw_fft" or "pre_extracted"
    harmonic_multipliers: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 6, 8, 10]
    )
    fft_window: int = 4096  # samples per FFT window (raw_fft mode)
    fft_step: int = 1024  # step between FFT windows (raw_fft mode)

    # For pre_extracted mode: column name patterns for harmonic features
    harmonic_column_patterns: List[str] = field(default_factory=list)
    # Exact harmonic column names (populated during training)
    harmonic_columns: List[str] = field(default_factory=list)
    pre_extracted_feature_source: str = "harmonic_columns"  # or "peak_bins"
    # For pair-input mode: precomputed top-K peak columns per channel
    pair_frequency_column_patterns: List[str] = field(default_factory=list)
    pair_amplitude_column_patterns: List[str] = field(default_factory=list)
    k_peaks: int = 5
    pair_embed_dim: int = 16
    f_max_rel: float = 12.0
    pair_sample_mode: str = "sliding_windows"
    peak_harmonic_bins: List[int] = field(default_factory=list)
    peak_bin_tolerance: float = 0.35

    # ── CNN architecture ──────────────────────────────────────────────
    cnn_window: int = 16  # time steps fed to CNN
    conv_channels: List[int] = field(default_factory=lambda: [16, 16])
    fc_hidden: int = 32
    kernel_size: int = 5

    # ── Training configuration ────────────────────────────────────────
    learning_rate_schedule: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"lr": 1e-3, "epochs": 30},
            {"lr": 1e-4, "epochs": 20},
        ]
    )
    batch_size: int = 16
    pos_weight: Optional[float] = None  # BCE pos_weight for class imbalance
    n_windows_per_sample: int = 2  # random windows per sample during training
    random_seed: int = 0  # Controls model init and shuffle order for reproducible retrains
    val_split: float = 0.2  # fraction held out for validation
    early_stopping_patience: int = 10  # epochs without val improvement
    allow_unlabelled_fallback: bool = False  # Raw casedata fallback is opt-in only

    # ── Persistence ───────────────────────────────────────────────────
    model_save_path: str = "data/models/harmonic_context.pt"
    enabled: bool = False  # Must be explicitly enabled
    decision_threshold: float = 0.5

    # ── Metadata (populated during training) ──────────────────────────
    n_harm_features: int = 0  # Set during training: n_channels * n_harmonics
    n_params: int = 0  # Set during training: len(context_param_keys)
    dataset_name: str = ""  # e.g. "casedata", "site_a_line2"
    trained_at: Optional[str] = None  # ISO timestamp
    training_metrics: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_harmonics(self) -> int:
        """Number of harmonic multipliers."""
        return len(self.harmonic_multipliers)

    def validate_window_alignment(self, sample_rate_hz: float = 1.0) -> Dict[str, Any]:
        """Check compatibility between FFT and CNN window sizes.

        Returns a dict with alignment info and any warnings.
        """
        result: Dict[str, Any] = {"valid": True, "warnings": []}

        if self.harmonic_mode == "raw_fft":
            if sample_rate_hz <= 0:
                result["valid"] = False
                result["warnings"].append("sample_rate_hz must be > 0")
                return result

            # How many FFT steps fit in a data window?
            window_samples_needed = (
                self.fft_window + (self.cnn_window - 1) * self.fft_step
            )
            window_seconds = window_samples_needed / sample_rate_hz
            result["window_samples_needed"] = window_samples_needed
            result["window_seconds_needed"] = round(window_seconds, 2)
            result["fft_time_resolution_s"] = round(self.fft_step / sample_rate_hz, 4)

            if sample_rate_hz < 100:
                result["warnings"].append(
                    f"Sample rate ({sample_rate_hz} Hz) is very low for FFT "
                    f"harmonic extraction.  Consider using harmonic_mode='pre_extracted'."
                )

        elif self.harmonic_mode == "pre_extracted":
            # Need at least cnn_window rows of harmonic features
            result["min_rows_needed"] = self.cnn_window
            if self.cnn_window > 60:
                result["warnings"].append(
                    f"cnn_window={self.cnn_window} is large for pre-extracted "
                    f"harmonics (typically 1 Hz data).  Consider reducing."
                )

        if not result["warnings"]:
            result["warnings"] = []

        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "target_label": self.target_label,
            "positive_labels": self.positive_labels,
            "input_channel_roles": self.input_channel_roles,
            "input_column_patterns": self.input_column_patterns,
            "input_columns": self.input_columns,
            "context_param_keys": self.context_param_keys,
            "context_param_sources": self.context_param_sources,
            "context_param_stats": self.context_param_stats,
            "scorer_kind": self.scorer_kind,
            "model_kind": self.model_kind,
            "harmonic_mode": self.harmonic_mode,
            "harmonic_multipliers": self.harmonic_multipliers,
            "fft_window": self.fft_window,
            "fft_step": self.fft_step,
            "harmonic_column_patterns": self.harmonic_column_patterns,
            "harmonic_columns": self.harmonic_columns,
            "pre_extracted_feature_source": self.pre_extracted_feature_source,
            "pair_frequency_column_patterns": self.pair_frequency_column_patterns,
            "pair_amplitude_column_patterns": self.pair_amplitude_column_patterns,
            "k_peaks": self.k_peaks,
            "pair_embed_dim": self.pair_embed_dim,
            "f_max_rel": self.f_max_rel,
            "pair_sample_mode": self.pair_sample_mode,
            "peak_harmonic_bins": self.peak_harmonic_bins,
            "peak_bin_tolerance": self.peak_bin_tolerance,
            "cnn_window": self.cnn_window,
            "conv_channels": self.conv_channels,
            "fc_hidden": self.fc_hidden,
            "kernel_size": self.kernel_size,
            "learning_rate_schedule": self.learning_rate_schedule,
            "batch_size": self.batch_size,
            "pos_weight": self.pos_weight,
            "n_windows_per_sample": self.n_windows_per_sample,
            "random_seed": self.random_seed,
            "val_split": self.val_split,
            "early_stopping_patience": self.early_stopping_patience,
            "allow_unlabelled_fallback": self.allow_unlabelled_fallback,
            "model_save_path": self.model_save_path,
            "enabled": self.enabled,
            "decision_threshold": self.decision_threshold,
            "n_harm_features": self.n_harm_features,
            "n_params": self.n_params,
            "dataset_name": self.dataset_name,
            "trained_at": self.trained_at,
            "training_metrics": self.training_metrics,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HarmonicContextConfig":
        """Deserialize from a dict (e.g., from JSON config)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ══════════════════════════════════════════════════════════════════════════════
# Dataset presets
# ══════════════════════════════════════════════════════════════════════════════


def casedata_stoppage_preset(**overrides: Any) -> HarmonicContextConfig:
    """Preset for casedata → stoppage prediction.

    Casedata CSVs are at ~1 Hz with pre-computed vibration harmonics from the
    CMS (Vibration_Harmonic_1-8 amplitude X/Y).  Cutting context params
    available from sensors: ``spindle_speed_mean``, ``feed_rate_mean``.

    Uses ``harmonic_mode='pre_extracted'`` since 1 Hz data cannot support
    raw FFT harmonic extraction at CNC-relevant frequencies.
    """
    # Filter unknown overrides to avoid TypeError from dataclass constructor
    valid_keys = HarmonicContextConfig.__dataclass_fields__
    overrides = {k: v for k, v in overrides.items() if k in valid_keys}
    defaults = dict(
        target_label="label",
        positive_labels=["pre_stoppage"],
        input_channel_roles=["primary_vibration"],
        input_column_patterns=[
            r"Vibration_Harmonic_\d+_[XY]_Amplitude",
            r"vib_severity_[xy]_mean",
        ],
        input_columns=[],  # Populated during data loading
        context_param_keys=["spindle_speed", "feed_rate"],
        context_param_sources={
            "spindle_speed": "spindle_speed_mean",
            "feed_rate": "feed_rate_mean",
        },
        harmonic_mode="pre_extracted",
        harmonic_column_patterns=[
            r"Vibration_Harmonic_\d+_[XY]_Amplitude",
        ],
        harmonic_columns=[],  # Populated during data loading
        cnn_window=16,
        conv_channels=[16, 16],
        fc_hidden=32,
        kernel_size=5,
        batch_size=16,
        pos_weight=3.0,  # Stoppage events are rare
        model_save_path="data/models/harmonic_context_casedata.pt",
        dataset_name="casedata",
        enabled=True,
    )
    defaults.update(overrides)
    return HarmonicContextConfig(**defaults)


def casedata_peak_context_preset(**overrides: Any) -> HarmonicContextConfig:
    """Preset for casedata context scoring from Vibration_Peak columns.

    This derives a fixed harmonic-style feature vector by binning the top-K
    peak amplitudes by relative frequency (f / fg) into harmonic bins.
    """
    valid_keys = HarmonicContextConfig.__dataclass_fields__
    overrides = {k: v for k, v in overrides.items() if k in valid_keys}
    defaults = dict(
        target_label="label",
        positive_labels=["pre_stoppage"],
        input_channel_roles=["primary_vibration"],
        input_column_patterns=[
            r"Vibration_Peak_\d+_[XY]_(Amplitude|Frequency)",
            r"vib_severity_[xy]_mean",
        ],
        input_columns=[],
        context_param_keys=["spindle_speed", "feed_rate"],
        context_param_sources={
            "spindle_speed": "spindle_speed_mean",
            "feed_rate": "feed_rate_mean",
        },
        harmonic_mode="pre_extracted",
        harmonic_column_patterns=[],
        harmonic_columns=[],
        pre_extracted_feature_source="peak_bins",
        pair_frequency_column_patterns=[
            r"Vibration_Peak_\d+_[XY]_Frequency",
        ],
        pair_amplitude_column_patterns=[
            r"Vibration_Peak_\d+_[XY]_Amplitude",
        ],
        k_peaks=5,
        f_max_rel=12.0,
        peak_harmonic_bins=[1, 2, 3, 4, 5],
        peak_bin_tolerance=0.35,
        cnn_window=16,
        conv_channels=[16, 16],
        fc_hidden=32,
        kernel_size=5,
        batch_size=16,
        pos_weight=3.0,
        model_save_path="data/models/harmonic_context_casedata_peaks.pt",
        dataset_name="casedata_peaks",
        enabled=True,
    )
    defaults.update(overrides)
    return HarmonicContextConfig(**defaults)


def site_a_line2_breakage_preset(**overrides: Any) -> HarmonicContextConfig:
    """Preset for Site_a_line2 → breakage prediction.

    Site_a_line2 merged CSVs contain pre-computed harmonic/vibration features
    from the DLG6CF sensor at ~1 Hz.  Context params available:
    ``spindle_speed`` (SpindleSpeedActual), ``feed_rate`` (Axis_FeedRate_actual),
    ``teeth_count`` (CNC_parameters_teeth_num).
    """
    valid_keys = HarmonicContextConfig.__dataclass_fields__
    overrides = {k: v for k, v in overrides.items() if k in valid_keys}
    defaults = dict(
        target_label="label",
        positive_labels=["pre_break", "break_event"],
        input_channel_roles=["primary_vibration"],
        input_column_patterns=[
            r"Accel_Severity_Acc_\d+",
            r"Chatter_Detection_Amplitude_Acc_\d+",
        ],
        input_columns=[],
        context_param_keys=["spindle_speed", "feed_rate", "teeth_count"],
        context_param_sources={
            "spindle_speed": "SpindleSpeedActual",
            "feed_rate": "Axis_FeedRate_actual",
            "teeth_count": "CNC_parameters_teeth_num",
        },
        harmonic_mode="pre_extracted",
        harmonic_column_patterns=[
            r"Vibration_Harmonic_\d+_Acc_\d+_Amplitude",
            r"Chatter_Detection_Amplitude_Acc_\d+",
        ],
        harmonic_columns=[],
        cnn_window=16,
        conv_channels=[16, 16],
        fc_hidden=32,
        kernel_size=5,
        batch_size=16,
        pos_weight=5.0,  # Breakage events are very rare
        model_save_path="data/models/harmonic_context_site_a_line2.pt",
        dataset_name="site_a_line2",
        enabled=True,
    )
    defaults.update(overrides)
    return HarmonicContextConfig(**defaults)


def stoppage_1hz_preset(**overrides: Any) -> HarmonicContextConfig:
    """Preset for 1 Hz stoppage-series data with precomputed harmonics.

    This matches the incoming casedata/olddata-style streams and the
    `stoppage_raw_series.npz` experiment artifact: low-rate time series with
    precomputed `Vibration_Harmonic_*` amplitudes rather than high-rate raw
    acceleration suitable for FFT extraction at train time.
    """
    valid_keys = HarmonicContextConfig.__dataclass_fields__
    overrides = {k: v for k, v in overrides.items() if k in valid_keys}
    defaults = dict(
        target_label="label",
        positive_labels=["pre_stoppage"],
        input_channel_roles=["primary_vibration"],
        input_column_patterns=[
            r"Vibration_Harmonic_\d+_[XY]_Amplitude",
            r"Vibration_Severity_[XY]",
            r"Chatter_Detection_Amplitude_[XY]",
        ],
        input_columns=[],
        context_param_keys=["spindle_speed", "feed_rate"],
        context_param_sources={
            "spindle_speed": "Spindle_Speed_Actual",
            "feed_rate": "Feed_Rate_Actual",
        },
        harmonic_mode="pre_extracted",
        harmonic_column_patterns=[
            r"Vibration_Harmonic_\d+_[XY]_Amplitude",
        ],
        harmonic_columns=[],
        cnn_window=16,
        conv_channels=[16, 16],
        fc_hidden=32,
        kernel_size=5,
        batch_size=16,
        pos_weight=3.0,
        model_save_path="data/models/harmonic_context_1hz.pt",
        dataset_name="stoppage_1hz",
        enabled=True,
    )
    defaults.update(overrides)
    return HarmonicContextConfig(**defaults)


def raw_accelerometer_preset(**overrides: Any) -> HarmonicContextConfig:
    """Preset for raw high-frequency accelerometer data (generic domain).

    For datasets with raw multi-channel accelerometer signals at high sample
    rates (e.g., 50 kHz), this preset uses ``harmonic_mode='raw_fft'`` to
    extract spindle harmonics via sliding FFT.
    """
    valid_keys = HarmonicContextConfig.__dataclass_fields__
    overrides = {k: v for k, v in overrides.items() if k in valid_keys}
    defaults = dict(
        target_label="label",
        positive_labels=["anomaly", "fault"],
        input_channel_roles=["primary_vibration"],
        input_column_patterns=[r"Channel_\d+", r"accel_[xyz]"],
        input_columns=[],
        context_param_keys=["spindle_speed", "feed_rate"],
        context_param_sources={},
        harmonic_mode="raw_fft",
        harmonic_multipliers=[1, 2, 3, 4, 6, 8, 10],
        fft_window=4096,
        fft_step=1024,
        harmonic_column_patterns=[],
        harmonic_columns=[],
        cnn_window=16,
        conv_channels=[16, 16],
        fc_hidden=32,
        kernel_size=5,
        batch_size=16,
        model_save_path="data/models/harmonic_context_raw.pt",
        dataset_name="raw_accelerometer",
        enabled=True,
    )
    defaults.update(overrides)
    return HarmonicContextConfig(**defaults)


def pair_raw_preset(**overrides: Any) -> HarmonicContextConfig:
    """Preset for pair-input training on labelled FFT peak parquet data."""
    valid_keys = HarmonicContextConfig.__dataclass_fields__
    overrides = {k: v for k, v in overrides.items() if k in valid_keys}
    defaults = dict(
        target_label="label",
        positive_labels=["pre_break", "break_event"],
        input_channel_roles=["primary_vibration"],
        input_column_patterns=[],
        input_columns=[],
        context_param_keys=["spindle_speed", "feed_rate", "teeth_count"],
        context_param_sources={
            "spindle_speed": "CNC_parameters_Programed_spindle_speed",
            "feed_rate": "Axis_FeedRate_commanded",
            "teeth_count": "CNC_parameters_teeth_num",
        },
        scorer_kind="pair",
        harmonic_mode="pre_extracted",
        harmonic_column_patterns=[],
        harmonic_columns=[],
        pair_frequency_column_patterns=[
            r"Accel_FFT_Acc\d+_range\d+_Frequencies_\d+",
        ],
        pair_amplitude_column_patterns=[
            r"Accel_FFT_Acc\d+_range\d+_Amplitudes_\d+",
        ],
        k_peaks=5,
        pair_embed_dim=16,
        f_max_rel=12.0,
        cnn_window=16,
        conv_channels=[32, 32],
        fc_hidden=32,
        kernel_size=3,
        batch_size=16,
        pos_weight=5.0,
        model_save_path="data/models/harmonic_pair_raw.pt",
        dataset_name="pair_raw",
        enabled=True,
    )
    defaults.update(overrides)
    return HarmonicContextConfig(**defaults)


def pair_casedata_preset(**overrides: Any) -> HarmonicContextConfig:
    """Preset for pair-input training on casedata vibration peak columns."""
    valid_keys = HarmonicContextConfig.__dataclass_fields__
    overrides = {k: v for k, v in overrides.items() if k in valid_keys}
    defaults = dict(
        target_label="label",
        positive_labels=["pre_stoppage"],
        input_channel_roles=["primary_vibration"],
        input_column_patterns=[],
        input_columns=[],
        context_param_keys=["spindle_speed", "feed_rate"],
        context_param_sources={
            "spindle_speed": "spindle_speed_mean",
            "feed_rate": "feed_rate_mean",
        },
        scorer_kind="pair",
        harmonic_mode="pre_extracted",
        harmonic_column_patterns=[],
        harmonic_columns=[],
        pair_frequency_column_patterns=[
            r"Vibration_Peak_\d+_[XY]_Frequency",
        ],
        pair_amplitude_column_patterns=[
            r"Vibration_Peak_\d+_[XY]_Amplitude",
        ],
        k_peaks=5,
        pair_embed_dim=16,
        f_max_rel=12.0,
        pair_sample_mode="trailing_window",
        cnn_window=16,
        conv_channels=[32, 32],
        fc_hidden=32,
        kernel_size=3,
        batch_size=16,
        pos_weight=3.0,
        model_save_path="data/models/harmonic_pair_casedata.pt",
        dataset_name="pair_casedata",
        enabled=True,
    )
    defaults.update(overrides)
    return HarmonicContextConfig(**defaults)


def pair_lfl_preset(**overrides: Any) -> HarmonicContextConfig:
    """Preset for the original LFL pair-input architecture.

    This keeps the peak-pair tensor format used by the integrated runtime,
    but restores the original parameter-conditioned pair encoder and the
    full five-parameter cutting vector ``[d, z, n, f, vf]``.
    """
    valid_keys = HarmonicContextConfig.__dataclass_fields__
    overrides = {k: v for k, v in overrides.items() if k in valid_keys}
    defaults = dict(
        target_label="label",
        positive_labels=["pre_stoppage"],
        input_channel_roles=["primary_vibration"],
        input_column_patterns=[],
        input_columns=[],
        context_param_keys=["d", "z", "n", "f", "vf"],
        context_param_sources={
            "d": "tool_diameter",
            "z": "num_teeth",
            "n": "spindle_speed_mean",
            "f": "feed_per_tooth",
            "vf": "feed_rate_mean",
        },
        scorer_kind="pair",
        model_kind="lfl_v2",
        harmonic_mode="pre_extracted",
        harmonic_column_patterns=[],
        harmonic_columns=[],
        pair_frequency_column_patterns=[
            r"Vibration_Peak_\d+_[XY]_Frequency",
        ],
        pair_amplitude_column_patterns=[
            r"Vibration_Peak_\d+_[XY]_Amplitude",
        ],
        k_peaks=5,
        pair_embed_dim=16,
        f_max_rel=12.0,
        pair_sample_mode="trailing_window",
        cnn_window=12,
        conv_channels=[16, 16],
        fc_hidden=32,
        kernel_size=5,
        batch_size=16,
        pos_weight=1.25,
        model_save_path="data/models/harmonic_pair_lfl.pt",
        dataset_name="pair_lfl",
        enabled=True,
    )
    defaults.update(overrides)
    return HarmonicContextConfig(**defaults)

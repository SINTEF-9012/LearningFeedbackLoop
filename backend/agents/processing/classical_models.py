"""
Classical Models — Seed models and reinforcement learning for anomaly detection.

# ===========================================================================
# DRAFT/PROTOTYPE - Tag: [PROTOTYPE_CLASSICAL_RL_V1]
# This module sketches classical ML + RL approaches to complement the
# existing pattern-based significance scoring.
#
# Architecture:
#   DatasetLoader → FeatureExtractor → SeedModel (Isolation Forest / LOF)
#     ↓                                       ↑ (online updates)
#   AnomalyScore → SignificanceScorer  ←──  RLAgent (reward from feedback)
#
# The seed model provides a baseline anomaly detector trained on real data.
# The RL agent learns a policy for weighting anomaly signals based on
# operator feedback (confirm/dismiss), closing the learning loop.
# ===========================================================================

Progression plan:
1. SeedModel: train Isolation Forest + Local Outlier Factor on case data
2. OnlineAnomalyDetector: sliding-window anomaly scoring with model adaptation
3. RLAgent: contextual bandit / Q-learning for feedback-driven weight tuning
4. Integration: feed classical model scores into SignificanceScorer.external_signals
"""

from __future__ import annotations

import logging
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..model_confidence import current_model_confidence

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed — classical models unavailable")


# ============================================================================
# Feature Extraction
# ============================================================================

# Features used across all models (derived from DatasetLoader.WindowData)
FEATURE_NAMES = [
    # --- Original 17 CNC features ---
    "power_spindle_mean",
    "power_spindle_max",
    "power_spindle_std",
    "power_y_mean",
    "power_y_max",
    "power_z_mean",
    "vib_severity_x_mean",
    "vib_severity_x_max",
    "vib_severity_y_mean",
    "vib_severity_y_max",
    "chatter_ratio",
    "power_active_mean",
    "power_active_std",
    "power_factor_mean",
    "spindle_speed_mean",
    "feed_rate_mean",
    "temp_head_mean",
    # --- Physics-based fault features (new) ---
    # Tool breakage: sudden HF burst, loss of periodicity
    "hf_energy_ratio",          # fraction of spectral energy above 500 Hz
    "impulse_crest_factor",     # max crest factor across vibration channels
    "kurtosis_max",             # max excess kurtosis (heavy-tailed impulse)
    "periodicity_strength",     # autocorrelation peak at expected tooth period
    # Chatter: modulated vibration, increased amplitude
    "modulation_depth",         # amplitude modulation index (envelope crest)
    "vib_amplitude_growth",     # ratio of current window RMS to session baseline
    "tp_harmonic_energy",       # energy at tooth-passing frequency harmonics
    # Chip adhesion / built-up edge: irregular tooth-passing pattern
    "harmonic_amplitude_cv",    # coefficient of variation of harmonic amplitudes
    "tp_amplitude_variance",    # normalised variance at tooth-passing frequency
    # Workpiece slip / clamping: shift at spindle frequency
    "spindle_order_amplitude",  # amplitude at 1× spindle frequency
    "spindle_phase_shift",      # phase change at spindle frequency (radians)
]

# Original feature count (for backward compatibility with pre-trained models)
N_ORIGINAL_FEATURES = 17


def features_from_dict(feature_dict: Dict[str, float]) -> np.ndarray:
    """Extract a fixed-order feature vector from a feature dictionary.

    Missing features are filled with 0.0 (neutral).
    """
    return np.array(
        [float(feature_dict.get(name, 0.0)) for name in FEATURE_NAMES],
        dtype=np.float64,
    )


def batch_features_from_df(
    df: "pd.DataFrame",
    col_map: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Vectorized feature extraction from a DataFrame → (N, D) float64 array.

    10-100× faster than calling ``features_from_dict()`` per row via iterrows.
    ``col_map`` maps CSV column names to FEATURE_NAMES entries (same format as
    ``BreakageFeatureExtractor._COL_MAP``).  When *None*, assumes columns already
    match FEATURE_NAMES.
    """
    import pandas as pd  # local import to keep module lightweight

    n = len(df)
    d = len(FEATURE_NAMES)
    out = np.zeros((n, d), dtype=np.float64)

    # Build reverse lookup: feature_name → csv_column
    if col_map is not None:
        feat_to_csv: Dict[str, str] = {v: k for k, v in col_map.items()}
    else:
        feat_to_csv = {name: name for name in FEATURE_NAMES}

    for fi, feat_name in enumerate(FEATURE_NAMES):
        csv_col = feat_to_csv.get(feat_name, feat_name)
        if csv_col in df.columns:
            out[:, fi] = pd.to_numeric(df[csv_col], errors="coerce").fillna(0.0).values
        # else stays 0.0

    # Handle chatter_ratio specially if source columns exist
    chatter_idx = None
    for fi, name in enumerate(FEATURE_NAMES):
        if name == "chatter_ratio":
            chatter_idx = fi
            break
    if chatter_idx is not None:
        cx_col = "chatter_amp_x_mean"
        cy_col = "chatter_amp_y_mean"
        if cx_col in df.columns and cy_col in df.columns:
            cx = pd.to_numeric(df[cx_col], errors="coerce").fillna(0.0).values
            cy = pd.to_numeric(df[cy_col], errors="coerce").fillna(0.0).values
            out[:, chatter_idx] = cx / (cy + 1e-10)

    return out


def features_from_window(window: Any) -> np.ndarray:
    """Extract feature vector from a DatasetLoader.WindowData instance."""
    if not window.features:
        window.compute_features()
    return features_from_dict(window.features)


# ============================================================================
# Seed Model — Unsupervised anomaly detection baseline
# ============================================================================

@dataclass
class SeedModelConfig:
    """Configuration for the seed anomaly detection model."""

    # Isolation Forest parameters
    n_estimators: int = 200
    contamination: float = 0.05  # Expected fraction of anomalies
    max_samples: str | int = "auto"  # Subsample size per tree
    random_state: int = 42

    # Local Outlier Factor parameters
    lof_n_neighbors: int = 20
    lof_contamination: float = 0.05

    # Ensemble weighting
    iforest_weight: float = 0.6
    lof_weight: float = 0.4

    # Training data sampling
    window_seconds: int = 10  # Window size for feature extraction
    stride_seconds: int = 5   # Stride between windows
    max_training_windows: int = 5000  # Cap on training set size


class SeedModel:
    """Unsupervised anomaly detection seed model.

    Combines Isolation Forest and Local Outlier Factor into an ensemble.
    Trained on real case data to establish a baseline anomaly detector
    that feeds scores into the existing SignificanceScorer as
    ``external_signals["anomaly_detector_score"]``.

    Usage:
        model = SeedModel()
        model.train_from_casedata(loader, ["OF00001", "OF00002"])
        score = model.score(feature_vector)  # → 0.0 (normal) to 1.0 (anomaly)
    """

    def __init__(self, config: Optional[SeedModelConfig] = None):
        self.config = config or SeedModelConfig()
        self._scaler: Optional[Any] = None  # StandardScaler
        self._iforest: Optional[Any] = None  # IsolationForest
        self._lof: Optional[Any] = None  # LocalOutlierFactor
        self._is_trained: bool = False
        self._training_stats: Dict[str, Any] = {}

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, X: np.ndarray) -> Dict[str, Any]:
        """Train the ensemble on a feature matrix.

        Args:
            X: (n_samples, n_features) array

        Returns:
            Training statistics dict
        """
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn required for SeedModel training")

        n_samples, n_features = X.shape
        logger.info(
            "SeedModel training: %d samples × %d features", n_samples, n_features
        )

        # Standardise features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Train Isolation Forest
        self._iforest = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
        )
        self._iforest.fit(X_scaled)

        # Train LOF (novelty detection mode)
        self._lof = LocalOutlierFactor(
            n_neighbors=min(self.config.lof_n_neighbors, n_samples - 1),
            contamination=self.config.lof_contamination,
            novelty=True,  # Required for .score_samples() on new data
        )
        self._lof.fit(X_scaled)

        self._is_trained = True

        # Compute baseline statistics
        if_scores = self._iforest.score_samples(X_scaled)
        lof_scores = self._lof.score_samples(X_scaled)

        self._training_stats = {
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "feature_names": FEATURE_NAMES,
            "iforest_score_mean": float(np.mean(if_scores)),
            "iforest_score_std": float(np.std(if_scores)),
            "lof_score_mean": float(np.mean(lof_scores)),
            "lof_score_std": float(np.std(lof_scores)),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "SeedModel trained: IF score mean=%.3f std=%.3f, LOF score mean=%.3f std=%.3f",
            self._training_stats["iforest_score_mean"],
            self._training_stats["iforest_score_std"],
            self._training_stats["lof_score_mean"],
            self._training_stats["lof_score_std"],
        )

        return self._training_stats

    def train_from_casedata(
        self,
        loader: Any,  # DatasetLoader
        operation_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Train on sliding windows extracted from real case data.

        Uses the bulk feature extraction method for efficiency — each CSV
        file is loaded once rather than once per window.

        Args:
            loader: DatasetLoader instance
            operation_ids: Specific operations to train on; None = all
        """
        from .dataset_loader import DatasetLoader

        if not isinstance(loader, DatasetLoader):
            raise TypeError("Expected DatasetLoader instance")

        ops = (
            [loader.get_operation(oid) for oid in operation_ids]
            if operation_ids
            else loader.list_operations()
        )

        all_features: List[np.ndarray] = []
        budget_per_op = max(
            50, self.config.max_training_windows // max(len(ops), 1)
        )

        for op_info in ops:
            logger.info(
                "Extracting training features from %s (budget=%d)...",
                op_info.operation_id,
                budget_per_op,
            )

            feat_dicts = loader.extract_bulk_features(
                op_info.operation_id,
                window_seconds=self.config.window_seconds,
                stride_seconds=self.config.stride_seconds,
                max_windows=budget_per_op,
            )

            for fd in feat_dicts:
                fv = features_from_dict(fd)
                if not np.all(np.isnan(fv)):
                    all_features.append(fv)

            if len(all_features) >= self.config.max_training_windows:
                break

        if len(all_features) < 10:
            raise ValueError(
                f"Insufficient training data: only {len(all_features)} windows extracted"
            )

        X = np.stack(all_features)
        # Replace NaN/inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return self.train(X)

    def score(self, features: np.ndarray) -> float:
        """Score a single feature vector for anomaly likelihood.

        Args:
            features: 1D array of length n_features

        Returns:
            Anomaly score in [0.0, 1.0] where 1.0 = most anomalous
        """
        if not self._is_trained:
            return 0.5  # Neutral when untrained

        x = features.reshape(1, -1)
        x_scaled = self._scaler.transform(x)

        # Isolation Forest: score_samples returns negative anomaly score
        # More negative = more anomalous
        if_raw = self._iforest.score_samples(x_scaled)[0]
        # Normalise to [0, 1]: use training stats for calibration
        if_mean = self._training_stats["iforest_score_mean"]
        if_std = self._training_stats["iforest_score_std"]
        if_z = (if_mean - if_raw) / max(if_std, 1e-8)  # Positive = anomalous
        if_z = float(np.clip(if_z, -4.0, 4.0))  # Clamp to prevent sigmoid saturation
        if_score = float(np.clip(1.0 / (1.0 + np.exp(-if_z)), 0.0, 1.0))

        # LOF: negative_outlier_factor-like score
        lof_raw = self._lof.score_samples(x_scaled)[0]
        lof_mean = self._training_stats["lof_score_mean"]
        lof_std = self._training_stats["lof_score_std"]
        lof_z = (lof_mean - lof_raw) / max(lof_std, 1e-8)
        lof_z = float(np.clip(lof_z, -4.0, 4.0))  # Clamp to prevent sigmoid saturation
        lof_score = float(np.clip(1.0 / (1.0 + np.exp(-lof_z)), 0.0, 1.0))

        # Weighted ensemble
        w_if = self.config.iforest_weight
        w_lof = self.config.lof_weight
        combined = (w_if * if_score + w_lof * lof_score) / (w_if + w_lof)

        return float(combined)

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Score an entire (N, D) feature matrix at once.

        Returns a 1-D array of anomaly scores in [0.0, 1.0].
        10-50× faster than calling ``score()`` per row because
        sklearn's ``score_samples`` vectorises over all rows.
        """
        if not self._is_trained:
            return np.full(X.shape[0], 0.5)

        X_scaled = self._scaler.transform(X)

        # Isolation Forest
        if_raw = self._iforest.score_samples(X_scaled)  # (N,)
        if_mean = self._training_stats["iforest_score_mean"]
        if_std = max(self._training_stats["iforest_score_std"], 1e-8)
        if_z = np.clip((if_mean - if_raw) / if_std, -4.0, 4.0)
        if_score = 1.0 / (1.0 + np.exp(-if_z))

        # LOF
        lof_raw = self._lof.score_samples(X_scaled)
        lof_mean = self._training_stats["lof_score_mean"]
        lof_std = max(self._training_stats["lof_score_std"], 1e-8)
        lof_z = np.clip((lof_mean - lof_raw) / lof_std, -4.0, 4.0)
        lof_score = 1.0 / (1.0 + np.exp(-lof_z))

        # Weighted ensemble
        w_if = self.config.iforest_weight
        w_lof = self.config.lof_weight
        combined = (w_if * if_score + w_lof * lof_score) / (w_if + w_lof)
        return combined.astype(np.float64)

    def score_dict(self, feature_dict: Dict[str, float]) -> float:
        """Score from a feature dictionary (convenience wrapper)."""
        return self.score(features_from_dict(feature_dict))

    def score_detailed(self, features: np.ndarray) -> Dict[str, Any]:
        """Return individual model scores plus ensemble.

        Public API so callers do not need to access private model
        internals (``_scaler``, ``_iforest``, ``_lof``).

        Returns:
            Dict with keys ``ensemble``, ``isolation_forest``, ``lof``
            each in [0.0, 1.0].
        """
        if not self._is_trained:
            return {"ensemble": 0.5, "isolation_forest": 0.5, "lof": 0.5}

        x = features.reshape(1, -1)
        x_scaled = self._scaler.transform(x)

        # Isolation Forest
        if_raw = self._iforest.score_samples(x_scaled)[0]
        if_mean = self._training_stats["iforest_score_mean"]
        if_std = self._training_stats["iforest_score_std"]
        if_z = (if_mean - if_raw) / max(if_std, 1e-8)
        if_z = float(np.clip(if_z, -4.0, 4.0))
        if_score = float(np.clip(1.0 / (1.0 + np.exp(-if_z)), 0.0, 1.0))

        # LOF
        lof_raw = self._lof.score_samples(x_scaled)[0]
        lof_mean = self._training_stats["lof_score_mean"]
        lof_std = self._training_stats["lof_score_std"]
        lof_z = (lof_mean - lof_raw) / max(lof_std, 1e-8)
        lof_z = float(np.clip(lof_z, -4.0, 4.0))
        lof_score = float(np.clip(1.0 / (1.0 + np.exp(-lof_z)), 0.0, 1.0))

        # Weighted ensemble
        w_if = self.config.iforest_weight
        w_lof = self.config.lof_weight
        ensemble = (w_if * if_score + w_lof * lof_score) / (w_if + w_lof)

        return {
            "ensemble": float(ensemble),
            "isolation_forest": if_score,
            "lof": lof_score,
        }

    def score_detailed_dict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """Return detailed scores from a feature dictionary (convenience wrapper)."""
        return self.score_detailed(features_from_dict(feature_dict))

    @property
    def n_training_samples(self) -> int:
        """Number of samples the model was trained on."""
        return int(self._training_stats.get("n_samples", 0))

    @property
    def feature_names(self) -> List[str]:
        """Feature names the model was trained with."""
        return list(self._training_stats.get("feature_names", []))

    def save(self, path: str | Path) -> None:
        """Persist the trained model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "scaler": self._scaler,
            "iforest": self._iforest,
            "lof": self._lof,
            "config": self.config,
            "training_stats": self._training_stats,
            "feature_names": FEATURE_NAMES,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("SeedModel saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a previously trained model from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._scaler = data["scaler"]
        self._iforest = data["iforest"]
        self._lof = data["lof"]
        self.config = data.get("config", SeedModelConfig())
        self._training_stats = data.get("training_stats", {})
        self._is_trained = True
        logger.info("SeedModel loaded from %s", path)


# ============================================================================
# RL Agent — Feedback-driven weight learning
# ============================================================================

@dataclass
class RLState:
    """State representation for the RL agent.

    Captures the context in which an anomaly decision is made,
    allowing the agent to learn context-dependent policies.
    """
    # Anomaly model outputs
    seed_model_score: float = 0.5
    harmonic_score: float = 0.0
    pattern_score: float = 0.0
    historical_prior: float = 0.5

    # Cutting context (discretised)
    operating_regime: str = "unknown"
    tool_type: str = "unknown"
    material: str = "unknown"

    # Recent history
    recent_alert_rate: float = 0.0  # Alerts per minute in last 10 min
    recent_dismiss_rate: float = 0.0  # Fraction of dismissed alerts

    def to_vector(self) -> np.ndarray:
        """Convert to numeric feature vector for Q-table lookup."""
        # Discretise continuous features into bins
        bins = [
            _discretise(self.seed_model_score, 5),
            _discretise(self.harmonic_score, 5),
            _discretise(self.pattern_score, 5),
            _discretise(self.historical_prior, 5),
            hash(self.operating_regime) % 8,
            hash(self.tool_type) % 8,
            hash(self.material) % 8,
            _discretise(self.recent_alert_rate, 4),
            _discretise(self.recent_dismiss_rate, 4),
        ]
        return np.array(bins, dtype=np.int32)

    def to_key(self) -> str:
        """Convert to a hashable string key for tabular Q-learning."""
        v = self.to_vector()
        return "|".join(str(x) for x in v)


def _discretise(value: float, n_bins: int) -> int:
    """Map [0, 1] float to integer bin."""
    return int(np.clip(value * n_bins, 0, n_bins - 1))


@dataclass
class RLAction:
    """Actions the RL agent can take."""
    # Weight adjustments for the scorer
    classical_weight_adj: float = 0.0  # Adjustment to classical model weight
    pattern_weight_adj: float = 0.0   # Adjustment to pattern rule weight
    threshold_adj: float = 0.0        # Adjustment to alert threshold

    @staticmethod
    def action_space() -> List["RLAction"]:
        """Enumerate the discrete action space.

        Actions represent weight adjustments that the RL agent can
        recommend.  The idea is that the agent learns which combination
        of weight shifts yields the best operator-aligned alerting
        behaviour.
        """
        actions = []
        for c_adj in [-0.05, 0.0, 0.05]:
            for p_adj in [-0.05, 0.0, 0.05]:
                for t_adj in [-0.02, 0.0, 0.02]:
                    actions.append(RLAction(
                        classical_weight_adj=c_adj,
                        pattern_weight_adj=p_adj,
                        threshold_adj=t_adj,
                    ))
        return actions


@dataclass
class RLConfig:
    """Hyperparameters for the RL agent."""
    learning_rate: float = 0.1        # Q-learning α
    discount_factor: float = 0.95     # γ
    exploration_rate: float = 0.2     # ε for ε-greedy
    exploration_decay: float = 0.999  # ε decay per episode
    min_exploration: float = 0.01     # Lower bound on ε
    reward_confirm: float = 1.0       # Reward for confirmed alert
    reward_dismiss: float = -0.5      # Penalty for dismissed alert
    reward_no_alert_normal: float = 0.1  # Reward for not alerting on normal
    reward_missed_anomaly: float = -1.0  # Penalty for missing a real anomaly


class RLAgent:
    """Tabular Q-learning agent for feedback-driven weight optimisation.

    The agent operates as a contextual bandit: given the current state
    (anomaly scores, cutting context, recent history), it selects weight
    adjustments that maximise long-term operator satisfaction.

    Rewards come from operator feedback:
    - Confirm alert → positive reward
    - Dismiss alert → negative reward
    - No alert on normal window → small positive reward

    Usage:
        agent = RLAgent()
        state = RLState(seed_model_score=0.8, ...)
        action = agent.select_action(state)
        # ... apply action, get feedback ...
        agent.update(state, action_idx, reward, next_state)
    """

    def __init__(self, config: Optional[RLConfig] = None):
        self.config = config or RLConfig()
        self._actions = RLAction.action_space()
        self._q_table: Dict[str, np.ndarray] = {}
        self._epsilon = self.config.exploration_rate
        self._episode_count: int = 0
        self._stats: Dict[str, Any] = {
            "total_updates": 0,
            "total_confirms": 0,
            "total_dismisses": 0,
            "cumulative_reward": 0.0,
        }

    @property
    def n_actions(self) -> int:
        return len(self._actions)

    def select_action(self, state: RLState) -> Tuple[int, RLAction]:
        """Select an action using ε-greedy policy.

        Returns:
            (action_index, action) tuple
        """
        key = state.to_key()

        # ε-greedy exploration
        if np.random.random() < self._epsilon:
            idx = np.random.randint(self.n_actions)
        else:
            q_values = self._get_q_values(key)
            idx = int(np.argmax(q_values))

        return idx, self._actions[idx]

    def get_recommended_action(self, state: RLState) -> RLAction:
        """Get the greedy (exploitation-only) recommended action."""
        key = state.to_key()
        q_values = self._get_q_values(key)
        idx = int(np.argmax(q_values))
        return self._actions[idx]

    def update(
        self,
        state: RLState,
        action_idx: int,
        reward: float,
        next_state: Optional[RLState] = None,
    ) -> None:
        """Q-learning update step.

        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
        """
        key = state.to_key()
        q_values = self._get_q_values(key)

        # Current Q value
        current_q = q_values[action_idx]

        # Future value estimate
        if next_state is not None:
            next_key = next_state.to_key()
            next_q = self._get_q_values(next_key)
            future_value = self.config.discount_factor * float(np.max(next_q))
        else:
            future_value = 0.0

        # Q-learning update
        td_error = reward + future_value - current_q
        q_values[action_idx] += self.config.learning_rate * td_error
        self._q_table[key] = q_values

        # Decay exploration
        self._epsilon = max(
            self.config.min_exploration,
            self._epsilon * self.config.exploration_decay,
        )
        self._episode_count += 1

        # Update stats
        self._stats["total_updates"] += 1
        self._stats["cumulative_reward"] += reward
        if reward > 0:
            self._stats["total_confirms"] += 1
        elif reward < 0:
            self._stats["total_dismisses"] += 1

    def compute_reward(self, feedback_action: str, was_alerted: bool) -> float:
        """Compute reward from operator feedback.

        Args:
            feedback_action: "confirm" or "dismiss"
            was_alerted: Whether the system triggered an alert
        """
        if was_alerted:
            if feedback_action == "confirm":
                return self.config.reward_confirm
            elif feedback_action == "dismiss":
                return self.config.reward_dismiss
        else:
            # No alert was triggered
            if feedback_action == "confirm":
                # Operator says it was real but we missed it
                return self.config.reward_missed_anomaly
            else:
                return self.config.reward_no_alert_normal

        return 0.0

    def get_weight_adjustments(self, state: RLState) -> Dict[str, float]:
        """Get the recommended weight adjustments for the scorer.

        Returns a dict that can be applied to SignificanceConfig weights.
        """
        action = self.get_recommended_action(state)
        return {
            "weight_classical_alert_adj": action.classical_weight_adj,
            "weight_pattern_rule_adj": action.pattern_weight_adj,
            "alert_threshold_adj": action.threshold_adj,
        }

    def _get_q_values(self, key: str) -> np.ndarray:
        """Get or initialise Q-values for a state."""
        if key not in self._q_table:
            self._q_table[key] = np.zeros(self.n_actions, dtype=np.float64)
        return self._q_table[key]

    def save(self, path: str | Path) -> None:
        """Persist the agent to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_table": {k: v.tolist() for k, v in self._q_table.items()},
            "epsilon": self._epsilon,
            "episode_count": self._episode_count,
            "config": self.config,
            "stats": self._stats,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("RLAgent saved to %s (%d states)", path, len(self._q_table))

    def load(self, path: str | Path) -> None:
        """Load agent state from disk."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._q_table = {
            k: np.array(v, dtype=np.float64)
            for k, v in data.get("q_table", {}).items()
        }
        self._epsilon = data.get("epsilon", self.config.exploration_rate)
        self._episode_count = data.get("episode_count", 0)
        self._stats = data.get("stats", self._stats)
        logger.info("RLAgent loaded from %s (%d states)", path, len(self._q_table))

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "epsilon": self._epsilon,
            "episode_count": self._episode_count,
            "q_table_size": len(self._q_table),
        }


# ============================================================================
# Online Anomaly Detector — Bridges classical models into the scorer
# ============================================================================

class OnlineAnomalyDetector:
    """Real-time anomaly scoring using seed model + RL-adjusted weights.

    This is the main integration point between the classical models and
    the existing SignificanceScorer. It:
    1. Scores new windows using the seed model
    2. Optionally adjusts scorer weights using the RL agent
    3. Returns an ``external_signals`` dict for the scorer

    Usage:
        detector = OnlineAnomalyDetector(seed_model, rl_agent)
        signals = detector.score_window(feature_dict, cutting_context)
        # signals is like {"anomaly_detector_score": 0.73, "model_confidence": 0.8}
        # Pass to scorer: scorer.score(..., external_signals=signals)
    """

    def __init__(
        self,
        seed_model: Optional[SeedModel] = None,
        rl_agent: Optional[RLAgent] = None,
        model_confidence_path: Optional[str | Path] = None,
    ):
        self.seed_model = seed_model
        self.rl_agent = rl_agent
        self.model_confidence_path = Path(model_confidence_path) if model_confidence_path else None
        self._recent_scores: List[float] = []
        self._max_history: int = 100

    def score_window(
        self,
        feature_dict: Dict[str, float],
        *,
        cutting_context: Optional[Dict[str, Any]] = None,
        pattern_score: float = 0.0,
        historical_prior: float = 0.5,
    ) -> Dict[str, Any]:
        """Score a window and return external signals for the scorer.

        Returns dict with:
            anomaly_detector_score: float [0, 1]
            model_confidence: float [0, 1]
            model_source: str
            weight_adjustments: dict (from RL agent if available)
        """
        signals: Dict[str, Any] = {}

        # 1. Seed model score
        if self.seed_model is not None and self.seed_model.is_trained:
            score = self.seed_model.score_dict(feature_dict)
            signals["anomaly_detector_score"] = round(score, 4)

            # Track for rolling confidence estimation
            self._recent_scores.append(score)
            if len(self._recent_scores) > self._max_history:
                self._recent_scores = self._recent_scores[-self._max_history:]

            # Confidence is learned from confirm/dismiss feedback on the
            # classical rule rather than a static training-sample heuristic.
            signals["model_confidence"] = current_model_confidence(self.model_confidence_path)
            signals["model_source"] = "seed_model_v1"

        # 2. RL agent adjustments
        if self.rl_agent is not None:
            recent_alert_rate = sum(
                1 for s in self._recent_scores[-20:] if s > 0.6
            ) / max(len(self._recent_scores[-20:]), 1)

            state = RLState(
                seed_model_score=signals.get("anomaly_detector_score", 0.5),
                pattern_score=pattern_score,
                historical_prior=historical_prior,
                operating_regime=(cutting_context or {}).get("operating_regime", "unknown"),
                tool_type=(cutting_context or {}).get("tool_type", "unknown"),
                material=(cutting_context or {}).get("workpiece_material", "unknown"),
                recent_alert_rate=recent_alert_rate,
            )

            adjustments = self.rl_agent.get_weight_adjustments(state)
            signals["weight_adjustments"] = adjustments

        return signals

    def process_feedback(
        self,
        feedback_action: str,
        was_alerted: bool,
        feature_dict: Dict[str, float],
        cutting_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Process operator feedback to update the RL agent.

        Called when an operator confirms or dismisses an alert.
        """
        if self.rl_agent is None:
            return

        score = 0.5
        if self.seed_model is not None and self.seed_model.is_trained:
            score = self.seed_model.score_dict(feature_dict)

        state = RLState(
            seed_model_score=score,
            operating_regime=(cutting_context or {}).get("operating_regime", "unknown"),
            tool_type=(cutting_context or {}).get("tool_type", "unknown"),
            material=(cutting_context or {}).get("workpiece_material", "unknown"),
        )

        reward = self.rl_agent.compute_reward(feedback_action, was_alerted)
        action_idx, _ = self.rl_agent.select_action(state)
        self.rl_agent.update(state, action_idx, reward)


# ============================================================================
# Factory / convenience
# ============================================================================

def create_seed_model(
    casedata_path: str | Path = "data/casedata",
    model_path: Optional[str | Path] = None,
    operation_ids: Optional[List[str]] = None,
    *,
    force_retrain: bool = False,
    lazy_training: bool = False,
) -> SeedModel:
    """Create or load a SeedModel.

    If a saved model exists at model_path and force_retrain is False,
    loads it. Otherwise trains from casedata.

    When ``lazy_training`` is True **and** no cached model is available,
    training runs on a background ``threading.Thread`` and an untrained
    ``SeedModel`` is returned immediately. Consumers must handle
    ``seed_model.is_trained is False`` — ``OnlineAnomalyDetector.score_window``
    already does: it simply omits the ``anomaly_detector_score`` signal
    until training completes. This unblocks orchestrator init for demos
    / tests where the ~30 s casedata read is unacceptable while keeping
    production behaviour unchanged (lazy_training defaults to False).
    """
    model = SeedModel()

    # Try loading existing
    if model_path and not force_retrain:
        mp = Path(model_path)
        if mp.exists():
            model.load(mp)
            return model

    if lazy_training:
        import threading

        def _train() -> None:
            try:
                from .dataset_loader import DatasetLoader

                loader = DatasetLoader(casedata_path)
                stats = model.train_from_casedata(loader, operation_ids)
                logger.info("Seed model trained (lazy): %s", stats)
                if model_path:
                    try:
                        model.save(model_path)
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.warning(
                            "Lazy seed model save failed (%s): %s",
                            model_path, exc,
                        )
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Lazy seed model training failed: %s", exc)

        thread = threading.Thread(
            target=_train, name="seed-model-lazy-train", daemon=True,
        )
        thread.start()
        # Expose the thread so callers / tests can join() if they need
        # to wait for completion deterministically.
        setattr(model, "_lazy_training_thread", thread)
        return model

    # Train from casedata (synchronous)
    from .dataset_loader import DatasetLoader

    loader = DatasetLoader(casedata_path)
    stats = model.train_from_casedata(loader, operation_ids)
    logger.info("Seed model trained: %s", stats)

    if model_path:
        model.save(model_path)

    return model


def create_online_detector(
    seed_model: Optional[SeedModel] = None,
    rl_agent: Optional[RLAgent] = None,
    rl_path: Optional[str | Path] = None,
    model_confidence_path: Optional[str | Path] = None,
) -> OnlineAnomalyDetector:
    """Create an OnlineAnomalyDetector with optional RL agent."""
    if rl_agent is None and rl_path:
        rp = Path(rl_path)
        if rp.exists():
            rl_agent = RLAgent()
            rl_agent.load(rp)

    if rl_agent is None:
        rl_agent = RLAgent()

    return OnlineAnomalyDetector(
        seed_model=seed_model,
        rl_agent=rl_agent,
        model_confidence_path=model_confidence_path,
    )

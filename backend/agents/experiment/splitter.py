"""Data splitting for the three-phase experiment."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Holds the three partitions of the feature dataset."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    eval_df: pd.DataFrame

    # Metadata
    train_ops: List[str] = field(default_factory=list)
    test_op: str = ""
    eval_op: str = ""
    n_train: int = 0
    n_test: int = 0
    n_eval: int = 0
    n_train_normal: int = 0
    n_train_pre_stoppage: int = 0
    n_test_normal: int = 0
    n_test_pre_stoppage: int = 0
    n_eval_normal: int = 0
    n_eval_pre_stoppage: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "train_ops": self.train_ops,
            "test_op": self.test_op,
            "eval_op": self.eval_op,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_eval": self.n_eval,
            "n_train_normal": self.n_train_normal,
            "n_train_pre_stoppage": self.n_train_pre_stoppage,
            "n_test_normal": self.n_test_normal,
            "n_test_pre_stoppage": self.n_test_pre_stoppage,
            "n_eval_normal": self.n_eval_normal,
            "n_eval_pre_stoppage": self.n_eval_pre_stoppage,
        }


def create_split(config: ExperimentConfig) -> DataSplit:
    """Load the features CSV and partition into train / test / eval.

    Validates that:
    - No operation overlap between splits
    - Both labels (`pre_stoppage`, `normal`) exist in test and eval
    - Training set is non-empty
    """
    csv_path = config.features_csv
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Features CSV not found: {csv_path}\n"
            f"Run  scripts/extract_pre_stoppage_patterns.py  first."
        )

    logger.info("Loading features from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info(
        "Loaded %d rows, operations: %s",
        len(df),
        sorted(df["operation_id"].unique()),
    )

    # Validate no overlap
    all_assigned = set(config.train_ops) | {config.test_op} | {config.eval_op}
    if len(all_assigned) != len(config.train_ops) + 2:
        raise ValueError(
            f"Operation overlap detected: train={config.train_ops}, "
            f"test={config.test_op}, eval={config.eval_op}"
        )

    # Split
    train_df = df[df["operation_id"].isin(config.train_ops)].copy()
    test_df = df[df["operation_id"] == config.test_op].copy()
    eval_df = df[df["operation_id"] == config.eval_op].copy()

    # Validate
    if len(train_df) == 0:
        raise ValueError(f"No training data for operations {config.train_ops}")
    if len(test_df) == 0:
        raise ValueError(f"No test data for operation {config.test_op}")
    if len(eval_df) == 0:
        raise ValueError(f"No eval data for operation {config.eval_op}")

    for name, subset in [("test", test_df), ("eval", eval_df)]:
        labels = set(subset["label"].unique())
        if "pre_stoppage" not in labels:
            logger.warning("%s set has no pre_stoppage samples", name)
        if "normal" not in labels:
            logger.warning("%s set has no normal samples", name)

    # Optional downsampling (for development iteration speed).
    # NOTE: assigning to ``locals()[name]`` does NOT rebind a function local in
    # CPython, so the previous loop silently discarded the downsample and ran
    # full-size. Reassign the real variables explicitly.
    if config.downsample_max > 0:
        rng = np.random.RandomState(config.random_seed)

        def _downsample(subset: pd.DataFrame) -> pd.DataFrame:
            if len(subset) <= config.downsample_max:
                return subset
            idx = rng.choice(len(subset), config.downsample_max, replace=False)
            return subset.iloc[sorted(idx)].reset_index(drop=True)

        train_df = _downsample(train_df)
        test_df = _downsample(test_df)
        eval_df = _downsample(eval_df)

    split = DataSplit(
        train_df=train_df.reset_index(drop=True),
        test_df=_sort_temporal(test_df).reset_index(drop=True),
        eval_df=_sort_temporal(eval_df).reset_index(drop=True),
        train_ops=list(config.train_ops),
        test_op=config.test_op,
        eval_op=config.eval_op,
        n_train=len(train_df),
        n_test=len(test_df),
        n_eval=len(eval_df),
        n_train_normal=int((train_df["label"] == "normal").sum()),
        n_train_pre_stoppage=int((train_df["label"] == "pre_stoppage").sum()),
        n_test_normal=int((test_df["label"] == "normal").sum()),
        n_test_pre_stoppage=int((test_df["label"] == "pre_stoppage").sum()),
        n_eval_normal=int((eval_df["label"] == "normal").sum()),
        n_eval_pre_stoppage=int((eval_df["label"] == "pre_stoppage").sum()),
    )

    logger.info(
        "Split: train=%d (%d normal, %d pre_stoppage), "
        "test=%d (%d normal, %d pre_stoppage), "
        "eval=%d (%d normal, %d pre_stoppage)",
        split.n_train, split.n_train_normal, split.n_train_pre_stoppage,
        split.n_test, split.n_test_normal, split.n_test_pre_stoppage,
        split.n_eval, split.n_eval_normal, split.n_eval_pre_stoppage,
    )
    return split


def _sort_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by event_timestamp if the column exists, preserving temporal order."""
    if "event_timestamp" in df.columns:
        return df.sort_values("event_timestamp")
    return df

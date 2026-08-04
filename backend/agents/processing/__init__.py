"""
Processing module - Real-time processing agents.

This module contains:
- compute: ComputeAgent for Fg/Fp force computation
- online: OnlineAgent for online learning/anomaly detection
- dataset_loader: DatasetLoader for reading real CNC sensor CSV data
- classical_models: SeedModel, RLAgent, OnlineAnomalyDetector
"""

from .compute import ComputeAgent
from .online import OnlineAgent

# [PROTOTYPE_CLASSICAL_RL_V1] - Lazy imports for classical models
# (depend on scikit-learn which may not be installed)
def _lazy_classical():
    from .classical_models import (
        SeedModel,
        RLAgent,
        OnlineAnomalyDetector,
        create_seed_model,
        create_online_detector,
    )
    return {
        "SeedModel": SeedModel,
        "RLAgent": RLAgent,
        "OnlineAnomalyDetector": OnlineAnomalyDetector,
        "create_seed_model": create_seed_model,
        "create_online_detector": create_online_detector,
    }

__all__ = [
    "ComputeAgent",
    "OnlineAgent",
]

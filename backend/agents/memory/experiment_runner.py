"""Live in-process experiment runner with PubSub progress streaming.

Instead of spawning a subprocess (which gives zero observability), this
module calls the same three-phase experiment pipeline in-process and
publishes structured progress events to the event bus so that connected
WebSocket clients get real-time updates.

Usage in the router::

    runner = LiveExperimentRunner(config_overrides=body, run_id="my-run")
    await runner.execute()  # publishes progress to bus
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .live_runner_base import LiveRunnerBase

logger = logging.getLogger(__name__)

# Project root (same as router.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# -------------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------------

class LiveExperimentRunner(LiveRunnerBase):
    """Run a three-phase stoppage experiment in-process with progress events.

    Parameters
    ----------
    config_overrides : dict
        Key-value pairs to merge into ExperimentConfig defaults.
    run_id : str
        Unique identifier for this run (used as PubSub channel suffix).
    sandbox_priors : bool
        If True (default), snapshot priors before the run and restore after.
    """

    def __init__(
        self,
        config_overrides: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        sandbox_priors: bool = True,
    ):
        resolved_run_id = run_id or f"stoppage_{time.strftime('%Y-%m-%d_%H%M')}"
        super().__init__(resolved_run_id)
        self.overrides = config_overrides or {}
        self.sandbox_priors = sandbox_priors

    # ------------------------------------------------------------------
    # Internal — synchronous (runs in executor thread)
    # ------------------------------------------------------------------

    def _run_sync(self) -> Dict[str, Any]:
        """Blocking three-phase pipeline.  Progress is emitted via _emit_sync."""
        # Lazy imports — these pull in pandas, sklearn, etc.
        from backend.agents.experiment.config import ExperimentConfig
        from backend.agents.experiment.evaluator import run_eval_phase, run_test_phase
        from backend.agents.experiment.metrics import compare_phases
        from backend.agents.experiment.metrics import compute_metrics
        from backend.agents.experiment.reporting import save_results
        from backend.agents.experiment.splitter import create_split
        from backend.agents.experiment.trainer import train_phase

        # 1. Build config ------------------------------------------------
        self._emit_sync("setup", "started", "Building experiment config …", pct=2)
        cfg = self._build_config(ExperimentConfig)

        # Check and report LLM/explanation status for API-mode runs.
        llm_status: Dict[str, Any] = {"generate_explanations": False}
        if getattr(cfg, "api_mode", False):
            try:
                from backend.agents.memory.orchestrator import get_orchestrator as _get_orch
                _orch = _get_orch()
                llm_status["generate_explanations"] = _orch.config.generate_explanations
                llm_status["llm_available"] = _orch.explainer.is_available()
                llm_status["llm_provider"] = getattr(_orch.explainer.config, "provider", "groq")
                llm_status["ollama_model"] = getattr(_orch.explainer.config, "model", "")
                if _orch.config.generate_explanations and not _orch.explainer.is_available():
                    _provider = llm_status["llm_provider"]
                    llm_status["llm_warning"] = (
                        f"LLM explanations are enabled but {_provider} is not reachable "
                        f"(model={_orch.explainer.config.model}). "
                        f"Explanations will use heuristic fallback text."
                    )
                elif not _orch.config.generate_explanations:
                    llm_status["llm_warning"] = (
                        "LLM explanations are disabled on the orchestrator. "
                        "Toggle the 🧠 LLM switch ON before running to get explanations."
                    )
            except Exception as exc:
                llm_status["llm_warning"] = f"Could not check LLM status: {exc}"

        self._emit_sync("setup", "completed", f"Config ready — run_dir: {cfg.run_dir}", pct=5,
                        detail={"config": self._safe_config_dict(cfg), "llm_status": llm_status})

        # 2. Sandbox priors -----------------------------------------------
        scorer = None
        if self.sandbox_priors:
            try:
                from backend.agents.memory.orchestrator import get_orchestrator
                orch = get_orchestrator()
                scorer = orch.scorer
                scorer.snapshot_priors(self.run_id)
                self._emit_sync("setup", "progress", "Priors sandboxed", pct=7)
            except Exception as exc:
                logger.warning("Prior sandbox failed (continuing without): %s", exc)

        try:
            return self._run_phases(cfg, train_phase, run_test_phase, run_eval_phase,
                                    compare_phases, save_results, create_split)
        finally:
            # Always restore priors even on failure
            if scorer is not None and scorer.is_sandboxed:
                scorer.restore_priors()

    def _run_phases(self, cfg, train_phase, run_test_phase, run_eval_phase,
                    compare_phases, save_results, create_split) -> Dict[str, Any]:
        """Execute the actual Phase 0-3 pipeline."""
        import pandas as pd  # noqa: F811
        from backend.agents.experiment.metrics import compute_metrics

        # Phase 0 — split -------------------------------------------------
        self._emit_sync("split", "started", "Splitting data …", pct=10)
        split = create_split(cfg)
        self._emit_sync("split", "completed",
                        f"Split: train={len(split.train_df)} test={len(split.test_df)} "
                        f"eval={len(split.eval_df)}",
                        pct=15,
                        detail=split.summary())

        # Phase 1 — train -------------------------------------------------
        self._emit_sync("train", "started", "Training seed model …", pct=18)
        train_result = train_phase(split.train_df, cfg)
        threshold = train_result.calibration.get("threshold", 0.5)
        self._emit_sync("train", "completed",
                        f"Threshold = {threshold:.3f}", pct=35,
                        detail={
                            "threshold": threshold,
                            "n_normal": train_result.n_normal_used,
                            "n_pre_stoppage": train_result.n_pre_stoppage_held,
                        })

        # Prepare normal features for online retraining
        from backend.agents.processing.classical_models import features_from_dict
        from backend.agents.processing.breakage_detector import BreakageFeatureExtractor
        normal_train_df = split.train_df[split.train_df["label"] == "normal"]
        train_normal_features = np.array([
            features_from_dict(BreakageFeatureExtractor.row_to_feature_dict(row))
            for _, row in normal_train_df.iterrows()
        ])

        # Phase 2 — test (no feedback) ------------------------------------
        self._emit_sync("test", "started",
                        f"Testing on {len(split.test_df)} rows (no feedback) …", pct=38)
        _test_score_buf: list = []
        def _test_cb(idx, total, score_snapshot=None):
            if score_snapshot is not None:
                _test_score_buf.append(score_snapshot)
                if len(_test_score_buf) >= 50:
                    self._emit_sync("scores", "progress", "",
                                    pct=38 + int((idx / max(total, 1)) * 22),
                                    detail={"samples": list(_test_score_buf), "phase": "test", "fold": 1})
                    _test_score_buf.clear()
            else:
                pct = 38 + int((idx / max(total, 1)) * 22)
                self._emit_sync("test", "progress",
                                f"Sample {idx}/{total}…", pct=pct)
        test_result = run_test_phase(
            split.test_df, cfg,
            threshold=threshold,
            train_normal_features=train_normal_features,
            progress_callback=_test_cb,
        )
        test_flagged = sum(1 for s in test_result.sample_results if s.predicted_positive)
        # Flush remaining test score buffer
        if _test_score_buf:
            self._emit_sync("scores", "progress", "", pct=60,
                            detail={"samples": list(_test_score_buf), "phase": "test", "fold": 1})
            _test_score_buf.clear()
        test_tp = sum(1 for s in test_result.sample_results if s.predicted_positive and s.label == 'pre_stoppage')
        test_fp = sum(1 for s in test_result.sample_results if s.predicted_positive and s.label == 'normal')
        test_metrics = compute_metrics(test_result)
        self._emit_sync("test", "completed",
                        f"Test done — flagged {test_flagged}/{test_result.n_samples}",
                        pct=60,
                        detail={
                            "n_samples": test_result.n_samples,
                            "n_flagged": test_flagged,
                            "tp": test_tp,
                            "fp": test_fp,
                            "f1": round(test_metrics.f1, 4),
                            "precision": round(test_metrics.precision, 4),
                            "recall": round(test_metrics.recall, 4),
                            "auc_roc": round(test_metrics.auc_roc, 4),
                        })

        # Phase 3 — eval (with feedback) ----------------------------------
        self._emit_sync("eval", "started",
                        f"Evaluating on {len(split.eval_df)} rows (with feedback) …", pct=63)

        _eval_score_buf: list = []

        def _eval_cb(idx, total, score_snapshot=None):
            if score_snapshot is not None:
                _eval_score_buf.append(score_snapshot)
                if len(_eval_score_buf) >= 50:
                    self._emit_sync("scores", "progress", "",
                                    pct=63 + int((idx / max(total, 1)) * 22),
                                    detail={"samples": list(_eval_score_buf), "phase": "eval", "fold": 1})
                    _eval_score_buf.clear()
            else:
                self._emit_sync("eval", "progress",
                                f"Sample {idx}/{total}\u2026",
                                pct=63 + int((idx / max(total, 1)) * 22))

        # Handle warm variant
        warm_priors = None
        if cfg.eval_variant == "warm":
            from backend.agents.experiment.evaluator import evaluate_phase
            self._emit_sync("eval", "progress", "Warm variant: running feedback on test set …", pct=65)
            warm_test = evaluate_phase(
                split.test_df, cfg,
                phase="warmup", feedback_enabled=True, threshold=threshold,
                train_normal_features=train_normal_features,
            )
            warm_path = cfg.run_dir / "warm_priors.json"
            final = warm_test.prior_history[-1] if warm_test.prior_history else {}
            warm_path.write_text(json.dumps({
                "pattern_priors": final,
                "feedback_counts": {},
            }, indent=2))
            warm_priors = warm_path

        eval_result = run_eval_phase(
            split.eval_df, cfg,
            threshold=threshold,
            warm_priors_path=warm_priors,
            train_normal_features=train_normal_features,
            progress_callback=_eval_cb,
        )
        eval_flagged = sum(1 for s in eval_result.sample_results if s.predicted_positive)
        # Flush remaining eval score buffer
        if _eval_score_buf:
            self._emit_sync("scores", "progress", "", pct=85,
                            detail={"samples": list(_eval_score_buf), "phase": "eval", "fold": 1})
            _eval_score_buf.clear()
        eval_tp = sum(1 for s in eval_result.sample_results if s.predicted_positive and s.label == 'pre_stoppage')
        eval_fp = sum(1 for s in eval_result.sample_results if s.predicted_positive and s.label == 'normal')
        eval_feedback = sum(1 for s in eval_result.sample_results if s.feedback_given)
        eval_explained = sum(1 for s in eval_result.sample_results if getattr(s, 'explanation', None))
        eval_llm = sum(1 for s in eval_result.sample_results if getattr(s, 'explanation_source', None) == 'llm')
        eval_fallback = sum(1 for s in eval_result.sample_results if getattr(s, 'explanation_source', None) == 'fallback')
        eval_metrics = compute_metrics(eval_result)
        # Thin-out prior history to ≤30 snapshots for WS transport
        prior_hist = eval_result.prior_history or []
        if len(prior_hist) > 30:
            step = max(1, len(prior_hist) // 30)
            prior_hist = prior_hist[::step] + [prior_hist[-1]]
        self._emit_sync("eval", "completed",
                        f"Eval done — flagged {eval_flagged}/{eval_result.n_samples}",
                        pct=85,
                        detail={
                            "n_samples": eval_result.n_samples,
                            "n_flagged": eval_flagged,
                            "tp": eval_tp,
                            "fp": eval_fp,
                            "n_feedback": eval_feedback,
                            "f1": round(eval_metrics.f1, 4),
                            "precision": round(eval_metrics.precision, 4),
                            "recall": round(eval_metrics.recall, 4),
                            "auc_roc": round(eval_metrics.auc_roc, 4),
                            "prior_history": prior_hist,
                            "n_explained": eval_explained,
                            "n_llm": eval_llm,
                            "n_fallback": eval_fallback,
                        })

        # Report ----------------------------------------------------------
        self._emit_sync("report", "started", "Generating report …", pct=88)
        comparison = compare_phases(test_result, eval_result)
        self._emit_sync("report", "progress", "Comparison ready", pct=90,
                        detail={
                            "delta_f1": round(comparison.delta_f1, 4),
                            "delta_precision": round(comparison.delta_precision, 4),
                            "delta_recall": round(comparison.delta_recall, 4),
                            "delta_auc_roc": round(comparison.delta_auc_roc, 4),
                            "pct_f1_improvement": round(comparison.pct_f1_improvement, 2),
                            "n_feedback_events": comparison.n_feedback_events,
                        })

        results_payload = {
            "config": {**self._safe_config_dict(cfg), "run_dir": str(cfg.run_dir)},
            "train": asdict(train_result) if hasattr(train_result, "__dataclass_fields__") else {},
            "test": self._phase_dict(test_result),
            "eval": self._phase_dict(eval_result),
            "comparison": comparison.to_dict() if hasattr(comparison, "to_dict") else self._phase_dict(comparison),
        }

        train_meta = asdict(train_result) if hasattr(train_result, "__dataclass_fields__") else {}
        save_results(cfg, comparison, test_result, eval_result, train_meta=train_meta)
        self._emit_sync("report", "completed", "Results saved", pct=95,
                        detail={"run_dir": str(cfg.run_dir)})

        results_payload["success"] = True
        return results_payload

    def _success_detail(self, result: Dict[str, Any]) -> Dict[str, Any]:
        disk_run_id = ""
        run_dir = result.get("config", {}).get("run_dir", "")
        if run_dir:
            try:
                exp_root = _PROJECT_ROOT / "data" / "breakage_patterns" / "stoppage_experiment"
                disk_run_id = str(Path(run_dir).relative_to(exp_root))
            except (ValueError, TypeError):
                disk_run_id = Path(run_dir).name if run_dir else ""
        return {"success": True, "disk_run_id": disk_run_id}

    # ------------------------------------------------------------------
    # Config builder
    # ------------------------------------------------------------------

    def _build_config(self, ConfigClass):
        """Merge user overrides into ExperimentConfig defaults."""
        import dataclasses as dc

        # Start with defaults
        kwargs: Dict[str, Any] = {}
        valid_fields = {f.name for f in dc.fields(ConfigClass)}

        for k, v in self.overrides.items():
            if k in valid_fields:
                kwargs[k] = v

        # Force output dir to include run_id for uniqueness
        cfg = ConfigClass(**kwargs)

        # Tag the output_dir with run_id so consecutive live runs
        # don't overwrite each other.  run_dir is a derived property
        # based on output_dir + ops/gap/window, so we place a
        # run_id-stamped subdirectory inside the default base.
        if "output_dir" not in self.overrides:
            cfg.output_dir = cfg.output_dir / self.run_id
        return cfg

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_config_dict(cfg) -> Dict[str, Any]:
        """Convert ExperimentConfig to a JSON-safe dict."""
        import dataclasses as dc
        d: Dict[str, Any] = {}
        for f in dc.fields(cfg):
            v = getattr(cfg, f.name)
            if isinstance(v, Path):
                d[f.name] = str(v)
            elif isinstance(v, (list, tuple)):
                d[f.name] = list(v)
            else:
                d[f.name] = v
        return d

    @staticmethod
    def _phase_dict(result) -> Dict[str, Any]:
        """Convert a PhaseResult to a serialisable dict."""
        try:
            import dataclasses as dc
            d = dc.asdict(result)
            # numpy arrays aren't JSON serialisable
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    d[k] = v.tolist()
            return d
        except Exception:
            return {"error": "Could not serialise phase result"}

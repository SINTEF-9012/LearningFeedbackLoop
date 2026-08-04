"""Live in-process breakage experiment runner with PubSub progress streaming.

Uses the **same 3-phase methodology** as the stoppage experiment runner:
  Phase 1 (train) — SeedModel on normal data, threshold calibration
  Phase 2 (test)  — baseline scoring with NO feedback
  Phase 3 (eval)  — scoring WITH feedback loop (prior updates, online retraining)

The breakage experiment wraps this in a LOOCV loop: for each fold the
held-out operation is used for *both* test and eval phases, and aggregate
metrics + per-fold comparisons are reported.

Usage in the router::

    runner = LiveBreakageExperimentRunner(
        dataset="site_a_line2", run_id="my-run",
    )
    await runner.execute()  # publishes progress to bus
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.agents.processing.tool_lookup import FAMILY_MACHINE_A1, resolve_tool_context

from .live_runner_base import LiveRunnerBase

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FEATURES_ROOT = _PROJECT_ROOT / "data" / "breakage_patterns"
_EXPERIMENT_ROOT = _PROJECT_ROOT / "data" / "experiment_snapshots"
_SPLITS_ROOT = _FEATURES_ROOT / "splits"
_DEFAULT_SITE_A_LINE2_SPLIT = "PART0001_excel"


def _round_optional(value: Any, digits: int = 4) -> Optional[float]:
    """Round finite numeric values and preserve None for absent fields."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return round(numeric, digits)


def _serialize_breakage_sample(sample: Any) -> Dict[str, Any]:
    """Serialise breakage experiment samples into the dashboard contract."""
    return {
        "sample_id": sample.sample_id,
        "label": sample.label,
        "operation_id": sample.operation_id,
        "tool_number": sample.tool_number,
        "memory_id": sample.memory_id,
        "significance_score": round(sample.significance_score, 4),
        "action": sample.action,
        "predicted_positive": sample.predicted_positive,
        "raw_model_score": round(sample.raw_model_score, 4),
        "pattern_rule_score": round(sample.pattern_rule_score, 4),
        "anomaly_z_score": round(sample.anomaly_z_score, 4),
        "prior_boost": round(sample.prior_boost, 4),
        "multi_rule_bonus": round(sample.multi_rule_bonus, 4),
        "n_rules_triggered": sample.n_rules_triggered,
        "detected_patterns": list(sample.detected_patterns or []),
        "supervised_score": round(sample.supervised_score, 4),
        "unsupervised_score": round(sample.unsupervised_score, 4),
        "combined_score": round(sample.combined_score, 4),
        "tool_prior": round(sample.tool_prior, 4),
        "tool_multiplier": round(sample.tool_multiplier, 4),
        "weight_supervised": round(sample.weight_supervised, 4),
        "weight_unsupervised": round(sample.weight_unsupervised, 4),
        "score_trace": list(sample.score_trace or []),
        "model_breakdown": sample.model_breakdown or {},
        "feedback_given": sample.feedback_given,
        "feedback_action": sample.feedback_action,
        "feedback_source": sample.feedback_source,
        "counterfactual_score": _round_optional(sample.counterfactual_score),
        "prediction_flipped": sample.prediction_flipped,
        "prior_snapshot": {
            key: round(float(val), 4) for key, val in (sample.prior_snapshot or {}).items()
        },
        "explanation": sample.explanation,
        "explanation_source": sample.explanation_source,
        "alert_line": sample.alert_line,
        "alert_line_source": sample.alert_line_source,
        "stored_in_memory": sample.stored_in_memory,
        "co_occurring_pairs": [list(pair) for pair in (sample.co_occurring_pairs or [])],
        "propagated_prior_deltas": {
            key: round(float(val), 4) for key, val in (sample.propagated_prior_deltas or {}).items()
        },
        "sindit_context": sample.sindit_context or None,
    }


def _ordered_unique_strings(values: List[Any]) -> List[str]:
    seen: Dict[str, None] = {}
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen[text] = None
    return list(seen.keys())


def _collect_pattern_keys_used(all_fold_results: List[Dict[str, Any]]) -> List[str]:
    pattern_keys: List[Any] = []
    for fold in all_fold_results:
        for phase_key in ("test", "eval"):
            phase = fold.get(phase_key, {})
            pattern_keys.extend(phase.get("discovered_pattern_keys") or [])
            for sample in phase.get("sample_results") or []:
                if isinstance(sample, dict):
                    pattern_keys.extend(sample.get("detected_patterns") or [])
    return _ordered_unique_strings(pattern_keys)


def _build_execution_summary(
    *,
    config: Dict[str, Any],
    sandbox_priors: bool,
    data_source: _DatasetSource,
    n_folds: int,
    n_feedback_events: int,
    pattern_keys_used: List[str],
) -> Dict[str, Any]:
    return {
        "mode": "api" if bool(config.get("api_mode")) else "local",
        "api_mode": bool(config.get("api_mode")),
        "api_mode_strict": bool(config.get("api_mode_strict")),
        "server_pattern_derivation": bool(config.get("api_use_server_patterns")),
        "feedback_scope_user_id": config.get("feedback_user_id"),
        "sandbox_priors": bool(sandbox_priors),
        "persist_shared_priors": bool(config.get("persist_shared_priors")),
        "data_source": data_source.kind,
        "split_name": data_source.split_name,
        "n_folds": int(n_folds),
        "n_feedback_events": int(n_feedback_events),
        "n_patterns_used": len(pattern_keys_used),
        "pattern_keys_used": list(pattern_keys_used),
    }


@dataclass(frozen=True)
class _DatasetSource:
    kind: str
    source_path: str
    dataset_name: str
    split_name: Optional[str] = None
    summary_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "kind": self.kind,
            "source_path": self.source_path,
            "dataset_name": self.dataset_name,
        }
        if self.split_name:
            payload["split_name"] = self.split_name
        if self.summary_path:
            payload["summary_path"] = self.summary_path
        return payload


class LiveBreakageExperimentRunner(LiveRunnerBase):
    """Run a breakage detection experiment in-process with progress events.

    Parameters
    ----------
    dataset : str
        ``"site_a_line2"`` or ``"casedata"``
    run_id : str
        Unique run identifier (used as PubSub channel suffix).
    sandbox_priors : bool
        Snapshot priors before the run and restore after.
    csv_path : str | None
        Explicit CSV path. If None, auto-resolves from dataset.
    label_scheme : str
        ``"original"`` — use site_a_line2_features.csv with pre_break/normal labels.
        ``"conservative"`` — use site_a_line2_features_conservative.csv, label_binary column.
        ``"conservative_3class"`` — use conservative CSV, label_3class column.
        ``"conservative_5class"`` — use conservative CSV, native 5-class label column.
        ``"v2"`` — tool-segmented (OF, tool) labels, site_a_line2_breakage_v2.csv
        (leakage-controlled; see docs/SITE_A_LINE2_DATASET_V2_2026-06-30.md).
        ``"v3"`` — sub-pass inspection-aligned refinement of v2,
        site_a_line2_breakage_v3.csv. Both bypass the split bundle.
    """

    # Supported label scheme → (csv_suffix, label_column_remap)
    _LABEL_SCHEMES: Dict[str, tuple] = {
        "original": ("site_a_line2_features.csv", None),
        "conservative": ("site_a_line2_features_conservative.csv", "label_binary"),
        "conservative_3class": ("site_a_line2_features_conservative.csv", "label_3class"),
        "conservative_5class": ("site_a_line2_features_conservative.csv", "label"),
        "v2": ("site_a_line2_breakage_v2.csv", None),
        "v3": ("site_a_line2_breakage_v3.csv", None),
    }

    # Schemes whose CSV is the dataset of record — the Excel-anchored split
    # bundle must not silently take precedence over them.
    _CSV_ONLY_SCHEMES = {"v2", "v3"}

    def __init__(
        self,
        dataset: str = "site_a_line2",
        run_id: Optional[str] = None,
        sandbox_priors: bool = True,
        csv_path: Optional[str] = None,
        label_scheme: str = "original",
        split_name: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        resolved_run_id = run_id or f"breakage_live_{int(time.time())}"
        super().__init__(resolved_run_id)
        self.dataset = dataset
        self.sandbox_priors = sandbox_priors
        self.csv_path = csv_path
        self.label_scheme = label_scheme
        self.split_name = split_name
        self.config_overrides = dict(config_overrides or {})
        self.config_overrides.setdefault("api_mode", True)
        self.config_overrides.setdefault("api_mode_strict", True)
        self.config_overrides.setdefault("experiment_fast_path", False)
        self.config_overrides.setdefault("api_use_server_patterns", True)
        self.config_overrides.setdefault("api_batch_size", 1)
        self.config_overrides.setdefault("persist_shared_priors", False)
        self.config_overrides.setdefault("feedback_user_id", f"experiment:{self.run_id}")

    # ------------------------------------------------------------------
    # Internal — synchronous (runs in executor thread)
    # ------------------------------------------------------------------

    def _run_sync(self) -> Dict[str, Any]:
        """Blocking breakage pipeline using the unified 3-phase methodology."""

        # ── Setup ──────────────────────────────────────────────────────
        self._emit_sync("setup", "started", "Loading breakage dataset …", pct=2)

        df, data_source = self._load_dataset_frame()
        csv_path = data_source.source_path

        # Drop non-numeric metadata columns that the stoppage trainer doesn't
        # know about, but keep the tool-context fields needed for tool priors
        # and API-mode cutting-context payloads.
        _keep_meta = {
            "operation_id",
            "label",
            "tool_number",
            "timestamp",
            "session",
            "condition",
            "machine_id",
            "machine_family",
            "tool_id",
            "tool_type",
            "tool_material",
            "sindit_tool_iri",
        }
        _drop = [
            c for c in df.columns
            if c not in _keep_meta and df[c].dtype == "object"
        ]
        if _drop:
            logger.info("Dropping non-numeric columns: %s", _drop)
            df = df.drop(columns=_drop)

        ops = sorted(df["operation_id"].unique().tolist())
        n_positive = int((df["label"] == "pre_stoppage").sum())
        n_normal = int((df["label"] == "normal").sum())

        self._emit_sync("setup", "completed",
                        f"Loaded {len(df)} samples, {n_positive} pre-break, "
                        f"{n_normal} normal, {len(ops)} operation(s)",
                        pct=5,
                        detail={
                            "n_samples": len(df),
                            "n_pre_break": n_positive,
                            "n_normal": n_normal,
                            "operations": ops,
                            "dataset": self.dataset,
                            "data_source": data_source.to_dict(),
                        })

        # ── Split — determine LOOCV folds ─────────────────────────────
        self._emit_sync("split", "started", "Determining LOOCV folds …", pct=8)

        if len(ops) == 1:
            folds = [(ops[0], ops)]
            self._emit_sync("split", "completed",
                            "Single operation — self-training mode",
                            pct=10,
                            detail={"n_folds": 1, "mode": "self-training"})
        else:
            folds = [(op, [o for o in ops if o != op]) for op in ops]
            self._emit_sync("split", "completed",
                            f"Leave-one-out CV: {len(folds)} folds",
                            pct=10,
                            detail={"n_folds": len(folds), "mode": "LOOCV"})

        # ── Sandbox priors ────────────────────────────────────────────
        scorer = None
        if self.sandbox_priors:
            try:
                from backend.agents.memory.orchestrator import get_orchestrator
                orch = get_orchestrator()
                scorer = orch.scorer
                scorer.snapshot_priors(self.run_id)
                self._emit_sync("setup", "progress", "Priors sandboxed", pct=12)
            except Exception as exc:
                logger.warning("Prior sandbox failed (continuing without): %s", exc)

        try:
            return self._run_folds(folds, df, csv_path, data_source)
        finally:
            if scorer is not None and scorer.is_sandboxed:
                scorer.restore_priors()

    def _success_message(self) -> str:
        return "Breakage experiment finished"

    def _success_detail(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True, "disk_run_id": result.get("disk_run_id", "")}

    def _load_dataset_frame(self) -> Tuple[pd.DataFrame, _DatasetSource]:
        """Load the requested dataset into the canonical breakage experiment schema."""
        split_dir = self._resolve_split_dir()
        if split_dir is not None:
            return self._load_split_dataset(split_dir)

        csv_path = self._resolve_csv()
        df = pd.read_csv(csv_path)

        scheme_info = self._LABEL_SCHEMES.get(self.label_scheme)
        if scheme_info and scheme_info[1] and scheme_info[1] in df.columns:
            src_col = scheme_info[1]
            logger.info("Label scheme '%s': using column '%s' as label", self.label_scheme, src_col)
            df["label"] = df[src_col]

        if self.label_scheme == "conservative_3class":
            df["label"] = df["label"].replace({
                "anomalous": "pre_stoppage",
                "suspect": "normal",
            })
        else:
            df["label"] = df["label"].replace({"pre_break": "pre_stoppage"})

        return df, _DatasetSource(
            kind="csv",
            source_path=str(csv_path),
            dataset_name=self.dataset,
        )

    def _resolve_split_dir(self) -> Optional[Path]:
        """Return the default split directory for Site_a_line2 when available."""
        if self.csv_path or self.dataset != "site_a_line2":
            return None
        if self.label_scheme in self._CSV_ONLY_SCHEMES:
            return None

        split_name = self.split_name or _DEFAULT_SITE_A_LINE2_SPLIT
        split_dir = _SPLITS_ROOT / split_name
        combined_csv = split_dir / "site_a_line2_PART0001_labeled.csv"
        if combined_csv.is_file():
            return split_dir
        return None

    def _load_split_dataset(self, split_dir: Path) -> Tuple[pd.DataFrame, _DatasetSource]:
        """Load the Excel-anchored Site_a_line2 split bundle and normalise its schema."""
        combined_csv = split_dir / "site_a_line2_PART0001_labeled.csv"
        summary_path = split_dir / "overall_summary.json"
        df = pd.read_csv(combined_csv)

        if "embargoed" in df.columns:
            embargoed = df["embargoed"].astype(str).str.lower().eq("true")
            df = df.loc[~embargoed].copy()

        label_map = {
            "pre_break": "pre_stoppage",
        }
        if self.label_scheme in {"conservative", "conservative_3class", "conservative_5class"}:
            label_map["tool_wear"] = "pre_stoppage"
        else:
            label_map["tool_wear"] = "normal"

        df["label"] = df["label"].replace(label_map)
        df = df[df["label"].isin(["normal", "pre_stoppage"])].copy()

        df["timestamp"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["operation_id"] = df["uf5_of"].map(lambda value: f"OF{int(value)}" if pd.notna(value) else "")

        context_df = df.apply(
            lambda row: pd.Series(
                resolve_tool_context(
                    FAMILY_MACHINE_A1,
                    row.get("Cnc_Tool_Number_RT"),
                    dataset_id="site_a_line2",
                    machine_id=row.get("session"),
                    raw_teeth=row.get("CNC_parameters_teeth_num"),
                )
            ),
            axis=1,
        )
        for column in context_df.columns:
            df[column] = context_df[column]
        if "tool_number" in df.columns:
            df["tool_number"] = pd.to_numeric(df["tool_number"], errors="coerce").astype("Int64")
        if "num_teeth" in df.columns:
            df["num_teeth"] = pd.to_numeric(df["num_teeth"], errors="coerce").astype("Int64")

        # Map the split's windowed raw metrics into the feature names used by
        # the existing breakage experiment pipeline. Missing richer features
        # remain at 0.0 instead of silently reusing the old session-level CSV.
        df["power_spindle_mean"] = pd.to_numeric(df.get("Spdl_actual_power"), errors="coerce").fillna(0.0)
        df["power_spindle_max"] = df["power_spindle_mean"]
        df["power_spindle_std"] = 0.0
        df["power_y_mean"] = 0.0
        df["power_y_max"] = 0.0
        df["power_z_mean"] = 0.0
        df["vib_severity_x_mean"] = pd.to_numeric(df.get("Accel_Severity_Acc_1_range1_Severity"), errors="coerce").fillna(0.0)
        df["vib_severity_x_max"] = df["vib_severity_x_mean"]
        df["vib_severity_y_mean"] = pd.to_numeric(df.get("Accel_Severity_Acc_2_range2_Severity"), errors="coerce").fillna(0.0)
        df["vib_severity_y_max"] = df["vib_severity_y_mean"]
        df["chatter_amp_x_mean"] = df["vib_severity_x_mean"]
        df["chatter_amp_y_mean"] = df["vib_severity_y_mean"]
        df["power_active_mean"] = df["power_spindle_mean"]
        df["power_active_std"] = 0.0
        df["power_factor_mean"] = 0.0
        df["spindle_actual_mean"] = pd.to_numeric(df.get("SpindleSpeedActual"), errors="coerce").fillna(0.0)
        df["feed_actual_mean"] = pd.to_numeric(df.get("Axis_FeedRate_actual"), errors="coerce").fillna(0.0)
        df["temperature_head_mean"] = pd.to_numeric(df.get("Spindle_Temperature_degreeCelsius_d1"), errors="coerce").fillna(0.0)

        logger.info(
            "Using split-backed breakage dataset '%s': %d rows across %d OFs",
            split_dir.name,
            len(df),
            df["operation_id"].nunique(),
        )
        return df, _DatasetSource(
            kind="split",
            source_path=str(combined_csv),
            dataset_name=self.dataset,
            split_name=split_dir.name,
            summary_path=str(summary_path) if summary_path.is_file() else None,
        )

    # ------------------------------------------------------------------

    def _resolve_csv(self) -> str:
        """Resolve the features CSV path."""
        if self.csv_path:
            csv_path = self.csv_path
        elif self.dataset == "site_a_line2":
            # Pick file based on label scheme
            scheme_info = self._LABEL_SCHEMES.get(self.label_scheme)
            if scheme_info:
                csv_path = str(_FEATURES_ROOT / scheme_info[0])
            else:
                csv_path = str(_FEATURES_ROOT / "site_a_line2_features.csv")
        else:
            csv_path = str(_FEATURES_ROOT / "breakage_features.csv")

        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"Features CSV not found: {csv_path}. "
                f"Run Feature Extraction first."
            )
        return csv_path

    # ------------------------------------------------------------------

    def _run_folds(
        self,
        folds: List[tuple],
        df: pd.DataFrame,
        csv_path: str,
        data_source: _DatasetSource,
    ) -> Dict[str, Any]:
        """Execute LOOCV with train → test → eval per fold."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from backend.agents.experiment.config import ExperimentConfig

        n_folds = len(folds)
        all_comparisons: List[Dict[str, Any]] = []
        all_fold_results: List[Dict[str, Any]] = []

        # Create a top-level run directory
        run_dir = _EXPERIMENT_ROOT / f"breakage_loocv_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build a reference config to read parallel_folds setting
        _ref_cfg = ExperimentConfig(features_csv=Path(csv_path))
        for k, v in self.config_overrides.items():
            if hasattr(_ref_cfg, k):
                try:
                    setattr(_ref_cfg, k, type(getattr(_ref_cfg, k))(v))
                except (ValueError, TypeError):
                    setattr(_ref_cfg, k, v)
        max_workers = max(1, getattr(_ref_cfg, "parallel_folds", 1))

        if max_workers > 1 and n_folds > 1:
            # Parallel fold execution
            logger.info("Running %d folds in parallel (max_workers=%d)", n_folds, max_workers)
            with ThreadPoolExecutor(max_workers=min(max_workers, n_folds)) as pool:
                futures = {
                    pool.submit(
                        self._run_single_fold,
                        fi, test_op, train_ops, df, csv_path, run_dir, n_folds,
                    ): fi
                    for fi, (test_op, train_ops) in enumerate(folds)
                }
                for future in as_completed(futures):
                    fi = futures[future]
                    try:
                        fold_result, comparison = future.result()
                        all_fold_results.append(fold_result)
                        all_comparisons.append(comparison)
                    except Exception as exc:
                        logger.error("Fold %d failed: %s", fi + 1, exc, exc_info=True)
                        self._emit_sync("fold", "error",
                                        f"Fold {fi + 1} failed: {exc}", pct=0)
            # Sort results by fold index so output order is deterministic
            all_fold_results.sort(key=lambda f: f["fold"])
            all_comparisons = [
                c for _, c in sorted(
                    zip([f["fold"] for f in all_fold_results], all_comparisons),
                    key=lambda x: x[0],
                )
            ]
        else:
            # Sequential fold execution (original behavior)
            for fi, (test_op, train_ops) in enumerate(folds):
                fold_result, comparison = self._run_single_fold(
                    fi, test_op, train_ops, df, csv_path, run_dir, n_folds,
                )
                all_fold_results.append(fold_result)
                all_comparisons.append(comparison)

        # ── Report — aggregate across folds ──────────────────────────
        self._emit_sync("report", "started", "Aggregating results …", pct=88)

        # Aggregate test vs eval metrics
        test_f1s = [f["test"]["f1"] for f in all_fold_results]
        eval_f1s = [f["eval"]["f1"] for f in all_fold_results]
        test_prec = [f["test"]["precision"] for f in all_fold_results]
        eval_prec = [f["eval"]["precision"] for f in all_fold_results]
        test_rec = [f["test"]["recall"] for f in all_fold_results]
        eval_rec = [f["eval"]["recall"] for f in all_fold_results]

        agg = {
            "test_f1_mean": round(float(np.mean(test_f1s)), 4),
            "eval_f1_mean": round(float(np.mean(eval_f1s)), 4),
            "test_precision_mean": round(float(np.mean(test_prec)), 4),
            "eval_precision_mean": round(float(np.mean(eval_prec)), 4),
            "test_recall_mean": round(float(np.mean(test_rec)), 4),
            "eval_recall_mean": round(float(np.mean(eval_rec)), 4),
            "delta_f1_mean": round(float(np.mean(eval_f1s)) - float(np.mean(test_f1s)), 4),
            "pct_f1_improvement": round(
                ((float(np.mean(eval_f1s)) - float(np.mean(test_f1s))) /
                 max(float(np.mean(test_f1s)), 1e-9)) * 100, 2
            ),
            "n_folds": n_folds,
        }
        n_feedback_events = sum(f["eval"].get("n_feedback", 0) for f in all_fold_results)
        pattern_keys_used = _collect_pattern_keys_used(all_fold_results)
        execution_summary = _build_execution_summary(
            config=self.config_overrides,
            sandbox_priors=self.sandbox_priors,
            data_source=data_source,
            n_folds=n_folds,
            n_feedback_events=n_feedback_events,
            pattern_keys_used=pattern_keys_used,
        )

        # Emit report progress with comparison data
        self._emit_sync("report", "progress", "Comparison ready", pct=90,
                        detail={
                            "delta_f1": agg["delta_f1_mean"],
                            "pct_f1_improvement": agg["pct_f1_improvement"],
                            "n_feedback_events": n_feedback_events,
                        })

        # Save results to disk
        results_payload: Dict[str, Any] = {
            "experiment": "breakage_detection_lfl",
            "experiment_type": "breakage",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "methodology": "3-phase (train/test-baseline/eval-feedback) × LOOCV",
            "dataset": {
                "csv_path": str(csv_path),
                "dataset_name": self.dataset,
                "data_source": data_source.kind,
                "split_name": data_source.split_name,
                "summary_path": data_source.summary_path,
            },
            "folds": all_fold_results,
            "aggregate": agg,
            "config": {
                "dataset": self.dataset,
                "csv_path": str(csv_path),
                "data_source": data_source.kind,
                "split_name": data_source.split_name,
                "run_dir": str(run_dir),
            },
            "summary": execution_summary,
        }

        (run_dir / "experiment_results.json").write_text(
            json.dumps(results_payload, indent=2, default=str)
        )

        try:
            disk_run_id = str(run_dir.relative_to(_EXPERIMENT_ROOT))
        except ValueError:
            disk_run_id = run_dir.name

        self._emit_sync("report", "completed",
                        f"Done — test F1={agg['test_f1_mean']:.3f} → "
                        f"eval F1={agg['eval_f1_mean']:.3f} "
                        f"(Δ{agg['delta_f1_mean']:+.3f})",
                        pct=95,
                        detail={
                            "f1": agg["eval_f1_mean"],
                            "precision": agg["eval_precision_mean"],
                            "recall": agg["eval_recall_mean"],
                            "test_f1": agg["test_f1_mean"],
                            "delta_f1": agg["delta_f1_mean"],
                            "pct_f1_improvement": agg["pct_f1_improvement"],
                            "summary": execution_summary,
                            "run_dir": str(run_dir),
                        })

        results_payload["success"] = True
        results_payload["disk_run_id"] = disk_run_id
        return results_payload

    # ------------------------------------------------------------------
    # Single-fold execution (extracted for parallel/sequential reuse)
    # ------------------------------------------------------------------

    def _run_single_fold(
        self,
        fi: int,
        test_op: str,
        train_ops: List[str],
        df: pd.DataFrame,
        csv_path: str,
        run_dir: Path,
        n_folds: int,
    ) -> tuple:
        """Run train → test → eval for one LOOCV fold.

        Returns ``(fold_result_dict, comparison_dict)``.
        """
        from backend.agents.experiment.config import ExperimentConfig
        from backend.agents.experiment.evaluator import evaluate_phase, _summarize_feedback_events
        from backend.agents.experiment.metrics import compare_phases, compute_metrics
        from backend.agents.experiment.trainer import train_phase
        from backend.agents.processing.classical_models import batch_features_from_df
        from backend.agents.processing.breakage_detector import _COL_MAP

        fold_pct_base = 15 + (fi / n_folds) * 70
        fold_pct_span = 70 / n_folds

        # ── (a) Train phase ──────────────────────────────────────
        pct_train = fold_pct_base
        self._emit_sync("train", "started",
                        f"Fold {fi+1}/{n_folds}: training on {len(train_ops)} ops …",
                        pct=pct_train)

        train_df = df[df["operation_id"].isin(train_ops)].copy()
        test_df = df[df["operation_id"] == test_op].copy()

        if test_df["label"].nunique() < 2:
            labels_present = test_df["label"].unique().tolist()
            logger.warning(
                "Fold %d: test op %s has only labels %s — metrics may be degenerate",
                fi + 1, test_op, labels_present,
            )

        fold_dir = run_dir / f"fold_{fi+1}_{test_op}"
        cfg = ExperimentConfig(
            features_csv=Path(csv_path),
            output_dir=fold_dir,
            train_ops=train_ops,
            test_op=test_op,
            eval_op=test_op,
        )
        for k, v in self.config_overrides.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, type(getattr(cfg, k))(v))
                except (ValueError, TypeError):
                    setattr(cfg, k, v)
        cfg.ensure_dirs()

        train_result = train_phase(train_df, cfg)
        threshold = train_result.calibration.get("threshold", 0.5)

        normal_train_df = train_df[train_df["label"] == "normal"]
        train_normal_features = batch_features_from_df(normal_train_df, col_map=_COL_MAP)

        n_train_normal = len(normal_train_df)
        n_train_pos = len(train_df[train_df["label"] == "pre_stoppage"])

        self._emit_sync("train", "completed",
                        f"Fold {fi+1}: threshold={threshold:.3f}, "
                        f"{n_train_normal} normal, {n_train_pos} positive",
                        pct=pct_train + fold_pct_span * 0.2,
                        detail={
                            "fold": fi + 1, "test_op": test_op,
                            "threshold": round(threshold, 4),
                            "n_train_normal": n_train_normal,
                            "n_train_positive": n_train_pos,
                        })

        # ── (b) Test phase — baseline, NO feedback ───────────────
        pct_test = fold_pct_base + fold_pct_span * 0.25
        self._emit_sync("test", "started",
                        f"Fold {fi+1}: testing {test_op} (no feedback) …",
                        pct=pct_test)

        _test_score_buf: List[Dict[str, Any]] = []

        def _test_progress(idx: int, total: int, score_snapshot=None) -> None:
            if score_snapshot is not None:
                _test_score_buf.append(score_snapshot)
                if len(_test_score_buf) >= 50:
                    self._emit_sync("scores", "progress", "",
                                    pct=pct_test + fold_pct_span * 0.2 * (idx / max(total, 1)),
                                    detail={"samples": list(_test_score_buf), "phase": "test", "fold": fi + 1})
                    _test_score_buf.clear()
            else:
                frac = idx / max(total, 1)
                self._emit_sync("test", "progress",
                                f"Fold {fi+1} test: {idx}/{total} samples",
                                pct=pct_test + fold_pct_span * 0.2 * frac)

        test_result = evaluate_phase(
            test_df, cfg,
            phase="test",
            feedback_enabled=False,
            threshold=threshold,
            train_normal_features=train_normal_features,
            progress_callback=_test_progress,
        )
        test_metrics = compute_metrics(test_result)
        test_flagged = sum(1 for s in test_result.sample_results if s.predicted_positive)
        # Flush remaining test score buffer
        if _test_score_buf:
            self._emit_sync("scores", "progress", "",
                            pct=pct_test + fold_pct_span * 0.2,
                            detail={"samples": list(_test_score_buf), "phase": "test", "fold": fi + 1})
            _test_score_buf.clear()

        self._emit_sync("test", "completed",
                        f"Fold {fi+1} test: F1={test_metrics.f1:.3f} "
                        f"({test_flagged}/{test_result.n_samples} flagged)",
                        pct=pct_test + fold_pct_span * 0.2,
                        detail={
                            "fold": fi + 1, "test_op": test_op,
                            "n_samples": test_result.n_samples,
                            "n_flagged": test_flagged,
                            "f1": round(test_metrics.f1, 4),
                            "precision": round(test_metrics.precision, 4),
                            "recall": round(test_metrics.recall, 4),
                            "auc_roc": round(test_metrics.auc_roc, 4),
                        })

        # ── (c) Eval phase — WITH feedback loop ──────────────────
        pct_eval = fold_pct_base + fold_pct_span * 0.5
        self._emit_sync("eval", "started",
                        f"Fold {fi+1}: evaluating {test_op} (with feedback) …",
                        pct=pct_eval)

        _eval_score_buf: List[Dict[str, Any]] = []

        def _eval_progress(idx: int, total: int, score_snapshot=None) -> None:
            if score_snapshot is not None:
                _eval_score_buf.append(score_snapshot)
                if len(_eval_score_buf) >= 50:
                    self._emit_sync("scores", "progress", "",
                                    pct=pct_eval + fold_pct_span * 0.3 * (idx / max(total, 1)),
                                    detail={"samples": list(_eval_score_buf), "phase": "eval", "fold": fi + 1})
                    _eval_score_buf.clear()
            else:
                frac = idx / max(total, 1)
                self._emit_sync("eval", "progress",
                                f"Fold {fi+1} eval: {idx}/{total} samples",
                                pct=pct_eval + fold_pct_span * 0.3 * frac)

        eval_result = evaluate_phase(
            test_df.copy(), cfg,
            phase="eval",
            feedback_enabled=True,
            threshold=threshold,
            train_normal_features=train_normal_features,
            progress_callback=_eval_progress,
        )
        eval_metrics = compute_metrics(eval_result)
        eval_flagged = sum(1 for s in eval_result.sample_results if s.predicted_positive)
        eval_feedback = sum(1 for s in eval_result.sample_results if s.feedback_given)
        # Flush remaining eval score buffer
        if _eval_score_buf:
            self._emit_sync("scores", "progress", "",
                            pct=pct_eval + fold_pct_span * 0.3,
                            detail={"samples": list(_eval_score_buf), "phase": "eval", "fold": fi + 1})
            _eval_score_buf.clear()

        prior_hist = eval_result.prior_history or []
        if len(prior_hist) > 30:
            step = max(1, len(prior_hist) // 30)
            prior_hist = prior_hist[::step] + [prior_hist[-1]]

        self._emit_sync("eval", "completed",
                        f"Fold {fi+1} eval: F1={eval_metrics.f1:.3f} "
                        f"({eval_flagged}/{eval_result.n_samples} flagged, "
                        f"{eval_feedback} feedback events)",
                        pct=pct_eval + fold_pct_span * 0.3,
                        detail={
                            "fold": fi + 1, "test_op": test_op,
                            "n_samples": eval_result.n_samples,
                            "n_flagged": eval_flagged,
                            "n_feedback": eval_feedback,
                            "f1": round(eval_metrics.f1, 4),
                            "precision": round(eval_metrics.precision, 4),
                            "recall": round(eval_metrics.recall, 4),
                            "auc_roc": round(eval_metrics.auc_roc, 4),
                            "prior_history": prior_hist,
                            "n_predictions_flipped": eval_result.n_predictions_flipped,
                            "n_model_retrains": eval_result.n_model_retrains,
                        })

        # ── Per-fold comparison ───────────────────────────────────
        comparison = compare_phases(test_result, eval_result)

        def _serialize_samples(result) -> List[Dict[str, Any]]:
            return [_serialize_breakage_sample(sr) for sr in result.sample_results]

        fold_result = {
            "fold": fi + 1,
            "test_operation": test_op,
            "train_operations": train_ops,
            "test": {
                "f1": round(test_metrics.f1, 4),
                "precision": round(test_metrics.precision, 4),
                "recall": round(test_metrics.recall, 4),
                "auc_roc": round(test_metrics.auc_roc, 4),
                "n_flagged": test_flagged,
                "n_samples": test_result.n_samples,
                "n_model_retrains": test_result.n_model_retrains,
                "feedback_events": list(test_result.feedback_events or []),
                "pattern_feedback_summary": _summarize_feedback_events(test_result.feedback_events or []),
                "all_propagated_deltas": list(test_result.all_propagated_deltas or []),
                "n_propagation_events": len(test_result.all_propagated_deltas or []),
                "n_discovered_patterns": test_result.n_discovered_patterns,
                "n_suppression_patterns": test_result.n_suppression_patterns,
                "discovered_pattern_keys": list(test_result.discovered_pattern_keys or []),
                "sample_results": _serialize_samples(test_result),
                "sindit_context_summary": test_result.sindit_context_summary,
            },
            "eval": {
                "f1": round(eval_metrics.f1, 4),
                "precision": round(eval_metrics.precision, 4),
                "recall": round(eval_metrics.recall, 4),
                "auc_roc": round(eval_metrics.auc_roc, 4),
                "n_flagged": eval_flagged,
                "n_samples": eval_result.n_samples,
                "n_feedback": eval_feedback,
                "n_predictions_flipped": eval_result.n_predictions_flipped,
                "n_model_retrains": eval_result.n_model_retrains,
                "feedback_events": list(eval_result.feedback_events or []),
                "pattern_feedback_summary": _summarize_feedback_events(eval_result.feedback_events or []),
                "all_propagated_deltas": list(eval_result.all_propagated_deltas or []),
                "n_propagation_events": len(eval_result.all_propagated_deltas or []),
                "n_discovered_patterns": eval_result.n_discovered_patterns,
                "n_suppression_patterns": eval_result.n_suppression_patterns,
                "discovered_pattern_keys": list(eval_result.discovered_pattern_keys or []),
                "sample_results": _serialize_samples(eval_result),
                "sindit_context_summary": eval_result.sindit_context_summary,
            },
            "comparison": {
                "delta_f1": round(comparison.delta_f1, 4),
                "delta_precision": round(comparison.delta_precision, 4),
                "delta_recall": round(comparison.delta_recall, 4),
                "pct_f1_improvement": round(comparison.pct_f1_improvement, 2),
                "n_feedback_events": comparison.n_feedback_events,
            },
        }
        return fold_result, comparison.to_dict()


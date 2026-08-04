from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

from backend.json_utils import finite_float as _safe_float, json_safe


logger = logging.getLogger(__name__)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_EXPERIMENT_ROOT = _PROJECT_ROOT / "data" / "breakage_patterns" / "stoppage_experiment"
_BREAKAGE_EXPERIMENT_ROOT = _PROJECT_ROOT / "data" / "experiment_snapshots"
_ALL_EXPERIMENT_ROOTS = [_EXPERIMENT_ROOT, _BREAKAGE_EXPERIMENT_ROOT]


def _resolve_run_dir(run_id: str) -> pathlib.Path:
    """Resolve a run_id to its on-disk directory, checking all experiment roots."""
    for root in _ALL_EXPERIMENT_ROOTS:
        candidate = root / run_id
        if candidate.is_dir():
            return candidate
    return _EXPERIMENT_ROOT / run_id


def _save_error_results(run_id: str, experiment_type: str, result: Dict[str, Any]) -> None:
    """Persist a minimal experiment_results.json when the experiment crashes."""
    try:
        run_base = _resolve_run_dir(run_id)
        if run_base.is_dir():
            candidates = list(run_base.rglob("*.joblib"))
            if candidates:
                target_dir = candidates[0].parent
            else:
                target_dir = run_base
        else:
            target_dir = run_base

        target_dir.mkdir(parents=True, exist_ok=True)
        error_file = target_dir / "experiment_results.json"
        if error_file.exists():
            return

        error_data: Dict[str, Any] = {
            "error": True,
            "success": False,
            "experiment_type": experiment_type,
            "run_id": run_id,
            "error_message": result.get("error", "Unknown error"),
            "traceback": result.get("traceback", ""),
            "config": result.get("config", {}),
            "comparison": {
                "test_f1": None,
                "eval_f1": None,
                "delta_f1": None,
                "pct_f1_improvement": None,
            },
        }
        error_file.write_text(json.dumps(error_data, indent=2, default=str), encoding="utf-8")
        logger.info("Saved error results to %s", error_file)
    except Exception:
        logger.debug("Could not save error results for %s", run_id, exc_info=True)


def _run_id_from_dir(d: pathlib.Path) -> str:
    """Use the directory path relative to its experiment root as the canonical run_id."""
    for root in _ALL_EXPERIMENT_ROOTS:
        try:
            return str(d.relative_to(root))
        except ValueError:
            continue
    return d.name


def _load_run_json(run_dir: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Load experiment_results.json from a run directory."""
    p = run_dir / "experiment_results.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _build_run_summary(run_id: str, data: Dict[str, Any], run_dir: pathlib.Path) -> Dict[str, Any]:
    """Build a RunSummary-shaped dict from experiment_results.json data."""
    if data.get("error") is True or data.get("success") is False:
        results_file = run_dir / "experiment_results.json"
        try:
            ts = results_file.stat().st_mtime
        except Exception:
            ts = 0
        return json_safe({
            "run_id": run_id,
            "experiment_type": data.get("experiment_type", "stoppage"),
            "config": data.get("config", {}),
            "eval_metrics": {},
            "test_metrics": {},
            "feedback_stats": {},
            "summary": data.get("summary", {}),
            "gap_s": data.get("config", {}).get("prediction_gap_s", 0),
            "timestamp": ts,
            "error": True,
            "error_message": data.get("error_message", data.get("error", "Unknown error")),
        })

    cfg = data.get("config", {})
    comparison = data.get("comparison", {})
    test_m = comparison.get("test", {})
    eval_m = comparison.get("eval", {})
    fb = comparison.get("feedback_stats", {})

    folds = data.get("folds", [])
    aggregate = data.get("aggregate", {})
    if not test_m and folds:
        test_m = {
            "f1": aggregate.get("test_f1_mean"),
            "precision": aggregate.get("test_precision_mean"),
            "recall": aggregate.get("test_recall_mean"),
            "n_samples": sum(f.get("test", {}).get("n_samples", 0) for f in folds),
        }
        test_aucs = [
            f.get("test", {}).get("auc_roc")
            for f in folds
            if f.get("test", {}).get("auc_roc") is not None
        ]
        if test_aucs:
            test_m["auc_roc"] = sum(test_aucs) / len(test_aucs)
        for cm_key in ("tp", "fp", "tn", "fn"):
            vals = [
                f.get("test", {}).get(cm_key)
                for f in folds
                if f.get("test", {}).get(cm_key) is not None
            ]
            if vals:
                test_m[cm_key] = sum(vals)
    if not eval_m and folds:
        eval_m = {
            "f1": aggregate.get("eval_f1_mean"),
            "precision": aggregate.get("eval_precision_mean"),
            "recall": aggregate.get("eval_recall_mean"),
            "n_samples": sum(f.get("eval", {}).get("n_samples", 0) for f in folds),
        }
        eval_aucs = [
            f.get("eval", {}).get("auc_roc")
            for f in folds
            if f.get("eval", {}).get("auc_roc") is not None
        ]
        if eval_aucs:
            eval_m["auc_roc"] = sum(eval_aucs) / len(eval_aucs)
        for cm_key in ("tp", "fp", "tn", "fn"):
            vals = [
                f.get("eval", {}).get(cm_key)
                for f in folds
                if f.get("eval", {}).get(cm_key) is not None
            ]
            if vals:
                eval_m[cm_key] = sum(vals)
    if not fb and folds:
        total_fb = sum(
            f.get("eval", {}).get("n_feedback", 0)
            or f.get("comparison", {}).get("n_feedback_events", 0)
            for f in folds
        )
        fb = {"n_events": total_fb} if total_fb else {}

    gap_s: Optional[float] = cfg.get("prediction_gap_s")
    if gap_s is None:
        name = run_id
        if "_gap" in name:
            try:
                gap_s = float(name.split("_gap")[1].replace("s", ""))
            except (ValueError, IndexError):
                gap_s = 0.0
        else:
            gap_s = 0.0

    results_file = run_dir / "experiment_results.json"
    try:
        ts = results_file.stat().st_mtime
    except Exception:
        ts = 0

    train_ops = cfg.get("train_ops", [])
    test_op = cfg.get("test_op", "")
    eval_op = cfg.get("eval_op", "")
    if not train_ops and folds:
        all_train = set()
        all_test = []
        for fold in folds:
            for t in fold.get("train_operations", []):
                all_train.add(t)
            top = fold.get("test_operation", "")
            if top:
                all_test.append(top)
        train_ops = sorted(all_train)
        test_op = ", ".join(all_test) if all_test else ""
        eval_op = test_op

    return json_safe({
        "run_id": run_id,
        "experiment_type": data.get("experiment_type", "stoppage"),
        "config": {
            "train_ops": train_ops,
            "test_op": test_op,
            "eval_op": eval_op,
            "eval_variant": cfg.get("eval_variant"),
            "noise_rate": cfg.get("noise_rate"),
            "feedback_every_n": cfg.get("feedback_every_n"),
            "prediction_gap_s": gap_s,
            "features_csv": cfg.get("features_csv"),
            "min_discrimination_ratio": cfg.get("min_discrimination_ratio"),
            "negative_sampling_enabled": cfg.get("negative_sampling_enabled"),
            "negative_sampling_rate": cfg.get("negative_sampling_rate"),
            "store_threshold": cfg.get("store_threshold"),
            "alert_threshold": cfg.get("alert_threshold"),
            "critical_threshold": cfg.get("critical_threshold"),
            "weight_protective_pattern": cfg.get("weight_protective_pattern"),
        },
        "eval_metrics": {
            "f1": eval_m.get("f1"),
            "precision": eval_m.get("precision"),
            "recall": eval_m.get("recall"),
            "auc_roc": eval_m.get("auc_roc"),
            "auc_pr": eval_m.get("auc_pr"),
            "balanced_accuracy": eval_m.get("balanced_accuracy"),
            "n_samples": eval_m.get("n_samples"),
            "tp": eval_m.get("tp"),
            "fp": eval_m.get("fp"),
            "tn": eval_m.get("tn"),
            "fn": eval_m.get("fn"),
        },
        "test_metrics": {
            "f1": test_m.get("f1"),
            "precision": test_m.get("precision"),
            "recall": test_m.get("recall"),
            "auc_roc": test_m.get("auc_roc"),
            "n_samples": test_m.get("n_samples"),
        },
        "feedback_stats": {
            "n_events": fb.get("n_events"),
            "n_confirms": fb.get("n_confirms"),
            "n_dismissals": fb.get("n_dismissals"),
            "accuracy": fb.get("accuracy"),
        },
        "summary": data.get("summary", {}),
        "gap_s": gap_s,
        "timestamp": ts,
    })


def _extract_pattern_polarity_counts(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Summarize calibrated pattern polarities from stored run JSON."""
    direct_sources = [
        data.get("polarity_counts"),
        (data.get("train_meta") or {}).get("polarity_counts") if isinstance(data.get("train_meta"), dict) else None,
        (data.get("train_phase") or {}).get("polarity_counts") if isinstance(data.get("train_phase"), dict) else None,
    ]
    for source in direct_sources:
        if isinstance(source, dict):
            counts = {
                "fault_supporting": int(source.get("fault_supporting", 0) or 0),
                "protective": int(source.get("protective", 0) or 0),
                "uninformative": int(source.get("uninformative", 0) or 0),
                "mixed": int(source.get("mixed", 0) or 0),
            }
            if any(counts.values()):
                return counts

    calibrated_sources = [
        data.get("calibrated_pattern_thresholds"),
        (data.get("train_meta") or {}).get("calibrated_pattern_thresholds") if isinstance(data.get("train_meta"), dict) else None,
        (data.get("train_phase") or {}).get("calibrated_pattern_thresholds") if isinstance(data.get("train_phase"), dict) else None,
    ]
    calibrated = next((source for source in calibrated_sources if isinstance(source, dict)), None)
    if not isinstance(calibrated, dict):
        return None

    counts = {
        "fault_supporting": 0,
        "protective": 0,
        "uninformative": 0,
        "mixed": 0,
    }
    for entry in calibrated.values():
        if not isinstance(entry, dict):
            continue

        polarity = entry.get("polarity") if isinstance(entry.get("polarity"), str) else None
        if not polarity:
            thresholds = entry.get("thresholds")
            threshold_polarities = set()
            if isinstance(thresholds, dict):
                for threshold in thresholds.values():
                    if isinstance(threshold, dict) and isinstance(threshold.get("polarity"), str):
                        threshold_polarities.add(threshold["polarity"])
            if len(threshold_polarities) == 1:
                polarity = next(iter(threshold_polarities))
            elif len(threshold_polarities) > 1:
                polarity = "mixed"

        if polarity in counts:
            counts[polarity] += 1

    return counts if any(counts.values()) else None


def _map_stored_sample_result(sr: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a stored sample-result dict to the dashboard sample shape."""
    return {
        "sample_id": sr.get("sample_id", ""),
        "label": sr.get("label", ""),
        "operation_id": sr.get("operation_id", ""),
        "tool_number": sr.get("tool_number", ""),
        "memory_id": sr.get("memory_id"),
        "significance_score": _safe_float(sr.get("significance_score")) or 0.0,
        "action": sr.get("action", ""),
        "predicted_positive": sr.get("predicted_positive", False),
        "raw_model_score": _safe_float(sr.get("raw_model_score")) or 0.0,
        "pattern_rule_score": _safe_float(sr.get("pattern_rule_score")) or 0.0,
        "anomaly_z_score": _safe_float(sr.get("anomaly_z_score")) or 0.0,
        "prior_boost": _safe_float(sr.get("prior_boost")) or 0.0,
        "multi_rule_bonus": _safe_float(sr.get("multi_rule_bonus")) or 0.0,
        "n_rules_triggered": sr.get("n_rules_triggered", 0),
        "score_trace": sr.get("score_trace", []),
        "detected_patterns": sr.get("detected_patterns", []),
        "supervised_score": _safe_float(sr.get("supervised_score")) or 0.0,
        "unsupervised_score": _safe_float(sr.get("unsupervised_score")) or 0.0,
        "combined_score": _safe_float(sr.get("combined_score")) or 0.0,
        "tool_prior": _safe_float(sr.get("tool_prior")) or 0.5,
        "tool_multiplier": _safe_float(sr.get("tool_multiplier")) or 1.0,
        "weight_supervised": _safe_float(sr.get("weight_supervised")) or 0.0,
        "weight_unsupervised": _safe_float(sr.get("weight_unsupervised")) or 0.0,
        "feedback_given": sr.get("feedback_given", False),
        "feedback_action": sr.get("feedback_action", ""),
        "feedback_source": sr.get("feedback_source", ""),
        "counterfactual_score": _safe_float(sr.get("counterfactual_score")) or 0.0,
        "prediction_flipped": sr.get("prediction_flipped", False),
        "prior_snapshot": sr.get("prior_snapshot", {}),
        "model_breakdown": sr.get("model_breakdown") or {},
        "stored_in_memory": sr.get("stored_in_memory", False),
        "co_occurring_pairs": sr.get("co_occurring_pairs", []),
        "propagated_prior_deltas": sr.get("propagated_prior_deltas", {}),
        "sindit_context": sr.get("sindit_context"),
        "explanation": sr.get("explanation"),
        "explanation_source": sr.get("explanation_source"),
        "alert_line": sr.get("alert_line"),
        "alert_line_source": sr.get("alert_line_source"),
    }


def _map_stored_phase_result(phase_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a stored phase dict to the dashboard phase-detail shape."""
    raw_samples = phase_data.get("sample_results", [])
    samples = [_map_stored_sample_result(sr) for sr in raw_samples]
    return {
        "phase": phase_data.get("phase", ""),
        "operation": phase_data.get("operation", ""),
        "n_samples": phase_data.get("n_samples", len(samples)),
        "threshold": phase_data.get("threshold", 0.5),
        "adapted_threshold": phase_data.get("adapted_threshold", 0.5),
        "prior_history": phase_data.get("prior_history", {}),
        "scores_positive": phase_data.get("scores_positive", []),
        "scores_negative": phase_data.get("scores_negative", []),
        "n_predictions_flipped": phase_data.get("n_predictions_flipped", 0),
        "weight_history": phase_data.get("weight_history", []),
        "tool_prior_history": phase_data.get("tool_prior_history", []),
        "n_model_retrains": phase_data.get("n_model_retrains", 0),
        "co_occurrence_graph": phase_data.get("co_occurrence_graph", {}),
        "stored_memories_count": phase_data.get("stored_memories_count", 0),
        "all_propagated_deltas": phase_data.get("all_propagated_deltas", []),
        "sindit_context_summary": phase_data.get("sindit_context_summary", {}),
        "samples": samples,
    }


def _build_stoppage_evaluation_from_json(run_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build EvaluationDetail from stored stoppage experiment JSON."""
    test_data = data.get("test_phase") or data.get("test")
    eval_data = data.get("eval_phase") or data.get("eval")
    if not test_data or not eval_data:
        return None

    test_samples_raw = test_data.get("sample_results", [])
    eval_samples_raw = eval_data.get("sample_results", [])
    if not test_samples_raw and not eval_samples_raw:
        return None

    return {
        "run_id": run_id,
        "test": _map_stored_phase_result(test_data),
        "eval": _map_stored_phase_result(eval_data),
    }


def _build_breakage_evaluation(run_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Build an EvaluationDetail-shaped response from stored breakage results."""
    cfg = data.get("config", {})
    comparison = data.get("comparison", {})
    folds = data.get("folds", [])
    prior_evo = data.get("prior_evolution", {})

    test_m = comparison.get("test", {})
    eval_m = comparison.get("eval", {})

    co_graph: Dict[str, int] = {}
    pattern_fire_counts: Dict[str, int] = {}
    total_memories = 0
    for fold in folds:
        for pair, cnt in fold.get("co_occurrence_graph", {}).items():
            co_graph[pair] = co_graph.get(pair, 0) + cnt
        for pat, cnt in fold.get("pattern_fire_counts", {}).items():
            pattern_fire_counts[pat] = pattern_fire_counts.get(pat, 0) + cnt
        total_memories += fold.get("memories_stored", 0)

    prior_history: Dict[str, List[float]] = {}
    for _op, patterns in prior_evo.items():
        for pat, values in patterns.items():
            if pat in prior_history:
                prior_history[pat].extend(values)
            else:
                prior_history[pat] = list(values)

    def _collect_fold_samples(phase_key: str) -> List[Dict[str, Any]]:
        all_samples: List[Dict[str, Any]] = []
        for fold in folds:
            phase_data = fold.get(phase_key, {})
            stored_samples = phase_data.get("sample_results", [])
            if stored_samples:
                for sr in stored_samples:
                    all_samples.append(_map_stored_sample_result(sr))
        return all_samples

    def _collect_feedback_events(phase_key: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for fold in folds:
            phase_data = fold.get(phase_key, {})
            for event in phase_data.get("feedback_events", []) or []:
                if isinstance(event, dict):
                    events.append(event)
        return events

    def _merge_pattern_feedback_summary(phase_key: str) -> Dict[str, Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for fold in folds:
            phase_data = fold.get(phase_key, {})
            summary = phase_data.get("pattern_feedback_summary", {})
            if not isinstance(summary, dict):
                continue
            for pattern_key, entry in summary.items():
                if not isinstance(pattern_key, str) or not isinstance(entry, dict):
                    continue
                bucket = merged.setdefault(pattern_key, {
                    "polarity": entry.get("polarity"),
                    "n_feedback_events": 0,
                    "n_confirms": 0,
                    "n_dismissals": 0,
                    "total_prior_delta": 0.0,
                    "mean_prior_delta": 0.0,
                    "max_abs_prior_delta": 0.0,
                    "last_prior": _safe_float(entry.get("last_prior")) or 0.0,
                })
                bucket["n_feedback_events"] += int(entry.get("n_feedback_events", 0) or 0)
                bucket["n_confirms"] += int(entry.get("n_confirms", 0) or 0)
                bucket["n_dismissals"] += int(entry.get("n_dismissals", 0) or 0)
                bucket["total_prior_delta"] = round(
                    float(bucket["total_prior_delta"]) + float(entry.get("total_prior_delta", 0.0) or 0.0),
                    4,
                )
                bucket["max_abs_prior_delta"] = round(
                    max(float(bucket["max_abs_prior_delta"]), abs(float(entry.get("max_abs_prior_delta", 0.0) or 0.0))),
                    4,
                )
                last_prior = _safe_float(entry.get("last_prior"))
                if last_prior is not None:
                    bucket["last_prior"] = last_prior
                if entry.get("polarity"):
                    bucket["polarity"] = entry.get("polarity")

        for bucket in merged.values():
            n_events = max(1, int(bucket["n_feedback_events"]))
            bucket["mean_prior_delta"] = round(float(bucket["total_prior_delta"]) / n_events, 4)
        return merged

    def _collect_phase_int(phase_key: str, field: str) -> int:
        total = 0
        for fold in folds:
            phase_data = fold.get(phase_key, {})
            total += int(phase_data.get(field, 0) or 0)
        return total

    def _collect_phase_list(phase_key: str, field: str) -> List[Any]:
        items: List[Any] = []
        for fold in folds:
            phase_data = fold.get(phase_key, {})
            values = phase_data.get(field, []) or []
            if isinstance(values, list):
                items.extend(values)
        return items

    def _collect_discovered_pattern_keys(phase_key: str) -> List[str]:
        seen: Dict[str, None] = {}
        for key in _collect_phase_list(phase_key, "discovered_pattern_keys"):
            if isinstance(key, str) and key not in seen:
                seen[key] = None
        return list(seen.keys())

    def _collect_sindit_summary(phase_key: str) -> Dict[str, Any]:
        n_normal = 0
        n_degraded = 0
        total = 0
        for fold in folds:
            phase_data = fold.get(phase_key, {})
            summary = phase_data.get("sindit_context_summary", {})
            n_normal += summary.get("n_normal", 0)
            n_degraded += summary.get("n_degraded", 0)
            total += summary.get("total", 0)
        if total == 0:
            return {}
        return {"n_normal": n_normal, "n_degraded": n_degraded, "total": total}

    test_samples = _collect_fold_samples("test")
    eval_samples = _collect_fold_samples("eval")
    test_feedback_events = _collect_feedback_events("test")
    eval_feedback_events = _collect_feedback_events("eval")
    test_feedback_summary = _merge_pattern_feedback_summary("test")
    eval_feedback_summary = _merge_pattern_feedback_summary("eval")

    if not eval_samples:
        for fold in folds:
            test_op = fold.get("test_operation", cfg.get("test_op", ""))
            final_priors = fold.get("final_priors", {})
            stub = {
                "sample_id": f"fold_{test_op}",
                "label": f"LOOCV fold ({fold.get('eval', {}).get('n_samples', 0)} samples)",
                "operation_id": test_op,
                "tool_number": "",
                "significance_score": fold.get("eval", {}).get("f1", 0.0),
                "action": "fold_summary",
                "predicted_positive": True,
                "raw_model_score": 0.0,
                "pattern_rule_score": 0.0,
                "anomaly_z_score": 0.0,
                "prior_boost": 0.0,
                "multi_rule_bonus": 0.0,
                "n_rules_triggered": len(fold.get("pattern_fire_counts", {})),
                "detected_patterns": list(fold.get("pattern_fire_counts", {}).keys()),
                "supervised_score": 0.0,
                "unsupervised_score": 0.0,
                "combined_score": 0.0,
                "tool_prior": 0.5,
                "tool_multiplier": 1.0,
                "weight_supervised": 0.0,
                "weight_unsupervised": 0.0,
                "feedback_given": True,
                "feedback_action": "fold_summary",
                "feedback_source": "breakage_loocv",
                "counterfactual_score": 0.0,
                "prediction_flipped": False,
                "prior_snapshot": final_priors,
                "stored_in_memory": False,
                "co_occurring_pairs": [],
                "propagated_prior_deltas": {},
                "sindit_context": None,
            }
            eval_samples.append(stub)
            test_samples.append(stub)

    def _make_phase(
        phase_name: str,
        metrics: Dict[str, Any],
        samples: List[Dict[str, Any]],
        sindit_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        feedback_events = test_feedback_events if phase_name == "test" else eval_feedback_events
        feedback_summary = test_feedback_summary if phase_name == "test" else eval_feedback_summary
        return {
            "phase": phase_name,
            "operation": cfg.get("test_op", ""),
            "n_samples": metrics.get("n_samples", 0),
            "threshold": 0.5,
            "adapted_threshold": 0.5,
            "prior_history": prior_history,
            "scores_positive": [],
            "scores_negative": [],
            "n_predictions_flipped": _collect_phase_int(phase_name, "n_predictions_flipped"),
            "weight_history": {},
            "tool_prior_history": {},
            "n_model_retrains": _collect_phase_int(phase_name, "n_model_retrains"),
            "co_occurrence_graph": co_graph,
            "stored_memories_count": total_memories,
            "all_propagated_deltas": _collect_phase_list(phase_name, "all_propagated_deltas"),
            "sindit_context_summary": sindit_summary,
            "feedback_events": feedback_events,
            "pattern_feedback_summary": feedback_summary,
            "n_propagation_events": _collect_phase_int(phase_name, "n_propagation_events"),
            "n_discovered_patterns": _collect_phase_int(phase_name, "n_discovered_patterns"),
            "n_suppression_patterns": _collect_phase_int(phase_name, "n_suppression_patterns"),
            "discovered_pattern_keys": _collect_discovered_pattern_keys(phase_name),
            "samples": samples,
        }

    return {
        "run_id": run_id,
        "test": _make_phase("test", test_m, test_samples, _collect_sindit_summary("test")),
        "eval": _make_phase("eval", eval_m, eval_samples, _collect_sindit_summary("eval")),
    }


__all__ = [
    "_ALL_EXPERIMENT_ROOTS",
    "_build_breakage_evaluation",
    "_build_run_summary",
    "_build_stoppage_evaluation_from_json",
    "_extract_pattern_polarity_counts",
    "_load_run_json",
    "_resolve_run_dir",
    "_run_id_from_dir",
    "_save_error_results",
]
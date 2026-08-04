"""Coverage audit helpers for tool-master lookup resolution.

This module powers the coverage smoke test described in
``docs/DATASET_CUTTING_PARAMS_AUDIT.md``. It measures, per operation / OF,
how often the tool-master can resolve diameter and teeth from real dataset rows.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Sequence

import pandas as pd

from .tool_lookup import (
    FAMILY_MACHINE_A1,
    lookup as lookup_tool_spec,
    resolve_machine_family,
)

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "site_b_olddata": 0.95,
    "site_a_casedata": 0.60,
    "site_a_line2": 0.60,
    "site_c": 0.40,
}


def collect_tool_lookup_coverage(
    repo_root: Path | str | None = None,
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    items = collect_coverage_items(root)
    summary = summarize_coverage_items(items, thresholds=thresholds)
    failures = build_threshold_failures(summary)
    return {
        "repo_root": str(root),
        "items": items,
        "summary": summary,
        "ok": not failures,
        "threshold_failures": failures,
    }


def collect_coverage_items(repo_root: Path | str) -> list[dict[str, Any]]:
    root = Path(repo_root)
    items: list[dict[str, Any]] = []
    items.extend(collect_casedata_coverage_items(root / "data" / "casedata"))
    items.extend(collect_casedata_coverage_items(root / "data" / "site_a"))
    items.extend(collect_casedata_coverage_items(root / "data" / "olddata", machine_id_override="olddata"))
    items.extend(collect_site_a_line2_coverage_items(root / "data" / "Site_a_line2"))
    return sorted(items, key=lambda item: (item["dataset"], item["machine_id"], item["operation_id"]))


def collect_casedata_coverage_items(
    root: Path | str,
    *,
    machine_id_override: str | None = None,
) -> list[dict[str, Any]]:
    base = Path(root)
    if not base.exists():
        logger.info("Coverage source %s not found; skipping", base)
        return []

    items: list[dict[str, Any]] = []
    for machine_id, operation_id, tyzbps_path in _iter_casedata_operation_sources(base, machine_id_override=machine_id_override):
        item = _build_casedata_item(machine_id, operation_id, tyzbps_path)
        if item is not None:
            items.append(item)
    return items


def collect_site_a_line2_coverage_items(root: Path | str) -> list[dict[str, Any]]:
    data_root = Path(root)
    monitored_root = data_root / "Monitored data"
    if not monitored_root.exists():
        logger.info("Site_a_line2 monitored data %s not found; skipping", monitored_root)
        return []

    items: list[dict[str, Any]] = []
    for session_dir in sorted(monitored_root.iterdir()):
        if not session_dir.is_dir() or session_dir.name.startswith("."):
            continue

        tyzbps_path = _first_matching_csv(session_dir, "TYZBPS")
        dlg6cf_path = _first_matching_csv(session_dir, "DLG6CF")
        if tyzbps_path is None:
            continue

        tyzbps = _read_csv_subset(
            tyzbps_path,
            requested_columns=[
                "Date",
                "UF5-Numero_de_pieza_OF",
                "Cnc_Tool_Number_RT",
                "SpindleSpeedActual",
                "Axis_FeedRate_actual",
            ],
        )
        if tyzbps.empty:
            continue

        tyzbps = tyzbps.rename(columns={
            "UF5-Numero_de_pieza_OF": "of_id",
            "Cnc_Tool_Number_RT": "tool_number",
            "SpindleSpeedActual": "spindle_speed",
            "Axis_FeedRate_actual": "feed_rate",
        })

        if dlg6cf_path is not None:
            dlg6cf = _read_csv_subset(
                dlg6cf_path,
                requested_columns=["Date", "CNC_parameters_teeth_num"],
            )
            if not dlg6cf.empty:
                dlg6cf = dlg6cf.rename(columns={"CNC_parameters_teeth_num": "raw_teeth"})
                tyzbps = pd.merge_asof(
                    tyzbps.sort_values("_timestamp"),
                    dlg6cf.sort_values("_timestamp")[["_timestamp", "raw_teeth"]],
                    on="_timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta(seconds=2),
                )
        if "raw_teeth" not in tyzbps.columns:
            tyzbps["raw_teeth"] = pd.NA

        tyzbps["tool_number"] = pd.to_numeric(tyzbps.get("tool_number"), errors="coerce").fillna(0).astype(int)
        tyzbps["of_id"] = tyzbps.get("of_id", "").map(_canonical_site_a_line2_operation_id)
        valid = tyzbps["tool_number"] > 0
        if not valid.any():
            continue

        working = tyzbps.loc[valid].copy()
        for operation_id, group in working.groupby("of_id", dropna=False):
            normalized_operation = str(operation_id).strip() or session_dir.name
            item = _build_site_a_line2_item(session_dir.name, normalized_operation, group)
            if item is not None:
                items.append(item)

    return items


def summarize_coverage_items(
    items: Sequence[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    effective_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        effective_thresholds.update(thresholds)

    datasets: dict[str, Any] = {}
    for dataset, threshold in effective_thresholds.items():
        dataset_items = [item for item in items if item.get("dataset") == dataset]
        valid_rows = sum(int(item.get("valid_tool_rows", 0)) for item in dataset_items)
        resolved_master_rows = sum(int(item.get("resolved_master_rows", 0)) for item in dataset_items)
        resolved_d_rows = sum(int(item.get("resolved_d_rows", 0)) for item in dataset_items)
        resolved_z_rows = sum(int(item.get("resolved_z_rows", 0)) for item in dataset_items)
        resolved_dz_rows = sum(int(item.get("resolved_dz_rows", 0)) for item in dataset_items)
        harmonic_ready_rows = sum(int(item.get("harmonic_ready_rows", 0)) for item in dataset_items)

        observed_tools = _merge_tool_sets(dataset_items, "unique_tools")
        resolved_master_tools = _merge_tool_sets(dataset_items, "resolved_master_tools")
        resolved_d_tools = _merge_tool_sets(dataset_items, "resolved_d_tools")
        resolved_z_tools = _merge_tool_sets(dataset_items, "resolved_z_tools")
        resolved_dz_tools = _merge_tool_sets(dataset_items, "resolved_dz_tools")
        harmonic_ready_tools = _merge_tool_sets(dataset_items, "harmonic_ready_tools")

        if observed_tools:
            coverage_master = _safe_ratio(len(resolved_master_tools), len(observed_tools))
            coverage_diameter = _safe_ratio(len(resolved_d_tools), len(observed_tools))
            coverage_teeth = _safe_ratio(len(resolved_z_tools), len(observed_tools))
            coverage_dz = _safe_ratio(len(resolved_dz_tools), len(observed_tools))
            harmonic_ready = _safe_ratio(len(harmonic_ready_tools), len(observed_tools))
        else:
            coverage_master = _safe_ratio(resolved_master_rows, valid_rows)
            coverage_diameter = _safe_ratio(resolved_d_rows, valid_rows)
            coverage_teeth = _safe_ratio(resolved_z_rows, valid_rows)
            coverage_dz = _safe_ratio(resolved_dz_rows, valid_rows)
            harmonic_ready = _safe_ratio(harmonic_ready_rows, valid_rows)
        passes = bool(dataset_items) and coverage_master >= threshold

        datasets[dataset] = {
            "threshold": threshold,
            "items": len(dataset_items),
            "observed_tools": observed_tools,
            "resolved_master_tools": resolved_master_tools,
            "resolved_d_tools": resolved_d_tools,
            "resolved_z_tools": resolved_z_tools,
            "resolved_dz_tools": resolved_dz_tools,
            "harmonic_ready_tools": harmonic_ready_tools,
            "valid_tool_rows": valid_rows,
            "resolved_master_rows": resolved_master_rows,
            "coverage_master": coverage_master,
            "resolved_d_rows": resolved_d_rows,
            "resolved_z_rows": resolved_z_rows,
            "resolved_dz_rows": resolved_dz_rows,
            "harmonic_ready_rows": harmonic_ready_rows,
            "coverage_diameter": coverage_diameter,
            "coverage_teeth": coverage_teeth,
            "coverage_dz": coverage_dz,
            "harmonic_ready": harmonic_ready,
            "passes": passes,
            "data_available": bool(dataset_items),
        }

    return {
        "datasets": datasets,
        "thresholds": effective_thresholds,
    }


def build_threshold_failures(summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    datasets = summary.get("datasets") or {}
    for dataset, payload in sorted(datasets.items()):
        if not payload.get("data_available"):
            failures.append(f"{dataset}: no data available for coverage audit")
            continue
        if not payload.get("passes"):
            failures.append(
                f"{dataset}: coverage_master={payload.get('coverage_master', 0.0):.3f} "
                f"< threshold={payload.get('threshold', 0.0):.3f}"
            )
    return failures


def _iter_casedata_operation_sources(
    root: Path,
    *,
    machine_id_override: str | None = None,
) -> Iterator[tuple[str, str, Path]]:
    direct_operations = sorted(
        child for child in root.iterdir() if child.is_dir() and child.name.startswith("OF")
    )
    if direct_operations:
        machine_id = machine_id_override or root.name
        for op_dir in direct_operations:
            tyzbps_path = _first_matching_csv(op_dir, "TYZBPS")
            if tyzbps_path is not None:
                yield machine_id, op_dir.name, tyzbps_path
        return

    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir() or case_dir.name.startswith("."):
            continue
        machine_id = case_dir.name
        for op_dir in sorted(case_dir.iterdir()):
            if not op_dir.is_dir() or not op_dir.name.startswith("OF"):
                continue
            tyzbps_path = _first_matching_csv(op_dir, "TYZBPS")
            if tyzbps_path is not None:
                yield machine_id, op_dir.name, tyzbps_path


def _build_casedata_item(machine_id: str, operation_id: str, tyzbps_path: Path) -> dict[str, Any] | None:
    frame = _read_csv_subset(
        tyzbps_path,
        requested_columns=[
            "timestamp",
            "Tool_Number",
            "Spindle_Speed_Actual",
            "Feed_Rate_Actual",
        ],
    )
    if frame.empty or "Tool_Number" not in frame.columns:
        return None

    frame = frame.rename(columns={
        "Tool_Number": "tool_number",
        "Spindle_Speed_Actual": "spindle_speed",
        "Feed_Rate_Actual": "feed_rate",
    })
    return _build_lookup_item(
        dataset=_dataset_name_for_machine(machine_id),
        machine_id=machine_id,
        machine_family=resolve_machine_family(machine_id),
        operation_id=operation_id,
        frame=frame,
        raw_teeth_column=None,
    )


def _build_site_a_line2_item(
    session_name: str,
    operation_id: str,
    frame: pd.DataFrame,
) -> dict[str, Any] | None:
    return _build_lookup_item(
        dataset="site_a_line2",
        machine_id=session_name,
        machine_family=FAMILY_MACHINE_A1,
        operation_id=operation_id,
        frame=frame,
        raw_teeth_column="raw_teeth",
    )


def _build_lookup_item(
    *,
    dataset: str,
    machine_id: str,
    machine_family: str,
    operation_id: str,
    frame: pd.DataFrame,
    raw_teeth_column: str | None,
) -> dict[str, Any] | None:
    if "tool_number" not in frame.columns:
        return None

    tool_numbers = pd.to_numeric(frame.get("tool_number"), errors="coerce").fillna(0).astype(int)
    valid = tool_numbers > 0
    valid_rows = int(valid.sum())
    if valid_rows == 0:
        return None

    spec_cache = {
        tool_number: lookup_tool_spec(machine_family, tool_number)
        for tool_number in sorted(set(int(value) for value in tool_numbers.loc[valid].tolist()))
    }
    diameter_values = tool_numbers.map(
        lambda value: spec_cache.get(int(value)).diameter_mm if int(value) in spec_cache and spec_cache.get(int(value)) is not None else None
    )
    teeth_values = tool_numbers.map(
        lambda value: spec_cache.get(int(value)).teeth if int(value) in spec_cache and spec_cache.get(int(value)) is not None else None
    )

    raw_teeth = None
    if raw_teeth_column and raw_teeth_column in frame.columns:
        raw_teeth = pd.to_numeric(frame.get(raw_teeth_column), errors="coerce")
        teeth_values = teeth_values.where(teeth_values.notna(), raw_teeth.where(raw_teeth > 0))

    spindle = pd.to_numeric(frame.get("spindle_speed"), errors="coerce") if "spindle_speed" in frame.columns else pd.Series(index=frame.index, dtype=float)
    feed = pd.to_numeric(frame.get("feed_rate"), errors="coerce") if "feed_rate" in frame.columns else pd.Series(index=frame.index, dtype=float)

    resolved_d = valid & diameter_values.notna()
    resolved_z = valid & teeth_values.notna()
    resolved_dz = resolved_d & resolved_z
    resolved_master = valid & tool_numbers.map(lambda value: spec_cache.get(int(value)) is not None)
    harmonic_ready = resolved_dz & spindle.notna() & feed.notna()

    unique_tools = sorted(set(int(value) for value in tool_numbers.loc[valid].tolist()))
    resolved_master_tools = [
        tool_number for tool_number in unique_tools
        if spec_cache.get(tool_number) is not None
    ]
    missing_tools = [
        tool_number for tool_number in unique_tools
        if spec_cache.get(tool_number) is None
    ]
    tools_missing_teeth = [
        tool_number for tool_number in unique_tools
        if spec_cache.get(tool_number) is not None and spec_cache[tool_number].teeth is None
    ]

    resolved_d_tools = [
        tool_number for tool_number in unique_tools
        if spec_cache.get(tool_number) is not None and spec_cache[tool_number].diameter_mm is not None
    ]
    resolved_z_tools = [
        tool_number for tool_number in unique_tools
        if _tool_has_teeth(tool_number, spec_cache, frame, raw_teeth_column)
    ]
    resolved_dz_tools = [
        tool_number for tool_number in resolved_d_tools
        if tool_number in resolved_z_tools
    ]
    harmonic_ready_tools = [
        tool_number for tool_number in resolved_dz_tools
        if _tool_has_spindle_and_feed(tool_number, tool_numbers, spindle, feed)
    ]

    return {
        "dataset": dataset,
        "machine_id": machine_id,
        "machine_family": machine_family,
        "operation_id": operation_id,
        "valid_tool_rows": valid_rows,
        "resolved_master_rows": int(resolved_master.sum()),
        "resolved_d_rows": int(resolved_d.sum()),
        "resolved_z_rows": int(resolved_z.sum()),
        "resolved_dz_rows": int(resolved_dz.sum()),
        "harmonic_ready_rows": int(harmonic_ready.sum()),
        "coverage_master": _safe_ratio(int(resolved_master.sum()), valid_rows),
        "coverage_diameter": _safe_ratio(int(resolved_d.sum()), valid_rows),
        "coverage_teeth": _safe_ratio(int(resolved_z.sum()), valid_rows),
        "coverage_dz": _safe_ratio(int(resolved_dz.sum()), valid_rows),
        "harmonic_ready": _safe_ratio(int(harmonic_ready.sum()), valid_rows),
        "unique_tools": unique_tools,
        "resolved_master_tools": resolved_master_tools,
        "resolved_d_tools": resolved_d_tools,
        "resolved_z_tools": resolved_z_tools,
        "resolved_dz_tools": resolved_dz_tools,
        "harmonic_ready_tools": harmonic_ready_tools,
        "missing_tools": missing_tools,
        "tools_missing_master_teeth": tools_missing_teeth,
    }


def _dataset_name_for_machine(machine_id: str) -> str:
    normalized = str(machine_id).strip().lower()
    if normalized.startswith("site_a"):
        return "site_a_casedata"
    if normalized.startswith("site_c"):
        return "site_c"
    return "site_b_olddata"


def _canonical_site_a_line2_operation_id(value: Any) -> str:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return ""
    return f"OF{numeric}" if numeric > 0 else ""


def _first_matching_csv(directory: Path, needle: str) -> Path | None:
    needle_upper = needle.upper()
    for candidate in sorted(directory.glob("*.csv")):
        if needle_upper in candidate.name.upper():
            return candidate
    return None


def _read_csv_subset(path: Path, *, requested_columns: Sequence[str]) -> pd.DataFrame:
    delimiter = _detect_delimiter(path)
    header = _read_header(path, delimiter)
    available_columns = [column for column in requested_columns if column in header]
    if not available_columns:
        return pd.DataFrame()

    frame = pd.read_csv(path, sep=delimiter, usecols=available_columns, low_memory=False)
    timestamp_column = next((column for column in ("timestamp", "Timestamp", "Date", "time", "Time") if column in frame.columns), None)
    if timestamp_column is not None:
        frame["_timestamp"] = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
        frame["_timestamp"] = frame["_timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        frame = frame.dropna(subset=["_timestamp"]).sort_values("_timestamp").reset_index(drop=True)
    return frame


def _detect_delimiter(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline()
    return ";" if first_line.count(";") > first_line.count(",") else ","


def _read_header(path: Path, delimiter: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        return next(reader, [])


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _merge_tool_sets(items: Sequence[dict[str, Any]], key: str) -> list[int]:
    merged: set[int] = set()
    for item in items:
        for value in item.get(key) or []:
            try:
                merged.add(int(value))
            except (TypeError, ValueError):
                continue
    return sorted(merged)


def _tool_has_teeth(
    tool_number: int,
    spec_cache: dict[int, Any],
    frame: pd.DataFrame,
    raw_teeth_column: str | None,
) -> bool:
    spec = spec_cache.get(tool_number)
    if spec is not None and spec.teeth is not None:
        return True
    if not raw_teeth_column or raw_teeth_column not in frame.columns:
        return False
    raw_teeth = pd.to_numeric(frame.loc[pd.to_numeric(frame.get("tool_number"), errors="coerce").fillna(0).astype(int) == tool_number, raw_teeth_column], errors="coerce")
    return bool((raw_teeth > 0).any())


def _tool_has_spindle_and_feed(
    tool_number: int,
    tool_numbers: pd.Series,
    spindle: pd.Series,
    feed: pd.Series,
) -> bool:
    mask = tool_numbers == tool_number
    if not bool(mask.any()):
        return False
    return bool((spindle.loc[mask].notna() & feed.loc[mask].notna()).any())


__all__ = [
    "DEFAULT_THRESHOLDS",
    "build_threshold_failures",
    "collect_casedata_coverage_items",
    "collect_coverage_items",
    "collect_site_a_line2_coverage_items",
    "collect_tool_lookup_coverage",
    "summarize_coverage_items",
]
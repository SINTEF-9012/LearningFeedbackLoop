"""Shared pytest configuration.

Some tests exercise tool-master enrichment, which needs a site-supplied
tool-data directory (``data/tools/``: machine-family registry plus tool
workbooks). No tool data ships with this repository — it is site-specific — so
those tests are skipped unless you point the suite at your own dataset.

This mirrors how the end-to-end tests skip themselves when no server is
running: the suite stays green out of the box and gets stricter as you add
data. Everything else — the streaming core, the memory/feedback loop, scoring,
retrieval, the graph layer — runs without any dataset.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "data" / "tools"


def tool_data_available() -> bool:
    """True when a site tool-master dataset is present."""
    return TOOLS_DIR.is_dir() and any(TOOLS_DIR.iterdir())


# Tests that cannot pass without a tool-master dataset.
REQUIRES_TOOL_DATA: frozenset[str] = frozenset({
    "test_merge_session_metadata_resolves_machine_twin_and_tool_context_for_live_casedata",
    "test_merge_session_metadata_resolves_generic_live_source_to_machine_and_tool",
    "test_load_split_dataset_normalizes_operations_and_labels",
    "test_load_split_dataset_original_scheme_keeps_only_hard_break_positive",
    "test_windowdata_cutting_context_uses_tool_lookup_for_known_case_family",
    "test_analyse_site_a_supports_loader_discovery_and_process_plan",
    "test_write_json_report_exports_weak_labels_with_plan_context",
    "test_extract_use_case_operation_sequences_from_repo_ppt",
    "test_resolve_runtime_metadata_classifies_site_a_casedata_dataset",
    "test_machine_family_registry_loads_default_yaml",
    "test_resolve_machine_family_uses_registry_and_slug_fallback",
    "test_start_demo_can_disambiguate_duplicate_operation_ids_by_case",
    "test_casedata_catalog_marks_harmonic_ready_operations",
    "test_casedata_catalog_uses_cutting_rows_for_tool_preview",
    "test_casedata_catalog_prefers_dominant_cutting_tool_for_preview",
    "test_casedata_catalog_marks_pair_preview_operations_as_ready",
    "test_start_demo_valid_tools_only_skips_invalid_first_operation",
    "test_start_demo_casedata_defaults_to_harmonic_ready_operation",
    "test_machine_a1_encoded_geometry_is_parsed",
    "test_machine_a1_ardatza_is_classified_as_tap",
    "test_press_c_ditto_description_keeps_previous_tool_family",
    "test_builder_b1_reviewed_merge_is_conservative",
    "test_press_c_split_header_is_parsed",
    "test_lookup_returns_copy_and_none_for_missing",
    "test_resolve_tool_context_uses_raw_teeth_fallback",
    "test_collect_site_a_line2_coverage_uses_raw_teeth_fallback",
    "test_collect_casedata_coverage_handles_direct_of_root",
    "test_collect_casedata_coverage_classifies_site_a_case_root",
})


def pytest_collection_modifyitems(config, items):
    if tool_data_available():
        return
    skip = pytest.mark.skip(
        reason="requires a site tool-master dataset under data/tools/ "
               "(not distributed — see tests/conftest.py)"
    )
    for item in items:
        if item.originalname in REQUIRES_TOOL_DATA or item.name in REQUIRES_TOOL_DATA:
            item.add_marker(skip)

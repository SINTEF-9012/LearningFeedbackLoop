from __future__ import annotations

from pathlib import Path

from scripts.extract_use_case_operation_sequence import extract_operation_sequences


def test_extract_use_case_operation_sequences_from_repo_ppt() -> None:
    ppt_path = Path("data/tools/UseCasesOperationSequence v2.pptx")

    rows = extract_operation_sequences(ppt_path)

    assert len(rows) > 80

    site_b_finish_cut_out = next(
        row for row in rows
        if row["use_case_id"] == 1 and row["operation_id"] == "OP045"
    )
    assert site_b_finish_cut_out["tool_numbers"] == [64]
    assert site_b_finish_cut_out["description"] == "FINISH CUT OUT"

    site_c_thread_m30 = next(
        row for row in rows
        if row["use_case_id"] == 2 and row["operation_id"] == "OP052"
    )
    assert site_c_thread_m30["tool_numbers"] == [2695]
    assert site_c_thread_m30["description"] == "THREADING M30"

    site_a_multi_tool = next(
        row for row in rows
        if row["use_case_id"] == 3
        and row["operation_id"] == "OP20"
        and row["description"] == "FACE MILLING OF UPPER SURFACES"
    )
    assert site_a_multi_tool["head"] == "BORING"
    assert site_a_multi_tool["tool_numbers"] == [2, 30, 44, 45]

    site_b_dual_tool = next(
        row for row in rows
        if row["use_case_id"] == 1
        and row["operation_id"] == "OP039"
        and row["description"].startswith("ROUGH MILLING")
    )
    assert site_b_dual_tool["tool_numbers"] == [60, 72]

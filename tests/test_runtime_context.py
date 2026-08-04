from __future__ import annotations

from backend.agents.sindit.runtime_context import resolve_runtime_metadata


def test_resolve_runtime_metadata_classifies_site_a_casedata_dataset() -> None:
    metadata = {
        "source": "simulated_casedata",
        "casedata": {
            "case_dir": "Site_a - MACHINE_A1 - CASE_A1",
        },
    }

    resolved = resolve_runtime_metadata(metadata, {"metadata": metadata})

    assert resolved["machine_id"] == "Site_a - MACHINE_A1 - CASE_A1"
    assert resolved["machine_family"] == "machine_a1"
    assert resolved["dataset_id"] == "site_a_casedata"
    assert resolved["source_dataset_id"] == "site_a_casedata"
    assert resolved["casedata"]["dataset_id"] == "site_a_casedata"
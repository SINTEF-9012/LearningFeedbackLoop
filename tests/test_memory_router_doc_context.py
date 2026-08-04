from __future__ import annotations

from backend.agents.memory.router import _doc_lines


def test_doc_lines_includes_evidence_entities():
    rendered = _doc_lines(
        [
            {
                "citation": "SITE_A / manual.pdf / p.4 / machine=MACHINE_A1",
                "text": "Check spindle imbalance before adjusting feed rate.",
                "evidence_entities": [
                    {"id": "entity-1", "name": "Spindle Imbalance", "type": "Symptom"},
                    {"id": "entity-2", "name": "Feed Rate", "type": "Parameter"},
                ],
            }
        ]
    )

    assert "Spindle Imbalance (Symptom)" in rendered
    assert "Feed Rate (Parameter)" in rendered
    assert "Check spindle imbalance" in rendered


def test_doc_lines_handles_missing_matches():
    assert _doc_lines([]) == "No documentation context available."
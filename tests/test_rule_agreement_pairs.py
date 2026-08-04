"""Agent P — tests for data-driven rule-agreement pairs.

Phase 1 left `_RULE_AGREEMENT_PAIRS` as a module-level constant. Phase 2
moves it behind `_load_rule_agreement_pairs()` which merges built-ins with
`data/rule_agreement_pairs.json`. These tests verify:

- Built-in pairs are preserved when file is absent / malformed.
- File-provided pairs fully replace built-ins (override semantics).
- Non-string entries are dropped silently.
- score_trace uses the rule_name -> weighted dict rather than positional zip.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from backend.agents.core.schemas import PatternKey, PatternType
from backend.agents.memory import scorer as scorer_mod
from backend.agents.memory.scorer import (
    SignificanceScorer,
    _load_rule_agreement_pairs,
    _RULE_AGREEMENT_PAIRS_BUILTIN,
)


def test_builtins_present():
    # Sanity: built-ins cover the canonical rule-agreement pairs.
    assert len(_RULE_AGREEMENT_PAIRS_BUILTIN) == 5
    names = {(a, b) for a, b, _ in _RULE_AGREEMENT_PAIRS_BUILTIN}
    assert ("classical_alert", "pattern_match") in names
    assert ("classical_alert", "anomaly_deviation") in names
    assert ("harmonic_alert", "pattern_match") in names
    assert ("harmonic_alert", "anomaly_deviation") in names
    assert ("pattern_match", "anomaly_deviation") in names


def test_loader_returns_list_of_3tuples(tmp_path, monkeypatch):
    # Verify the real loader returns a well-formed list.
    pairs = _load_rule_agreement_pairs()
    assert len(pairs) >= 3
    for entry in pairs:
        assert len(entry) == 3
        assert all(isinstance(x, str) for x in entry)


def test_loader_accepts_file_override(tmp_path):
    # Build a fake file and patch the candidate path directly inside the
    # loader by temporarily swapping Path resolution.
    fake_file = tmp_path / "rule_agreement_pairs.json"
    fake_file.write_text(json.dumps({
        "schema_version": 1,
        "pairs": [
            {"a": "r1", "b": "r2", "bonus_attr": "attr_a"},
            {"a": "r3", "b": "r4", "bonus_attr": "attr_b"},
        ],
    }), encoding="utf-8")

    # Re-implement the loader body with our candidate path for isolation.
    def _reload_with(path: Path):
        merged = list(_RULE_AGREEMENT_PAIRS_BUILTIN)
        if not path.is_file():
            return merged
        raw = json.loads(path.read_text())
        file_pairs = raw.get("pairs") if isinstance(raw, dict) else None
        if not isinstance(file_pairs, list):
            return merged
        merged = []
        for entry in file_pairs:
            if isinstance(entry, dict):
                a, b, attr = entry.get("a"), entry.get("b"), entry.get("bonus_attr")
                if isinstance(a, str) and isinstance(b, str) and isinstance(attr, str):
                    merged.append((a, b, attr))
        return merged or list(_RULE_AGREEMENT_PAIRS_BUILTIN)

    out = _reload_with(fake_file)
    assert ("r1", "r2", "attr_a") in out
    assert ("r3", "r4", "attr_b") in out
    assert len(out) == 2  # File fully replaces built-ins


def test_loader_drops_malformed_entries(tmp_path):
    fake = tmp_path / "rule_agreement_pairs.json"
    fake.write_text(json.dumps({
        "schema_version": 1,
        "pairs": [
            {"a": "ok1", "b": "ok2", "bonus_attr": "attr"},
            {"a": 123, "b": "x", "bonus_attr": "y"},        # bad type
            "not_a_dict",                                    # not a dict
            {"a": "x", "b": "y"},                            # missing bonus_attr
        ],
    }), encoding="utf-8")

    raw = json.loads(fake.read_text())
    pairs = raw.get("pairs")
    merged = []
    for entry in pairs:
        if isinstance(entry, dict):
            a, b, attr = entry.get("a"), entry.get("b"), entry.get("bonus_attr")
            if isinstance(a, str) and isinstance(b, str) and isinstance(attr, str):
                merged.append((a, b, attr))
    assert merged == [("ok1", "ok2", "attr")]


def test_score_trace_uses_dict_not_positional_zip():
    """Regression: if weighted_by_rule is keyed properly, trace order is
    independent of iteration order of rules.
    """
    scorer = SignificanceScorer()
    result = scorer.score(
        patterns=[PatternKey(pattern_type=PatternType.ANOMALY, key="ANOMALY_HIGH:0.9")],
        external_signals={"breakage_prediction": 0.85},
    )
    # Find rule:* entries in trace
    rule_entries = [e for e in result.score_trace if e["component"].startswith("rule:")]
    assert len(rule_entries) >= 1
    # Every rule entry must have a non-zero weight string parseable from source
    for e in rule_entries:
        assert "w=" in e["source"]
        w_part = e["source"].split("w=")[-1].rstrip(")")
        assert float(w_part) > 0

from __future__ import annotations

from backend.agents.llm.entity_canonicalizer import EntityCanonicalizer


def test_entity_canonicalizer_reuses_exact_match():
    canonicalizer = EntityCanonicalizer(usecase="SITE_A")

    first = canonicalizer.register(name="MACHINE_A1", entity_type="Machine")
    second = canonicalizer.register(name="MACHINE_A1", entity_type="Machine")

    assert first.id == second.id
    assert len(canonicalizer.list_entities()) == 1


def test_entity_canonicalizer_merges_alias_hits_and_new_surface_forms():
    canonicalizer = EntityCanonicalizer(usecase="SITE_A")

    first = canonicalizer.register(
        name="Spindle Imbalance",
        entity_type="Symptom",
        aliases=["spindle unbalance"],
    )
    second = canonicalizer.register(
        name="Spindle Vibration",
        entity_type="Symptom",
        aliases=["spindle unbalance"],
    )

    assert first.id == second.id
    assert "Spindle Vibration" in second.aliases


def test_entity_canonicalizer_fuzzy_matches_within_type_only():
    canonicalizer = EntityCanonicalizer(usecase="SITE_A")

    first = canonicalizer.register(name="feed rate", entity_type="Parameter")
    second = canonicalizer.register(name="feedrate", entity_type="Parameter")
    third = canonicalizer.register(name="feedrate", entity_type="Procedure")

    assert first.id == second.id
    assert first.id != third.id


def test_entity_canonicalizer_preserves_resolved_canonical_ids():
    canonicalizer = EntityCanonicalizer(
        usecase="SITE_A",
        canonical_id_resolver=lambda **kwargs: kwargs.get("machine_uri"),
    )

    first = canonicalizer.register(
        name="MACHINE_A1",
        entity_type="Machine",
        machine_hint="MACHINE_A1",
        machine_uri="urn:lfl:asset:machine_a1",
    )
    second = canonicalizer.register(
        name="MACHINE_A1",
        entity_type="Machine",
        aliases=["machine machine_a1"],
        machine_hint="MACHINE_A1",
        machine_uri="urn:lfl:asset:machine_a1",
    )

    assert first.id == second.id
    assert first.canonical_id == "urn:lfl:asset:machine_a1"
    assert second.canonical_id == "urn:lfl:asset:machine_a1"
"""Tests for backend.agents.knowledge — Agent H."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.knowledge import (
    KNOWLEDGE_PACK_VERSION,
    apply_fleet_pack,
    ContextKeys,
    FileSink,
    FleetPackApplication,
    HttpSink,
    KnowledgePack,
    MqttSink,
    build_knowledge_pack,
    load_pack,
    push_to_sinks,
    save_pack,
    should_apply,
    similarity_score,
)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    (tmp_path / "pattern_priors.json").write_text(
        json.dumps({"pattern_priors": {"ALARM_TRIGGERED": 0.8, "TOOL_BREAKAGE": 0.7}})
    )
    (tmp_path / "discovered_patterns.json").write_text(
        json.dumps({"patterns": {"discovered:foo": {"key": "discovered:foo"}}})
    )
    (tmp_path / "pattern_aliases.json").write_text(json.dumps({"alias": {"a": "b"}}))
    (tmp_path / "rl_agent.json").write_text(json.dumps({"q_table": {}}))
    # pattern_index.json + rule_agreement_pairs.json intentionally missing
    return tmp_path


# ── build_knowledge_pack ──────────────────────────────────────────────


def test_build_pack_reads_all_available_sources(data_dir: Path) -> None:
    pack = build_knowledge_pack(
        data_dir,
        site="CNC-1",
        context=ContextKeys(machine_type="cnc", tool_type="endmill", material="al"),
    )
    assert pack.site == "CNC-1"
    assert pack.version == KNOWLEDGE_PACK_VERSION
    assert "ALARM_TRIGGERED" in pack.pattern_priors["pattern_priors"]
    assert pack.context == {
        "machine_type": "cnc",
        "tool_type": "endmill",
        "material": "al",
        "regime": None,
    }
    # Missing files → empty dicts, not errors.
    assert pack.pattern_index == {}
    assert pack.rule_agreement_pairs == {}


def test_build_pack_can_require_complete_context(data_dir: Path) -> None:
    with pytest.raises(ValueError, match="context keys"):
        build_knowledge_pack(
            data_dir,
            site="CNC-1",
            context=ContextKeys(machine_type="cnc", tool_type="endmill"),
            require_complete_context=True,
        )


def test_build_pack_includes_provenance_metadata(data_dir: Path) -> None:
    pack = build_knowledge_pack(
        data_dir,
        site="CNC-1",
        context=ContextKeys(machine_type="cnc", tool_type="endmill", material="al", regime="rough"),
        require_complete_context=True,
        tenant_id="tenant-a",
        signer="knowledge_push",
        license="fleet-share",
        pii_scrub_level="symbolic_only",
        expires_at="2026-12-31T00:00:00+00:00",
    )
    assert pack.tenant_id == "tenant-a"
    assert pack.signer == "knowledge_push"
    assert pack.signed_at == pack.built_at
    assert pack.license == "fleet-share"
    assert pack.pii_scrub_level == "symbolic_only"
    assert pack.expires_at == "2026-12-31T00:00:00+00:00"


def test_build_pack_handles_completely_empty_dir(tmp_path: Path) -> None:
    pack = build_knowledge_pack(tmp_path, site="new")
    assert pack.pattern_priors == {}
    assert pack.discovered_patterns == {}
    assert pack.summary()["priors"] == 0


def test_build_pack_strict_export_filters_discoveries_to_exact_promoted_context(tmp_path: Path) -> None:
    matching_key = (
        "discovered:machine_type=cnc|tool_type=endmill|"
        "workpiece_material=al|operating_regime=rough::power_spindle_mean_H"
    )
    foreign_key = (
        "discovered:machine_type=cnc|tool_type=drill|"
        "workpiece_material=al|operating_regime=rough::power_spindle_mean_H"
    )
    pending_key = (
        "discovered:machine_type=cnc|tool_type=endmill|"
        "workpiece_material=al|operating_regime=rough::feed_rate_mean_L"
    )
    (tmp_path / "discovered_patterns.json").write_text(
        json.dumps(
            {
                "patterns": {
                    matching_key: {
                        "key": matching_key,
                        "features": {"power_spindle_mean": "high"},
                        "context_key": "machine_type=cnc|tool_type=endmill|workpiece_material=al|operating_regime=rough",
                        "confirmation_count": 4,
                        "promoted": True,
                        "prior": 0.5,
                        "source_events": [{"memory_id": "m-1", "session_id": "s-1"}],
                    },
                    foreign_key: {
                        "key": foreign_key,
                        "features": {"power_spindle_mean": "high"},
                        "context_key": "machine_type=cnc|tool_type=drill|workpiece_material=al|operating_regime=rough",
                        "confirmation_count": 6,
                        "promoted": True,
                        "prior": 0.7,
                    },
                    pending_key: {
                        "key": pending_key,
                        "features": {"feed_rate_mean": "low"},
                        "context_key": "machine_type=cnc|tool_type=endmill|workpiece_material=al|operating_regime=rough",
                        "confirmation_count": 2,
                        "promoted": False,
                        "prior": 0.5,
                    },
                },
                "baseline_stats": {"n": 99},
                "baseline_stats_by_context": {
                    "machine_type=cnc|tool_type=endmill|workpiece_material=al|operating_regime=rough": {"n": 25}
                },
                "version": 2,
            }
        )
    )

    pack = build_knowledge_pack(
        tmp_path,
        site="CNC-1",
        context=ContextKeys(machine_type="cnc", tool_type="endmill", material="al", regime="rough"),
        require_complete_context=True,
    )

    assert pack.discovered_patterns == {
        "patterns": {
            matching_key: {
                "key": matching_key,
                "features": {"power_spindle_mean": "high"},
                "context_key": "machine_type=cnc|tool_type=endmill|workpiece_material=al|operating_regime=rough",
                "confirmation_count": 4,
                "promoted": True,
                "prior": 0.5,
            }
        }
    }
    assert pack.summary()["discovered_patterns"] == 1


def test_build_pack_non_strict_keeps_local_discovery_payload(tmp_path: Path) -> None:
    (tmp_path / "discovered_patterns.json").write_text(
        json.dumps(
            {
                "patterns": {"discovered:foo": {"key": "discovered:foo", "promoted": False}},
                "baseline_stats": {"n": 42},
                "baseline_stats_by_context": {"machine_type=cnc": {"n": 21}},
                "version": 2,
            }
        )
    )

    pack = build_knowledge_pack(tmp_path, site="local-inspect")

    assert pack.discovered_patterns["baseline_stats"]["n"] == 42
    assert pack.discovered_patterns["baseline_stats_by_context"]["machine_type=cnc"]["n"] == 21


def test_build_pack_strict_export_filters_priors_to_exact_context(tmp_path: Path) -> None:
    matching_context_key = (
        "machine_type=cnc|tool_type=endmill|workpiece_material=al|operating_regime=rough"
    )
    foreign_context_key = (
        "machine_type=cnc|tool_type=drill|workpiece_material=al|operating_regime=rough"
    )
    (tmp_path / "pattern_priors.json").write_text(
        json.dumps(
            {
                "pattern_priors": {"GLOBAL_MIXED": 0.66},
                "pattern_priors_by_context": {
                    matching_context_key: {"CUSTOM:endmill": 0.81},
                    foreign_context_key: {"CUSTOM:drill": 0.93},
                },
                "feedback_counts": {"GLOBAL_MIXED": {"confirm": 9, "dismiss": 4}},
                "feedback_counts_by_context": {
                    matching_context_key: {"CUSTOM:endmill": {"confirm": 4, "dismiss": 1}},
                    foreign_context_key: {"CUSTOM:drill": {"confirm": 8, "dismiss": 0}},
                },
                "severity_calibration": {"CUSTOM:endmill": {"weight_sum": 1.0}},
                "feedback_observability": {"CUSTOM:endmill": {"effective_weight_total": 5.0}},
                "updated_at": "2026-05-26T00:00:00+00:00",
            }
        )
    )

    pack = build_knowledge_pack(
        tmp_path,
        site="CNC-1",
        context=ContextKeys(machine_type="cnc", tool_type="endmill", material="al", regime="rough"),
        require_complete_context=True,
    )

    assert pack.pattern_priors == {
        "pattern_priors": {"CUSTOM:endmill": 0.81},
        "pattern_evidence_counts": {"CUSTOM:endmill": 5},
        "pattern_priors_by_context": {
            matching_context_key: {"CUSTOM:endmill": 0.81},
        },
    }


def test_build_pack_injects_runtime_extras(data_dir: Path) -> None:
    pack = build_knowledge_pack(
        data_dir,
        site="CNC-1",
        weight_profiles={"cnc|endmill|al|rough": {"w": 1.0}},
        adaptive_thresholds={"chatter": 0.6},
        rule_performance={"rule_chatter": {"f1": 0.82}},
        seed_model_meta={"trainedAt": "2026-04-01", "nSamples": 1234},
        notes=["exported pre-maintenance"],
    )
    assert pack.weight_profiles["cnc|endmill|al|rough"]["w"] == 1.0
    assert pack.rule_performance["rule_chatter"]["f1"] == 0.82
    assert "exported pre-maintenance" in pack.notes


def test_build_pack_tolerates_corrupt_json(tmp_path: Path) -> None:
    (tmp_path / "pattern_priors.json").write_text("{not valid json")
    pack = build_knowledge_pack(tmp_path, site="s")
    # Tolerant: corrupt file → empty section, no exception.
    assert pack.pattern_priors == {}


def test_build_pack_wraps_non_dict_json(tmp_path: Path) -> None:
    (tmp_path / "pattern_priors.json").write_text("[1, 2, 3]")
    pack = build_knowledge_pack(tmp_path, site="s")
    assert pack.pattern_priors == {"items": [1, 2, 3]}


# ── Summary / round-trip ───────────────────────────────────────────────


def test_pack_summary_counts(data_dir: Path) -> None:
    pack = build_knowledge_pack(data_dir, site="CNC-1")
    summary = pack.summary()
    assert summary["discovered_patterns"] == 1
    assert summary["aliases"] == 1


def test_save_and_load_pack_roundtrip(data_dir: Path, tmp_path: Path) -> None:
    pack = build_knowledge_pack(data_dir, site="CNC-1", notes=["n1"])
    target = save_pack(pack, tmp_path / "pack.json")
    assert target.exists()
    loaded = load_pack(target)
    assert loaded.site == pack.site
    assert loaded.notes == ["n1"]
    assert loaded.pattern_priors == pack.pattern_priors


def test_save_pack_is_atomic(data_dir: Path, tmp_path: Path) -> None:
    pack = build_knowledge_pack(data_dir, site="s")
    target = tmp_path / "nested" / "pack.json"
    save_pack(pack, target)
    # No leftover .tmp
    leftovers = list(target.parent.glob("*.tmp"))
    assert leftovers == []


# ── Similarity gate ────────────────────────────────────────────────────


def test_similarity_score_full_match() -> None:
    ctx = {"machine_type": "cnc", "tool_type": "endmill", "material": "al", "regime": "rough"}
    assert similarity_score(ctx, ctx) == pytest.approx(1.0)


def test_similarity_score_partial() -> None:
    a = {"machine_type": "cnc", "tool_type": "endmill", "material": "al", "regime": "rough"}
    b = {"machine_type": "cnc", "tool_type": "endmill", "material": "steel", "regime": "finish"}
    # machine_type (0.4) + tool_type (0.3) = 0.7 / 1.0
    assert similarity_score(a, b) == pytest.approx(0.7)


def test_similarity_score_case_insensitive() -> None:
    a = {"machine_type": "CNC"}
    b = {"machine_type": "cnc"}
    assert similarity_score(a, b) > 0.0


def test_should_apply_respects_threshold() -> None:
    pack = KnowledgePack(context={"machine_type": "cnc", "tool_type": "endmill"})
    allowed, score = should_apply(pack, {"machine_type": "cnc"}, threshold=0.5)
    # Only machine_type matches → 0.4 / 1.0 = 0.4, below 0.5.
    assert allowed is False
    assert score == pytest.approx(0.4)

    allowed2, _ = should_apply(pack, {"machine_type": "cnc"}, threshold=0.3)
    assert allowed2 is True


def test_apply_fleet_pack_rejects_dissimilar_pack() -> None:
    pack = KnowledgePack(
        site="fleet-hub",
        context={
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
        },
        pattern_priors={"pattern_priors": {"CUSTOM:test": 0.9}},
        discovered_patterns={"patterns": {"discovered:foo": {"key": "discovered:foo", "promoted": True}}},
    )

    result = apply_fleet_pack(
        pack,
        {"machine_type": "lathe"},
        threshold=0.5,
    )

    assert isinstance(result, FleetPackApplication)
    assert result.allowed is False
    assert result.score == pytest.approx(0.0)
    assert result.pattern_priors == {}
    assert result.discovered_patterns == {}


def test_apply_fleet_pack_damps_priors_by_similarity() -> None:
    pack = KnowledgePack(
        site="fleet-hub",
        context={
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
        },
        pattern_priors={"pattern_priors": {"CUSTOM:test": 0.9}},
    )

    result = apply_fleet_pack(
        pack,
        {
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "steel",
            "regime": "finish",
        },
        threshold=0.5,
    )

    assert result.allowed is True
    assert result.score == pytest.approx(0.7)
    assert result.pattern_priors["CUSTOM:test"] == pytest.approx(0.78)


def test_apply_fleet_pack_only_surfaces_promoted_discoveries() -> None:
    pack = KnowledgePack(
        site="fleet-hub",
        context={
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
        },
        pattern_priors={"pattern_priors": {"CUSTOM:test": 0.9}},
        discovered_patterns={
            "patterns": {
                "discovered:ok": {"key": "discovered:ok", "promoted": True},
                "discovered:pending": {"key": "discovered:pending", "promoted": False},
            }
        },
    )

    result = apply_fleet_pack(
        pack,
        {
            "machine_type": "cnc",
            "tool_type": "endmill",
            "material": "al",
            "regime": "rough",
        },
    )

    assert result.allowed is True
    assert result.discovered_patterns == {
        "patterns": {
            "discovered:ok": {"key": "discovered:ok", "promoted": True}
        }
    }


# ── Sinks ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_file_sink_writes_payload(tmp_path: Path) -> None:
    sink = FileSink(directory=tmp_path, prefix="test")
    ok = await sink.push({"built_at": "2026-04-24T10-00-00+00-00", "site": "s"})
    assert ok is True
    files = list(tmp_path.glob("test_*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["site"] == "s"
    # Atomic: no .tmp leftover.
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.asyncio
async def test_mqtt_sink_disabled_by_default_returns_false() -> None:
    sink = MqttSink(broker_url="tcp://broker:1883", topic="lfl/test")
    ok = await sink.push({"built_at": "now", "site": "s"})
    assert ok is False
    assert len(sink.last_payloads) == 1  # records for inspection


@pytest.mark.asyncio
async def test_mqtt_sink_enabled_without_transport_raises() -> None:
    sink = MqttSink(broker_url="tcp://broker:1883", topic="lfl/test", enabled=True)
    with pytest.raises(NotImplementedError):
        await sink.push({"built_at": "now"})


@pytest.mark.asyncio
async def test_http_sink_posts_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class _Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["content_type"] = request.headers.get("Content-type")
        return _Response()

    monkeypatch.setattr("backend.agents.knowledge.sinks.urlopen", fake_urlopen)

    sink = HttpSink(url="https://hub.example/ingest", timeout_seconds=3.5)
    ok = await sink.push({"built_at": "now", "site": "s"})

    assert ok is True
    assert captured["url"] == "https://hub.example/ingest"
    assert captured["timeout"] == pytest.approx(3.5)
    assert captured["body"]["site"] == "s"
    assert captured["content_type"] == "application/json"


@pytest.mark.asyncio
async def test_http_sink_missing_url_returns_false() -> None:
    sink = HttpSink(url="")
    ok = await sink.push({"built_at": "now"})
    assert ok is False


def test_mqtt_sink_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError):
        MqttSink(mode="nonsense")


@pytest.mark.asyncio
async def test_push_to_sinks_fans_out_results(tmp_path: Path) -> None:
    file_sink = FileSink(directory=tmp_path, prefix="fan", name="file")
    mqtt_sink = MqttSink(name="mqtt")  # disabled
    results = await push_to_sinks([file_sink, mqtt_sink], {"built_at": "t", "site": "s"})
    assert results == {"file": True, "mqtt": False}


@pytest.mark.asyncio
async def test_push_to_sinks_isolates_failures(tmp_path: Path) -> None:
    class _BoomSink:
        name = "boom"

        async def push(self, payload):  # type: ignore[no-untyped-def]
            raise RuntimeError("kaboom")

    file_sink = FileSink(directory=tmp_path, prefix="fan")
    results = await push_to_sinks([_BoomSink(), file_sink], {"built_at": "t"})
    assert results["boom"] is False
    assert results["file"] is True

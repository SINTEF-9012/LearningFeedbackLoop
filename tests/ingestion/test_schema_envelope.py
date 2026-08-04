from __future__ import annotations

from backend.agents.memory.feature_stream_bridge import (
    _augment_payload_with_provider,
    _coerce_feature_payload,
    _merge_session_metadata,
    _should_apply_stoppage_predictor,
    create_memory_event_from_feature,
)
from backend.agents.sindit.asset_catalog import build_machine_asset
from backend.ingestion.schema import FrameEnvelope, envelope_to_dict


def test_envelope_to_dict_omits_none_fields():
    envelope = FrameEnvelope(
        kind="tag_sample",
        session_id="session-1",
        ts_unix=10.0,
        position=5,
        fs=1.0,
        signals={"Power_Spindle": 12.5},
    )

    payload = envelope_to_dict(envelope)

    assert payload["kind"] == "tag_sample"
    assert payload["session_id"] == "session-1"
    assert payload["signals"] == {"Power_Spindle": 12.5}
    assert payload["schema_version"] == 1
    assert payload["source"] == "unknown"
    assert "frame" not in payload
    assert "window_seconds" not in payload


def test_coerce_feature_payload_copies_legacy_dict():
    legacy = {"session_id": "session-2", "position": 3, "external_signals": {"x": 1.0}}

    payload = _coerce_feature_payload(legacy)

    assert payload == legacy
    assert payload is not legacy


def test_frame_envelope_round_trips_into_memory_event():
    envelope = FrameEnvelope(
        kind="tag_sample",
        session_id="session-3",
        ts_unix=20.0,
        position=5,
        fs=2.0,
        window_seconds=1.5,
        signals={"Power_Spindle": 8.0},
        patterns=["FAULT:breakage"],
        external_signals={"breakage_prediction": 0.7},
        batch={"batch_id": "batch-7", "unit_index": 1, "unit_count": 4, "recipe_id": "recipe-a"},
    )

    payload = _coerce_feature_payload(envelope)
    event = create_memory_event_from_feature(
        payload["session_id"],
        payload,
        {"fs": payload["fs"]},
    )

    assert event.session_id == "session-3"
    assert event.time_range.i0 == 2
    assert event.time_range.i1 == 5
    assert event.time_range.t0 == 1.0
    assert event.time_range.t1 == 2.5
    assert [pattern.key for pattern in event.patterns] == ["FAULT:breakage"]
    assert event.external_signals["breakage_prediction"] == 0.7
    assert payload["batch"]["batch_id"] == "batch-7"
    assert event.batch is not None
    assert event.batch.batch_id == "batch-7"
    assert event.batch.unit_index == 1
    assert event.batch.unit_count == 4
    assert event.batch.recipe_id == "recipe-a"


def test_signal_only_frame_envelope_synthesizes_legacy_frame_payload():
    envelope = FrameEnvelope(
        kind="tag_sample",
        session_id="session-4",
        ts_unix=30.0,
        position=9,
        fs=1.0,
        signals={"Power_Spindle": 14.0, "Feed_Rate_Actual": 120.0},
    )

    payload = _coerce_feature_payload(envelope)

    assert payload["signals"] == {"Power_Spindle": 14.0, "Feed_Rate_Actual": 120.0}
    assert payload["frame"] == {
        "t": 9.0,
        "i": 9,
        "fs": 1.0,
        "Power_Spindle": 14.0,
        "Feed_Rate_Actual": 120.0,
    }


def test_raw_tag_sample_payload_does_not_synthesize_generic_metrics():
    envelope = FrameEnvelope(
        kind="tag_sample",
        session_id="session-5",
        ts_unix=40.0,
        position=12,
        fs=1.0,
        signals={
            "Power_Spindle": 18.0,
            "Feed_Rate_Actual": 120.0,
            "Spindle_Speed_Actual": 600.0,
        },
    )

    payload = _augment_payload_with_provider(_coerce_feature_payload(envelope))

    assert "metrics" not in payload
    assert "patterns" not in payload


def test_merge_session_metadata_preserves_casedata_and_live_context():
    envelope = FrameEnvelope(
        kind="tag_sample",
        session_id="session-6",
        ts_unix=50.0,
        position=15,
        fs=1.0,
        signals={
            "Feed_Rate_Actual": 150.0,
            "Spindle_Speed_Actual": 720.0,
        },
        metadata={
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
            "casedata": {
                "operation_id": "OF00001",
                "tool_id": "CASE_C1",
            },
        },
    )

    payload = _coerce_feature_payload(envelope)
    session_meta = _merge_session_metadata({}, payload)
    event = create_memory_event_from_feature(payload["session_id"], payload, session_meta)

    assert event.cutting_context is not None
    assert event.cutting_context.tool_id == "CASE_C1"
    assert event.cutting_context.spindle_speed == 720.0
    assert event.cutting_context.feed_rate == 150.0
    assert event.cutting_context.extra["operation_id"] == "OF00001"


def test_merge_session_metadata_resolves_machine_twin_and_tool_context_for_live_casedata():
    machine_id = "Site_b - MACHINE_B1 - CASE_B1"
    envelope = FrameEnvelope(
        kind="tag_sample",
        session_id="session-6c",
        ts_unix=60.0,
        position=18,
        fs=1.0,
        signals={
            "Feed_Rate_Actual": 240.0,
            "Spindle_Speed_Actual": 810.0,
            "Tool_Number": 6.0,
        },
        metadata={
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
            "casedata": {
                "operation_id": "OF00001",
                "case_dir": machine_id,
            },
        },
    )

    payload = _coerce_feature_payload(envelope)
    session_meta = _merge_session_metadata({}, payload)
    event = create_memory_event_from_feature(payload["session_id"], payload, session_meta)

    assert session_meta["dataset_id"] == "site_b_casedata"
    assert session_meta["machine_family"] == "builder_b12"
    assert session_meta["machine_uri"] == build_machine_asset(machine_id, label=machine_id)["uri"]
    assert session_meta["sindit_asset_iri"] == session_meta["machine_uri"]
    assert event.cutting_context is not None
    assert event.cutting_context.machine_id == machine_id
    assert event.cutting_context.tool_id == "T06"
    assert event.cutting_context.tool_diameter == 65.0
    assert event.cutting_context.num_teeth == 1
    assert event.cutting_context.extra["sindit_tool_iri"] == "urn:lfl:tool:builder_b12-t6"


def test_merge_session_metadata_resolves_generic_live_source_to_machine_and_tool():
    machine_id = "SITE_C - MACHINE_C1 - CASE_C1"
    envelope = FrameEnvelope(
        kind="tag_sample",
        session_id="session-6d",
        ts_unix=65.0,
        position=20,
        fs=1.0,
        signals={
            "tool_number": 2432.0,
            "feed_rate": 90.0,
            "spindle_speed": 500.0,
        },
        metadata={
            "sample_frequency": 1.0,
            "source": "mqtt",
            "machine_id": machine_id,
            "mqtt": {
                "topic": "site_c/live",
            },
        },
    )

    payload = _coerce_feature_payload(envelope)
    session_meta = _merge_session_metadata({}, payload)
    event = create_memory_event_from_feature(payload["session_id"], payload, session_meta)

    assert session_meta["dataset_id"] == "site_c_casedata"
    assert session_meta["machine_family"] == "press_c-20-0482-010"
    assert session_meta["machine_uri"] == build_machine_asset(machine_id, label=machine_id)["uri"]
    assert event.cutting_context is not None
    assert event.cutting_context.machine_id == machine_id
    assert event.cutting_context.tool_diameter == 20.5
    assert event.cutting_context.tool_id == "T2432"
    assert event.cutting_context.extra["sindit_tool_iri"] == "urn:lfl:tool:press_c-20-0482-010-t2432"


def test_simulated_casedata_payload_skips_stoppage_predictor_guard():
    envelope = FrameEnvelope(
        kind="tag_sample",
        session_id="session-6b",
        ts_unix=55.0,
        position=16,
        fs=1.0,
        signals={"Feed_Rate_Actual": 150.0},
        metadata={
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
            "casedata": {"operation_id": "OF00001"},
        },
    )

    payload = _coerce_feature_payload(envelope)
    session_meta = _merge_session_metadata({}, payload)

    assert _should_apply_stoppage_predictor(session_meta, payload) is False


def test_named_ratio_pattern_is_bucketed_for_window_payload():
    payload = _augment_payload_with_provider(
        {
            "session_id": "session-7",
            "frame": {
                "fs": 10.0,
                "Power_Spindle": [3.7, 3.7, 3.7, 3.7],
                "Power_Y": [1.0, 1.0, 1.0, 1.0],
            },
        }
    )

    assert "patterns" in payload
    assert "RATIO_Power_Spindle_Power_Y:2-5" in payload["patterns"]
    assert all(not pattern.startswith("RATIO_ch0_ch1:") for pattern in payload["patterns"])


def test_near_zero_ratio_denominator_is_suppressed_for_window_payload():
    payload = _augment_payload_with_provider(
        {
            "session_id": "session-8",
            "frame": {
                "fs": 10.0,
                "Power_Spindle": [3.0, 3.0, 3.0, 3.0],
                "Power_Y": [0.0, 0.0, 0.0, 0.0],
            },
        }
    )

    assert all(not pattern.startswith("RATIO_") for pattern in payload.get("patterns", []))
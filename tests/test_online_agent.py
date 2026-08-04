from backend.agents.processing.online import OnlineAgent
from backend.ingestion.schema import FrameEnvelope


def test_online_agent_extracts_features_from_frame_envelope_signals():
    agent = OnlineAgent()
    envelope = FrameEnvelope(
        kind="tag_sample",
        session_id="session-1",
        ts_unix=1.0,
        position=5,
        fs=1.0,
        signals={"spindle_speed": 1200.0, "feed_rate": 80.0},
        metadata={},
    )

    features = agent._extract_features(envelope)

    assert features["spindle_speed"] == 1200.0
    assert features["feed_rate"] == 80.0


def test_online_agent_reads_label_from_envelope_metadata():
    agent = OnlineAgent()
    envelope = FrameEnvelope(
        kind="tag_sample",
        session_id="session-1",
        ts_unix=1.0,
        position=5,
        fs=1.0,
        signals={"spindle_speed": 1200.0},
        metadata={"label": "normal"},
    )

    assert agent._event_get(envelope, "label") == "normal"


def test_online_agent_extracts_features_from_legacy_dict_payload():
    agent = OnlineAgent()

    features = agent._extract_features(
        {
            "payload": {
                "freqs": [1.0, 2.0, 3.0],
                "ensemble_score": 0.8,
            }
        }
    )

    assert features["freqs_mean"] == 2.0
    assert features["freqs_std"] > 0.0
    assert features["ensemble_score"] == 0.8
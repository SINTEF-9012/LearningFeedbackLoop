from __future__ import annotations

import json

import backend.agents.llm.entity_extractor as extractor_module
from backend.agents.llm.entity_extractor import EntityExtractor


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_entity_extractor_filters_to_closed_vocab(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "entities": [
                                {"name": "MACHINE_A1", "type": "Machine", "aliases": ["MACHINE_A1"]},
                                {"name": "Chatter", "type": "Symptom", "aliases": []},
                                {"name": "Mystery", "type": "UnknownThing", "aliases": []},
                            ],
                            "relations": [
                                {
                                    "src_name": "Chatter",
                                    "src_type": "Symptom",
                                    "dst_name": "MACHINE_A1",
                                    "dst_type": "Machine",
                                    "rel_type": "APPLIES_TO",
                                    "confidence": 0.8,
                                },
                                {
                                    "src_name": "Mystery",
                                    "src_type": "UnknownThing",
                                    "dst_name": "MACHINE_A1",
                                    "dst_type": "Machine",
                                    "rel_type": "CAUSES",
                                    "confidence": 0.7,
                                },
                            ],
                        }
                    )
                }
            }
        ]
    }

    monkeypatch.setattr(extractor_module.httpx, "post", lambda *args, **kwargs: _FakeResponse(payload))

    extractor = EntityExtractor(
        enabled=True,
        groq_api_key="secret",
        groq_api_url="https://groq.example.test",
        model="test-model",
        timeout=1.0,
    )
    result = extractor.extract_from_chunk(
        "MACHINE_A1 operator manual: chatter may occur during roughing and applies to this machine.",
        usecase="SITE_A",
        machine_hint="MACHINE_A1",
    )

    assert [entity.type for entity in result.entities] == ["Machine", "Symptom"]
    assert len(result.relations) == 1
    assert result.relations[0].rel_type == "APPLIES_TO"
    assert result.relations[0].confidence == 0.8
    assert result.warnings


def test_entity_extractor_returns_warning_on_invalid_json(monkeypatch):
    payload = {"choices": [{"message": {"content": "not-json"}}]}
    monkeypatch.setattr(extractor_module.httpx, "post", lambda *args, **kwargs: _FakeResponse(payload))

    extractor = EntityExtractor(
        enabled=True,
        groq_api_key="secret",
        groq_api_url="https://groq.example.test",
    )
    result = extractor.extract_from_chunk(
        "This is a long enough chunk to force extraction even though the model will reply incorrectly.",
        usecase="SITE_A",
    )

    assert result.entities == []
    assert result.relations == []
    assert result.warnings
    assert result.warnings[0].startswith("extractor_error:")


def test_entity_extractor_skips_disabled_calls(monkeypatch):
    called = {"value": False}

    def _fake_post(*args, **kwargs):
        called["value"] = True
        return _FakeResponse({})

    monkeypatch.setattr(extractor_module.httpx, "post", _fake_post)

    extractor = EntityExtractor(enabled=False, groq_api_key="secret")
    result = extractor.extract_from_chunk(
        "This chunk is long enough to call the extractor, but the feature is disabled.",
        usecase="SITE_A",
    )

    assert result.entities == []
    assert result.relations == []
    assert called["value"] is False
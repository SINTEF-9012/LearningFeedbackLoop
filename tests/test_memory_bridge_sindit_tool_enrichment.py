from __future__ import annotations

from typing import Any, Dict, List, Tuple

from backend.agents.memory import feature_stream_bridge


class _FakeSinditProvider:
    def __init__(self):
        self.calls: List[Tuple[str, str | None]] = []

    async def enrich_context(self, context_dict: Dict[str, Any], asset_iri: str | None = None):
        self.calls.append(("machine", asset_iri))
        if asset_iri:
            context_dict["spindle_speed"] = 1800.0
        return context_dict

    async def enrich_tool_properties(self, context_dict: Dict[str, Any], *, tool_iri: str):
        self.calls.append(("tool", tool_iri))
        context_dict["tool_diameter"] = 65.0
        context_dict["num_teeth"] = 1
        return context_dict


def test_bridge_applies_machine_then_tool_sindit_enrichment():
    provider = _FakeSinditProvider()
    feature_stream_bridge.set_sindit_provider(provider)
    try:
        event = feature_stream_bridge.create_memory_event_from_feature(
            session_id="sess-1",
            payload={},
            session_meta={
                "sample_frequency": 1.0,
                "machine_iri": "urn:lfl:asset:site_b-1",
                "casedata": {
                    "cutting_context": {
                        "tool_id": "T6",
                        "extra": {
                            "sindit_tool_iri": "urn:lfl:tool:builder_b12-t6",
                        },
                    },
                },
            },
        )

        assert event.cutting_context is not None
        assert event.cutting_context.tool_id == "T6"
        assert event.cutting_context.spindle_speed == 1800.0
        assert event.cutting_context.tool_diameter == 65.0
        assert event.cutting_context.num_teeth == 1
        assert provider.calls == [
            ("machine", "urn:lfl:asset:site_b-1"),
            ("tool", "urn:lfl:tool:builder_b12-t6"),
        ]
    finally:
        feature_stream_bridge.set_sindit_provider(None)
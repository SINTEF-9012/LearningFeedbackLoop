from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.core.context import CuttingContext, OperatingRegime
from backend.agents.core.schemas import PatternKey, PatternType, TimeRange
from backend.agents.memory.orchestrator import MemoryEvent, MemoryEventOrchestrator, OrchestratorConfig
from backend.agents.memory.scorer import SignificanceAction, SignificanceResult


@pytest.mark.asyncio
async def test_orchestrator_auto_composes_reconfig_for_significant_event(tmp_path: Path) -> None:
    outbox_path = tmp_path / "reconfig_outbox.jsonl"
    orchestrator = MemoryEventOrchestrator(
        config=OrchestratorConfig(
            use_classical_models=False,
            enable_harmonic_scorer=False,
            dispatch_alerts=False,
            compose_reconfig_proposals=True,
            reconfig_outbox_path=str(outbox_path),
        )
    )
    orchestrator.scorer.score = lambda **_: SignificanceResult(
        is_significant=True,
        score=0.81,
        action=SignificanceAction.ALERT,
        reasons=["stub significant"],
        triggered_rules=["pattern_rule"],
    )

    event = MemoryEvent(
        session_id="session-auto-reconfig",
        time_range=TimeRange(i0=0, i1=1, t0=0.0, t1=1.0, fs=1.0),
        patterns=[
            PatternKey(
                key="SPINDLE_LOAD_RAMP",
                pattern_type=PatternType.CUSTOM,
                confidence=0.74,
            )
        ],
        cutting_context=CuttingContext(
            machine_type="cnc",
            tool_type="endmill",
            workpiece_material="al",
            operating_regime=OperatingRegime.ROUGHING,
        ),
        batch={
            "batch_id": "batch-42",
            "unit_index": 3,
            "unit_count": 12,
            "recipe_id": "recipe-a",
        },
    )

    result = await orchestrator.process_event(event)

    assert result.significant is True
    assert result.reconfig_proposal_id is not None
    stored_rows = [json.loads(line) for line in outbox_path.read_text().strip().splitlines()]
    assert stored_rows[-1]["proposal_id"] == result.reconfig_proposal_id
    assert stored_rows[-1]["parameter_deltas"][0]["parameter"] == "feed_rate"
    assert stored_rows[-1]["tool_actions"][0]["action"] == "inspect"
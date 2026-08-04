#!/usr/bin/env python3
"""
Demo script for the LLM Memory Pipeline.

# ===========================================================================
# [PROTOTYPE_LLM_MEMORY_V1] - Demo/Test Script
# This script demonstrates the memory system with dummy data.
# ===========================================================================

Usage:
    # Run directly (uses in-memory orchestrator)
    python scripts/demo_memory_pipeline.py

    # Enable LLM explanations (requires Ollama)
    python scripts/demo_memory_pipeline.py --enable-llm

    # Or run the FastAPI server first, then use curl:
    cd backend && uvicorn app:app --reload
    # Then in another terminal:
    python scripts/demo_memory_pipeline.py --use-api
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path

# Ensure repo root is on sys.path so `import backend...` works when running from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.core.schemas import PatternKey, PatternType, TimeRange
from backend.agents.core.context import CuttingContext, OperatingRegime
from backend.agents.memory.scorer import SignificanceScorer, SignificanceConfig
from backend.agents.memory.orchestrator import (
    MemoryEventOrchestrator,
    MemoryEvent,
    OrchestratorConfig,
)
from backend.agents.memory.feedback import MemoryFeedbackRequest, FeedbackAction


# ============================================================================
# Dummy Data Generators
# ============================================================================

def make_dummy_time_range(start_sec: float = 0.0, duration_sec: float = 1.0, fs: float = 10000.0) -> TimeRange:
    """Create a TimeRange for a window of data."""
    samples = int(duration_sec * fs)
    return TimeRange(
        i0=int(start_sec * fs),
        i1=int(start_sec * fs) + samples,
        t0=start_sec,
        t1=start_sec + duration_sec,
        fs=fs,
    )


def make_dummy_cutting_context(
    spindle_rpm: float = 8500,
    num_teeth: int = 3,
    depth_mm: float = 2.5,
) -> CuttingContext:
    """Create typical cutting context for milling."""
    return CuttingContext(
        tool_type="end_mill",
        tool_diameter=10.0,
        num_teeth=num_teeth,
        spindle_speed=spindle_rpm,
        feed_rate=1200.0,  # mm/min
        feed_per_tooth=0.047,  # mm/tooth
        cutting_speed=267.0,  # m/min (for 10mm tool at 8500rpm)
        axial_depth=depth_mm,
        radial_depth=5.0,
        workpiece_material="steel",
        workpiece_hardness=35.0,  # HRC
        machine_id="DMG_MORI_01",
        operating_regime=OperatingRegime.ROUGHING if depth_mm > 2.0 else OperatingRegime.FINISHING,
    )


def make_dummy_patterns_normal() -> list[PatternKey]:
    """Normal operation patterns - should NOT trigger alerts."""
    return [
        PatternKey(pattern_type=PatternType.SPECTRAL_PEAK, key="SPECTRAL_PEAK_425Hz", source_metric="Fx"),
        PatternKey(pattern_type=PatternType.RATIO, key="RATIO_Fx_Fy:1.2", source_metric="ratio"),
    ]


def make_dummy_patterns_anomaly() -> list[PatternKey]:
    """Anomalous patterns - SHOULD trigger alerts."""
    return [
        PatternKey(pattern_type=PatternType.ANOMALY, key="ANOMALY_HIGH", source_metric="model"),
        PatternKey(pattern_type=PatternType.RATIO, key="RATIO_Fx_Fy:>5", source_metric="ratio"),
        PatternKey(pattern_type=PatternType.CUSTOM, key="CHATTER_DETECTED_512Hz", source_metric="fft"),
    ]


def make_dummy_patterns_breakage_warning() -> list[PatternKey]:
    """Breakage prediction patterns - high priority."""
    return [
        PatternKey(pattern_type=PatternType.CUSTOM, key="BREAKAGE_IMMINENT", source_metric="classifier"),
        PatternKey(pattern_type=PatternType.ANOMALY, key="SPIKE_RATE:>10", source_metric="Fz"),
    ]


# ============================================================================
# Demo Scenarios
# ============================================================================

async def demo_scenario_1_normal_operation(orchestrator: MemoryEventOrchestrator):
    """Scenario 1: Normal operation - should be stored but not alerted."""
    print("\n" + "="*70)
    print("SCENARIO 1: Normal Operation")
    print("="*70)
    
    event = MemoryEvent(
        session_id="demo_session_001",
        time_range=make_dummy_time_range(start_sec=0.0),
        patterns=make_dummy_patterns_normal(),
        cutting_context=make_dummy_cutting_context(),
        channels=["Fx", "Fy", "Fz"],
    )
    
    result = await orchestrator.process_event(event)
    
    print(f"  Processed: {result.processed}")
    print(f"  Significant: {result.significant}")
    print(f"  Score: {result.significance_score:.3f}")
    print(f"  Action: {result.action.value}")
    print(f"  Memory ID: {result.memory_id}")
    print(f"  Alert Dispatched: {result.alert_dispatched}")
    
    return result


async def demo_scenario_2_anomaly_detected(orchestrator: MemoryEventOrchestrator):
    """Scenario 2: Anomaly detected - should trigger alert."""
    print("\n" + "="*70)
    print("SCENARIO 2: Anomaly Detected (Chatter + High Force Ratio)")
    print("="*70)
    
    event = MemoryEvent(
        session_id="demo_session_001",
        time_range=make_dummy_time_range(start_sec=5.0),
        patterns=make_dummy_patterns_anomaly(),
        cutting_context=make_dummy_cutting_context(spindle_rpm=8500, depth_mm=3.0),
        channels=["Fx", "Fy", "Fz"],
    )
    
    result = await orchestrator.process_event(event)
    
    print(f"  Processed: {result.processed}")
    print(f"  Significant: {result.significant}")
    print(f"  Score: {result.significance_score:.3f}")
    print(f"  Action: {result.action.value}")
    print(f"  Memory ID: {result.memory_id}")
    print(f"  Explanation: {result.explanation}")
    print(f"  Similar memories found: {len(result.similar_memories)}")
    print(f"  Alert Dispatched: {result.alert_dispatched}")
    
    return result


async def demo_scenario_3_external_signal(orchestrator: MemoryEventOrchestrator):
    """Scenario 3: External classical model triggers alert."""
    print("\n" + "="*70)
    print("SCENARIO 3: External Breakage Prediction Signal")
    print("="*70)
    
    result = await orchestrator.process_external_signal(
        session_id="demo_session_001",
        signal_type="breakage_prediction",
        signal_value=0.85,  # 85% probability
        metadata={
            "machining": {
                "n": 8500,
                "z": 3,
                "ap": 2.5,
                "type": "end_mill",
            }
        },
    )
    
    print(f"  Processed: {result.processed}")
    print(f"  Significant: {result.significant}")
    print(f"  Score: {result.significance_score:.3f}")
    print(f"  Action: {result.action.value}")
    print(f"  Memory ID: {result.memory_id}")
    print(f"  Alert Dispatched: {result.alert_dispatched}")
    
    return result


async def demo_scenario_4_user_feedback(orchestrator: MemoryEventOrchestrator, memory_id: str):
    """Scenario 4: User provides feedback on a memory."""
    print("\n" + "="*70)
    print("SCENARIO 4: User Confirms Memory as Significant")
    print("="*70)
    
    request = MemoryFeedbackRequest(
        action=FeedbackAction.CONFIRM,
        user_id="operator_john",
        reason="This was actual chatter, had to reduce feed rate",
    )
    
    response = await orchestrator.feedback_handler.process_feedback(
        memory_id=memory_id,
        request=request,
    )
    
    print(f"  Success: {response.success}")
    print(f"  Feedback ID: {response.feedback_id}")
    print(f"  Message: {response.message}")
    print(f"  Updated fields: {response.updated_fields}")
    
    # Show updated pattern priors
    print(f"\n  Updated Pattern Priors:")
    for pattern, prior in orchestrator.scorer._pattern_priors.items():
        print(f"    {pattern}: {prior:.3f}")
    
    return response


async def demo_scenario_5_subsequent_event_with_learning(orchestrator: MemoryEventOrchestrator, score_before: float):
    """Scenario 5: Same pattern appears again - prior should boost score."""
    print("\n" + "="*70)
    print("SCENARIO 5: Same Pattern Reappears (Testing Learning)")
    print("="*70)
    
    # Re-use anomaly patterns
    event = MemoryEvent(
        session_id="demo_session_002",  # New session
        time_range=make_dummy_time_range(start_sec=0.0),
        patterns=make_dummy_patterns_anomaly(),
        cutting_context=make_dummy_cutting_context(),
        channels=["Fx", "Fy", "Fz"],
    )
    
    result = await orchestrator.process_event(event)
    
    score_after = result.significance_score
    score_delta = score_after - score_before
    
    print(f"  Score BEFORE feedback: {score_before:.3f}")
    print(f"  Score AFTER feedback:  {score_after:.3f}")
    print(f"  Score INCREASE:        {score_delta:+.3f} {'✓ Learning effect visible!' if score_delta > 0 else '(no change)'}")
    print(f"  Action: {result.action.value}")
    print(f"  Similar memories found: {len(result.similar_memories)}")
    if result.similar_memories:
        print(f"  Top similar memory: {result.similar_memories[0].memory.id}")
    
    return result


# ============================================================================
# Main Demo Runner
# ============================================================================

async def run_demo(*, enable_llm: bool = False, dispatch_alerts: bool = False):
    """Run all demo scenarios."""
    print("\n" + "#"*70)
    print("# LLM MEMORY PIPELINE DEMO")
    print("# [PROTOTYPE_LLM_MEMORY_V1]")
    print("#"*70)
    
    # Create orchestrator with config
    config = OrchestratorConfig(
        always_store=False,  # Only store significant events
        min_score_for_retrieval=0.3,
        top_k_similar=5,
        # Default to fast/local-friendly behavior.
        # Use --enable-llm if you want to call Ollama for explanations.
        generate_explanations=bool(enable_llm),
        # Default off: local mode has no websocket clients anyway.
        dispatch_alerts=bool(dispatch_alerts),
    )
    
    orchestrator = MemoryEventOrchestrator(config=config)
    
    # Run scenarios
    result1 = await demo_scenario_1_normal_operation(orchestrator)
    result2 = await demo_scenario_2_anomaly_detected(orchestrator)
    result3 = await demo_scenario_3_external_signal(orchestrator)
    
    # Save baseline score before feedback
    baseline_score = result2.significance_score
    
    # Feedback on anomaly event
    if result2.memory_id:
        await demo_scenario_4_user_feedback(orchestrator, result2.memory_id)
    
    # Test learning effect - pass baseline score to show improvement
    await demo_scenario_5_subsequent_event_with_learning(orchestrator, baseline_score)
    
    # Summary
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)
    
    all_memories = orchestrator.list_memories()
    print(f"  Total memories stored: {len(all_memories)}")
    print(f"  Pattern priors learned: {len(orchestrator.scorer._pattern_priors)}")
    
    print("\n  Stored Memories:")
    for mem in all_memories:
        patterns = ", ".join([p.key for p in mem.pattern_keys[:2]])
        score = mem.metadata.get("significance_score", 0) if mem.metadata else 0
        print(f"    [{mem.id[:8]}...] score={score:.2f} patterns={patterns}")
    
    print("\n" + "#"*70)
    print("# DEMO COMPLETE")
    print("#"*70 + "\n")


# ============================================================================
# API-based Demo (when server is running)
# ============================================================================

def demo_via_api():
    """Demo using HTTP API calls (requires server running)."""
    import requests
    
    BASE_URL = "http://localhost:8000/agent/memory"
    
    print("\n" + "#"*70)
    print("# API-BASED DEMO (requires: uvicorn app:app --reload)")
    print("#"*70 + "\n")
    
    # Event 1: Normal
    print("Sending normal event...")
    resp = requests.post(f"{BASE_URL}/events", json={
        "session_id": "api_demo_001",
        "pattern_keys": ["SPECTRAL_PEAK_425Hz", "RATIO_Fx_Fy:1.2"],
        "time_range": {"i0": 0, "i1": 10000, "t0": 0.0, "t1": 1.0, "fs": 10000},
        "cutting_context": {
            "spindle_speed": 8500,
            "num_teeth": 3,
            "axial_depth": 2.5,
            "tool_type": "end_mill",
        },
    })
    print(f"  Response: {resp.json()}")
    
    # Event 2: Anomaly
    print("\nSending anomaly event...")
    resp = requests.post(f"{BASE_URL}/events", json={
        "session_id": "api_demo_001",
        "pattern_keys": ["ANOMALY_HIGH", "RATIO_Fx_Fy:>5", "CHATTER_DETECTED"],
        "time_range": {"i0": 50000, "i1": 60000, "t0": 5.0, "t1": 6.0, "fs": 10000},
        "cutting_context": {
            "spindle_speed": 8500,
            "num_teeth": 3,
            "axial_depth": 3.0,
        },
    })
    result = resp.json()
    print(f"  Response: {result}")
    
    memory_id = result.get("memory_id")
    
    # Feedback
    if memory_id:
        print(f"\nConfirming memory {memory_id}...")
        resp = requests.post(f"{BASE_URL}/{memory_id}/confirm", params={
            "user_id": "demo_operator",
            "reason": "Confirmed chatter",
        })
        print(f"  Response: {resp.json()}")
    
    # List memories
    print("\nListing all memories...")
    resp = requests.get(f"{BASE_URL}/")
    print(f"  Response: {json.dumps(resp.json(), indent=2)}")
    
    # Stats
    print("\nGetting stats...")
    resp = requests.get(f"{BASE_URL}/stats/overview")
    print(f"  Response: {json.dumps(resp.json(), indent=2)}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM Memory Pipeline Demo")
    ap.add_argument("--use-api", action="store_true", help="Use the FastAPI server endpoints instead of in-process mode")
    ap.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable LLM explanations in in-process mode (requires Ollama; can be slow depending on model)",
    )
    ap.add_argument(
        "--dispatch-alerts",
        action="store_true",
        help="Dispatch alerts in in-process mode (useful only if you have websocket subscribers)",
    )
    args = ap.parse_args()

    if args.use_api:
        demo_via_api()
    else:
        asyncio.run(run_demo(enable_llm=args.enable_llm, dispatch_alerts=args.dispatch_alerts))

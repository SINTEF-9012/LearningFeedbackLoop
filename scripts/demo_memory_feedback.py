#!/usr/bin/env python3
"""
Memory System Demo - Interactive demonstration of memory storage and feedback.

This script demonstrates:
1. How events are processed and scored for significance
2. How memories are stored and retrieved
3. How operator feedback affects pattern priors (learning)
4. How similar memories are retrieved based on patterns
5. How the LLM generates explanations for significant events

=============================================================================
DEMO STRUCTURE
=============================================================================

The demo uses JSON input files from scripts/demo_data/:
  - event_1_normal.json         → Normal operation (low significance)
  - event_2_chatter.json        → Chatter detected (high significance)
  - event_3_classical_alert.json → External model alert
  - event_4_similar_to_chatter.json → Similar to event 2 (tests retrieval)
  - feedback_confirm.json       → Confirm feedback template
  - feedback_dismiss.json       → Dismiss feedback template
  - feedback_comment.json       → Comment feedback template

=============================================================================
USAGE
=============================================================================

Default: Run against API (recommended, includes LLM explanations)
    uvicorn backend.app:app --reload --port 8000
    python scripts/demo_memory_feedback.py

Option 2: Run in-process (no server needed, no LLM explanations)
    python scripts/demo_memory_feedback.py --local

Options:
    --local         Use in-process mode (no server, no LLM)
    --url URL       API base URL (default: http://localhost:8000)
    --pause         Pause between steps for explanation
    --verbose       Show full JSON responses
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# For API mode
try:
    import httpx
except ImportError:
    httpx = None

# Colors for terminal output - disabled when not a TTY
class Colors:
    # Check if stdout is a terminal
    _use_colors = sys.stdout.isatty()
    
    @classmethod
    def disable(cls):
        """Disable all colors."""
        cls._use_colors = False
        cls._update_colors()
    
    @classmethod
    def _update_colors(cls):
        """Update color values based on _use_colors flag."""
        if cls._use_colors:
            cls.HEADER = '\033[95m'
            cls.BLUE = '\033[94m'
            cls.CYAN = '\033[96m'
            cls.GREEN = '\033[92m'
            cls.YELLOW = '\033[93m'
            cls.RED = '\033[91m'
            cls.ENDC = '\033[0m'
            cls.BOLD = '\033[1m'
            cls.DIM = '\033[2m'
        else:
            cls.HEADER = ''
            cls.BLUE = ''
            cls.CYAN = ''
            cls.GREEN = ''
            cls.YELLOW = ''
            cls.RED = ''
            cls.ENDC = ''
            cls.BOLD = ''
            cls.DIM = ''

# Initialize colors based on TTY detection
Colors._update_colors()

# Default values (will be updated by _update_colors)
Colors.HEADER = '\033[95m' if Colors._use_colors else ''
Colors.BLUE = '\033[94m' if Colors._use_colors else ''
Colors.CYAN = '\033[96m' if Colors._use_colors else ''
Colors.GREEN = '\033[92m' if Colors._use_colors else ''
Colors.YELLOW = '\033[93m' if Colors._use_colors else ''
Colors.RED = '\033[91m' if Colors._use_colors else ''
Colors.ENDC = '\033[0m' if Colors._use_colors else ''
Colors.BOLD = '\033[1m' if Colors._use_colors else ''
Colors.DIM = '\033[2m' if Colors._use_colors else ''


def print_header(text: str):
    """Print a section header."""
    print(f"\n{Colors.HEADER}{'='*70}")
    print(f" {text}")
    print(f"{'='*70}{Colors.ENDC}\n")


def print_step(step: int, text: str):
    """Print a numbered step."""
    print(f"{Colors.CYAN}[Step {step}]{Colors.ENDC} {Colors.BOLD}{text}{Colors.ENDC}")


def print_explanation(text: str):
    """Print an explanation in dim text."""
    for line in text.split('\n'):
        print(f"  {Colors.DIM}→ {line}{Colors.ENDC}")


def print_result(label: str, value: Any, color: str = Colors.GREEN):
    """Print a result value."""
    print(f"  {color}{label}:{Colors.ENDC} {value}")


def print_json(data: dict, indent: int = 2):
    """Print formatted JSON."""
    print(json.dumps(data, indent=indent, default=str))


def load_json_file(filename: str) -> dict:
    """Load a JSON file from demo_data directory."""
    script_dir = Path(__file__).parent
    filepath = script_dir / "demo_data" / filename
    with open(filepath) as f:
        data = json.load(f)
    # Remove _description and _explanation fields for API calls
    return {k: v for k, v in data.items() if not k.startswith('_')}


def load_json_with_description(filename: str) -> tuple[dict, str, str]:
    """Load JSON file and return (data, description, explanation)."""
    script_dir = Path(__file__).parent
    filepath = script_dir / "demo_data" / filename
    with open(filepath) as f:
        raw = json.load(f)
    desc = raw.get('_description', filename)
    expl = raw.get('_explanation', '')
    data = {k: v for k, v in raw.items() if not k.startswith('_')}
    return data, desc, expl


# =============================================================================
# API CLIENT (for --api mode)
# =============================================================================

class MemoryAPIClient:
    """HTTP client for memory system API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        if httpx is None:
            raise ImportError("httpx required for API mode: pip install httpx")
        self.base_url = self._normalize_base_url(base_url)
        self.client = httpx.Client(timeout=60.0)  # Longer timeout for LLM
        self.llm_available = self._check_llm_availability()

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        """Normalize a user-provided URL to the API base (scheme://host[:port]).

        Common user mistake: passing a router-prefixed URL like
        http://localhost:8000/agent/memory.
        """
        url = (url or "").rstrip("/")
        for suffix in ("/agent/memory", "/agent"):
            if url.endswith(suffix):
                url = url[: -len(suffix)].rstrip("/")
        return url

    @staticmethod
    def _normalize_event_payload(event: dict) -> dict:
        """Accept either {pattern_keys:[...]} or legacy {patterns:[...]} inputs."""
        if not isinstance(event, dict):
            return event
        if (not event.get("pattern_keys")) and event.get("patterns"):
            event = dict(event)
            event["pattern_keys"] = event.get("patterns")
        return event
    
    def _check_llm_availability(self) -> bool:
        """Check if LLM service is available."""
        try:
            resp = self.client.get(f"{self.base_url}/agent/memory/llm/status", timeout=5.0)
            if resp.status_code == 200:
                return resp.json().get('available', False)
        except Exception:
            pass
        return False
    
    def process_event(self, event: dict) -> dict:
        """POST /agent/memory/events"""
        event = self._normalize_event_payload(event)
        resp = self.client.post(f"{self.base_url}/agent/memory/events", json=event)
        resp.raise_for_status()
        return resp.json()
    
    def get_memory(self, memory_id: str) -> dict:
        """GET /agent/memory/{id}"""
        resp = self.client.get(f"{self.base_url}/agent/memory/{memory_id}")
        resp.raise_for_status()
        return resp.json()
    
    def add_feedback(self, memory_id: str, feedback: dict) -> dict:
        """PATCH /agent/memory/{id}/feedback"""
        resp = self.client.patch(
            f"{self.base_url}/agent/memory/{memory_id}/feedback", 
            json=feedback
        )
        resp.raise_for_status()
        return resp.json()
    
    def list_memories(self, session_id: Optional[str] = None) -> dict:
        """GET /agent/memory/ or /agent/memory/session/{session_id}"""
        if session_id:
            resp = self.client.get(f"{self.base_url}/agent/memory/session/{session_id}")
        else:
            resp = self.client.get(f"{self.base_url}/agent/memory/")
        resp.raise_for_status()
        return resp.json()
    
    def get_stats(self) -> dict:
        """GET /agent/memory/stats/overview"""
        resp = self.client.get(f"{self.base_url}/agent/memory/stats/overview")
        resp.raise_for_status()
        return resp.json()
    
    def reset_priors(self) -> dict:
        """POST /agent/memory/scorer/reset-priors"""
        resp = self.client.post(f"{self.base_url}/agent/memory/scorer/reset-priors")
        resp.raise_for_status()
        return resp.json()
    
    def delete_all(self) -> dict:
        """DELETE /agent/memory/"""
        resp = self.client.delete(f"{self.base_url}/agent/memory/")
        resp.raise_for_status()
        return resp.json()
    
    def explain_memory(self, memory_id: str) -> dict:
        """POST /agent/memory/{id}/explain - Get LLM explanation for a memory."""
        try:
            resp = self.client.post(
                f"{self.base_url}/agent/memory/{memory_id}/explain",
                timeout=60.0  # LLM can take time
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {'explanation': None, 'error': str(e)}


# =============================================================================
# IN-PROCESS CLIENT (default mode, no server needed)
# =============================================================================

class InProcessClient:
    """Direct orchestrator access for testing without server."""
    
    def __init__(self, enable_llm: bool = False):
        # Add project root to path for imports
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Import here to avoid import errors if deps missing
        from backend.agents.core.schemas import PatternKey, PatternType, TimeRange
        from backend.agents.memory.orchestrator import (
            MemoryEventOrchestrator,
            MemoryEvent,
            OrchestratorConfig,
            get_orchestrator,
        )
        from backend.agents.memory.feedback import MemoryFeedbackRequest, FeedbackAction
        from backend.agents.memory.scorer import SignificanceConfig
        
        self.PatternKey = PatternKey
        self.PatternType = PatternType
        self.TimeRange = TimeRange
        self.MemoryEvent = MemoryEvent
        self.MemoryFeedbackRequest = MemoryFeedbackRequest
        self.FeedbackAction = FeedbackAction
        self.enable_llm = enable_llm
        
        # Create fresh orchestrator for demo
        config = OrchestratorConfig(
            always_store=False,
            min_score_for_retrieval=0.2,
            top_k_similar=5,
            generate_explanations=enable_llm,  # Enable LLM if requested
            dispatch_alerts=False,  # Skip WebSocket for demo
        )
        self.orchestrator = MemoryEventOrchestrator(config=config)
        self.llm_available = enable_llm and self.orchestrator.explainer.is_available()
    
    def _dict_to_event(self, data: dict) -> "MemoryEvent":
        """Convert dict to MemoryEvent."""
        time_range = self.TimeRange(**data.get('time_range', {
            'i0': 0, 'i1': 1000, 't0': 0.0, 't1': 1.0, 'fs': 10000.0
        }))
        
        pattern_keys = data.get('pattern_keys')
        if not pattern_keys and data.get('patterns'):
            # Accept legacy/alternate key name used by some docs and examples.
            pattern_keys = data.get('patterns')
        patterns = [
            self.PatternKey(pattern_type=self.PatternType.CUSTOM, key=k)
            for k in (pattern_keys or [])
        ]
        
        # Handle cutting context
        cutting_context = None
        if data.get('cutting_context'):
            from backend.agents.core.context import CuttingContext
            cutting_context = CuttingContext(**data['cutting_context'])
        
        return self.MemoryEvent(
            session_id=data.get('session_id', 'demo-session'),
            time_range=time_range,
            patterns=patterns,
            cutting_context=cutting_context,
            external_signals=data.get('external_signals') or {},
            channels=data.get('channels', []),
        )
    
    def process_event(self, event: dict) -> dict:
        """Process an event through the orchestrator."""
        mem_event = self._dict_to_event(event)
        # Use asyncio.run() for Python 3.12+ compatibility.
        # asyncio.get_event_loop() is brittle when no loop is set.
        result = asyncio.run(self.orchestrator.process_event(mem_event))
        return {
            'processed': result.processed,
            'significant': result.significant,
            'memory_id': result.memory_id,
            'significance_score': result.significance_score,
            'action': result.action.value,
            'explanation': result.explanation,
            'similar_memory_count': len(result.similar_memories),
        }
    
    def get_memory(self, memory_id: str) -> dict:
        """Get memory by ID."""
        mem = self.orchestrator.get_memory(memory_id)
        if mem is None:
            return {'error': 'Not found'}
        return {
            'memory': {
                'id': mem.id,
                'session_id': mem.session_id,
                'annotation_text': mem.annotation_text,
                'pattern_keys': [p.key for p in mem.pattern_keys],
                'tags': mem.tags,
                'metadata': mem.metadata,
            },
            'feedback_stats': self.orchestrator.feedback_handler.get_feedback_stats(memory_id),
        }
    
    def add_feedback(self, memory_id: str, feedback: dict) -> dict:
        """Add feedback to a memory."""
        action = self.FeedbackAction(feedback['action'])
        request = self.MemoryFeedbackRequest(
            action=action,
            user_id=feedback.get('user_id', 'operator'),
            comment=feedback.get('comment'),
            reason=feedback.get('reason'),
        )
        result = asyncio.run(self.orchestrator.feedback_handler.process_feedback(memory_id, request))
        return {
            'success': result.success,
            'feedback_id': result.feedback_id,
            'message': result.message,
            'updated_fields': result.updated_fields,
        }
    
    def list_memories(self, session_id: Optional[str] = None) -> dict:
        """List memories."""
        memories = self.orchestrator.list_memories(session_id)
        return {
            'total_count': len(memories),
            'memories': [
                {
                    'id': m.id,
                    'patterns': [p.key for p in m.pattern_keys],
                    'significance_score': m.metadata.get('significance_score') if m.metadata else None,
                }
                for m in memories
            ],
        }
    
    def get_stats(self) -> dict:
        """Get system stats including pattern priors."""
        priors = dict(list(self.orchestrator.scorer._pattern_priors.items())[:10])
        return {
            'total_memories': len(self.orchestrator._memories),
            'scorer_priors': priors,
        }
    
    def reset_priors(self) -> dict:
        """Reset pattern priors."""
        self.orchestrator.scorer._pattern_priors.clear()
        return {'message': 'Priors reset'}
    
    def delete_all(self) -> dict:
        """Delete all memories."""
        count = len(self.orchestrator._memories)
        self.orchestrator._memories.clear()
        self.orchestrator.scorer._pattern_priors.clear()
        return {'deleted_count': count}
    
    def explain_memory(self, memory_id: str) -> dict:
        """Get LLM explanation for a memory."""
        if not self.enable_llm:
            return {'explanation': None, 'error': 'LLM not enabled in local mode'}
        
        memory = self.orchestrator.get_memory(memory_id)
        if not memory:
            return {'explanation': None, 'error': 'Memory not found'}
        
        try:
            explanation = self.orchestrator.explainer.generate_memory_summary(
                memory=memory,
                context=None,
            )
            return {'explanation': explanation, 'memory_id': memory_id}
        except Exception as e:
            return {'explanation': None, 'error': str(e)}


# =============================================================================
# DEMO RUNNER
# =============================================================================

def run_demo(client, pause: bool = False, verbose: bool = False, confirm_repeats: int = 1):
    """Run the complete demo."""
    
    def maybe_pause():
        if pause:
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    # Check if LLM is available
    llm_available = getattr(client, 'llm_available', False)
    is_local = client.__class__.__name__ == "InProcessClient"
    
    # =========================================================================
    print_header("MEMORY SYSTEM DEMO - Storage & Feedback")
    print(f"""
This demo shows how the memory system:
  1. Scores events for significance
  2. Stores significant events as memories
  3. Learns from operator feedback (confirm/dismiss)
  4. Retrieves similar past memories
  5. Generates LLM explanations for significant events

Input files are in: scripts/demo_data/
LLM Status: {Colors.GREEN + 'Available' + Colors.ENDC if llm_available else Colors.YELLOW + 'Not available (fallback explanations)' + Colors.ENDC}
Mode: {Colors.YELLOW + 'LOCAL (in-process; no server websockets)' + Colors.ENDC if is_local else Colors.GREEN + 'API (server-backed; websockets enabled if you connect first)' + Colors.ENDC}
    """)
    if is_local:
        print_explanation("Note: LOCAL mode does not send WebSocket alerts to the server.")
        print_explanation("If you want scripts/visualize_memory_alerts.py to show output, run the server and run this demo in API mode.")
    maybe_pause()
    
    # =========================================================================
    print_header("SETUP: Reset system state")
    print_step(0, "Clearing all memories and resetting pattern priors")
    
    result = client.delete_all()
    print_result("Deleted", f"{result.get('deleted_count', 0)} memories")
    
    result = client.reset_priors()
    print_result("Priors", "Reset to defaults")
    maybe_pause()
    
    # =========================================================================
    print_header("PART 1: Event Significance Scoring")
    
    # --- Event 1: Normal operation ---
    print_step(1, "Processing event_1_normal.json")
    data, desc, expl = load_json_with_description("event_1_normal.json")
    print_explanation(f"Description: {desc}")
    print_explanation(expl)
    
    if verbose:
        print(f"\n{Colors.DIM}Request:{Colors.ENDC}")
        print_json(data)
    
    if not data.get("pattern_keys") and not data.get("patterns") and not data.get("external_signals"):
        print(f"\n  {Colors.YELLOW}Warning:{Colors.ENDC} event has no pattern_keys/patterns/external_signals; it will likely score 0/ignore")
    result = client.process_event(data)
    print(f"\n  {Colors.BOLD}Result:{Colors.ENDC}")
    print_result("Significance Score", f"{result['significance_score']:.3f}")
    print_result("Action", result['action'], 
                 Colors.GREEN if result['action'] == 'ignore' else Colors.YELLOW)
    print_result("Stored", "Yes" if result['memory_id'] else "No")
    
    normal_memory_id = result.get('memory_id')
    maybe_pause()
    
    # --- Event 2: Chatter detected ---
    print_step(2, "Processing event_2_chatter.json")
    data, desc, expl = load_json_with_description("event_2_chatter.json")
    print_explanation(f"Description: {desc}")
    print_explanation(expl)
    
    if not data.get("pattern_keys") and not data.get("patterns") and not data.get("external_signals"):
        print(f"\n  {Colors.YELLOW}Warning:{Colors.ENDC} event has no pattern_keys/patterns/external_signals; it will likely score 0/ignore")
    result = client.process_event(data)
    print(f"\n  {Colors.BOLD}Result:{Colors.ENDC}")
    print_result("Significance Score", f"{result['significance_score']:.3f}")
    print_result("Action", result['action'],
                 Colors.RED if result['action'] in ('alert', 'critical') else Colors.YELLOW)
    print_result("Memory ID", result.get('memory_id', 'None'))
    
    # Show explanation from orchestrator (may be from LLM or fallback)
    if result.get('explanation'):
        print(f"\n  {Colors.CYAN}{Colors.BOLD}LLM Explanation:{Colors.ENDC}")
        print(f"  {Colors.CYAN}\"{result['explanation']}\"{Colors.ENDC}")
    
    chatter_memory_id = result.get('memory_id')
    maybe_pause()
    
    # --- Event 3: Classical model alert ---
    print_step(3, "Processing event_3_classical_alert.json")
    data, desc, expl = load_json_with_description("event_3_classical_alert.json")
    print_explanation(f"Description: {desc}")
    print_explanation(expl)
    
    if not data.get("pattern_keys") and not data.get("patterns") and not data.get("external_signals"):
        print(f"\n  {Colors.YELLOW}Warning:{Colors.ENDC} event has no pattern_keys/patterns/external_signals; it will likely score 0/ignore")
    result = client.process_event(data)
    print(f"\n  {Colors.BOLD}Result:{Colors.ENDC}")
    print_result("Significance Score", f"{result['significance_score']:.3f}")
    print_result("Action", result['action'],
                 Colors.RED if result['action'] in ('alert', 'critical') else Colors.YELLOW)
    print_result("Memory ID", result.get('memory_id', 'None'))
    
    classical_memory_id = result.get('memory_id')
    maybe_pause()
    
    # =========================================================================
    print_header("PART 2: Pattern Priors BEFORE Feedback")
    
    print_step(4, "Checking current pattern priors")
    print_explanation("Before any feedback, pattern priors are at default values.")
    print_explanation("The scorer uses these priors to boost patterns that have been confirmed as significant.")
    
    stats = client.get_stats()
    print(f"\n  {Colors.BOLD}Current State:{Colors.ENDC}")
    print_result("Total Memories", stats.get('total_memories', 0))
    print(f"  {Colors.GREEN}Pattern Priors:{Colors.ENDC}")
    priors = stats.get('scorer_priors', {})
    if priors:
        for pattern, prior in priors.items():
            print(f"    {pattern}: {prior:.3f}")
    else:
        print(f"    {Colors.DIM}(no priors yet - will be set by feedback){Colors.ENDC}")
    
    maybe_pause()

    # =========================================================================
    print_header("PART 2B: Baseline Run (Before Feedback)")

    print_step(4.5, "Processing event_4_similar_to_chatter.json (baseline)")
    data, desc, expl = load_json_with_description("event_4_similar_to_chatter.json")
    print_explanation(f"Description: {desc}")
    print_explanation(expl)
    print_explanation("This is a baseline run BEFORE feedback. We'll run it again AFTER confirmation and compare scores.")

    if not data.get("pattern_keys") and not data.get("patterns") and not data.get("external_signals"):
        print(f"\n  {Colors.YELLOW}Warning:{Colors.ENDC} event has no pattern_keys/patterns/external_signals; it will likely score 0/ignore")
    baseline_similar = client.process_event(data)
    baseline_similar_score = float(baseline_similar.get("significance_score", 0.0) or 0.0)
    print(f"\n  {Colors.BOLD}Baseline Result:{Colors.ENDC}")
    print_result("Significance Score", f"{baseline_similar_score:.3f}")
    print_result("Action", baseline_similar.get('action', ''),
                 Colors.RED if baseline_similar.get('action') in ('alert', 'critical') else Colors.YELLOW)
    maybe_pause()
    
    # =========================================================================
    print_header("PART 3: Operator Feedback - CONFIRM")
    
    if chatter_memory_id:
        print_step(5, f"Confirming memory {chatter_memory_id[:8]}...")
        data, desc, expl = load_json_with_description("feedback_confirm.json")
        print_explanation(f"Description: {desc}")
        print_explanation(expl)

        confirm_repeats = max(1, int(confirm_repeats or 1))
        last_feedback_result = None
        for i in range(confirm_repeats):
            last_feedback_result = client.add_feedback(chatter_memory_id, data)
            if confirm_repeats > 1:
                print(f"  Applied confirm {i+1}/{confirm_repeats}")

        print(f"\n  {Colors.BOLD}Feedback Result:{Colors.ENDC}")
        print_result("Success", (last_feedback_result or {}).get('success', False))
        print_result("Message", (last_feedback_result or {}).get('message', ''))
        
        # Show updated priors
        print_step(6, "Checking pattern priors AFTER confirmation")
        print_explanation("Patterns from the confirmed memory should now have boosted priors.")
        
        stats = client.get_stats()
        print(f"\n  {Colors.GREEN}Updated Pattern Priors:{Colors.ENDC}")
        priors = stats.get('scorer_priors', {})
        if priors:
            for pattern, prior in priors.items():
                boost_indicator = f" {Colors.GREEN}↑ BOOSTED{Colors.ENDC}" if prior > 0.5 else ""
                print(f"    {pattern}: {prior:.3f}{boost_indicator}")
        else:
            print(f"    {Colors.DIM}(no changes visible){Colors.ENDC}")
    else:
        print(f"  {Colors.RED}Skipping: No chatter memory was stored{Colors.ENDC}")
    
    maybe_pause()
    
    # =========================================================================
    print_header("PART 4: Testing Learned Priors")
    
    print_step(7, "Processing event_4_similar_to_chatter.json (after feedback)")
    data, desc, expl = load_json_with_description("event_4_similar_to_chatter.json")
    print_explanation(f"Description: {desc}")
    print_explanation(expl)
    print_explanation("This event has similar patterns to the confirmed chatter event.")
    print_explanation("The boosted pattern priors may increase its significance score.")
    print_explanation("Note: If other detection rules already fire strongly, the score delta can be small; the priors themselves should still clearly change.")
    
    result = client.process_event(data)
    after_score = float(result.get("significance_score", 0.0) or 0.0)
    print(f"\n  {Colors.BOLD}Result:{Colors.ENDC}")
    print_result("Significance Score", f"{after_score:.3f}")
    if baseline_similar_score is not None:
        delta = after_score - baseline_similar_score
        delta_color = Colors.GREEN if delta > 0.005 else Colors.YELLOW
        print_result("Δ Score vs Baseline", f"{delta:+.3f}", delta_color)
    print_result("Action", result['action'],
                 Colors.RED if result['action'] in ('alert', 'critical') else Colors.YELLOW)
    print_result("Similar Memories Found", result.get('similar_memory_count', 0))
    
    if result.get('similar_memory_count', 0) > 0:
        print(f"\n  {Colors.CYAN}→ The retriever found similar past memories!{Colors.ENDC}")
    
    similar_memory_id = result.get('memory_id')
    maybe_pause()
    
    # =========================================================================
    print_header("PART 5: Operator Feedback - DISMISS")
    
    if classical_memory_id:
        print_step(8, f"Dismissing memory {classical_memory_id[:8]}...")
        data, desc, expl = load_json_with_description("feedback_dismiss.json")
        print_explanation(f"Description: {desc}")
        print_explanation(expl)
        
        result = client.add_feedback(classical_memory_id, data)
        print(f"\n  {Colors.BOLD}Feedback Result:{Colors.ENDC}")
        print_result("Success", result.get('success', False))
        print_result("Message", result.get('message', ''))
        
        # Show how dismissal affects the memory
        print_step(9, "Checking dismissed memory state")
        mem_result = client.get_memory(classical_memory_id)
        if 'feedback_stats' in mem_result:
            fb_stats = mem_result['feedback_stats']
            print(f"  {Colors.RED}Feedback Stats:{Colors.ENDC}")
            print(f"    Confirmed: {fb_stats.get('confirmed', False)}")
            print(f"    Dismissed: {fb_stats.get('dismissed', False)}")
    else:
        print(f"  {Colors.RED}Skipping: No classical alert memory was stored{Colors.ENDC}")
    
    maybe_pause()
    
    # =========================================================================
    print_header("PART 6: Adding Operator Comments")
    
    if chatter_memory_id:
        print_step(10, f"Adding comment to memory {chatter_memory_id[:8]}...")
        data, desc, expl = load_json_with_description("feedback_comment.json")
        print_explanation(f"Description: {desc}")
        print_explanation(expl)
        
        result = client.add_feedback(chatter_memory_id, data)
        print(f"\n  {Colors.BOLD}Comment Added:{Colors.ENDC}")
        print_result("Success", result.get('success', False))
        
        # Show the enriched memory
        mem_result = client.get_memory(chatter_memory_id)
        if 'memory' in mem_result:
            mem = mem_result['memory']
            print(f"\n  {Colors.CYAN}Enriched Memory:{Colors.ENDC}")
            print(f"    ID: {mem.get('id', '')[:8]}...")
            print(f"    Patterns: {mem.get('pattern_keys', [])}")
            print(f"    Annotation: {mem.get('annotation_text', '')[:60]}...")
    
    maybe_pause()
    
    # =========================================================================
    print_header("FINAL: Memory System Overview")
    
    print_step(11, "Final system state")
    
    stats = client.get_stats()
    list_result = client.list_memories()
    
    print(f"\n  {Colors.BOLD}Summary:{Colors.ENDC}")
    print_result("Total Memories Stored", stats.get('total_memories', list_result.get('total_count', 0)))
    
    print(f"\n  {Colors.BOLD}Stored Memories:{Colors.ENDC}")
    for mem in list_result.get('memories', [])[:5]:
        score = mem.get('significance_score')
        score_str = f"{score:.3f}" if score else "N/A"
        patterns = mem.get('patterns', [])[:2]
        print(f"    • {mem['id'][:8]}... score={score_str} patterns={patterns}")
    
    print(f"\n  {Colors.BOLD}Final Pattern Priors:{Colors.ENDC}")
    priors = stats.get('scorer_priors', {})
    if priors:
        for pattern, prior in list(priors.items())[:5]:
            print(f"    {pattern}: {prior:.3f}")
    
    # =========================================================================
    print_header("DEMO COMPLETE")
    print(f"""
{Colors.GREEN}Key Takeaways:{Colors.ENDC}

1. {Colors.BOLD}Significance Scoring{Colors.ENDC}: Events are scored based on patterns,
   external signals, and learned priors. Only significant events are stored.

2. {Colors.BOLD}Pattern Priors{Colors.ENDC}: When operators confirm events, the pattern 
   priors are boosted. This makes similar future patterns more likely to 
   be flagged as significant.

3. {Colors.BOLD}Feedback Learning{Colors.ENDC}: The system learns from feedback:
   - CONFIRM → boost pattern priors (increase future sensitivity)
   - DISMISS → reduce pattern priors (decrease false positives)
   - COMMENT → enrich memory with domain knowledge

4. {Colors.BOLD}Similar Memory Retrieval{Colors.ENDC}: When new events are processed,
   the retriever finds similar past memories to provide context.

{Colors.CYAN}Try modifying the JSON files in scripts/demo_data/ and re-running!{Colors.ENDC}
    """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Memory System Demo")
    parser.add_argument('--local', action='store_true', 
                        help='Use in-process mode (no server, no LLM)')
    parser.add_argument('--pause', action='store_true',
                        help='Pause between steps for explanation')
    parser.add_argument('--verbose', action='store_true',
                        help='Show full JSON requests/responses')
    parser.add_argument('--confirm-repeats', type=int, default=1,
                        help='Apply CONFIRM feedback N times to amplify learning (default: 1)')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output (auto-detected when piping)')
    parser.add_argument('--url', default='http://localhost:8000',
                        help='API base URL (default: http://localhost:8000)')
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        Colors.disable()
    
    if args.local:
        print("Using in-process mode (no server, no LLM explanations)")
        try:
            client = InProcessClient(enable_llm=False)
        except ImportError as e:
            print(f"Error: {e}")
            print("Make sure you're running from the project root.")
            sys.exit(1)
    else:
        # Default: Try API mode first
        print(f"Connecting to API: {args.url}")
        try:
            client = MemoryAPIClient(args.url)
            # Quick health check
            client.get_stats()
            print(f"{Colors.GREEN}✓ Connected to API{Colors.ENDC}")
            if client.llm_available:
                print(f"{Colors.GREEN}✓ LLM service available{Colors.ENDC}")
            else:
                print(f"{Colors.YELLOW}⚠ LLM service not available (using fallback explanations){Colors.ENDC}")
        except ImportError as e:
            print(f"Error: {e}")
            print("Install httpx: pip install httpx")
            sys.exit(1)
        except Exception as e:
            print(f"\n{Colors.YELLOW}Could not connect to API: {e}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Falling back to in-process mode...{Colors.ENDC}\n")
            print(f"{Colors.YELLOW}NOTE:{Colors.ENDC} This local fallback will not produce server WebSocket alerts.")
            print(f"      To enable alerts, start the server and re-run without --local.")
            try:
                client = InProcessClient(enable_llm=False)
            except ImportError as e:
                print(f"Error: {e}")
                print("Make sure you're running from the project root.")
                sys.exit(1)
    
    try:
        run_demo(client, pause=args.pause, verbose=args.verbose, confirm_repeats=args.confirm_repeats)
    except Exception as e:
        if httpx and isinstance(e, httpx.ConnectError):
            print(f"\n{Colors.RED}Error: Connection lost to {args.url}")
            print(f"Make sure the server is running:{Colors.ENDC}")
            print("  uvicorn backend.app:app --reload --port 8000")
            sys.exit(1)
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
        raise


if __name__ == "__main__":
    main()

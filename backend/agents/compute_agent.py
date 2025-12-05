from typing import Any, Dict
import numpy as np

import sys
import os
# Add parent directory to path for sibling module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from computation import compute_fg_fp_for_window_session_multi_ref


class ComputeAgent:
    """Thin wrapper exposing compute functions as an agent.

    Methods:
      handle_request(session_id, action, args, context)
    """

    def __init__(self):
        pass

    async def handle_request(self, session_id: str, action: str, args: Dict[str, Any], context: Dict[str, Any]):
        # For now, only support 'amplitudes' action that forwards to compute_fg_fp_for_window_session_multi_ref
        if action in ("amplitudes", "fg-fp", "compute_amplitudes"):
            # Expect args to contain 'session' (a session reference) or the router will supply session
            session = args.get("session") or context.get("session")
            if session is None:
                raise ValueError("ComputeAgent requires a 'session' reference in args or context")

            # forward request dict to compute helper
            req = args.get("request", {})
            # Ensure arrays are numpy arrays where appropriate (the compute function tolerates lists but we'll convert)
            # compute_fg_fp_for_window_session_multi_ref expects session containing 'data' mapping to arrays/lists
            res = compute_fg_fp_for_window_session_multi_ref(session, req)
            return res

        raise ValueError(f"Unsupported compute action: {action}")

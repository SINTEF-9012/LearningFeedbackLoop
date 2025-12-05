"""Agents package: registry and helpers.

This package contains small agent implementations and a router used to
dispatch incoming requests to agent handlers. Files added here are
lightweight scaffolding to be extended later.
"""

from . import router  # expose router for app wiring

__all__ = ["router"]

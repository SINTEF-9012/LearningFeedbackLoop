"""Shared helpers for JSON-safe backend payloads."""

from __future__ import annotations

import math
from typing import Any


def finite_float(value: Any) -> float | None:
    """Convert a value to float and return None for non-finite values."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with None for JSON payloads."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value
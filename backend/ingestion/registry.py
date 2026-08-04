"""Registry for session-backed stream sources."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .mqtt_source import MqttStreamSource
from .simulated_casedata import SimulatedCasedataSource
from .simulated_file import SimulatedFileSource


SourceFactory = Callable[..., Any]

_SOURCE_FACTORIES: Dict[str, SourceFactory] = {
    "mqtt": MqttStreamSource,
    "simulated_file": SimulatedFileSource,
    "simulated_casedata": SimulatedCasedataSource,
}


def register_source(name: str, factory: SourceFactory) -> None:
    _SOURCE_FACTORIES[name] = factory


def create_source(name: str, sessions: Dict[str, Dict[str, Any]], **kwargs: Any) -> Any:
    if name not in _SOURCE_FACTORIES:
        raise KeyError(f"Unknown stream source: {name}")
    return _SOURCE_FACTORIES[name](sessions, **kwargs)


def registered_sources() -> list[str]:
    return sorted(_SOURCE_FACTORIES)
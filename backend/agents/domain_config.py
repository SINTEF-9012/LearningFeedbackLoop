"""
Domain Configuration Registry — data-driven adaptation for different sensor domains.

Instead of hardcoding channel names, feature extractors, and fault classifiers
for CNC machining, domain profiles encapsulate all domain-specific knowledge.
The system auto-detects or selects a profile at runtime, keeping the core
orchestration layer (scorer, memory, feedback) fully domain-agnostic.

Domain definitions live in YAML packs (``domain_packs/*.yaml``) — the single
source of truth. This module provides only the schema (``DomainConfig`` /
``ChannelRole`` / fault config) and the loading, auto-detection and selection
machinery; it does not hardcode any domain.

Usage:
    from backend.agents.domain_config import get_active_domain

    domain = get_active_domain(channel_names=session_channels)
    vib = window.get(domain.get_channel("primary_vibration"), np.array([]))
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fault indicator sigmoid configuration
# ---------------------------------------------------------------------------

@dataclass
class FaultIndicatorConfig:
    """One sigmoid-scored indicator within a fault type."""
    feature_name: str          # key in the feature dict (e.g. "hf_energy_ratio")
    sigmoid_center: float      # inflection point
    sigmoid_steepness: float   # slope (negative = inverted)
    weight: float              # contribution to composite fault score
    display_name: str = ""     # human-readable label (e.g. "HF energy burst")


@dataclass
class FaultTypeConfig:
    """Configuration for a single fault type (e.g. tool breakage, chatter)."""
    name: str                               # e.g. "tool_breakage"
    pattern_key: str                        # e.g. "fault:tool_breakage"
    severity: float                         # 0–1, used for alert triage
    indicators: List[FaultIndicatorConfig] = field(default_factory=list)
    dominant_threshold: float = 0.35        # score above this → dominant fault
    description: str = ""


# ---------------------------------------------------------------------------
# Channel role abstraction
# ---------------------------------------------------------------------------

class ChannelRole(str, Enum):
    """Semantic roles that channels can play, independent of actual names."""
    PRIMARY_VIBRATION = "primary_vibration"
    PRIMARY_POWER = "primary_power"
    SECONDARY_POWER = "secondary_power"
    CHATTER_AMPLITUDE = "chatter_amplitude"
    ACTIVE_POWER = "active_power"
    SPINDLE_SPEED = "spindle_speed"
    FEED_RATE = "feed_rate"
    TEMPERATURE = "temperature"
    GENERIC = "generic"


# ---------------------------------------------------------------------------
# Domain profile
# ---------------------------------------------------------------------------

@dataclass
class DomainConfig:
    """
    Complete domain-specific configuration.

    Encapsulates:
    - Channel name ↔ semantic role mapping
    - Fault type definitions (with indicator sigmoids)
    - Feature aliases (map generic names → domain-specific names)
    - Pattern keys tracked by the experiment framework
    - Columns that leak the target event (for experiment data leakage guard)
    - Channels used for the baseline z-score anomaly detector
    """
    name: str
    display_name: str = ""

    # Channel mapping: role → preferred channel name in the session data
    channel_roles: Dict[str, str] = field(default_factory=dict)

    # Optional fallback aliases for the same semantic role.
    channel_role_aliases: Dict[str, List[str]] = field(default_factory=dict)

    # Fault types registered for this domain
    fault_types: List[FaultTypeConfig] = field(default_factory=list)

    # Channels used by the z-score baseline anomaly detector
    z_score_channels: List[str] = field(default_factory=list)

    # Feature name aliases: canonical name → domain-specific feature name
    feature_aliases: Dict[str, str] = field(default_factory=dict)

    # Pattern keys tracked by the experiment framework
    pattern_keys: List[str] = field(default_factory=list)

    # Columns that encode the target event (data leakage)
    leaky_columns: List[str] = field(default_factory=list)

    # Metadata columns (not features)
    metadata_columns: List[str] = field(default_factory=list)

    # Known channel names that identify this domain (for auto-detection)
    signature_channels: Set[str] = field(default_factory=set)

    # ── Helpers ──────────────────────────────────────────────────────

    def get_channel(self, role: str, default: str = "") -> str:
        """Return the preferred channel name for a semantic role."""
        return self.channel_roles.get(role, default)

    def resolve_channel(
        self,
        role: str,
        available_channels: Optional[Iterable[str]] = None,
        default: str = "",
    ) -> str:
        """Return the first available channel candidate for a semantic role."""
        candidates: List[str] = []

        preferred = self.get_channel(role)
        if preferred:
            candidates.append(preferred)

        for alias in self.channel_role_aliases.get(role, []):
            if alias and alias not in candidates:
                candidates.append(alias)

        if not candidates:
            return default

        if available_channels is None:
            return candidates[0]

        available = set(available_channels)
        for candidate in candidates:
            if candidate in available:
                return candidate

        return default

    # Explicit severity overrides for pattern_keys not covered by fault_types.
    # Preserves backward compatibility with experiment scorer values.
    pattern_key_severities: Dict[str, float] = field(default_factory=dict)

    @property
    def fault_severity(self) -> Dict[str, float]:
        """Return {pattern_key: severity} mapping for all registered faults."""
        out: Dict[str, float] = {}
        for ft in self.fault_types:
            out[ft.pattern_key] = ft.severity
        # Apply explicit overrides for standalone pattern keys
        out.update(self.pattern_key_severities)
        # Fill remaining pattern_keys with a moderate default
        for pk in self.pattern_keys:
            if pk not in out:
                out[pk] = 0.80
        return out

    @property
    def fault_names(self) -> List[str]:
        return [ft.name for ft in self.fault_types]


# ---------------------------------------------------------------------------
# Domain registry — populated from YAML packs (domain_packs/*.yaml).
#
# YAML is the single source of truth for every domain (channel roles, fault
# indicators, signature channels). This module only supplies the schema above and
# the loading / detection / selection machinery. A tiny built-in fallback exists
# solely so the registry is never empty if the YAML directory is unreadable.
# ---------------------------------------------------------------------------

# Emergency fallback only — not edited per-domain; real domains live in YAML
# (see domain_packs/generic.yaml, which overrides this on load).
_FALLBACK_GENERIC_DOMAIN = DomainConfig(
    name="generic",
    display_name="Generic Time-Series",
)

_DOMAIN_REGISTRY: Dict[str, DomainConfig] = {"generic": _FALLBACK_GENERIC_DOMAIN}
_packs_loaded = False


def _ensure_packs_loaded() -> None:
    """Load every YAML domain pack into the registry on first access (idempotent).

    Imported lazily to avoid a circular import (``domain_packs`` imports the schema
    from this module). If loading fails, the built-in generic fallback remains so
    callers never see an empty registry.
    """
    global _packs_loaded
    if _packs_loaded:
        return
    _packs_loaded = True  # set first so register_domain calls don't re-enter
    try:
        from backend.agents.domain_pack_loader import register_packs, DEFAULT_PACK_DIR
        register_packs(DEFAULT_PACK_DIR)
    except Exception:
        logger.exception(
            "Failed to load domain packs from YAML; using built-in generic fallback"
        )


def _generic_domain() -> DomainConfig:
    """The generic/fallback domain (from generic.yaml when loaded, else built-in)."""
    return _DOMAIN_REGISTRY.get("generic", _FALLBACK_GENERIC_DOMAIN)


def register_domain(name: str, config: DomainConfig) -> None:
    """Register or override a domain profile in the registry.

    This is the hook the YAML pack loader uses, and is also available for
    programmatic / test registration of additional domains at runtime.
    """
    _DOMAIN_REGISTRY[name] = config
    logger.info("Registered domain profile: %s (%s)", name, config.display_name)


def get_domain(name: str) -> Optional[DomainConfig]:
    """Return a registered domain by name (loading YAML packs first), or None."""
    _ensure_packs_loaded()
    return _DOMAIN_REGISTRY.get(name)


def list_domains() -> List[str]:
    """Return names of all registered domain profiles."""
    _ensure_packs_loaded()
    return list(_DOMAIN_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def detect_domain_from_channels(channel_names: List[str]) -> DomainConfig:
    """Pick the best domain profile by matching channel names.

    Scores each registered domain by how many of its *signature_channels*
    appear in the supplied channel list. Falls back to the generic domain.
    """
    _ensure_packs_loaded()
    if not channel_names:
        return _generic_domain()

    names_set = set(channel_names)
    best_domain: DomainConfig = _generic_domain()
    best_overlap = 0

    for domain in _DOMAIN_REGISTRY.values():
        if not domain.signature_channels:
            continue
        overlap = len(names_set & domain.signature_channels)
        if overlap > best_overlap:
            best_overlap = overlap
            best_domain = domain

    # Require at least 2 matching channels to be confident
    if best_overlap >= 2:
        logger.info(
            "Auto-detected domain '%s' (%d/%d signature channels matched)",
            best_domain.name, best_overlap, len(best_domain.signature_channels),
        )
        return best_domain

    logger.info(
        "No domain profile matched channels %s — using generic profile",
        channel_names[:5],
    )
    return _generic_domain()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_active_domain: Optional[DomainConfig] = None


def get_active_domain(
    channel_names: Optional[List[str]] = None,
    force: bool = False,
) -> DomainConfig:
    """Return the active domain profile.

    Resolution order:
    1. ``DOMAIN_PROFILE`` environment variable (exact name or "auto")
    2. Auto-detection from *channel_names* if provided
    3. Cached active domain from a previous call
    4. ``cnc_machining`` (the default machining domain), else generic

    Set *force=True* to re-evaluate even if a domain is cached.
    """
    global _active_domain

    if _active_domain is not None and not force:
        return _active_domain

    _ensure_packs_loaded()

    env_profile = os.environ.get("DOMAIN_PROFILE", "").strip().lower()

    if env_profile and env_profile != "auto":
        if env_profile in _DOMAIN_REGISTRY:
            _active_domain = _DOMAIN_REGISTRY[env_profile]
            logger.info("Domain profile set from env: %s", _active_domain.name)
            return _active_domain
        else:
            logger.warning(
                "DOMAIN_PROFILE='%s' not found in registry (available: %s). "
                "Falling back to auto-detection.",
                env_profile, list(_DOMAIN_REGISTRY.keys()),
            )

    if channel_names:
        _active_domain = detect_domain_from_channels(channel_names)
        return _active_domain

    # Ultimate fallback: the machining default if loaded, else generic.
    _active_domain = _DOMAIN_REGISTRY.get("cnc_machining") or _generic_domain()
    return _active_domain


def reset_active_domain() -> None:
    """Clear the cached active domain (for testing)."""
    global _active_domain
    _active_domain = None

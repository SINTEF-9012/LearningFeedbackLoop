"""YAML-backed domain packs.

YAML is the single source of truth for every domain: the shipped domains
(``cnc.yaml`` for SITE_C/Site_b/SITE_A, ``site_a_line2.yaml``, ``generic.yaml``) and any
new process are all defined as YAML files under ``domain_packs/`` — no domain is
hardcoded in Python. The loader returns
:class:`backend.agents.domain_config.DomainConfig` instances so every existing
scorer/feature/alert path keeps working, and ``register_packs`` registers them into
the domain registry (loaded lazily by ``domain_config`` on first use).

Pack schema (all sections optional unless noted). Channel names must match the
dataset; use ``channel_role_aliases`` to cover naming variants of similar machines:

.. code-block:: yaml

    name: my_domain                            # required
    display_name: "My Domain"
    channel_roles:
        primary_vibration: "Vibration_Severity_X"
        primary_power: "Power_Spindle"
    channel_role_aliases:                      # cover similar-but-not-identical machines
        primary_power: ["Spindle_Power"]
    signature_channels: ["Vibration_Severity_X", "Power_Spindle"]
    z_score_channels: ["Power_Spindle"]
    feature_aliases:
        canonical_name: domain_specific_name
    pattern_keys: ["SPINDLE_POWER_SURGE"]
    pattern_key_severities:
        SPINDLE_POWER_SURGE: 0.9
    leaky_columns: ["label"]
    metadata_columns: ["sample_id", "timestamp"]
    thresholds:
        chatter_ratio_threshold: 5.0
        severity_alert_threshold: 0.7
    fault_types:
        - name: tool_breakage
          pattern_key: fault:tool_breakage
          severity: 0.95
          description: "..."
          dominant_threshold: 0.35
          indicators:
            - feature_name: hf_energy_ratio
              sigmoid_center: 0.15
              sigmoid_steepness: 20.0
              weight: 0.35
              display_name: "HF energy burst"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from backend.agents.domain_config import (
    DomainConfig,
    FaultIndicatorConfig,
    FaultTypeConfig,
    register_domain,
)

logger = logging.getLogger(__name__)


DEFAULT_PACK_DIR = Path(__file__).resolve().parents[2] / "domain_packs"


class DomainPackError(ValueError):
    """Raised for structurally invalid pack YAML."""


def _require(pack: Mapping[str, Any], key: str, path: Path) -> Any:
    if key not in pack:
        raise DomainPackError(f"Pack {path} missing required key {key!r}")
    return pack[key]


def _parse_indicator(raw: Mapping[str, Any], path: Path) -> FaultIndicatorConfig:
    required = ("feature_name", "sigmoid_center", "sigmoid_steepness", "weight")
    for k in required:
        if k not in raw:
            raise DomainPackError(f"Pack {path}: indicator missing {k!r}")
    return FaultIndicatorConfig(
        feature_name=str(raw["feature_name"]),
        sigmoid_center=float(raw["sigmoid_center"]),
        sigmoid_steepness=float(raw["sigmoid_steepness"]),
        weight=float(raw["weight"]),
        display_name=str(raw.get("display_name", "")),
    )


def _parse_fault(raw: Mapping[str, Any], path: Path) -> FaultTypeConfig:
    required = ("name", "pattern_key", "severity")
    for k in required:
        if k not in raw:
            raise DomainPackError(f"Pack {path}: fault missing {k!r}")
    indicators = [
        _parse_indicator(ind, path) for ind in (raw.get("indicators") or [])
    ]
    return FaultTypeConfig(
        name=str(raw["name"]),
        pattern_key=str(raw["pattern_key"]),
        severity=float(raw["severity"]),
        indicators=indicators,
        dominant_threshold=float(raw.get("dominant_threshold", 0.35)),
        description=str(raw.get("description", "")),
    )


def load_pack(path: str | Path) -> DomainConfig:
    """Load one YAML domain pack and return a :class:`DomainConfig`.

    Missing optional sections default to empty lists/dicts. Unknown
    top-level keys are ignored with a DEBUG log so pack authors can
    sketch fields before the loader is extended without breaking the
    boot path.
    """
    pack_path = Path(path)
    try:
        raw = yaml.safe_load(pack_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise DomainPackError(f"Pack {pack_path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise DomainPackError(f"Pack {pack_path} top level must be a mapping")

    name = str(_require(raw, "name", pack_path))
    fault_types = [_parse_fault(ft, pack_path) for ft in (raw.get("fault_types") or [])]
    thresholds = dict(raw.get("thresholds") or {})

    config = DomainConfig(
        name=name,
        display_name=str(raw.get("display_name", name.replace("_", " ").title())),
        channel_roles={str(k): str(v) for k, v in (raw.get("channel_roles") or {}).items()},
        channel_role_aliases={
            str(k): [str(v) for v in (vals or [])]
            for k, vals in (raw.get("channel_role_aliases") or {}).items()
        },
        fault_types=fault_types,
        z_score_channels=[str(c) for c in (raw.get("z_score_channels") or [])],
        feature_aliases={str(k): str(v) for k, v in (raw.get("feature_aliases") or {}).items()},
        pattern_keys=[str(pk) for pk in (raw.get("pattern_keys") or [])],
        leaky_columns=[str(c) for c in (raw.get("leaky_columns") or [])],
        metadata_columns=[str(c) for c in (raw.get("metadata_columns") or [])],
        signature_channels=set(str(c) for c in (raw.get("signature_channels") or [])),
        pattern_key_severities={str(k): float(v) for k, v in (raw.get("pattern_key_severities") or {}).items()},
    )

    # Attach thresholds out-of-band — DomainConfig doesn't reserve a
    # field for them, so we stash on an attribute readable by callers
    # and on feature_aliases-adjacent metadata would conflict. A plain
    # dynamic attribute keeps the static schema unchanged.
    object.__setattr__(config, "thresholds", thresholds)
    # Mark provenance so callers (e.g. the /domain router) can report the source
    # without inferring it from whether thresholds happen to be present.
    object.__setattr__(config, "loaded_from_yaml", True)

    # Warn on unknown keys to help catch typos.
    known = {
        "name", "display_name", "channel_roles", "channel_role_aliases",
        "signature_channels", "z_score_channels", "feature_aliases", "pattern_keys",
        "leaky_columns", "metadata_columns", "pattern_key_severities",
        "thresholds", "fault_types",
    }
    for key in raw:
        if key not in known:
            logger.debug("Pack %s: ignoring unknown key %r", pack_path, key)

    return config


def load_packs(directory: str | Path = DEFAULT_PACK_DIR) -> Dict[str, DomainConfig]:
    """Load every ``*.yaml`` under *directory*. Returns {name: config}.

    Never raises if the directory is missing; returns empty dict.
    Individual pack failures are logged and skipped so one bad file
    doesn't block the rest.
    """
    root = Path(directory)
    out: Dict[str, DomainConfig] = {}
    if not root.is_dir():
        logger.info("load_packs: directory %s not found; skipping", root)
        return out
    for path in sorted(root.glob("*.yaml")):
        try:
            cfg = load_pack(path)
            out[cfg.name] = cfg
        except DomainPackError:
            logger.exception("load_packs: failed to parse %s", path)
    return out


def register_packs(directory: str | Path = DEFAULT_PACK_DIR) -> List[str]:
    """Load and register every pack; return names registered.

    YAML packs are the source of truth and DO override a same-named built-in domain
    (e.g. ``cnc.yaml`` replaces the Python ``cnc_machining`` default). The YAML must
    therefore stay aligned to the data — see ``domain_packs/cnc.yaml``. Idempotent:
    calling twice overwrites the previous registration (reload behaviour during dev).
    """
    packs = load_packs(directory)
    for name, cfg in packs.items():
        register_domain(name, cfg)
    return sorted(packs.keys())


def get_threshold(config: DomainConfig, key: str, default: float) -> float:
    """Read a named threshold from the pack, falling back to *default*.

    Safe to call on packs that weren't loaded via :func:`load_pack` —
    returns *default* when the attribute is absent.
    """
    thresholds = getattr(config, "thresholds", None) or {}
    try:
        return float(thresholds.get(key, default))
    except (TypeError, ValueError):
        return float(default)

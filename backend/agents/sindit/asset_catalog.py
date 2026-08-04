"""SINDIT asset-catalog builders — Agent F (2026-04-24).

Typed helpers to construct SINDIT 2.0 payloads for every asset type LFL
uses to enrich the current-state digital twin. The builders are pure
(dict-in, dict-out) so they can be unit-tested without a running SINDIT
instance. A single :func:`sync_catalog` coroutine then hands a
``SinditCatalog`` to a live :class:`~backend.agents.sindit.client.SinditClient`
for upsert.

Per the plan (Agent F, 2026-04-23):

- Add asset types: ``Tool``, ``Workpiece``, ``Fixture``, ``Spindle``,
  ``ControllerProgram``.
- Add model-metadata asset (``urn:lfl:model:<name>``) so model lineage
  lives beside the twin.
- Parameterise the hardcoded machine URN (``urn:lfl:asset:cnc-machine-1``).
- Do **not** push ``Memory`` / ``Feedback`` / ``Pattern-history`` into
  SINDIT — those stay in Neo4j.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


# ── SAMM / SINDIT type URNs ────────────────────────────────────────────

SAMM_ASSET_TYPE = "urn:samm:sindit.sintef.no:1.0.0#AbstractAsset"
SAMM_PROPERTY_TYPE = "urn:samm:sindit.sintef.no:1.0.0#AbstractAssetProperty"

# LFL-specific asset sub-types (used only in payload metadata so downstream
# queries can filter; SINDIT itself keys everything off the SAMM base type).
LFL_ASSET_KIND_KEY = "lflAssetKind"
LFL_MACHINE_URI_FMT = "urn:lfl:asset:{machine_id}"
LFL_TOOL_URI_FMT = "urn:lfl:tool:{tool_id}"
LFL_WORKPIECE_URI_FMT = "urn:lfl:workpiece:{workpiece_id}"
LFL_FIXTURE_URI_FMT = "urn:lfl:fixture:{fixture_id}"
LFL_SPINDLE_URI_FMT = "urn:lfl:spindle:{spindle_id}"
LFL_PROGRAM_URI_FMT = "urn:lfl:program:{program_id}"
LFL_MODEL_URI_FMT = "urn:lfl:model:{model_name}"
LFL_PROP_URI_FMT = "urn:lfl:property:{parent_id}:{property_name}"
LFL_RELATION_TYPES = {
    "HAS_TOOL",
    "HAS_WORKPIECE",
    "HAS_FIXTURE",
    "HAS_SPINDLE",
    "RUNS_PROGRAM",
    "TRAINED_FOR",
}


# ── Normalisation ──────────────────────────────────────────────────────


def _slug(value: str) -> str:
    """Normalise free-form IDs into URN-safe tokens."""
    out = []
    for ch in str(value or "").strip():
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch.lower())
        elif ch.isspace() or ch in (":", "/", "\\", ".", ","):
            out.append("-")
    slug = "".join(out).strip("-")
    return slug or "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_description(text: Optional[str]) -> str:
    return (text or "").strip()


# ── Asset payload builders ─────────────────────────────────────────────


def build_machine_asset(
    machine_id: str,
    *,
    label: Optional[str] = None,
    make: Optional[str] = None,
    model: Optional[str] = None,
    controller_version: Optional[str] = None,
    max_rpm: Optional[float] = None,
    site: Optional[str] = None,
    operator_shift: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a :class:`Machine` asset payload.

    ``machine_id`` is slugified into the URN so callers can pass either
    ``"CNC-1"`` or ``"cnc_1"`` and get the same asset.
    """
    mid = _slug(machine_id)
    uri = LFL_MACHINE_URI_FMT.format(machine_id=mid)
    payload: Dict[str, Any] = {
        "uri": uri,
        "label": label or machine_id or mid,
        "assetType": SAMM_ASSET_TYPE,
        "assetDescription": _clean_description(description)
            or "LFL-monitored CNC machine (current state).",
        LFL_ASSET_KIND_KEY: "Machine",
    }
    extras: Dict[str, Any] = {}
    if make is not None:
        extras["make"] = make
    if model is not None:
        extras["model"] = model
    if controller_version is not None:
        extras["controllerVersion"] = controller_version
    if max_rpm is not None:
        extras["maxRpm"] = float(max_rpm)
    if site is not None:
        extras["site"] = site
    if operator_shift is not None:
        extras["operatorShift"] = operator_shift
    payload["metadata"] = extras
    return payload


def build_tool_asset(
    tool_id: str,
    *,
    label: Optional[str] = None,
    geometry: Optional[str] = None,
    material: Optional[str] = None,
    diameter_mm: Optional[float] = None,
    teeth: Optional[int] = None,
    max_feed_mm_min: Optional[float] = None,
    max_rpm: Optional[float] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    tid = _slug(tool_id)
    payload: Dict[str, Any] = {
        "uri": LFL_TOOL_URI_FMT.format(tool_id=tid),
        "label": label or tool_id or tid,
        "assetType": SAMM_ASSET_TYPE,
        "assetDescription": _clean_description(description) or "Cutting tool.",
        LFL_ASSET_KIND_KEY: "Tool",
    }
    extras: Dict[str, Any] = {}
    if geometry is not None:
        extras["geometry"] = geometry
    if material is not None:
        extras["material"] = material
    if diameter_mm is not None:
        extras["diameterMm"] = float(diameter_mm)
    if teeth is not None:
        extras["teeth"] = int(teeth)
    if max_feed_mm_min is not None:
        extras["maxFeedMmPerMin"] = float(max_feed_mm_min)
    if max_rpm is not None:
        extras["maxRpm"] = float(max_rpm)
    payload["metadata"] = extras
    return payload


def build_workpiece_asset(
    workpiece_id: str,
    *,
    label: Optional[str] = None,
    material: Optional[str] = None,
    hardness: Optional[str] = None,
    geometry: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    wid = _slug(workpiece_id)
    payload = {
        "uri": LFL_WORKPIECE_URI_FMT.format(workpiece_id=wid),
        "label": label or workpiece_id or wid,
        "assetType": SAMM_ASSET_TYPE,
        "assetDescription": _clean_description(description) or "Workpiece.",
        LFL_ASSET_KIND_KEY: "Workpiece",
    }
    extras: Dict[str, Any] = {}
    if material is not None:
        extras["material"] = material
    if hardness is not None:
        extras["hardness"] = hardness
    if geometry is not None:
        extras["geometry"] = geometry
    payload["metadata"] = extras
    return payload


def build_fixture_asset(
    fixture_id: str,
    *,
    label: Optional[str] = None,
    fixture_type: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    fid = _slug(fixture_id)
    payload = {
        "uri": LFL_FIXTURE_URI_FMT.format(fixture_id=fid),
        "label": label or fixture_id or fid,
        "assetType": SAMM_ASSET_TYPE,
        "assetDescription": _clean_description(description) or "Workpiece fixture.",
        LFL_ASSET_KIND_KEY: "Fixture",
    }
    extras: Dict[str, Any] = {}
    if fixture_type is not None:
        extras["fixtureType"] = fixture_type
    payload["metadata"] = extras
    return payload


def build_spindle_asset(
    spindle_id: str,
    *,
    label: Optional[str] = None,
    max_rpm: Optional[float] = None,
    power_kw: Optional[float] = None,
    bearing_type: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    sid = _slug(spindle_id)
    payload = {
        "uri": LFL_SPINDLE_URI_FMT.format(spindle_id=sid),
        "label": label or spindle_id or sid,
        "assetType": SAMM_ASSET_TYPE,
        "assetDescription": _clean_description(description) or "Machine spindle.",
        LFL_ASSET_KIND_KEY: "Spindle",
    }
    extras: Dict[str, Any] = {}
    if max_rpm is not None:
        extras["maxRpm"] = float(max_rpm)
    if power_kw is not None:
        extras["powerKw"] = float(power_kw)
    if bearing_type is not None:
        extras["bearingType"] = bearing_type
    payload["metadata"] = extras
    return payload


def build_controller_program_asset(
    program_id: str,
    *,
    label: Optional[str] = None,
    controller_version: Optional[str] = None,
    program_source: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    pid = _slug(program_id)
    payload = {
        "uri": LFL_PROGRAM_URI_FMT.format(program_id=pid),
        "label": label or program_id or pid,
        "assetType": SAMM_ASSET_TYPE,
        "assetDescription": _clean_description(description) or "Controller program.",
        LFL_ASSET_KIND_KEY: "ControllerProgram",
    }
    extras: Dict[str, Any] = {}
    if controller_version is not None:
        extras["controllerVersion"] = controller_version
    if program_source is not None:
        extras["programSource"] = program_source
    payload["metadata"] = extras
    return payload


def build_model_metadata_asset(
    model_name: str,
    *,
    label: Optional[str] = None,
    trained_at: Optional[str] = None,
    n_samples: Optional[int] = None,
    current_f1: Optional[float] = None,
    dataset: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a ModelMetadata asset. Lineage lives beside the twin.

    No model weights ever ship here — only metadata. Weights stay in
    ``data/models/`` and are exported separately via the knowledge pack
    (Agent H).
    """
    mid = _slug(model_name)
    payload = {
        "uri": LFL_MODEL_URI_FMT.format(model_name=mid),
        "label": label or model_name or mid,
        "assetType": SAMM_ASSET_TYPE,
        "assetDescription": _clean_description(description) or f"LFL model metadata: {model_name}.",
        LFL_ASSET_KIND_KEY: "ModelMetadata",
    }
    extras: Dict[str, Any] = {
        "trainedAt": trained_at or _now_iso(),
    }
    if n_samples is not None:
        extras["nSamples"] = int(n_samples)
    if current_f1 is not None:
        extras["currentF1"] = float(current_f1)
    if dataset is not None:
        extras["dataset"] = dataset
    payload["metadata"] = extras
    return payload


# ── Property / relationship builders ──────────────────────────────────


def build_property(
    parent_uri: str,
    property_name: str,
    *,
    value: Any = None,
    unit: str = "",
    data_type: str = "float",
    label: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a property payload attached to ``parent_uri``.

    ``parent_uri`` must be a full SINDIT IRI (e.g. the ``uri`` field of
    one of the asset builders above). The property URI is derived from
    the parent's tail segment + property name so duplicates are stable.
    """
    parent_tail = parent_uri.rsplit(":", 1)[-1] or "root"
    prop_uri = LFL_PROP_URI_FMT.format(
        parent_id=_slug(parent_tail),
        property_name=_slug(property_name),
    )
    return {
        "uri": prop_uri,
        "label": label or property_name.replace("_", " ").title(),
        "propertyName": property_name,
        "propertyValue": "" if value is None else str(value),
        "propertyUnit": unit,
        "propertyDataType": data_type,
        "propertyValueTimestamp": timestamp or _now_iso(),
        "assetUri": parent_uri,
    }


def build_relationship(
    source_uri: str,
    target_uri: str,
    relationship_type: str,
) -> Dict[str, Any]:
    """Build a relationship payload.

    ``relationship_type`` must be one of :data:`LFL_RELATION_TYPES` to
    keep the SINDIT ontology disciplined; raises ``ValueError`` on an
    unknown type so mistakes surface at build time, not on the wire.
    """
    if relationship_type not in LFL_RELATION_TYPES:
        raise ValueError(
            f"Unknown relationship_type={relationship_type!r}; "
            f"allowed={sorted(LFL_RELATION_TYPES)}"
        )
    # SINDIT persists a relationship as its own node and needs an absolute IRI
    # for it; without one it auto-generates a bare UUID and rejects it
    # ("Not a valid (absolute) IRI"). Derive a stable URN from the endpoints +
    # type so re-imports are idempotent (ISS-42a).
    rel_key = hashlib.sha1(
        f"{source_uri}|{relationship_type}|{target_uri}".encode("utf-8")
    ).hexdigest()[:20]
    return {
        "uri": f"urn:lfl:rel:{rel_key}",
        "sourceUri": source_uri,
        "targetUri": target_uri,
        "relationshipType": relationship_type,
    }


# ── Sensor → SINDIT property catalog ──────────────────────────────────
# Source-of-truth for feature-field → {label, unit} mapping. Replaces the
# hardcoded ``_SENSOR_FIELDS`` map that used to live in
# ``backend/agents/sindit/bridge.py`` (plan point 6, Agent F). Consumers
# should import this constant rather than redefining their own map.

SENSOR_PROPERTY_CATALOG: Dict[str, Dict[str, str]] = {
    "spindle_speed":   {"label": "Spindle Speed",   "unit": "rpm"},
    "feed_rate":       {"label": "Feed Rate",       "unit": "mm/min"},
    "depth_of_cut":    {"label": "Depth of Cut",    "unit": "mm"},
    "vibration_x":     {"label": "Vibration X",     "unit": "g"},
    "vibration_y":     {"label": "Vibration Y",     "unit": "g"},
    "vibration_z":     {"label": "Vibration Z",     "unit": "g"},
    "acoustic_rms":    {"label": "Acoustic RMS",    "unit": "dB"},
    "current_draw":    {"label": "Current Draw",    "unit": "A"},
    "temperature":     {"label": "Temperature",     "unit": "°C"},
    "tool_wear_pct":   {"label": "Tool Wear",       "unit": "%"},
    "force_x":         {"label": "Force X",         "unit": "N"},
    "force_y":         {"label": "Force Y",         "unit": "N"},
    "force_z":         {"label": "Force Z",         "unit": "N"},
}


def get_sensor_property_meta(field_name: str) -> Dict[str, str]:
    """Return ``{label, unit}`` for a sensor field, with sensible defaults.

    Unknown fields receive a title-cased label derived from the field
    name and an empty unit, matching the pre-refactor fallback behaviour
    in ``SinditBridge._ensure_property``.
    """
    meta = SENSOR_PROPERTY_CATALOG.get(field_name)
    if meta is not None:
        return dict(meta)
    return {
        "label": field_name.replace("_", " ").title(),
        "unit": "",
    }


# ── Catalog (bundle of upsertable items) ──────────────────────────────


@dataclass
class SinditCatalog:
    """Collection of SINDIT payloads to upsert in one pass.

    Instances are composed by calling builders and appending the return
    values. Order is preserved — assets should come before properties
    and properties before relationships so target IRIs exist by the time
    SINDIT receives the relationship call.
    """

    assets: List[Dict[str, Any]] = field(default_factory=list)
    properties: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience composition helpers
    # ------------------------------------------------------------------

    def add_asset(self, asset: Dict[str, Any]) -> "SinditCatalog":
        self.assets.append(asset)
        return self

    def add_property(self, prop: Dict[str, Any]) -> "SinditCatalog":
        self.properties.append(prop)
        return self

    def add_relationship(self, rel: Dict[str, Any]) -> "SinditCatalog":
        self.relationships.append(rel)
        return self

    def extend(self, other: "SinditCatalog") -> "SinditCatalog":
        self.assets.extend(other.assets)
        self.properties.extend(other.properties)
        self.relationships.extend(other.relationships)
        return self

    def uris(self) -> List[str]:
        """Return all asset + property IRIs in the catalog."""
        out: List[str] = []
        for a in self.assets:
            uri = a.get("uri")
            if isinstance(uri, str):
                out.append(uri)
        for p in self.properties:
            uri = p.get("uri")
            if isinstance(uri, str):
                out.append(uri)
        return out

    def summary(self) -> Dict[str, int]:
        return {
            "assets": len(self.assets),
            "properties": len(self.properties),
            "relationships": len(self.relationships),
        }


async def sync_catalog(client: Any, catalog: SinditCatalog) -> Dict[str, int]:
    """Upsert the full catalog via a :class:`SinditClient`.

    Returns a summary of per-kind success counts. Never raises — each
    failure is logged by the client and recorded as a miss. Skipping
    duplicates is left to SINDIT (the API is upsert-by-URI).
    """
    import logging

    log = logging.getLogger(__name__)
    results = {"assets_ok": 0, "properties_ok": 0, "relationships_ok": 0,
               "assets_fail": 0, "properties_fail": 0, "relationships_fail": 0}

    for asset in catalog.assets:
        try:
            ok = await client.post_asset(asset)
        except Exception:  # pragma: no cover - defensive
            log.exception("sync_catalog: post_asset %s failed", asset.get("uri"))
            ok = None
        results["assets_ok" if ok else "assets_fail"] += 1

    for prop in catalog.properties:
        try:
            ok = await client.post_property(prop)
        except Exception:
            log.exception("sync_catalog: post_property %s failed", prop.get("uri"))
            ok = None
        results["properties_ok" if ok else "properties_fail"] += 1

    for rel in catalog.relationships:
        try:
            ok = await client.post_relationship(rel)
        except Exception:
            log.exception("sync_catalog: post_relationship failed")
            ok = None
        results["relationships_ok" if ok else "relationships_fail"] += 1

    return results


# ── High-level composition: default machine kit ───────────────────────


def build_default_cnc_kit(
    machine_id: str,
    *,
    tool_id: str = "default-tool",
    workpiece_id: str = "default-workpiece",
    fixture_id: Optional[str] = None,
    spindle_id: Optional[str] = None,
    program_id: Optional[str] = None,
    machine_overrides: Optional[Dict[str, Any]] = None,
    tool_overrides: Optional[Dict[str, Any]] = None,
    workpiece_overrides: Optional[Dict[str, Any]] = None,
) -> SinditCatalog:
    """Bootstrap a minimal current-state kit for one machine.

    Mirrors the hardcoded bridge bootstrap, now parameterised. Emits
    Machine + Tool + Workpiece plus optional Fixture / Spindle /
    ControllerProgram, and the canonical ``HAS_*`` relationships
    pointing from the machine.
    """
    catalog = SinditCatalog()
    machine = build_machine_asset(machine_id, **(machine_overrides or {}))
    tool = build_tool_asset(tool_id, **(tool_overrides or {}))
    workpiece = build_workpiece_asset(workpiece_id, **(workpiece_overrides or {}))
    catalog.add_asset(machine).add_asset(tool).add_asset(workpiece)
    catalog.add_relationship(build_relationship(machine["uri"], tool["uri"], "HAS_TOOL"))
    catalog.add_relationship(build_relationship(machine["uri"], workpiece["uri"], "HAS_WORKPIECE"))

    if fixture_id is not None:
        fx = build_fixture_asset(fixture_id)
        catalog.add_asset(fx)
        catalog.add_relationship(build_relationship(machine["uri"], fx["uri"], "HAS_FIXTURE"))
    if spindle_id is not None:
        sp = build_spindle_asset(spindle_id)
        catalog.add_asset(sp)
        catalog.add_relationship(build_relationship(machine["uri"], sp["uri"], "HAS_SPINDLE"))
    if program_id is not None:
        pg = build_controller_program_asset(program_id)
        catalog.add_asset(pg)
        catalog.add_relationship(build_relationship(machine["uri"], pg["uri"], "RUNS_PROGRAM"))

    return catalog

"""Build and optionally import the tool master into SINDIT.

Usage:

    python -m backend.agents.sindit.import_tool_master --dry-run
    python -m backend.agents.sindit.import_tool_master
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from backend.agents.config import SINDIT_API_URL, SINDIT_TIMEOUT_S
from backend.agents.processing.tool_lookup import (
    MACHINE_FAMILIES_PATH,
    ToolSpec,
    load_machine_family_registry,
    load_tool_master,
)

from .asset_catalog import (
    SinditCatalog,
    build_machine_asset,
    build_property,
    build_relationship,
    build_tool_asset,
    build_workpiece_asset,
    sync_catalog,
)
from .client import SinditClient

logger = logging.getLogger(__name__)

_SITE_B_CASEDATA_MACHINE_IDS = (
    "Site_b - MACHINE_B1 - CASE_B1",
    "Site_b - MACHINE_B2 - CASE_B2",
)

_SHARED_WORKPIECE_SPECS = (
    {
        "workpiece_id": "site_b-casedata-shared-workpiece",
        "label": "Site_b casedata shared workpiece",
        "description": "Shared workpiece represented across Site_b MACHINE_B1 / CASE_B1 and MACHINE_B2 / CASE_B2.",
        "machine_ids": _SITE_B_CASEDATA_MACHINE_IDS,
        "properties": (
            ("DatasetId", "site_b_casedata", "string"),
            ("DatasetLabel", "Site_b casedata", "string"),
            ("SharedAcrossMachines", True, "boolean"),
        ),
    },
)


def build_tool_master_catalog(
    *,
    master: Mapping[tuple[str, int], ToolSpec] | None = None,
    family_to_machine_ids: Mapping[str, Sequence[str]] | None = None,
    imported_at: str | None = None,
) -> SinditCatalog:
    """Build a SINDIT catalog containing tool assets and HAS_TOOL edges."""
    tool_master = master or load_tool_master()
    family_map = family_to_machine_ids or load_machine_family_registry()
    imported_ts = imported_at or _now_iso()

    catalog = SinditCatalog()
    machine_assets: dict[str, dict] = {}

    for machine_ids in family_map.values():
        for machine_id in machine_ids:
            if machine_id in machine_assets:
                continue
            asset = build_machine_asset(machine_id, label=machine_id)
            machine_assets[machine_id] = asset
            catalog.add_asset(asset)

    _add_shared_workpiece_assets(catalog, machine_assets)

    for (family, tool_number), spec in sorted(tool_master.items()):
        tool_asset = build_tool_asset(
            tool_id=f"{family}-t{tool_number}",
            label=_tool_label(family, tool_number, spec),
            geometry=spec.tool_type,
            material=spec.tool_substrate,
            diameter_mm=spec.diameter_mm,
            teeth=spec.teeth,
            description=spec.description,
        )
        catalog.add_asset(tool_asset)

        for machine_id in family_map.get(family, []):
            machine_asset = machine_assets.get(machine_id)
            if machine_asset is None:
                machine_asset = build_machine_asset(machine_id, label=machine_id)
                machine_assets[machine_id] = machine_asset
                catalog.add_asset(machine_asset)
            catalog.add_relationship(
                build_relationship(machine_asset["uri"], tool_asset["uri"], "HAS_TOOL")
            )

        if spec.diameter_mm is not None:
            catalog.add_property(_build_labeled_property(
                tool_asset["uri"], "ToolDiameter", spec.diameter_mm, "mm", "float"
            ))
        if spec.teeth is not None:
            catalog.add_property(_build_labeled_property(
                tool_asset["uri"], "NumberOfTeeth", spec.teeth, "", "int"
            ))
        if spec.tool_type:
            catalog.add_property(_build_labeled_property(
                tool_asset["uri"], "ToolType", spec.tool_type, "", "string"
            ))
        if spec.tool_length_mm is not None:
            catalog.add_property(_build_labeled_property(
                tool_asset["uri"], "ToolLength", spec.tool_length_mm, "mm", "float"
            ))
        if spec.tool_substrate:
            catalog.add_property(_build_labeled_property(
                tool_asset["uri"], "ToolMaterial", spec.tool_substrate, "", "string"
            ))
        if spec.tool_id:
            catalog.add_property(_build_labeled_property(
                tool_asset["uri"], "MasterToolID", spec.tool_id, "", "string"
            ))
        catalog.add_property(_build_labeled_property(
            tool_asset["uri"], "LastImportedAt", imported_ts, "", "string"
        ))
        catalog.add_property(_build_labeled_property(
            tool_asset["uri"], "SourceWorkbook", spec.source, "", "string"
        ))

    return catalog


async def import_tool_master(
    *,
    master: Mapping[tuple[str, int], ToolSpec] | None = None,
    family_to_machine_ids: Mapping[str, Sequence[str]] | None = None,
    machine_families_path: Path | None = None,
    imported_at: str | None = None,
    base_url: str = SINDIT_API_URL,
    timeout_s: float = SINDIT_TIMEOUT_S,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, int]:
    """Authenticate and upsert the full tool-master catalog into SINDIT."""
    family_map = family_to_machine_ids or load_machine_family_registry(machine_families_path)
    catalog = build_tool_master_catalog(
        master=master,
        family_to_machine_ids=family_map,
        imported_at=imported_at,
    )
    username = username or os.environ.get("SINDIT_USERNAME", "sindit")
    password = password or os.environ.get("SINDIT_PASSWORD", "sindit")

    async with SinditClient(base_url=base_url, timeout=timeout_s) as client:
        ok = await client.authenticate(username, password)
        if not ok:
            raise RuntimeError(f"SINDIT auth failed for {base_url}")
        return await sync_catalog(client, catalog)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-families", default=str(MACHINE_FAMILIES_PATH))
    parser.add_argument("--api-url", default=SINDIT_API_URL)
    parser.add_argument("--timeout", type=float, default=SINDIT_TIMEOUT_S)
    parser.add_argument("--username", default=os.environ.get("SINDIT_USERNAME", "sindit"))
    parser.add_argument("--password", default=os.environ.get("SINDIT_PASSWORD", "sindit"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-master", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    family_map = load_machine_family_registry(Path(args.machine_families), refresh=True)
    master = load_tool_master(refresh=args.refresh_master)
    catalog = build_tool_master_catalog(
        master=master,
        family_to_machine_ids=family_map,
    )

    if args.dry_run:
        payload = {
            "machine_families_path": str(Path(args.machine_families)),
            "family_count": len(family_map),
            "family_to_machine_counts": {
                family: len(machine_ids) for family, machine_ids in family_map.items()
            },
            "tool_master_entries": len(master),
            "catalog": catalog.summary(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    results = asyncio.run(import_tool_master(
        master=master,
        family_to_machine_ids=family_map,
        base_url=args.api_url,
        timeout_s=args.timeout,
        username=args.username,
        password=args.password,
    ))
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _build_labeled_property(
    parent_uri: str,
    property_name: str,
    value: object,
    unit: str,
    data_type: str,
) -> dict:
    return build_property(
        parent_uri,
        property_name,
        value=value,
        unit=unit,
        data_type=data_type,
        label=property_name,
    )


def _tool_label(family: str, tool_number: int, spec: ToolSpec) -> str:
    description = (spec.description or "").strip()
    if description:
        return f"{family.title()} T{tool_number} - {description}"
    return f"{family.title()} T{tool_number}"


def _add_shared_workpiece_assets(
    catalog: SinditCatalog,
    machine_assets: Mapping[str, dict],
) -> None:
    for spec in _SHARED_WORKPIECE_SPECS:
        matched_machine_ids = [machine_id for machine_id in spec["machine_ids"] if machine_id in machine_assets]
        if not matched_machine_ids:
            continue

        workpiece_asset = build_workpiece_asset(
            spec["workpiece_id"],
            label=spec["label"],
            description=spec["description"],
        )
        catalog.add_asset(workpiece_asset)

        for property_name, property_value, data_type in spec.get("properties", ()):  # type: ignore[union-attr]
            catalog.add_property(
                _build_labeled_property(
                    workpiece_asset["uri"],
                    property_name,
                    property_value,
                    "",
                    data_type,
                )
            )

        for machine_id in matched_machine_ids:
            catalog.add_relationship(
                build_relationship(machine_assets[machine_id]["uri"], workpiece_asset["uri"], "HAS_WORKPIECE")
            )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
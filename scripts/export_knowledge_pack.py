"""CLI: export a knowledge pack for one site.

Usage::

    python -m scripts.export_knowledge_pack \\
        --data-dir data/ --site CNC-1 \\
        --machine-type cnc --tool-type endmill --material aluminium \\
        --out knowledge_pack.json

The script is a thin wrapper around
:func:`backend.agents.knowledge.build_knowledge_pack` — all real logic
lives in the library so tests can cover it without subprocess
gymnastics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.agents.knowledge import (
    ContextKeys,
    build_knowledge_pack,
    save_pack,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export an LFL knowledge pack.")
    p.add_argument("--data-dir", default="data", help="Directory holding priors/patterns JSON files.")
    p.add_argument("--site", required=True, help="Site identifier, e.g. CNC-1.")
    p.add_argument("--machine-type")
    p.add_argument("--tool-type")
    p.add_argument("--material")
    p.add_argument("--regime")
    p.add_argument("--note", action="append", default=[], help="Optional note; repeat for multiple.")
    p.add_argument("--out", default="knowledge_pack.json")
    p.add_argument("--summary-only", action="store_true", help="Print summary counts instead of writing file.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    ctx = ContextKeys(
        machine_type=args.machine_type,
        tool_type=args.tool_type,
        material=args.material,
        regime=args.regime,
    )
    pack = build_knowledge_pack(args.data_dir, site=args.site, context=ctx, notes=args.note)
    if args.summary_only:
        print(json.dumps(pack.summary(), indent=2))
        return 0
    target = save_pack(pack, Path(args.out))
    print(f"wrote knowledge pack: {target}")
    print(json.dumps(pack.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

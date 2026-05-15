"""Machine tool-list lookups.

Each machine ships an Excel sheet describing which tool is mounted in each
spindle pocket. For inference on real OF data we need to map the
``Tool_Number`` reported by the controller to a (diameter_mm, n_inserts)
pair so the model receives the same ``d`` (diameter) and ``z`` (teeth)
parameters it was trained on.

Currently only the Komatsu sheet is wired up; Goimek/WWR parsers can be
added by mirroring ``load_komatsu_tool_list``.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

import pandas as pd

# "Ø80", "Ø 65", "Ø 80 +0,1/-0", "Ø100", ...
_DIAM_RE = re.compile(r"Ø\s*(\d+(?:[\.,]\d+)?)")


def _parse_diameter(desc: str) -> float | None:
    if not isinstance(desc, str):
        return None
    m = _DIAM_RE.search(desc)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))


@lru_cache(maxsize=8)
def load_komatsu_tool_list(path: str) -> dict[int, dict]:
    """Return ``{pocket_number: {tool, description, diameter_mm, n_inserts}}``.

    The Komatsu sheet has the real header on row index 7 (with a banner above
    it). We read the file with ``header=7`` and select columns positionally so
    we don't depend on the exact column name capitalisation.
    """
    df = pd.read_excel(path, header=7)
    # Columns in order: TOOL, DESCRIPTION, POCKET NUMBER, INSERT TYPE,
    # NUMBER OF INSERTS, SCREW REF.
    df = df.iloc[:, :6]
    df.columns = [
        "tool",
        "description",
        "pocket",
        "insert_type",
        "n_inserts",
        "screw_ref",
    ]
    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        pocket = row["pocket"]
        if pd.isna(pocket):
            continue
        try:
            pocket = int(pocket)
        except (TypeError, ValueError):
            continue
        desc = row["description"]
        diameter = _parse_diameter(desc)
        teeth = row["n_inserts"]
        try:
            teeth = int(teeth) if not pd.isna(teeth) else None
        except (TypeError, ValueError):
            teeth = None
        out[pocket] = {
            "tool": str(row["tool"]) if not pd.isna(row["tool"]) else None,
            "description": str(desc) if not pd.isna(desc) else None,
            "diameter_mm": diameter,
            "n_inserts": teeth,
        }
    return out


def find_komatsu_tool_list(workspace_root: str) -> str | None:
    """Locate ``Komatsu_Tool List Reviewed v2.xlsx`` under ``workspace_root``."""
    candidates = [
        os.path.join(workspace_root, "Machine docs", "KOM", "Tool list",
                     "Komatsu_Tool List Reviewed v2.xlsx"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

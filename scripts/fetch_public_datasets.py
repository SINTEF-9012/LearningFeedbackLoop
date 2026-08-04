#!/usr/bin/env python3
"""Fetch + inventory the public machining datasets (Phase 1 of the integration plan).

See docs/PUBLIC_DATASET_INTEGRATION_PLAN_2026-07-17.md. Two datasets:

  figshare14  "A new open dataset from a milling process" (Sci Data 2025, CC BY 4.0)
              14 tools run to failure, 968 cycles. figshare article 28589216.
              The 1.8 MB FeatureAndMetadata_Milling.csv (cycle-level aggregates of all
              20 channels + CycleToFailure labels) is enough to build the features CSV —
              the 25.3 GB raw_data.zip is only needed for the FFT physics features.

  denkena     "Multivariate time series data of milling processes with varying tool
              wear and machine tools" (Data in Brief 2023, CC BY 4.0). 3 machines,
              9 tools, 6,418 HDF5 files, measured flank wear VB in each filename.
              Mendeley zpxs87bjt8 v3.

Mendeley constraint (verified 2026-07-17): the public files API returns only the first
1,000 entries alphabetically (= machine 1 only) with no working pagination, and the
bulk-zip URL is not guessable. The script therefore fetches what the API exposes and
prints the exact browser URL for the one manual "Download All" click; it can then
unpack and verify that zip against filelist.csv.

Usage
-----
    python scripts/fetch_public_datasets.py                  # small files + inventory
    python scripts/fetch_public_datasets.py --raw            # + big downloads (25 GB / 1.6 GB)
    python scripts/fetch_public_datasets.py --dataset denkena
    python scripts/fetch_public_datasets.py --inventory-only # no network, just report disk

Raw data lands in data/public/<dataset>/ (gitignored via `data/`). Never commit it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PUBLIC_DIR = ROOT / "data" / "public"

FIGSHARE_ARTICLE = "28589216"
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"
FIGSHARE_DIR = PUBLIC_DIR / "figshare14"
# Only these are needed to build the features CSV; raw_data.zip is opt-in.
FIGSHARE_SMALL = {"metadata.xlsx", "FeatureAndMetadata_Milling.csv"}
FIGSHARE_RAW = "raw_data.zip"  # 25.3 GB

MENDELEY_ID = "zpxs87bjt8"
MENDELEY_VERSION = "3"
MENDELEY_API = (f"https://data.mendeley.com/public-api/datasets/{MENDELEY_ID}/files"
                f"?folder_id=root&version={MENDELEY_VERSION}")
MENDELEY_PAGE = f"https://data.mendeley.com/datasets/{MENDELEY_ID}/{MENDELEY_VERSION}"
DENKENA_DIR = PUBLIC_DIR / "denkena_wear"
DENKENA_ZIP_GLOB = "*.zip"  # the manually-downloaded "Download All" archive

CHUNK = 1 << 20


# ---------------------------------------------------------------------------
# download helpers
# ---------------------------------------------------------------------------

def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, size: Optional[int] = None,
              checksum: Optional[str] = None, algo: str = "md5") -> bool:
    """Idempotent download: skip when dest exists with matching checksum/size."""
    if dest.exists():
        if checksum and _hash_file(dest, algo) == checksum:
            print(f"    [ok, cached] {dest.name}")
            return True
        if checksum is None and size is not None and dest.stat().st_size == size:
            print(f"    [ok, cached by size] {dest.name}")
            return True
        print(f"    [redownload] {dest.name} (checksum/size mismatch)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        done = 0
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if size and size > 100 * CHUNK and done % (200 * CHUNK) < CHUNK:
                    print(f"      … {done / 1e9:.1f} / {size / 1e9:.1f} GB", flush=True)
    if checksum and _hash_file(tmp, algo) != checksum:
        tmp.unlink()
        print(f"    [FAILED] {dest.name}: checksum mismatch after download")
        return False
    tmp.rename(dest)
    print(f"    [downloaded] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


# ---------------------------------------------------------------------------
# figshare14
# ---------------------------------------------------------------------------

def fetch_figshare(raw: bool) -> None:
    print(f"\n== figshare14 (article {FIGSHARE_ARTICLE}) -> {FIGSHARE_DIR.relative_to(ROOT)}")
    meta = requests.get(FIGSHARE_API, timeout=60).json()
    (FIGSHARE_DIR / "article_metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (FIGSHARE_DIR / "article_metadata.json").write_text(json.dumps(meta, indent=2))
    for f in meta["files"]:
        wanted = f["name"] in FIGSHARE_SMALL or (raw and f["name"] == FIGSHARE_RAW)
        if not wanted:
            continue
        # figshare's computed_md5 is empty for the big zip; supplied_md5 is authoritative
        _download(f["download_url"], FIGSHARE_DIR / f["name"],
                  size=f["size"], checksum=f.get("supplied_md5") or None, algo="md5")
    if not raw:
        zip_size = next((f["size"] for f in meta["files"] if f["name"] == FIGSHARE_RAW), 0)
        print(f"    [skipped] {FIGSHARE_RAW} ({zip_size / 1e9:.1f} GB) — rerun with --raw, "
              f"or download manually:")
        print(f"      https://ndownloader.figshare.com/files/"
              f"{next(f['id'] for f in meta['files'] if f['name'] == FIGSHARE_RAW)}")


# ---------------------------------------------------------------------------
# denkena
# ---------------------------------------------------------------------------

def fetch_denkena(raw: bool) -> None:
    print(f"\n== denkena (Mendeley {MENDELEY_ID} v{MENDELEY_VERSION}) -> "
          f"{DENKENA_DIR.relative_to(ROOT)}")
    listing: List[Dict[str, Any]] = requests.get(MENDELEY_API, timeout=120).json()
    DENKENA_DIR.mkdir(parents=True, exist_ok=True)
    (DENKENA_DIR / "api_listing.json").write_text(json.dumps(listing, indent=2))

    by_name = {f["filename"]: f for f in listing}
    fl = by_name.get("filelist.csv")
    if fl:
        _download(fl["content_details"]["download_url"], DENKENA_DIR / "filelist.csv",
                  size=fl["size"], checksum=fl["content_details"].get("sha256_hash"),
                  algo="sha256")

    h5s = [f for f in listing if f["filename"].endswith(".h5")]
    api_bytes = sum(f["size"] for f in h5s)
    if raw:
        print(f"    fetching the {len(h5s)} API-visible .h5 files "
              f"({api_bytes / 1e9:.2f} GB — machine 1 only, see below)")
        for f in h5s:
            _download(f["content_details"]["download_url"], DENKENA_DIR / f["filename"],
                      size=f["size"], checksum=f["content_details"].get("sha256_hash"),
                      algo="sha256")
    else:
        print(f"    [skipped] {len(h5s)} API-visible .h5 files ({api_bytes / 1e9:.2f} GB) "
              f"— rerun with --raw")

    # The API exposes only the first 1,000 files (alphabetical => machine 1).
    # The full 6,418-file set needs the one manual click:
    print(f"""
    MANUAL STEP for the full 3-machine set (API caps at machine 1):
      1. open  {MENDELEY_PAGE}
      2. click "Download All", save the zip into {DENKENA_DIR.relative_to(ROOT)}/
      3. rerun this script — it unpacks the zip and verifies against filelist.csv""")
    _unpack_denkena_zip()


def _unpack_denkena_zip() -> None:
    """Unpack a manually-downloaded Mendeley 'Download All' zip, if present."""
    zips = sorted(DENKENA_DIR.glob(DENKENA_ZIP_GLOB))
    if not zips:
        return
    for zp in zips:
        print(f"    unpacking {zp.name} …")
        with zipfile.ZipFile(zp) as z:
            for member in z.namelist():
                name = Path(member).name  # flatten any zip-internal folder
                if not name or not (name.endswith(".h5") or name.endswith(".csv")):
                    continue
                dest = DENKENA_DIR / name
                if dest.exists():
                    continue
                with z.open(member) as src, dest.open("wb") as out:
                    while chunk := src.read(CHUNK):
                        out.write(chunk)
        print(f"    unpacked {zp.name} — you may delete it to reclaim space")


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------

def inventory() -> None:
    print("\n================ INVENTORY ================")
    _inventory_figshare()
    _inventory_denkena()
    print("===========================================")


def _inventory_figshare() -> None:
    csv = FIGSHARE_DIR / "FeatureAndMetadata_Milling.csv"
    print("\n-- figshare14")
    if not csv.exists():
        print("   (not downloaded)")
        return
    import pandas as pd
    # semicolon-separated; row 0 is a junk 'Column1;Column2;…' line, row 1 is the header
    d = pd.read_csv(csv, sep=";", skiprows=1)
    sensors = sorted({" - ".join(c.split(" - ")[:-1]) for c in d.columns if " - " in c})
    meta_cols = [c for c in d.columns if " - " not in c]
    per_tool = d.groupby("TollIndex").size()
    print(f"   cycles: {len(d)}   tools: {d['TollIndex'].nunique()} "
          f"(cycles/tool min={per_tool.min()} max={per_tool.max()})   "
          f"tool types: {sorted(d['MillingToolType'].unique())}")
    print(f"   channels: {len(sensors)} x aggregates "
          f"{sorted({c.split(' - ')[-1] for c in d.columns if ' - ' in c})}")
    print(f"   label cols: {[c for c in meta_cols if 'Failure' in c]}   NaNs: {int(d.isna().sum().sum())}")
    raw = FIGSHARE_DIR / FIGSHARE_RAW
    print(f"   raw_data.zip: {'present' if raw.exists() else 'absent (only needed for FFT physics features)'}")


def _inventory_denkena() -> None:
    print("\n-- denkena")
    fl = DENKENA_DIR / "filelist.csv"
    if not fl.exists():
        print("   (not downloaded)")
        return
    import pandas as pd
    d = pd.read_csv(fl)
    h5_on_disk = {p.name for p in DENKENA_DIR.glob("*.h5")}
    per_m = d.groupby("machine").agg(files=("filename", "size"),
                                     vb_max=("wear", "max"),
                                     tools=("tool", "nunique"))
    print(f"   filelist: {len(d)} files, machines {sorted(d['machine'].unique())}, "
          f"{d.groupby(['machine', 'tool']).ngroups} (machine,tool) pairs, "
          f"VB {d['wear'].min()}-{d['wear'].max()} um")
    for m, row in per_m.iterrows():
        have = sum(1 for n in d[d['machine'] == m]['filename'] if n in h5_on_disk)
        print(f"   machine {m}: {row['files']} files ({row['tools']} tools, "
              f"VB<= {row['vb_max']}) — on disk: {have}")
    missing = len(d) - sum(1 for n in d['filename'] if n in h5_on_disk)
    if missing:
        print(f"   MISSING {missing} .h5 files — see the manual-download step above")
    sample = next(iter(sorted(h5_on_disk)), None)
    if sample:
        try:
            import h5py
            with h5py.File(DENKENA_DIR / sample) as h:
                def walk(name, obj):
                    if hasattr(obj, "shape"):
                        print(f"     {name}: shape={obj.shape} dtype={obj.dtype}")
                print(f"   sample {sample} structure:")
                h.visititems(walk)
        except Exception as exc:  # noqa: BLE001
            print(f"   (could not open sample h5: {exc})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["figshare14", "denkena", "all"], default="all")
    ap.add_argument("--raw", action="store_true",
                    help="also fetch the big artifacts (figshare 25.3 GB zip; "
                         "denkena API-visible 1.6 GB)")
    ap.add_argument("--inventory-only", action="store_true",
                    help="no network; report what is on disk")
    args = ap.parse_args()

    if not args.inventory_only:
        if args.dataset in ("figshare14", "all"):
            fetch_figshare(args.raw)
        if args.dataset in ("denkena", "all"):
            fetch_denkena(args.raw)
    inventory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

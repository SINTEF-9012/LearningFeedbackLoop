from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple


def is_operation_archive(path: Path) -> bool:
    return path.is_file() and path.name.startswith("OF") and path.name.endswith(".tar.gz")


def _link_or_copy_dir(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.symlink(source.resolve(), dest, target_is_directory=True)
    except OSError:
        shutil.copytree(source, dest, dirs_exist_ok=True)


def _extract_operation_archive(archive_path: Path, dest_dir: Path) -> bool:
    extracted = False
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            member_name = Path(member.name).name
            if not member_name:
                continue
            fileobj = archive.extractfile(member)
            if fileobj is None:
                continue
            dest_path = dest_dir / member_name
            with fileobj, dest_path.open("wb") as handle:
                shutil.copyfileobj(fileobj, handle)
            extracted = True
    return extracted


def _stage_case_with_archives(source_case_dir: Path, dest_case_dir: Path) -> Dict[str, int]:
    summary = {"linked_dirs": 0, "extracted_archives": 0}
    staged_ops: set[str] = set()

    for entry in sorted(source_case_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("OF"):
            _link_or_copy_dir(entry, dest_case_dir / entry.name)
            staged_ops.add(entry.name)
            summary["linked_dirs"] += 1

    for entry in sorted(source_case_dir.iterdir()):
        if not is_operation_archive(entry):
            continue
        op_id = entry.name.removesuffix(".tar.gz")
        if op_id in staged_ops:
            continue
        if _extract_operation_archive(entry, dest_case_dir / op_id):
            staged_ops.add(op_id)
            summary["extracted_archives"] += 1

    return summary


def prepare_analysis_root(
    data_dir: Path,
    *,
    case: Optional[str] = None,
    include_archives: bool = True,
) -> Tuple[Path, Optional[TemporaryDirectory], Dict[str, int]]:
    if not include_archives or not data_dir.exists():
        return data_dir, None, {"linked_dirs": 0, "extracted_archives": 0}

    if case is not None:
        case_dirs = [data_dir / case] if (data_dir / case).is_dir() else []
    else:
        case_dirs = [
            entry
            for entry in sorted(data_dir.iterdir())
            if entry.is_dir() and not entry.name.startswith(".")
        ]

    if not any(
        any(is_operation_archive(child) for child in case_dir.iterdir())
        for case_dir in case_dirs
    ):
        return data_dir, None, {"linked_dirs": 0, "extracted_archives": 0}

    temp_root = TemporaryDirectory(prefix="stoppage_case_")
    staged_root = Path(temp_root.name) / data_dir.name
    staged_root.mkdir(parents=True, exist_ok=True)

    summary = {"linked_dirs": 0, "extracted_archives": 0}
    for source_case_dir in case_dirs:
        case_summary = _stage_case_with_archives(
            source_case_dir,
            staged_root / source_case_dir.name,
        )
        summary["linked_dirs"] += case_summary["linked_dirs"]
        summary["extracted_archives"] += case_summary["extracted_archives"]

    return staged_root, temp_root, summary
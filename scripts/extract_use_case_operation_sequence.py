from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "tools" / "UseCasesOperationSequence v2.pptx"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "tools" / "use_case_operation_sequences.json"

_SLIDE_NAME_RE = re.compile(r"slide(\d+)\.xml$", re.IGNORECASE)
_USE_CASE_RE = re.compile(r"Use Case\s+(\d+)", re.IGNORECASE)
_SETUP_RE = re.compile(r"^SETUP\b", re.IGNORECASE)
_OPERATION_RE = re.compile(r"^OP\d+(?:\.\d+)?$", re.IGNORECASE)
_UUID_PREFIX_RE = re.compile(r"^\{[0-9A-F-]+\}", re.IGNORECASE)
_SLIDE_FRACTION_RE = re.compile(r"^\(\d+/\d+\)$")


def extract_operation_sequences(path: Path | str = DEFAULT_INPUT) -> list[dict[str, Any]]:
    pptx_path = Path(path)
    current_use_case: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []

    for slide_number, lines in _iter_slide_lines(pptx_path):
        use_case = _extract_use_case_context(lines)
        if use_case is not None:
            current_use_case = use_case
        if current_use_case is None:
            continue

        table_rows = _parse_sequence_slide(lines)
        for row_index, row in enumerate(table_rows, start=1):
            row["source_pptx"] = pptx_path.name
            row["slide_number"] = slide_number
            row["slide_row_index"] = row_index
            row["use_case_id"] = current_use_case["use_case_id"]
            row["use_case_title"] = current_use_case["use_case_title"]
            rows.append(row)

    return rows


def write_operation_sequences(
    input_path: Path | str = DEFAULT_INPUT,
    output_path: Path | str = DEFAULT_OUTPUT,
) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = extract_operation_sequences(input_path)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return out_path


def _iter_slide_lines(path: Path) -> list[tuple[int, list[str]]]:
    slides: list[tuple[int, list[str]]] = []
    with ZipFile(path) as archive:
        for name in archive.namelist():
            match = _SLIDE_NAME_RE.search(name)
            if not match or not name.startswith("ppt/slides/"):
                continue
            slide_number = int(match.group(1))
            xml = archive.read(name).decode("utf-8", errors="ignore")
            text = html.unescape(xml.replace("</a:t>", "\n"))
            text = re.sub(r"<[^>]+>", "", text)
            lines = [_normalize_line(line) for line in text.splitlines()]
            slides.append((slide_number, [line for line in lines if line]))
    slides.sort(key=lambda item: item[0])
    return slides


def _extract_use_case_context(lines: list[str]) -> dict[str, Any] | None:
    use_case_id: int | None = None
    for line in lines:
        match = _USE_CASE_RE.search(line)
        if match:
            use_case_id = int(match.group(1))
            break
    if use_case_id is None:
        return None

    ignore = {
        f"Use Case {use_case_id}",
        "Production data",
        "Main difficulties",
        "Process steps",
        "Cycle time",
    }
    for line in lines:
        if line in ignore:
            continue
        if line.startswith("Cycle time:"):
            continue
        if line.startswith("style.visibility"):
            continue
        if _USE_CASE_RE.search(line):
            continue
        return {"use_case_id": use_case_id, "use_case_title": line}

    return {"use_case_id": use_case_id, "use_case_title": f"Use Case {use_case_id}"}


def _parse_sequence_slide(lines: list[str]) -> list[dict[str, Any]]:
    if "OPERATION" not in lines or "TOOL" not in lines or "DESCRIPTION" not in lines:
        return []

    has_head = "HEAD" in lines
    description_index = max(index for index, value in enumerate(lines) if value == "DESCRIPTION")
    tokens = [token for token in lines[description_index + 1:] if not _is_non_data_line(token)]

    rows: list[dict[str, Any]] = []
    current_setup: str | None = None
    previous_operation: str | None = None
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if _SETUP_RE.match(token):
            current_setup = token
            index += 1
            continue

        if has_head:
            row, next_index = _parse_headed_row(tokens, index, previous_operation)
        else:
            row, next_index = _parse_simple_row(tokens, index)

        if row is None:
            index += 1
            continue

        row["setup"] = current_setup
        previous_operation = str(row["operation_id"])
        rows.append(row)
        index = next_index

    return rows


def _parse_simple_row(tokens: list[str], index: int) -> tuple[dict[str, Any] | None, int]:
    if index + 3 >= len(tokens) or not _OPERATION_RE.match(tokens[index]):
        return None, index

    operation_id = tokens[index]
    op_type = tokens[index + 1]
    tool_raw, next_index = _consume_tool_tokens(tokens, index + 2)
    if next_index >= len(tokens):
        return None, len(tokens)
    description = tokens[next_index]
    return {
        "operation_id": operation_id,
        "head": None,
        "op_type": op_type,
        "tool_raw": tool_raw,
        "tool_numbers": _extract_tool_numbers(tool_raw),
        "description": description,
    }, next_index + 1


def _parse_headed_row(
    tokens: list[str],
    index: int,
    previous_operation: str | None,
) -> tuple[dict[str, Any] | None, int]:
    if _OPERATION_RE.match(tokens[index]):
        if index + 4 >= len(tokens):
            return None, len(tokens)
        operation_id = tokens[index]
        head = tokens[index + 1]
        op_type = tokens[index + 2]
        tool_raw, next_index = _consume_tool_tokens(tokens, index + 3)
    else:
        if previous_operation is None or index + 3 >= len(tokens):
            return None, index
        operation_id = previous_operation
        head = tokens[index]
        op_type = tokens[index + 1]
        tool_raw, next_index = _consume_tool_tokens(tokens, index + 2)

    if next_index >= len(tokens):
        return None, len(tokens)
    description = tokens[next_index]
    return {
        "operation_id": operation_id,
        "head": head,
        "op_type": op_type,
        "tool_raw": tool_raw,
        "tool_numbers": _extract_tool_numbers(tool_raw),
        "description": description,
    }, next_index + 1


def _consume_tool_tokens(tokens: list[str], start: int) -> tuple[str, int]:
    parts = [tokens[start]]
    index = start + 1
    while index < len(tokens) and _looks_like_tool_continuation(tokens[index]):
        parts.append(tokens[index])
        index += 1
    return _normalize_tool_text(" ".join(parts)), index


def _looks_like_tool_continuation(token: str) -> bool:
    compact = token.strip()
    if not compact:
        return False
    if not re.fullmatch(r"[0-9()/\sA-Z.-]+", compact):
        return False
    letters = re.sub(r"[^A-Z]", "", compact)
    return letters in {"", "M", "S", "MS", "SM"}


def _normalize_tool_text(value: str) -> str:
    text = _normalize_line(value)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+/\s*", "/", text)
    return text


def _extract_tool_numbers(value: str) -> list[int]:
    return [int(match) for match in re.findall(r"\d+", value)]


def _is_non_data_line(token: str) -> bool:
    if token.startswith("style.visibility"):
        return True
    if token in {"Process", "sequence", "OPERATION", "HEAD", "OP TYPE", "TOOL", "DESCRIPTION"}:
        return True
    if token.startswith("DAS") or token.startswith("192.168."):
        return True
    if _SLIDE_FRACTION_RE.match(token):
        return True
    if _UUID_PREFIX_RE.match(token):
        return True
    return False


def _normalize_line(value: str) -> str:
    text = " ".join(str(value).strip().split())
    text = _UUID_PREFIX_RE.sub("", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract operation/tool sequences from the case-study PowerPoint.")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="Path to the input .pptx file")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to the output JSON file")
    args = parser.parse_args()

    output_path = write_operation_sequences(args.input, args.output)
    rows = extract_operation_sequences(args.input)
    print(f"Wrote {len(rows)} operation-sequence rows to {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple


DEFAULT_PRIORS_PATH = Path(__file__).resolve().parents[1] / "data" / "pattern_priors.json"

_PURGE_PATTERNS = (
    re.compile(r"^freq:ch\d", re.IGNORECASE),
    re.compile(r"^amp:ch\d", re.IGNORECASE),
    re.compile(r"^temporal:ch\d", re.IGNORECASE),
    re.compile(r"^spectral:ch\d", re.IGNORECASE),
    re.compile(r"^kurtosis:ch\d", re.IGNORECASE),
    re.compile(r"^RATIO_ch\d+_ch\d+", re.IGNORECASE),
    re.compile(r"^(fault|hypothesis):", re.IGNORECASE),
)


def should_purge_key(key: str) -> bool:
    return any(pattern.search(key) for pattern in _PURGE_PATTERNS)


def purge_prior_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    priors = payload.get("pattern_priors", {}) if isinstance(payload, dict) else {}
    feedback_counts = payload.get("feedback_counts", {}) if isinstance(payload, dict) else {}

    if not isinstance(priors, dict):
        priors = {}
    if not isinstance(feedback_counts, dict):
        feedback_counts = {}

    removed_prior_keys = [key for key in priors if should_purge_key(key)]
    removed_feedback_keys = [key for key in feedback_counts if should_purge_key(key)]

    cleaned = dict(payload)
    cleaned["pattern_priors"] = {
        key: value for key, value in priors.items()
        if key not in set(removed_prior_keys)
    }
    cleaned["feedback_counts"] = {
        key: value for key, value in feedback_counts.items()
        if key not in set(removed_feedback_keys)
    }
    cleaned["bootstrap_seeded"] = bool(payload.get("bootstrap_seeded", False))
    cleaned["prior_source"] = payload.get("prior_source", "feedback_runtime")
    cleaned["updated_at"] = datetime.now(timezone.utc).isoformat()

    summary = {
        "removed_pattern_priors": len(removed_prior_keys),
        "removed_feedback_counts": len(removed_feedback_keys),
        "removed_prior_keys": removed_prior_keys,
        "removed_feedback_keys": removed_feedback_keys,
        "remaining_pattern_priors": len(cleaned["pattern_priors"]),
        "remaining_feedback_counts": len(cleaned["feedback_counts"]),
    }
    return cleaned, summary


def purge_priors_file(path: str | Path = DEFAULT_PRIORS_PATH) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {resolved}")

    cleaned, summary = purge_prior_payload(payload)

    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(cleaned, handle, indent=2)
        handle.write("\n")

    return cleaned, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove generic/demo-learned priors from pattern_priors.json")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PRIORS_PATH),
        help="Path to pattern_priors.json (default: data/pattern_priors.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without rewriting the file",
    )
    args = parser.parse_args()

    resolved = Path(args.path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {resolved}")

    cleaned, summary = purge_prior_payload(payload)
    if not args.dry_run:
        with resolved.open("w", encoding="utf-8") as handle:
            json.dump(cleaned, handle, indent=2)
            handle.write("\n")

    print(f"path={resolved}")
    print(f"removed_pattern_priors={summary['removed_pattern_priors']}")
    print(f"removed_feedback_counts={summary['removed_feedback_counts']}")
    print(f"remaining_pattern_priors={summary['remaining_pattern_priors']}")
    print(f"remaining_feedback_counts={summary['remaining_feedback_counts']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
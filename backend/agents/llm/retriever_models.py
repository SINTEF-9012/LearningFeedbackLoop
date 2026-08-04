"""Typed request models for RetrieverAgent actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class QueryArgs:
    query: str = ""
    top_k: int = 5

    @classmethod
    def from_dict(cls, args: Dict[str, Any]) -> "QueryArgs":
        raw_top_k = args.get("top_k", 5)
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError):
            top_k = 5
        if top_k < 1:
            top_k = 1

        return cls(
            query=str(args.get("query") or args.get("q", "")),
            top_k=top_k,
        )


@dataclass(frozen=True)
class DocsQueryArgs(QueryArgs):
    usecase: Optional[str] = None
    source_filter: Optional[str] = None
    machine: Optional[str] = None
    document_type: Optional[str] = None

    @classmethod
    def from_dict(cls, args: Dict[str, Any]) -> "DocsQueryArgs":
        base = QueryArgs.from_dict(args)
        return cls(
            query=base.query,
            top_k=base.top_k,
            usecase=args.get("usecase"),
            source_filter=args.get("source_filter"),
            machine=args.get("machine"),
            document_type=args.get("document_type"),
        )

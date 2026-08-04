"""LLM output guardrails — Tier 1 (deterministic, post-generation check).

# ===========================================================================
# [LLM_GUARDRAILS_V1] - Output rail for the operator-facing free-text channel
# ===========================================================================

After the :class:`LLMExplainer` produces an operator-facing explanation, the
orchestrator runs the text through :class:`OutputGuardrail` *before* it is
persisted, broadcast, or shown to an operator. The structured machine-facing
path (``memory/reconfig.py``) is already operator-gated; these guardrails cover
the **LLM free-text channel**.

Tier 1 (this module) is **rule/lexical and fully deterministic** — no model
call, no network, low-latency, auditable. Five checks map onto a guardrail
action:

1. **Structure**        — required content present (it must read as a grounded
                          explanation, not an empty/garbage string).
2. **Grounding**        — entities named in the text (tool ids, fault / pattern
                          names, numeric thresholds) must appear in the evidence
                          pack ``ctx``; an out-of-pack claim is flagged.
3. **Machine-control**  — imperative machine commands (change feed, override
                          spindle, modify the NC program, "stop the machine"…)
                          are **blocked**: machine-facing actions must come from
                          the structured reconfig path, never free text.
4. **Uncertainty**      — if ``ctx`` indicates incomplete context (missing
                          tool / material / program / score), the text must carry
                          an uncertainty statement; if absent it is **annotated**
                          with a standard caveat.
5. **Scorer-consistency** — the text must not assert a severity / action that
                          contradicts the scorer's own action in ``ctx`` (e.g.
                          says "critical" when the action is not critical).

**Known limitation (record this, do not silently rely on Tier 1 alone):**
Tier-1 grounding is *lexical*, not semantic — it catches obvious unsupported
tokens and unsafe keywords but not paraphrased hallucination. A pluggable
**Tier 2** hook (``semantic_checker``) is provided for a later NLI-entailment
or small LLM-judge faithfulness pass; it is **not implemented here** and is
``None`` by default so Tier 1 ships standalone.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional, Set

logger = logging.getLogger(__name__)

GuardrailAction = Literal["pass", "annotate", "block"]

# Standard caveat appended when the evidence pack is incomplete and the LLM did
# not hedge on its own (check 4).
UNCERTAINTY_CAVEAT = (
    "Note: some process context (e.g. tool, material, or program) was "
    "unavailable, so this assessment is provisional — verify on the machine "
    "before acting."
)

# Standard text substituted when an explanation is blocked outright (check 3).
# The orchestrator prefers its own deterministic fallback; this is only used if
# the caller asks the guardrail itself for replacement text.
BLOCKED_REPLACEMENT = (
    "An automated explanation was withheld because it contained an unsupported "
    "or unsafe instruction. Review the underlying evidence directly."
)


# ---------------------------------------------------------------------------
# Machine-control blocklist (check 3)
# ---------------------------------------------------------------------------
# Imperative verbs that, combined with a machine-control object, constitute a
# direct machine instruction. Free text must never instruct the machine; those
# actions belong to the structured, operator-gated reconfig path.
_MC_VERBS = (
    r"set|change|increase|decrease|reduce|raise|lower|adjust|override|modify|"
    r"update|edit|rewrite|program|reprogram|disable|enable|bypass|disengage"
)
_MC_OBJECTS = (
    r"feed(?:\s*rate)?|spindle(?:\s*speed)?|speed|rpm|program|nc\s*program|"
    r"g-?code|m-?code|offset|tool\s*offset|parameter|override|coolant|"
    r"feed\s*override|depth\s*of\s*cut|axis"
)

_MACHINE_CONTROL_PATTERNS = (
    # "set/change/... <object>" — imperative verb followed (loosely) by a
    # machine-control object within a short span.
    re.compile(
        rf"\b(?:{_MC_VERBS})\b(?:\s+\w+){{0,4}}?\s+(?:the\s+)?(?:{_MC_OBJECTS})\b",
        re.IGNORECASE,
    ),
    # Explicit "stop/halt/pause/e-stop the machine/spindle/program".
    re.compile(
        r"\b(?:stop|halt|pause|e-?stop|shut\s*down|shutdown|kill)\b"
        r"(?:\s+\w+){0,3}?\s+(?:the\s+)?(?:machine|spindle|program|cut|tool|axis)\b",
        re.IGNORECASE,
    ),
    # Direct NC / G-code edits phrased without a leading verb-object pair.
    re.compile(r"\bmodify\s+the\s+nc\b", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# Lexical helpers
# ---------------------------------------------------------------------------
# A numeric token in the text (supports decimals and signed values).
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
# A "tool id"-shaped token, e.g. T12, T-12, TOOL 7, tool#3.
_TOOL_ID_RE = re.compile(r"\b(?:tool\s*#?\s*|t-?)(\d{1,4})\b", re.IGNORECASE)
# Severity words used in scorer-consistency (check 5).
_SEVERITY_WORDS = ("critical", "severe", "emergency", "catastrophic")
# Uncertainty / hedging vocabulary used in check 4.
_UNCERTAINTY_WORDS = (
    "uncertain", "unclear", "provisional", "tentative", "may ", "might ",
    "possibly", "likely", "appears", "appear ", "seems", "could ", "estimate",
    "estimated", "unconfirmed", "preliminary", "verify", "unavailable",
    "without ", "unknown", "limited context", "insufficient",
)


@dataclass
class GuardrailResult:
    """Outcome of running :class:`OutputGuardrail` over an explanation.

    Attributes
    ----------
    action:
        ``"pass"``     — text is acceptable as-is.
        ``"annotate"`` — text is acceptable but ``text`` has been modified
                         (e.g. an uncertainty caveat appended).
        ``"block"``    — text must not be shown; the caller should substitute a
                         deterministic fallback.
    reasons:
        Human / audit readable list of which checks fired and why.
    text:
        The (possibly annotated / redacted) text. For ``block`` this is the
        original text echoed back for audit; callers substitute their own
        fallback rather than display it.
    checks:
        Per-check verdicts (check name → "pass"/"annotate"/"block"), for audit.
    """

    action: GuardrailAction = "pass"
    reasons: List[str] = field(default_factory=list)
    text: str = ""
    checks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializable audit record (no raw text — that lives on the memory)."""
        return {
            "action": self.action,
            "reasons": list(self.reasons),
            "checks": dict(self.checks),
        }


class OutputGuardrail:
    """Deterministic Tier-1 output rail for LLM explanations.

    Parameters
    ----------
    semantic_checker:
        Optional **Tier-2** hook. If provided, it is invoked *after* the Tier-1
        checks as ``semantic_checker(text, ctx, tier1_result) -> GuardrailResult``
        and its result is returned in place of the Tier-1 result. Default
        ``None`` — Tier 2 is **not implemented here**; this is only a pluggable
        seam for a later NLI / LLM-judge faithfulness pass. Tier 1 is fully
        functional without it.
    block_on_ungrounded:
        When True, an out-of-pack entity (check 2) escalates to ``block``
        instead of ``annotate``. Default False (lexical grounding is
        approximate — see the module docstring's known-limitation note).
    """

    def __init__(
        self,
        *,
        semantic_checker: Optional[Callable[[str, Any, "GuardrailResult"], "GuardrailResult"]] = None,
        block_on_ungrounded: bool = False,
        soft_block: bool = False,
    ) -> None:
        self.semantic_checker = semantic_checker
        self.block_on_ungrounded = block_on_ungrounded
        # Demo soft-mode: downgrade the hard-block checks (machine-control,
        # structure) to *annotate* so the LLM explanation is still shown (with the
        # violation flagged in the audit trail) rather than replaced by a
        # deterministic fallback. Safety intent is preserved for the audit; the
        # text passes through. Gated by env so production stays strict.
        self.soft_block = soft_block

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def check(self, text: str, ctx: Any) -> GuardrailResult:
        """Run all Tier-1 checks (then the optional Tier-2 hook) over ``text``.

        Pure / deterministic and defensive about missing ``ctx`` fields. Any
        unexpected error is swallowed and reported as a ``pass`` with a reason,
        so the guardrail can never break the loop (the orchestrator also wraps
        the call in try/except per the codebase's degrade-to-fallback rule).
        """
        original = str(text or "")
        result = GuardrailResult(action="pass", reasons=[], text=original, checks={})

        try:
            # --- Check 3 first: machine-control is the hard stop ---------
            mc_reason = self._check_machine_control(original)
            if mc_reason:
                mc_verdict = "annotate" if self.soft_block else "block"
                result.checks["machine_control"] = mc_verdict
                result.action = self._escalate(result.action, mc_verdict)
                result.reasons.append(mc_reason)
                # No point annotating text we are going to drop, but keep
                # evaluating the other checks for a complete audit trail.
            else:
                result.checks["machine_control"] = "pass"

            # --- Check 1: structure ------------------------------------
            struct_reason = self._check_structure(original)
            if struct_reason:
                struct_verdict = "annotate" if self.soft_block else "block"
                result.checks["structure"] = struct_verdict
                result.reasons.append(struct_reason)
                result.action = self._escalate(result.action, struct_verdict)
            else:
                result.checks["structure"] = "pass"

            # --- Check 2: grounding ------------------------------------
            ground_reasons = self._check_grounding(original, ctx)
            if ground_reasons:
                verdict = "block" if self.block_on_ungrounded else "annotate"
                result.checks["grounding"] = verdict
                result.reasons.extend(ground_reasons)
                result.action = self._escalate(result.action, verdict)
            else:
                result.checks["grounding"] = "pass"

            # --- Check 5: scorer consistency ---------------------------
            consistency_reasons = self._check_scorer_consistency(original, ctx)
            if consistency_reasons:
                result.checks["scorer_consistency"] = "annotate"
                result.reasons.extend(consistency_reasons)
                result.action = self._escalate(result.action, "annotate")
            else:
                result.checks["scorer_consistency"] = "pass"

            # --- Check 4: uncertainty enforcement ----------------------
            # Only meaningful when we are not already blocking the text.
            needs_caveat = self._needs_uncertainty(original, ctx)
            if needs_caveat:
                result.checks["uncertainty"] = "annotate"
                result.reasons.append(
                    "Evidence pack is incomplete and the explanation lacked an "
                    "uncertainty statement; appended a standard caveat."
                )
                if result.action != "block":
                    result.text = self._append_caveat(result.text)
                result.action = self._escalate(result.action, "annotate")
            else:
                result.checks["uncertainty"] = "pass"

        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("OutputGuardrail check errored, passing through: %s", exc)
            return GuardrailResult(
                action="pass",
                reasons=[f"guardrail_error: {type(exc).__name__}"],
                text=original,
                checks={"error": "pass"},
            )

        # --- Tier-2 hook (optional, not implemented here) --------------
        if self.semantic_checker is not None:
            try:
                tier2 = self.semantic_checker(original, ctx, result)
                if isinstance(tier2, GuardrailResult):
                    return tier2
                logger.debug("semantic_checker returned non-GuardrailResult; ignoring")
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Tier-2 semantic_checker errored, keeping Tier-1: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Action escalation: pass < annotate < block
    # ------------------------------------------------------------------

    @staticmethod
    def _escalate(current: GuardrailAction, candidate: GuardrailAction) -> GuardrailAction:
        order = {"pass": 0, "annotate": 1, "block": 2}
        return current if order[current] >= order[candidate] else candidate

    # ------------------------------------------------------------------
    # Check 1: structure
    # ------------------------------------------------------------------

    @staticmethod
    def _check_structure(text: str) -> Optional[str]:
        """Reject empty / degenerate output.

        The explainer emits prose (its structured JSON is already flattened to a
        sentence stream before it reaches us), so we require it to read as a
        minimally substantive explanation rather than validate fixed sections.
        """
        stripped = text.strip()
        if not stripped:
            return "Explanation is empty."
        # Too short to be a grounded explanation (a couple of words).
        if len(stripped) < 20 or len(stripped.split()) < 4:
            return "Explanation is too short to be a grounded assessment."
        # Looks like an un-flattened raw JSON blob / fenced code — not prose.
        if stripped.startswith("{") or stripped.startswith("```"):
            return "Explanation is not in the expected prose form."
        return None

    # ------------------------------------------------------------------
    # Check 2: grounding (lexical)
    # ------------------------------------------------------------------

    def _check_grounding(self, text: str, ctx: Any) -> List[str]:
        """Flag tool ids and numeric thresholds in the text that are absent
        from the evidence pack. Lexical only — see module known-limitation.
        """
        reasons: List[str] = []
        pack_numbers, pack_tool_ids, pack_blob = self._evidence_index(ctx)

        # Out-of-pack tool ids.
        for match in _TOOL_ID_RE.finditer(text):
            tool_num = match.group(1)
            if tool_num not in pack_tool_ids and tool_num not in pack_numbers:
                reasons.append(
                    f"Explanation names tool id 'T{tool_num}' that is not in the "
                    f"evidence pack."
                )

        # If the pack carries no numeric evidence at all, skip numeric grounding
        # (nothing to compare against — avoids false positives on empty packs).
        if pack_numbers:
            seen_unsupported: Set[str] = set()
            for raw_num in _NUMBER_RE.findall(text):
                norm = self._normalize_number(raw_num)
                if norm is None:
                    continue
                if norm in seen_unsupported:
                    continue
                if not self._number_supported(norm, pack_numbers):
                    seen_unsupported.add(norm)
            if seen_unsupported:
                reasons.append(
                    "Explanation cites numeric value(s) not present in the "
                    f"evidence pack: {', '.join(sorted(seen_unsupported))}."
                )

        return reasons

    # ------------------------------------------------------------------
    # Check 3: machine-control blocklist
    # ------------------------------------------------------------------

    @staticmethod
    def _check_machine_control(text: str) -> Optional[str]:
        for pat in _MACHINE_CONTROL_PATTERNS:
            m = pat.search(text)
            if m:
                return (
                    "Explanation contains a direct machine-control instruction "
                    f"('{m.group(0).strip()}'); machine actions must come from "
                    f"the structured reconfig path, not free text."
                )
        return None

    # ------------------------------------------------------------------
    # Check 4: uncertainty enforcement
    # ------------------------------------------------------------------

    def _needs_uncertainty(self, text: str, ctx: Any) -> bool:
        if not self._context_incomplete(ctx):
            return False
        return not self._has_uncertainty_statement(text)

    @staticmethod
    def _context_incomplete(ctx: Any) -> bool:
        """True when the evidence pack is missing key process context."""
        if ctx is None:
            return True
        # No scorer result at all.
        if getattr(ctx, "significance", None) is None:
            return True
        cc = getattr(ctx, "cutting_context", None)
        if cc is None:
            return True
        # Missing tool / material / program-ish identity → provisional.
        tool = getattr(cc, "tool_type", None) or getattr(cc, "tool_number", None)
        material = getattr(cc, "workpiece_material", None)
        if not tool or not material:
            return True
        return False

    @staticmethod
    def _has_uncertainty_statement(text: str) -> bool:
        low = text.lower()
        return any(word in low for word in _UNCERTAINTY_WORDS)

    @staticmethod
    def _append_caveat(text: str) -> str:
        base = text.rstrip()
        if not base:
            return UNCERTAINTY_CAVEAT
        if UNCERTAINTY_CAVEAT in base:
            return base
        sep = " " if base.endswith((".", "!", "?")) else ". "
        return f"{base}{sep}{UNCERTAINTY_CAVEAT}"

    # ------------------------------------------------------------------
    # Check 5: scorer consistency
    # ------------------------------------------------------------------

    def _check_scorer_consistency(self, text: str, ctx: Any) -> List[str]:
        sig = getattr(ctx, "significance", None) if ctx is not None else None
        if sig is None:
            return []
        action = self._action_value(getattr(sig, "action", None))
        if action is None:
            return []
        low = text.lower()
        asserts_critical = any(w in low for w in _SEVERITY_WORDS)
        if asserts_critical and action != "critical":
            return [
                "Explanation asserts a critical/severe severity but the scorer "
                f"action is '{action}'."
            ]
        return []

    # ------------------------------------------------------------------
    # Evidence-pack indexing helpers
    # ------------------------------------------------------------------

    def _evidence_index(self, ctx: Any):
        """Build (set of normalized numbers, set of tool-id digits, text blob)
        from the evidence pack, defensively handling missing fields.
        """
        numbers: Set[str] = set()
        tool_ids: Set[str] = set()
        blob_parts: List[str] = []

        if ctx is None:
            return numbers, tool_ids, ""

        def add_number(val: Any) -> None:
            try:
                norm = self._normalize_number(str(val))
            except Exception:
                norm = None
            if norm is not None:
                numbers.add(norm)

        # Feature evidence: values + thresholds.
        fe = getattr(ctx, "feature_evidence", None) or {}
        if isinstance(fe, dict):
            for ev_list in fe.values():
                for ev in ev_list or []:
                    if not isinstance(ev, dict):
                        continue
                    for key in ("value", "threshold"):
                        if ev.get(key) is not None:
                            add_number(ev[key])
                    feat = ev.get("feature")
                    if feat:
                        blob_parts.append(str(feat))

        # Classical model scores.
        cm = getattr(ctx, "classical_model", None) or {}
        if isinstance(cm, dict):
            for k, v in cm.items():
                blob_parts.append(str(k))
                add_number(v)

        # Feedback stats (confirm/dismiss counts, prior).
        fs = getattr(ctx, "feedback_stats", None) or {}
        if isinstance(fs, dict):
            for stats in fs.values():
                if isinstance(stats, dict):
                    for v in stats.values():
                        add_number(v)

        # Raw metrics excerpt.
        rm = getattr(ctx, "raw_metrics_excerpt", None) or {}
        if isinstance(rm, dict):
            for k, v in rm.items():
                blob_parts.append(str(k))
                add_number(v)

        # Significance score.
        sig = getattr(ctx, "significance", None)
        if sig is not None and getattr(sig, "score", None) is not None:
            add_number(sig.score)

        # Pattern keys / names.
        for pk in getattr(ctx, "pattern_keys", None) or []:
            blob_parts.append(str(pk))

        # Cutting context — tool number / type contributes tool ids + numbers.
        cc = getattr(ctx, "cutting_context", None)
        if cc is not None:
            for attr in ("tool_type", "tool_number", "workpiece_material"):
                val = getattr(cc, attr, None)
                if val is not None:
                    blob_parts.append(str(val))
                    for m in _TOOL_ID_RE.finditer(str(val)):
                        tool_ids.add(m.group(1))
                    for m in _NUMBER_RE.finditer(str(val)):
                        norm = self._normalize_number(m.group(0))
                        if norm is not None:
                            tool_ids.add(norm)
            for attr in ("spindle_speed", "axial_depth", "tooth_passing_freq"):
                try:
                    val = getattr(cc, attr, None)
                except Exception:
                    val = None
                if val is not None:
                    add_number(val)

        return numbers, tool_ids, " ".join(blob_parts)

    # ------------------------------------------------------------------
    # Numeric matching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_number(raw: str) -> Optional[str]:
        """Normalize a numeric token to a canonical string for comparison.

        Returns integer form for whole numbers (``"15.0" -> "15"``) and a
        2-decimal form otherwise (``"0.823" -> "0.82"``). Returns None for
        non-numeric input.
        """
        try:
            f = float(raw)
        except (TypeError, ValueError):
            return None
        if f == int(f):
            return str(int(f))
        return f"{f:.2f}"

    @staticmethod
    def _number_supported(norm: str, pack_numbers: Set[str]) -> bool:
        """A text number is 'supported' if it (or a close rounding) is in the
        pack. We ignore small integers and percentages, which are commonly
        derived (e.g. '56%' exceeded-by, counts like '1', '2'), to keep the
        lexical check from over-flagging.
        """
        if norm in pack_numbers:
            return True
        try:
            val = float(norm)
        except ValueError:
            return True  # unparseable → don't flag
        # Ignore small magnitudes — derived counts / percentages / ordinals.
        if abs(val) <= 100 and val == int(val):
            return True
        # Accept if any pack number rounds to the same 1-decimal form.
        one_dp = f"{val:.1f}"
        for pn in pack_numbers:
            try:
                if f"{float(pn):.1f}" == one_dp:
                    return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _action_value(action: Any) -> Optional[str]:
        """Normalize a SignificanceAction (enum or str) to its string value."""
        if action is None:
            return None
        val = getattr(action, "value", None)
        if val is not None:
            return str(val).lower()
        return str(action).lower()


__all__ = [
    "GuardrailResult",
    "OutputGuardrail",
    "UNCERTAINTY_CAVEAT",
    "BLOCKED_REPLACEMENT",
]

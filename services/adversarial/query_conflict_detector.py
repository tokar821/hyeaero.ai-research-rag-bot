"""Deterministic detection of conflicting or misleading query structure."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from services.adversarial.budget_conflict_normalizer import (
    BudgetFeasibility,
    normalize_budget_conflicts,
)


class ConflictType(str, Enum):
    BUDGET_MODEL_INFEASIBLE = "BUDGET_MODEL_INFEASIBLE"
    INTENT_MIXED = "INTENT_MIXED"
    MODEL_AMBIGUOUS = "MODEL_AMBIGUOUS"
    TEMPORAL_CONTRADICTION = "TEMPORAL_CONTRADICTION"
    VALUATION_BUY_CONTRADICTION = "VALUATION_BUY_CONTRADICTION"
    BUDGET_SEMANTIC_CONFLICT = "BUDGET_SEMANTIC_CONFLICT"


class ConflictSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


_ULTRA_PREMIUM_RE = re.compile(
    r"\b(?:g\s*700|g700|g\s*650|g650|global\s*7500|falcon\s*8x|global\s*6500)\b",
    re.I,
)
_BUY_RE = re.compile(
    r"(?is)\b(?:buy|purchase|good\s+deal|fair\s+price|overpriced|should\s+i\s+buy|listed\s+at)\b",
)
_COMPARE_RE = re.compile(r"(?is)\b(?:vs\.?|versus|compare|comparison|better\s+than)\b")
_VALUATION_RE = re.compile(
    r"(?is)\b(?:worth|valuation|market\s+value|how\s+much\s+is)\b",
)
_CHARTER_RE = re.compile(r"(?is)\b(?:charter|on\s+demand|fractional)\b")
_CHEAP_NOW_RE = re.compile(r"(?is)\b(?:cheap\s+now|low\s+price\s+today|buy\s+now\s+cheap)\b")
_RISE_LATER_RE = re.compile(
    r"(?is)\b(?:will\s+increase|going\s+up|prices\s+rising|more\s+expensive\s+next\s+year)\b",
)
_AMBIGUOUS_MODEL_RE = re.compile(
    r"(?is)\b(?:cheap\s+gulfstream|baby\s+g650|longitude\s+jet|cheapest\s+private\s+jet|"
    r"like\s+a\s+citation|gulfstream\s+alternative)\b",
)
_BARE_LONGITUDE_RE = re.compile(r"(?is)\b(?<!citation\s)longitude\b")


@dataclass
class QueryConflictReport:
    conflict_type: Tuple[ConflictType, ...] = field(default_factory=tuple)
    severity: ConflictSeverity = ConflictSeverity.LOW
    resolved_intent_override: Optional[str] = None
    normalized_query_tokens: Tuple[str, ...] = field(default_factory=tuple)
    details: Tuple[str, ...] = field(default_factory=tuple)

    def add(self, ctype: ConflictType, detail: str, *, severity: ConflictSeverity) -> None:
        types = list(self.conflict_type)
        if ctype not in types:
            types.append(ctype)
        self.conflict_type = tuple(types)
        details = list(self.details)
        details.append(detail)
        self.details = tuple(details)
        if severity == ConflictSeverity.HIGH or (
            severity == ConflictSeverity.MEDIUM and self.severity == ConflictSeverity.LOW
        ):
            self.severity = severity
        elif severity == ConflictSeverity.MEDIUM and self.severity != ConflictSeverity.HIGH:
            self.severity = ConflictSeverity.MEDIUM


def _tokenize(q: str) -> Tuple[str, ...]:
    from services.consultant.recommendation_engine import detect_models_from_text

    tokens: List[str] = list(detect_models_from_text(q or ""))
    for m in re.findall(r"\$?\d+(?:\.\d+)?\s*(?:m|million)?", q or "", re.I):
        tokens.append(m.strip())
    return tuple(dict.fromkeys(tokens))


def detect_query_conflicts(
    query: str,
    *,
    budget_state: Optional[object] = None,
    resolved_models: Optional[List[str]] = None,
) -> QueryConflictReport:
    """Scan query for structured conflicts (deterministic)."""
    q = (query or "").strip()
    report = QueryConflictReport(normalized_query_tokens=_tokenize(q))

    bs = budget_state
    if bs is None:
        bs = normalize_budget_conflicts(q, resolved_models=resolved_models)

    if bs.feasibility == BudgetFeasibility.INFEASIBLE:
        report.add(
            ConflictType.BUDGET_MODEL_INFEASIBLE,
            bs.reason or "acquisition budget incompatible with model class",
            severity=ConflictSeverity.HIGH,
        )
    elif bs.feasibility == BudgetFeasibility.SEMANTICALLY_CONFLICTED:
        report.add(
            ConflictType.BUDGET_SEMANTIC_CONFLICT,
            bs.reason or "semantically conflicted budget signals",
            severity=ConflictSeverity.MEDIUM,
        )

    intents: List[str] = []
    if _BUY_RE.search(q):
        intents.append("buy")
    if _COMPARE_RE.search(q):
        intents.append("compare")
    if _VALUATION_RE.search(q):
        intents.append("valuation")
    if _CHARTER_RE.search(q):
        intents.append("charter")
    core = [i for i in intents if i in ("buy", "compare", "valuation")]
    if len(core) >= 2:
        report.add(
            ConflictType.INTENT_MIXED,
            f"mixed intents: {','.join(core)}",
            severity=ConflictSeverity.MEDIUM if len(core) == 2 else ConflictSeverity.HIGH,
        )

    if _AMBIGUOUS_MODEL_RE.search(q) or _BARE_LONGITUDE_RE.search(q):
        report.add(
            ConflictType.MODEL_AMBIGUOUS,
            "ambiguous or shorthand aircraft reference",
            severity=ConflictSeverity.MEDIUM,
        )

    if _CHEAP_NOW_RE.search(q) and _RISE_LATER_RE.search(q):
        report.add(
            ConflictType.TEMPORAL_CONTRADICTION,
            "cheap now vs rising prices later",
            severity=ConflictSeverity.MEDIUM,
        )

    acq = getattr(bs, "acquisition_cap_musd", None)
    if _VALUATION_RE.search(q) and _BUY_RE.search(q) and acq is not None:
        report.add(
            ConflictType.VALUATION_BUY_CONTRADICTION,
            "valuation shape combined with explicit acquisition budget",
            severity=ConflictSeverity.LOW,
        )

    return report

"""Plan multi-intent queries without changing primary dispatch ordering."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MultiIntentPlan:
    primary_intent: str
    secondary_intents: List[str] = field(default_factory=list)
    overlays: List[str] = field(default_factory=list)
    preserve_primary: bool = True
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_intent": self.primary_intent,
            "secondary_intents": list(self.secondary_intents),
            "overlays": list(self.overlays),
            "preserve_primary": self.preserve_primary,
            "notes": list(self.notes),
        }


_COMPARE_RE = re.compile(r"\b(?:compare|vs\.?|versus|comparison)\b", re.I)
_BUY_RE = re.compile(
    r"\b(?:"
    r"better\s+buy|good\s+buy|should\s+i\s+buy|buy\s+decision|"
    r"which\s+(?:is|one)\s+(?:the\s+)?better\s+buy|worth\s+buying"
    r")\b",
    re.I,
)
_VALUATION_RE = re.compile(
    r"\b(?:valuation|worth|market\s+value|resale|pricing)\b",
    re.I,
)
_TIMING_RE = re.compile(
    r"\b(?:market\s+trend|timing|buy\s+timing|when\s+to\s+buy|forward\s+pricing|temporal)\b",
    re.I,
)
_VALUATION_AND_TIMING_RE = re.compile(
    r"\b(?:valuation\s+and\s+buy\s+timing|value\s+and\s+timing)\b",
    re.I,
)


def plan_multi_intent(query: str) -> MultiIntentPlan:
    """Detect compound intents and plan additive overlays for the primary responder."""
    q = (query or "").strip()
    low = q.lower()

    has_compare = bool(_COMPARE_RE.search(q))
    has_buy = bool(_BUY_RE.search(q))
    has_valuation = bool(_VALUATION_RE.search(q))
    has_timing = bool(_TIMING_RE.search(q)) or "market trend" in low

    secondary: List[str] = []
    overlays: List[str] = []
    notes: List[str] = []

    if has_compare and has_buy:
        secondary.append("buy_decision")
        overlays.append("buy_read")
        notes.append("Comparison primary; acquisition read appended for both models.")

    if has_compare and has_timing:
        secondary.append("temporal")
        overlays.append("temporal")
        notes.append("Comparison primary; market trend overlay appended.")

    if has_compare and has_valuation and "buy_read" not in overlays:
        secondary.append("valuation")
        overlays.append("valuation_snapshot")
        notes.append("Comparison primary; valuation snapshot appended.")

    if _VALUATION_AND_TIMING_RE.search(q) or (has_valuation and has_timing and not has_compare):
        primary = "valuation"
        if has_timing:
            secondary.append("temporal")
            overlays.append("temporal")
        notes.append("Valuation primary with buy-timing overlay.")
        return MultiIntentPlan(
            primary_intent=primary,
            secondary_intents=secondary,
            overlays=overlays,
            notes=notes,
        )

    if has_compare:
        primary = "comparison"
    elif has_buy:
        primary = "buy_decision"
    elif has_valuation:
        primary = "valuation"
    else:
        primary = "mission"

    return MultiIntentPlan(
        primary_intent=primary,
        secondary_intents=secondary,
        overlays=overlays,
        notes=notes,
    )


__all__ = ["MultiIntentPlan", "plan_multi_intent"]

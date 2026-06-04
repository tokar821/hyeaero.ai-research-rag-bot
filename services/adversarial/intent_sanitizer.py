"""Deterministic intent priority when multiple intents appear in one query."""

from __future__ import annotations

import re
from typing import Optional, Set

_BUY_RE = re.compile(
    r"(?is)\b(?:buy|purchase|good\s+deal|fair\s+price|overpriced|should\s+i\s+buy)\b",
)
_COMPARE_RE = re.compile(r"(?is)\b(?:vs\.?|versus|compare|comparison)\b")
_VALUATION_RE = re.compile(r"(?is)\b(?:worth|valuation|market\s+value)\b")


def _detect_intent_tags(query: str) -> Set[str]:
    q = query or ""
    tags: Set[str] = set()
    if _BUY_RE.search(q):
        tags.add("buy")
    if _COMPARE_RE.search(q):
        tags.add("compare")
    if _VALUATION_RE.search(q):
        tags.add("valuation")
    return tags


def sanitize_intents(query: str, *, existing_override: Optional[str] = None) -> Optional[str]:
    """
    Return deterministic intent override only when multiple core intents conflict.

    Priority (highest wins): BUY > COMPARE > VALUATION
    """
    if existing_override:
        return existing_override

    tags = _detect_intent_tags(query)
    if len(tags) < 2:
        return None

    if "buy" in tags:
        if "compare" in tags:
            return "buy"
        if "valuation" in tags:
            return "buy"
    if "compare" in tags and "valuation" in tags:
        return "compare"
    return None

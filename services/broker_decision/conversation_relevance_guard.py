"""Detect answers that dump specs without answering the buyer's question."""

from __future__ import annotations

import re
from typing import Optional

from services.broker_decision.decision_intent_detector import DecisionIntent, detect_decision_intent

_CATALOG_DUMP_RE = re.compile(
    r"(?im)^(?:verified catalog comparison|side by side:|aircraft options:|mission fit:)"
)
_SPEC_BULLET_HEAVY_RE = re.compile(
    r"(?im)^\s*[•\-]\s+.+:\s+.+\b(?:class|nm|seats|operating cost band)\b",
)
_INSUFFICIENT_LEAD_RE = re.compile(
    r"(?im)^(?:insufficient verified|INSUFFICIENT_DATA|clarification_required)",
)
_NO_DIRECT_ANSWER_RE = re.compile(
    r"(?im)^(?:overview|analysis|recommendation|verdict:)\s*$",
)


def _first_paragraph(text: str) -> str:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    return parts[0] if parts else (text or "").strip()[:400]


def is_catalog_or_spec_dump(answer: str, *, query: str = "") -> bool:
    """True when answer looks like catalog/spec output without a buyer decision."""
    body = (answer or "").strip()
    if not body:
        return False

    intent = detect_decision_intent(query)
    if intent in (DecisionIntent.NONE,):
        return False

    first = _first_paragraph(body).lower()
    # Good decision answers start with direct language.
    decision_starters = (
        "no",
        "yes",
        "possibly",
        "at $",
        "a ",
        "i would",
        "if you",
        "with about",
        "stretch",
        "buy when",
        "not rush",
    )
    if any(first.startswith(s) for s in decision_starters):
        return False

    if _INSUFFICIENT_LEAD_RE.search(first):
        return True
    if _CATALOG_DUMP_RE.search(body):
        return True
    if len(_SPEC_BULLET_HEAVY_RE.findall(body)) >= 2 and not re.search(
        r"(?i)\b(?:realistic|overpay|budget|should i|recommend|would focus|stop looking)\b",
        first,
    ):
        return True
    if _NO_DIRECT_ANSWER_RE.search(body[:200]):
        return True

    # Comparison table without verdict language in opening.
    if re.search(r"(?i)\bvs\.?\b", query) and intent not in (
        DecisionIntent.NONE,
    ):
        if "choose" not in first and "lean" not in first and "would" not in first:
            if _SPEC_BULLET_HEAVY_RE.search(body):
                return True

    return False


def should_synthesize_decision(answer: str, *, query: str) -> bool:
    """Whether Phase 41 should rewrite or prepend a decision-led answer."""
    if not (query or "").strip():
        return False
    intent = detect_decision_intent(query)
    if intent == DecisionIntent.NONE:
        return False
    if is_catalog_or_spec_dump(answer, query=query):
        return True
    if _INSUFFICIENT_LEAD_RE.search((answer or "")[:300]):
        return True
    if len((answer or "").strip()) < 120:
        return True
    return False


__all__ = ["is_catalog_or_spec_dump", "should_synthesize_decision"]

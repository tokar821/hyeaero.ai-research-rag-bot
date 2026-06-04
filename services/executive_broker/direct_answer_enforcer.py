"""Enforce a direct first sentence for decision-shaped questions.

Presentation-only enforcement: does not alter routing, valuation, market math, temporal math,
adversarial logic, or intent classification. It only reshapes the *final* executive prose so the
first sentence answers the user's question directly (Yes/No/Probably/I would/I would not).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

_TRIGGER_RE = re.compile(
    r"(?is)^\s*(?:"
    r"can\s+i(?:\s+realistically)?(?:\s+buy|\s+afford|\s+get)?|"
    r"should\s+i|"
    r"is\s+it\s+realistic|"
    r"is\s+this\s+a\s+good\s+deal|"
    r"what\s+would\s+you\s+buy|"
    r"is\s+now\s+a\s+good\s+time"
    r")\b"
)

_DIRECT_START_RE = re.compile(r"(?is)^\s*(?:yes\.|no\.|probably\.|i would\b|i would not\b)")
_PRIMARY_RE = re.compile(r"(?is)^\s*my primary recommendation would be\b")

_BUDGET_IN_QUERY_RE = re.compile(
    r"(?is)\b(?:for|under|below|around|about|at|budget\s+is)\s+\$?\s*"
    r"(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)\b"
    r"(?:\s*[-–]\s*\$?\s*(?P<amt2>\d+(?:\.\d+)?)\s*(?P<unit2>m|mm|million|mil|k)\b)?"
)


def _best_direct_sentence(query: str, data_used: Dict[str, Any]) -> Optional[str]:
    bd = data_used.get("broker_decision")
    if isinstance(bd, dict):
        direct = str(bd.get("direct_answer") or "").strip()
        if direct:
            first = re.split(r"\n\s*\n", direct)[0].strip()
            if first:
                return first

    # Fall back to adversarial infeasibility when present.
    adv = data_used.get("adversarial") or {}
    if isinstance(adv, dict) and adv.get("budget_feasibility") == "INFEASIBLE":
        return "No."

    # If query includes an explicit budget and we have an executive primary far outside it, answer "No."
    m = _BUDGET_IN_QUERY_RE.search(query or "")
    if m:
        try:
            val = float(m.group("amt2") or m.group("amt"))
            unit = (m.group("unit2") or m.group("unit") or "m").lower()
            cap = val / 1000.0 if unit == "k" else val
        except (TypeError, ValueError):
            cap = None
        if cap is not None:
            rec = data_used.get("executive_recommendation") or {}
            model = str(rec.get("primary_recommendation") or "").strip()
            if model and cap <= 8 and re.search(r"(?is)\b(?:g700|g650)\b", model):
                return "No."

    # For "should I" timing queries, allow a broker stance.
    if re.search(r"(?is)^\s*should\s+i\b", query or ""):
        return "Probably."

    return None


def enforce_direct_answer(
    answer: str,
    *,
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    q = (query or "").strip()
    text = (answer or "").strip()
    du = data_used if isinstance(data_used, dict) else {}

    if not q or not text or not _TRIGGER_RE.search(q):
        return text

    # Already direct — leave as-is.
    if _DIRECT_START_RE.search(text):
        return text

    direct = _best_direct_sentence(q, du)
    if not direct:
        return text

    # If answer starts with executive primary, prepend the direct sentence.
    if _PRIMARY_RE.search(text):
        return f"{direct}\n\n{text}".strip()

    # Otherwise, prepend and keep answer.
    return f"{direct}\n\n{text}".strip()


__all__ = ["enforce_direct_answer"]


"""
Pinpoint factual turns — seats, range, single-spec, used price for one model.

Keeps answers tight and strips advisory boilerplate appended by safety fallbacks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

_PINPOINT_QUERY_RE = re.compile(
    r"(?:"
    r"how\s+many\s+(?:seats?|passengers?|pax)|"
    r"(?:what'?s|what\s+is)\s+(?:the\s+)?range\b|"
    r"\brange\s+of\s+(?:a|the)\s+|"
    r"(?:price|cost|asking)\s+of\s+(?:a\s+)?used\b|"
    r"used\s+.*\bprice\b|"
    r"(?:max(?:imum)?\s+)?(?:cruise\s+)?speed\s+of|"
    r"how\s+(?:fast|quick|high)"
    r")",
    re.I,
)

_GOOD_FIT_BLOCK_RE = re.compile(
    r"\n\s*✅\s*GOOD\s+FIT\b[\s\S]*$",
    re.I,
)
_ASSUMING_BLOCK_RE = re.compile(
    r"(?:\n\s*)?Assuming\s+6[–-]8\s+passengers[\s\S]*$",
    re.I,
)
_REALISTIC_FITS_BLOCK_RE = re.compile(
    r"(?:\n\s*)?"
    r"(?:Assuming\s+6[–-]8\s+passengers\s+and\s+)?"
    r"typical\s+business-use\s+constraints\s*\(no\s+extreme\s+hot/high\)\s*,?\s*"
    r"here\s+are\s+a\s+few\s+realistic\s+fits:\s*"
    r"[\s\S]*?"
    r"(?=\n\s*(?:For |On |With |I['']d |My |The |If you |What['']s |\Z))",
    re.I,
)
_CONSULTANT_INSIGHT_RE = re.compile(
    r"\n\s*Consultant\s+Insight:[\s\S]*$",
    re.I,
)
_BOTTOM_LINE_RE = re.compile(
    r"\n\s*Bottom\s+Line:[\s\S]*?(?=\n\s*Consultant\s+Insight:|\Z)",
    re.I,
)
_CLOSER_RE = re.compile(
    r"\n\s*(if you have|feel free to ask|let me know if|happy to go deeper)[\s\S]*$",
    re.I,
)


def is_pinpoint_factual_turn(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when the user asked for one fact field, not an open advisory brief."""
    q = (query or "").strip()
    if not q or not _PINPOINT_QUERY_RE.search(q):
        return False

    du = data_used if isinstance(data_used, dict) else {}
    fine = str(du.get("consultant_fine_intent") or du.get("fine_intent") or "").lower()
    if fine in (
        "aircraft_specs",
        "market_question",
        "market_pricing",
        "technical_spec",
    ):
        return True

    ql = q.lower()
    if re.search(r"\b(compare|versus|vs\.?|best\s+jet|recommend|show\s+me)\b", ql):
        return False
    if re.search(r"\b(seats?|passengers?|range|price|asking|cost)\b", ql):
        return True
    return False


def strip_advisory_boilerplate(answer: str) -> str:
    """Remove appended GOOD FIT / stock shortlist blocks from any consultant reply."""
    s = (answer or "").strip()
    if not s:
        return s
    patterns = (
        _GOOD_FIT_BLOCK_RE,
        _ASSUMING_BLOCK_RE,
        _REALISTIC_FITS_BLOCK_RE,
        _BOTTOM_LINE_RE,
        _CONSULTANT_INSIGHT_RE,
        _CLOSER_RE,
    )
    for _ in range(4):
        prev = s
        for pat in patterns:
            s = pat.sub("", s).strip()
        if s == prev:
            break
    s = re.sub(r"\s*✅\s*GOOD\s+FIT\s*", "", s, flags=re.I).strip()
    s = re.sub(
        r"\n\s*here are a few realistic fits:\s*",
        "\n",
        s,
        flags=re.I,
    ).strip()
    return s


def _pinpoint_field(query: str) -> Optional[str]:
    ql = (query or "").lower()
    if re.search(r"\b(seats?|passengers?|pax)\b", ql):
        return "seats"
    if re.search(r"\brange\b", ql):
        return "range"
    if re.search(r"\b(price|cost|asking)\b", ql):
        return "price"
    return None


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _extract_pinpoint_sentences(answer: str, field: str) -> str:
    sentences = _split_sentences(strip_advisory_boilerplate(answer))
    if not sentences:
        return (answer or "").strip()

    if field == "seats":
        kept = [
            s
            for s in sentences
            if re.search(r"\b(seats?|passengers?|pax|accommodat|configuration)\b", s, re.I)
            and not re.search(r"\b(nm|knot|mach|cruise\s+speed|nautical\s+miles)\b", s, re.I)
        ]
        max_s, max_w = 2, 45
    elif field == "range":
        kept = [
            s
            for s in sentences
            if re.search(r"\b(range|nm|nautical|kilometer|mile)\b", s, re.I)
        ]
        out: list[str] = []
        for s in kept[:2]:
            s = re.sub(r",?\s*when\s+flying\b.*$", "", s, flags=re.I)
            s = re.sub(r",?\s*at\s+a\s+(?:typical\s+)?cruise\b.*$", "", s, flags=re.I)
            s = re.sub(r",?\s*accommodat(?:ing|es)?\b.*$", "", s, flags=re.I)
            s = re.sub(r",?\s*with\s+\d+\s+passengers?.*$", "", s, flags=re.I)
            s = re.sub(r",?\s*configured\s+for\s+\d+.*$", "", s, flags=re.I)
            s = re.sub(r",?\s*and\s+three\s+crew.*$", "", s, flags=re.I)
            if s.strip():
                out.append(s.strip())
        kept = out or kept
        max_s, max_w = 1, 40
    else:
        kept = sentences[:2]
        max_s, max_w = 2, 55

    if not kept:
        kept = sentences[:1]

    text = " ".join(kept[:max_s]).strip()
    words = re.findall(r"\b\w+\b", text)
    if len(words) > max_w:
        m = re.search(r"^(.{0,320}?[.!?])", text + " ")
        text = (m.group(1) if m else " ".join(words[:max_w])).strip()
    return text


def trim_pinpoint_answer(answer: str, *, max_words: int = 85, query: str = "") -> str:
    field = _pinpoint_field(query)
    if field:
        extracted = _extract_pinpoint_sentences(answer, field)
        if extracted:
            return extracted
    s = strip_advisory_boilerplate(answer)
    words = re.findall(r"\b\w+\b", s)
    if len(words) > max_words:
        m = re.search(r"^(.{0,400}?[.!?])\s", s + " ")
        if m:
            return m.group(1).strip()
    return s


def enforce_pinpoint_answer(
    answer: str,
    *,
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    if not is_pinpoint_factual_turn(query, data_used):
        return answer
    return trim_pinpoint_answer(answer, query=query or "")

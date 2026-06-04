"""
Broker fallback messages — translate internal system prose into broker language.

Internal codes and diagnostic strings must never reach the client.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Patterns that indicate internal-only content (for classification / tests).
INTERNAL_MARKERS: Tuple[str, ...] = (
    "INSUFFICIENT_DATA",
    "deterministic execution",
    "verified catalog",
    "verified aircraft",
    "mission kernel",
    "catalog authority",
    "catalog contrast",
    "authority dispatch",
    "confidence threshold",
    "safety check",
    "comparison safety",
    "temporal confidence low",
    "CLARIFICATION_REQUIRED",
    "INFEASIBLE_BUDGET_CONSTRAINT",
    "MARKET_CONTEXT_AVAILABLE",
    "Adversarial input normalized",
    "catalog names required",
    "mission authority kernel",
    "operational synthesis",
    "approved shortlist",
)

# Ordered (pattern, replacement) — first match wins per line where noted.
_FULL_MESSAGE_REPLACEMENTS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?is)^\s*Insufficient verified data for deterministic execution\.\s*"
            r"(?:Verified catalog comparison requires two recognized aircraft models\.?\s*)?$"
        ),
        "Which aircraft are you comparing against? I need both models named clearly to give you a side-by-side read.",
    ),
    (
        re.compile(
            r"(?is)^\s*Insufficient verified data for deterministic execution\.\s*"
            r"Tier-peer alternatives require a verified catalog target aircraft\.?\s*$"
        ),
        "Tell me which aircraft you want alternatives for — I need a specific model to suggest credible replacements.",
    ),
    (
        re.compile(
            r"(?is)^\s*Insufficient verified data for deterministic execution\.\s*"
            r"Structured buy-decision analysis requires a resolved model and ask price\.?\s*$"
        ),
        "To assess the deal I need the aircraft model, year if you have it, and the asking price.",
    ),
    (
        re.compile(
            r"(?is)^\s*Insufficient verified data for deterministic execution\.\s*"
            r"Structured valuation requires a resolved aircraft model and verified market context\.?\s*$"
        ),
        "Which aircraft and year should I value? I need a specific model to pull market context.",
    ),
    (
        re.compile(
            r"(?is)^\s*Insufficient verified data for deterministic execution\.\s*"
            r"Fleet portfolio analysis requires at least two verified aircraft in fleet input\.?\s*$"
        ),
        "For fleet analysis I need at least two aircraft identified in your fleet.",
    ),
    (
        re.compile(
            r"(?is)^\s*Insufficient verified data for deterministic execution\.\s*"
            r"Multi-criteria optimization requires at least two verified candidate aircraft\.?\s*$"
        ),
        "To compare options I need at least two aircraft named in your request.",
    ),
    (
        re.compile(
            r"(?is)^\s*Insufficient verified data for deterministic execution\.?\s*$"
        ),
        "I don't have enough information to give a reliable recommendation.",
    ),
    (
        re.compile(
            r"(?is)^\s*INSUFFICIENT_DATA:\s*Insufficient verified aircraft data for this request\.\s*"
            r"(?:Mission Fit:.*?)?Verdict:\s*INSUFFICIENT_DATA\s*$"
        ),
        "I don't have enough information to give a reliable recommendation.",
    ),
    (
        re.compile(
            r"(?is)^\s*INSUFFICIENT_DATA:\s*No verified aircraft available\.?\s*$"
        ),
        "I don't have enough information to give a reliable recommendation.",
    ),
    (
        re.compile(
            r"(?is)^\s*INSUFFICIENT_DATA:\s*Insufficient verified aircraft data to produce a comparison\.?\s*$"
        ),
        "I can compare them, but I need the second aircraft model first.",
    ),
    (
        re.compile(
            r"(?is)^\s*Insufficient verified aircraft data to produce a comparison for those aircraft\.?\s*$"
        ),
        "I can compare them, but I need the second aircraft model first.",
    ),
    (
        re.compile(
            r"(?is)^\s*Comparison requires two recognized aircraft models\.?\s*$"
        ),
        "Which aircraft are you comparing against?",
    ),
    (
        re.compile(
            r"(?is)^\s*Verified catalog comparison requires two recognized aircraft models\.?\s*$"
        ),
        "Which aircraft are you comparing against?",
    ),
    (
        re.compile(
            r"(?is)^\s*Deterministic execution unavailable\.?\s*$"
        ),
        "I can't confidently identify the aircraft from that description.",
    ),
)

_LINE_REPLACEMENTS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^INSUFFICIENT_DATA:\s*", re.I | re.M),
        "",
    ),
    (
        re.compile(r"^VERDICT:\s*INSUFFICIENT_DATA\s*$", re.I | re.M),
        "",
    ),
    (
        re.compile(r"^Verdict:\s*INSUFFICIENT_DATA\s*$", re.I | re.M),
        "",
    ),
    (
        re.compile(r"^Verdict:\s*CLARIFICATION_REQUIRED\s*$", re.I | re.M),
        "",
    ),
    (
        re.compile(r"^Verdict:\s*INFEASIBLE_BUDGET_CONSTRAINT\s*$", re.I | re.M),
        "",
    ),
    (
        re.compile(r"^Verdict:\s*MARKET_CONTEXT_AVAILABLE\s*$", re.I | re.M),
        "",
    ),
    (
        re.compile(r"^CLARIFICATION_REQUIRED:\s*", re.I | re.M),
        "",
    ),
    (
        re.compile(r"^Verified catalog comparison:\s*$", re.I | re.M),
        "",
    ),
    (
        re.compile(r"\(verified catalog\)", re.I),
        "",
    ),
    (
        re.compile(r"\(catalog operating index\)", re.I),
        "",
    ),
    (
        re.compile(r"\bcatalog practical\b", re.I),
        "typical",
    ),
    (
        re.compile(r"\bverified operating cost band\b", re.I),
        "operating cost band",
    ),
    (
        re.compile(r"\bverified range\b", re.I),
        "range",
    ),
    (
        re.compile(r"\bverified profile\b", re.I),
        "profile",
    ),
    (
        re.compile(r"\bverified catalog\b", re.I),
        "market",
    ),
    (
        re.compile(r"\bverified aircraft data\b", re.I),
        "aircraft data",
    ),
    (
        re.compile(r"\bverified tier[- ]peer\b", re.I),
        "comparable",
    ),
    (
        re.compile(r"\bverified tier peers\b", re.I),
        "comparable models",
    ),
    (
        re.compile(r"\bmission kernel synthesis\b", re.I),
        "",
    ),
    (
        re.compile(r"\bcatalog authority\b", re.I),
        "market reference",
    ),
    (
        re.compile(r"\bcatalog contrast\b", re.I),
        "side-by-side comparison",
    ),
    (
        re.compile(r"\bdeterministic execution\b", re.I),
        "",
    ),
    (
        re.compile(r"\bAdversarial input normalized\b", re.I),
        "Your request mixes a few constraints",
    ),
    (
        re.compile(r"\bResolve budget vs model class before a buy verdict\b", re.I),
        "Clarify the budget and model before I can assess the deal",
    ),
    (
        re.compile(r"\bAircraft:\s*\(constraint review\)\s*$", re.I | re.M),
        "",
    ),
    (
        re.compile(r"\bAircraft:\s*\(clarification required\)\s*$", re.I | re.M),
        "",
    ),
    (
        re.compile(
            r"Insufficient verified aircraft data to produce a comparison for ([^.]+)\.\s*",
            re.I,
        ),
        r"I can work with \1, but I need the other aircraft named clearly before I compare them. ",
    ),
    (
        re.compile(
            r"Insufficient verified aircraft data to produce a comparison\.\s*",
            re.I,
        ),
        "I can compare them, but I need the second aircraft model first. ",
    ),
    (
        re.compile(
            r"Insufficient verified aircraft data for deterministic recommendation\.?",
            re.I,
        ),
        "I don't have enough information to give a reliable recommendation.",
    ),
    (
        re.compile(r"Insufficient verified market comps in synced data\.?", re.I),
        "I don't have enough recent market comps loaded for this model.",
    ),
    (
        re.compile(r"Insufficient verified market comps\.?", re.I),
        "I don't have enough recent market comps for this model.",
    ),
    (
        re.compile(r"Listing depth below band threshold; catalog band shown for orientation only\.?", re.I),
        "Listing depth is thin — treat the band as directional only.",
    ),
    (
        re.compile(r"Confidence:\s*MODERATE \(catalog authority — not live listing median\)", re.I),
        "Confidence: moderate (reference band — not live listing median)",
    ),
)

_INFEASIBLE_BUDGET_RE = re.compile(
    r"(?is)Market Reality:\s*\n\s*-\s*(.+?)\s*\n\s*(?:Verdict:\s*INFEASIBLE_BUDGET_CONSTRAINT)?"
)

_CLARIFICATION_QUERY_PATTERNS: Dict[str, str] = {
    "cheap_gulfstream": (
        "When you say 'cheap Gulfstream,' are you looking for:\n"
        "• the least expensive Gulfstream available today\n"
        "• a specific model such as a G450 or G550\n"
        "• or simply something cheaper than a G650?"
    ),
    "g700_cheaper": (
        "Cheaper in what sense?\n\n"
        "Range, cabin size, operating cost, or acquisition price?"
    ),
    "forgot_second_model": (
        "Which aircraft are you comparing against? "
        "Name both models and I will give you a side-by-side read."
    ),
    "longitude_solo": (
        "When you say 'Longitude,' do you mean the Citation Longitude? "
        "If so, tell me what you want to do — compare it, value one, or find alternatives."
    ),
    "budget_mission": (
        "With a $20M budget, are you prioritizing range, cabin size, operating cost, or a specific mission profile? "
        "That will narrow the field quickly."
    ),
}


def classify_occurrence(text: str) -> str:
    """Return INTERNAL ONLY or USER FACING for a string fragment."""
    low = (text or "").lower()
    for marker in INTERNAL_MARKERS:
        if marker.lower() in low:
            return "INTERNAL ONLY"
    return "USER FACING"


def contains_internal_language(text: str) -> bool:
    return classify_occurrence(text) == "INTERNAL ONLY"


def _match_clarification_query(query: str) -> Optional[str]:
    q = (query or "").strip().lower()
    if not q:
        return None
    if re.search(r"\bcheap\s+gulfstream\b", q):
        return _CLARIFICATION_QUERY_PATTERNS["cheap_gulfstream"]
    if re.search(r"\bg700\b", q) and re.search(r"\bcheaper\b", q):
        return _CLARIFICATION_QUERY_PATTERNS["g700_cheaper"]
    if re.search(r"\bcompare\b", q) and re.search(
        r"\b(?:forgot|missing|second|other)\b.*\bmodel\b|\bmodel\b.*\b(?:forgot|missing|second)\b", q
    ):
        return _CLARIFICATION_QUERY_PATTERNS["forgot_second_model"]
    if re.search(r"\bcompare\b", q) and re.search(r"\bforgot\b|\bsecond\b", q):
        return _CLARIFICATION_QUERY_PATTERNS["forgot_second_model"]
    if re.fullmatch(r"(?:the\s+)?longitude(?:\s+jet)?\??", q) or q == "longitude jet":
        return _CLARIFICATION_QUERY_PATTERNS["longitude_solo"]
    if re.search(r"\$20m\b|\b20\s*m\b|\b20\s*million\b", q) and re.search(
        r"\b(?:buy|budget|what\s+jet|should\s+i)\b", q
    ):
        return _CLARIFICATION_QUERY_PATTERNS["budget_mission"]
    return None


def _translate_infeasible_budget(text: str) -> str:
    m = _INFEASIBLE_BUDGET_RE.search(text or "")
    if not m:
        if "INFEASIBLE_BUDGET_CONSTRAINT" in (text or "").upper():
            return (
                "That budget and aircraft class don't line up. "
                "Tell me your real ceiling or the model band you are targeting and I will reset the search."
            )
        return text
    reason = m.group(1).strip().rstrip(".")
    if reason.lower().startswith("budget and aircraft class appear incompatible"):
        return (
            "That budget and aircraft class don't line up. "
            "Tell me your real ceiling or the model band you are targeting."
        )
    return f"{reason.rstrip('.')}. Tell me your real budget ceiling or a model band that fits it."


def translate_internal_messages(text: str) -> str:
    """Replace known internal fallback strings with broker language."""
    out = (text or "").strip()
    if not out:
        return out

    for pat, repl in _FULL_MESSAGE_REPLACEMENTS:
        if pat.fullmatch(out) or pat.search(out):
            candidate = pat.sub(repl, out).strip()
            if candidate != out:
                out = candidate
                break

    if "INFEASIBLE_BUDGET_CONSTRAINT" in out.upper() or re.search(
        r"budget and aircraft class appear incompatible", out, re.I
    ):
        out = _translate_infeasible_budget(out)

    for pat, repl in _LINE_REPLACEMENTS:
        out = pat.sub(repl, out)

    out = re.sub(r"\bINSUFFICIENT_DATA\b", "", out, flags=re.I)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    return out.strip()


def broker_fallback_for_query(query: str, answer: str) -> Optional[str]:
    """
    Return a targeted clarification when the query is ambiguous and the answer
    looks like a system failure or empty clarification block.
    """
    clarification = _match_clarification_query(query)
    if not clarification:
        return None

    low = (answer or "").lower()
    weak = (
        not answer
        or len(answer.strip()) < 40
        or contains_internal_language(answer)
        or "clarification" in low
        or "second aircraft model" in low
        or "don't have enough" in low
        or "which aircraft are you comparing" in low
    )
    if weak:
        return clarification
    return None


def apply_broker_fallbacks(text: str, *, query: str = "") -> str:
    """Full fallback pass: translate internals, then query-specific clarification."""
    out = translate_internal_messages(text)
    targeted = broker_fallback_for_query(query, out)
    if targeted:
        return targeted
    if not out and query:
        targeted = _match_clarification_query(query)
        if targeted:
            return targeted
    if not out:
        return "I don't have enough information to give a reliable recommendation."
    return out


__all__ = [
    "INTERNAL_MARKERS",
    "apply_broker_fallbacks",
    "broker_fallback_for_query",
    "classify_occurrence",
    "contains_internal_language",
    "translate_internal_messages",
]

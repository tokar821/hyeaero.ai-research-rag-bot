"""Classify buyer decision questions — not routing intents."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, Optional


class DecisionIntent(str, Enum):
    REALISTICITY_CHECK = "REALISTICITY_CHECK"
    BUY_OR_WAIT = "BUY_OR_WAIT"
    OVERPAY_CHECK = "OVERPAY_CHECK"
    ALTERNATIVE_DISCOVERY = "ALTERNATIVE_DISCOVERY"
    BUDGET_MATCH = "BUDGET_MATCH"
    STRETCH_BUDGET = "STRETCH_BUDGET"
    GENERAL_ACQUISITION = "GENERAL_ACQUISITION"
    NONE = "NONE"


_CAN_GET_RE = re.compile(
    r"(?is)\bcan\s+i\s+(?:get|buy|afford)\s+(?:a|an)?\s*(?P<tail>.+?)\s+"
    r"(?:for|at|under|below)\s+\$?\s*(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)?\b",
)
_UNDER_MODEL_RE = re.compile(
    r"(?is)\b(?P<model>[A-Za-z][\w\s+\-]{1,35}?)\s+under\s+\$?\s*"
    r"(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)?\b",
)
_FOUND_LISTING_RE = re.compile(
    r"(?is)\b(?:saw|found|listing|asking|listed)\s+(?:a|an|one\s+for)?\s*"
    r"(?:\$|usd\s*)?(?P<price>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)?",
)
_BUY_WAIT_RE = re.compile(
    r"(?is)\b(?:should\s+i\s+buy|should\s+i\s+wait|buy\s+now\s+or\s+wait|wait\s+until|wait\s+one\s+year|"
    r"wait\s+six\s+months|wait\s+for\s+more\s+inventory|wait\s+or\s+buy|prices?\s+(?:are\s+)?(?:soft|rising)|"
    r"market\s+trend|timing|good\s+time\s+to\s+buy|good\s+entry\s+point|seller'?s?\s+market|"
    r"financing\s+rates|found\s+a\s+deal|keep\s+looking)\b",
)
_STRETCH_RE = re.compile(
    r"(?is)\b(?:stretch\s+budget|stretch\s+for|worth\s+stretching|over\s+my\s+budget)\b",
)
_LIKE_CHEAPER_RE = re.compile(r"(?is)\b(?:like\s+.+\s+but\s+cheaper|cheaper\s+than|alternative\s+to)\b")
_BUDGET_MATCH_RE = re.compile(
    r"(?is)\b(?:what\s+(?:can|should)\s+i\s+buy|best\s+(?:jet|aircraft)|smartest\s+jet)\s+"
    r"(?:for|around|about|under|with)?\s*\$?\s*(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil)?\b",
)
_BUDGET_ONLY_RE = re.compile(
    r"(?is)\b(?:for|around|about|under|with)\s+\$?\s*(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil)\b",
)
_GULFSTREAM_BUDGET_RE = re.compile(
    r"(?is)\b(?:gulfstream|dassault|falcon)\s+(?:under|below|for)\s+\$?\s*"
    r"(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil)?\b",
)


def _to_musd(amount: str, unit: str) -> Optional[float]:
    try:
        val = float(amount)
    except ValueError:
        return None
    u = (unit or "m").lower()
    if u == "k":
        return val / 1000.0
    if val < 1000:
        return val
    return val / 1_000_000.0 if val >= 10_000 else val


def detect_decision_intent(query: str, *, data_used: Optional[Dict[str, Any]] = None) -> DecisionIntent:
    """Classify the buyer's underlying decision question."""
    q = (query or "").strip()
    if not q:
        return DecisionIntent.NONE

    du = data_used or {}
    cc = du.get("client_context") or du.get("broker_conversation_context") or {}
    if isinstance(cc, dict) and cc.get("remembered_budget_musd") is not None:
        if re.search(r"(?is)\b(?:what\s+should\s+i\s+buy|best\s+jet|what\s+can\s+i\s+buy)\b", q):
            return DecisionIntent.BUDGET_MATCH

    adv = du.get("adversarial") or {}
    if adv.get("budget_feasibility") == "INFEASIBLE":
        return DecisionIntent.REALISTICITY_CHECK

    if re.search(r"(?is)\bwhat\s+should\s+i\s+buy\b", q):
        return DecisionIntent.BUDGET_MATCH
    if re.search(
        r"(?is)\b(?:coast.?to.?coast|nonstop|passengers?|pax)\b",
        q,
    ) and re.search(r"(?is)\b(?:buy|budget|\$\s*\d)\b", q):
        return DecisionIntent.BUDGET_MATCH

    if _BUY_WAIT_RE.search(q):
        return DecisionIntent.BUY_OR_WAIT
    if _STRETCH_RE.search(q):
        return DecisionIntent.STRETCH_BUDGET
    if _LIKE_CHEAPER_RE.search(q):
        return DecisionIntent.ALTERNATIVE_DISCOVERY
    br = du.get("broker_reasoning") or {}
    if br.get("intent_expansion", {}).get("alternative_search"):
        return DecisionIntent.ALTERNATIVE_DISCOVERY

    _category_budget = re.search(
        r"(?is)\b(?:best\s+(?:jet|aircraft|super-?midsize\s+jet|falcon|challenger|citation)|cheap\s+gulfstream)\b",
        q,
    )

    if re.search(r"(?is)\bbest\s+super-?midsize\b", q) and re.search(
        r"(?is)\b(?:under|below)\s+\$?\s*\d+", q
    ):
        return DecisionIntent.BUDGET_MATCH

    if re.search(r"(?is)\bbest\s+light\s+jet\b", q) and re.search(
        r"(?is)\b(?:under|below|for)\s+\$?\s*\d+", q
    ):
        return DecisionIntent.BUDGET_MATCH

    if re.search(r"(?is)\bbest\s+falcon\b", q) and re.search(
        r"(?is)\b(?:under|below|for)\s+\$?\s*\d+", q
    ):
        return DecisionIntent.BUDGET_MATCH

    if _CAN_GET_RE.search(q) or (_UNDER_MODEL_RE.search(q) and not _category_budget):
        return DecisionIntent.REALISTICITY_CHECK

    if _FOUND_LISTING_RE.search(q) or re.search(
        r"(?is)\b(?:is\s+that\s+realistic|overpay|too\s+much|good\s+price|fair\s+price)\b", q
    ):
        return DecisionIntent.OVERPAY_CHECK

    if re.search(
        r"(?is)\b(?:g\d{3}|gulfstream|falcon|citation|challenger|longitude|latitude)\s+for\s+\$?\s*\d+",
        q,
    ) and re.search(r"(?is)\b(?:plausible|realistic|possible|feasible)\b", q):
        return DecisionIntent.REALISTICITY_CHECK

    if _BUDGET_MATCH_RE.search(q) or (
        _BUDGET_ONLY_RE.search(q)
        and re.search(r"(?is)\b(?:buy|what|best|smartest)\b", q)
    ):
        return DecisionIntent.BUDGET_MATCH

    if re.search(r"(?is)\b(?:passengers?|pax)\b", q) and re.search(
        r"(?is)\$\s*\d+(?:\.\d+)?\s*(?:m|mm|million|mil)\b", q
    ):
        return DecisionIntent.BUDGET_MATCH

    if _GULFSTREAM_BUDGET_RE.search(q):
        return DecisionIntent.ALTERNATIVE_DISCOVERY

    if re.search(r"(?is)\b(?:good\s+deal|worth\s+it|should\s+i\s+buy)\b", q):
        return DecisionIntent.OVERPAY_CHECK

    return DecisionIntent.GENERAL_ACQUISITION


def extract_price_context(query: str) -> Dict[str, Any]:
    """Extract model/budget/ask hints from query for decision builder."""
    q = query or ""
    ctx: Dict[str, Any] = {}

    m_get = _CAN_GET_RE.search(q)
    if m_get:
        ctx["model_phrase"] = (m_get.group("tail") or "").strip()
        if m_get.group("amt"):
            ctx["budget_musd"] = _to_musd(m_get.group("amt"), m_get.group("unit") or "m")
    _category_budget = re.search(
        r"(?is)\b(?:best\s+(?:jet|aircraft|super-?midsize\s+jet)|cheap\s+gulfstream)\b",
        q,
    )
    m_under = _UNDER_MODEL_RE.search(q)
    if m_under and not _category_budget:
        ctx["model_phrase"] = (m_under.group("model") or "").strip()
        if m_under.group("amt"):
            ctx["budget_musd"] = _to_musd(m_under.group("amt"), m_under.group("unit") or "m")

    fm = _FOUND_LISTING_RE.search(q)
    if fm:
        ctx["ask_musd"] = _to_musd(fm.group("price"), fm.group("unit") or "m")

    bm = _BUDGET_MATCH_RE.search(q) or _BUDGET_ONLY_RE.search(q)
    if bm and bm.group("amt"):
        ctx["budget_musd"] = _to_musd(bm.group("amt"), bm.group("unit") or "m")

    gm = _GULFSTREAM_BUDGET_RE.search(q)
    if gm and gm.group("amt"):
        ctx["budget_musd"] = _to_musd(gm.group("amt"), gm.group("unit") or "m")
        ctx["manufacturer"] = "Gulfstream" if "gulfstream" in q.lower() else "Dassault"

    return ctx


__all__ = ["DecisionIntent", "detect_decision_intent", "extract_price_context"]

"""
Phase 55 — execution category guards (routing / layer priority only).

Does not change ranking, RAG retrieval, or aircraft selection algorithms.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, Optional

_TAIL_CATEGORIES = frozenset(
    {
        "tail_lookup",
        "tail_ownership",
        "registry_lookup",
        "tail_history",
    }
)

_OWNERSHIP_RE = re.compile(
    r"(?is)\b(?:who\s+owns|who\s+is\s+the\s+owner|owner\s+of|ownership\s+of|registered\s+owner)\b"
)
_REGISTRY_RE = re.compile(
    r"(?is)\b(?:registry|faa\s+registry|n-number|tail\s+number|registration\s+record)\b"
)
_HISTORY_RE = re.compile(
    r"(?is)\b(?:history\s+of|accident\s+history|incident\s+history|damage\s+history)\b"
)
_COMPARISON_RE = re.compile(
    r"(?is)\b(?:\bvs\.?\b|versus|compare\s+.+\s+(?:to|vs|and)\s+|side\s+by\s+side)\b"
)
_COMPARISON_BUY_RE = re.compile(
    r"(?is)\b(?:which\s+should\s+i\s+buy|what\s+would\s+you\s+choose|best\s+option|"
    r"which\s+one\s+should\s+i\s+buy|what\s+should\s+i\s+buy)\b"
)
_MISSION_SIGNAL_RE = re.compile(
    r"(?is)\b(?:passengers?|pax)\b.{0,80}\b(?:to|from)\b|\b(?:from|between)\b.{0,80}\b(?:to)\b|"
    r"\bcoast.?to.?coast\b|\bnonstop\b"
)
_LISTING_PRICE_RE = re.compile(
    r"(?is)\b(?:asking|listed|for\s+sale|year\s+\d{4})\b.*\$|(?:\d{4})\s+\w+\s+asking\b"
)


class BrokerExecutionCategory(str, Enum):
    TAIL_LOOKUP = "tail_lookup"
    TAIL_OWNERSHIP = "tail_ownership"
    REGISTRY_LOOKUP = "registry_lookup"
    TAIL_HISTORY = "tail_history"
    COMPARISON = "comparison"
    MISSION = "mission"
    LISTING = "listing"
    ACQUISITION = "acquisition"
    GENERAL = "general"


def _extract_registrations(query: str) -> list[str]:
    try:
        from rag.aviation_tail import find_strict_tail_candidates_in_text

        return list(find_strict_tail_candidates_in_text(query or "") or [])
    except Exception:
        return list(dict.fromkeys(re.findall(r"\bN[A-Z0-9]{3,6}\b", (query or "").upper())))


def classify_broker_execution_category(
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> BrokerExecutionCategory:
    q = (query or "").strip()
    ql = q.lower()
    du = data_used if isinstance(data_used, dict) else {}

    try:
        from services.market_reality.listing_detector import detect_listing_signal, ListingMode

        sig = detect_listing_signal(q)
        if sig.mode == ListingMode.TAIL_INVESTIGATION and sig.registrations:
            if _OWNERSHIP_RE.search(q):
                return BrokerExecutionCategory.TAIL_OWNERSHIP
            if _HISTORY_RE.search(q):
                return BrokerExecutionCategory.TAIL_HISTORY
            return BrokerExecutionCategory.TAIL_LOOKUP
    except Exception:
        pass

    regs = _extract_registrations(q)
    if regs:
        try:
            from services.broker_execution.tail_depth_mode import classify_tail_depth_mode, TailDepthMode

            tdepth, _ = classify_tail_depth_mode(q)
            if tdepth == TailDepthMode.OWNER:
                return BrokerExecutionCategory.TAIL_OWNERSHIP
            if tdepth == TailDepthMode.SALE_STATUS:
                return BrokerExecutionCategory.TAIL_LOOKUP
            if tdepth == TailDepthMode.COMPARISON:
                return BrokerExecutionCategory.COMPARISON
            if tdepth in (TailDepthMode.ACQUISITION, TailDepthMode.ACQUISITION_RISKS):
                return BrokerExecutionCategory.ACQUISITION
            if tdepth == TailDepthMode.ENGINE_PROGRAM:
                return BrokerExecutionCategory.GENERAL
            if tdepth == TailDepthMode.IMAGES:
                return BrokerExecutionCategory.GENERAL
            if tdepth == TailDepthMode.MARKET_PRICE:
                return BrokerExecutionCategory.LISTING
            if tdepth == TailDepthMode.DETAIL:
                return BrokerExecutionCategory.TAIL_LOOKUP
            if tdepth == TailDepthMode.SUMMARY:
                return BrokerExecutionCategory.TAIL_LOOKUP
            # CONTEXT and other analytical tail intents — not tail_lookup card.
            if _OWNERSHIP_RE.search(q):
                return BrokerExecutionCategory.TAIL_OWNERSHIP
            if _HISTORY_RE.search(q):
                return BrokerExecutionCategory.TAIL_HISTORY
            return BrokerExecutionCategory.GENERAL
        except Exception:
            pass

        if _OWNERSHIP_RE.search(q):
            return BrokerExecutionCategory.TAIL_OWNERSHIP
        if _HISTORY_RE.search(q):
            return BrokerExecutionCategory.TAIL_HISTORY
        if _REGISTRY_RE.search(q):
            return BrokerExecutionCategory.REGISTRY_LOOKUP
        if not re.search(r"(?is)\b(?:buy|asking|listed|good\s+deal)\b", ql):
            return BrokerExecutionCategory.GENERAL

    if _COMPARISON_RE.search(q) and not _COMPARISON_BUY_RE.search(q):
        return BrokerExecutionCategory.COMPARISON

    if re.search(
        r"(?is)\b(?:seller\s+reduced|reduced\s+from|cut\s+from|dropped\s+from|"
        r"price\s+(?:cut|drop)|lowered\s+the\s+ask)\b",
        q,
    ):
        return BrokerExecutionCategory.LISTING

    if _LISTING_PRICE_RE.search(q) or (
        re.search(r"(?is)\b(?:listing|asking|realistic)\b", ql) and re.search(r"\$\s*\d", q)
    ):
        return BrokerExecutionCategory.LISTING

    if _MISSION_SIGNAL_RE.search(q) and not regs:
        if not re.search(r"(?is)\bwhat\s+should\s+i\s+buy\b", ql) or re.search(
            r"(?is)\b(?:passengers?|pax|to|from|nonstop|coast)\b", ql
        ):
            return BrokerExecutionCategory.MISSION

    if re.search(
        r"(?is)\b(?:what\s+should\s+i\s+buy|best\s+jet|can\s+i\s+buy|should\s+i\s+buy)\b",
        ql,
    ):
        return BrokerExecutionCategory.ACQUISITION

    collapsed = du.get("intent_collapse") or {}
    if isinstance(collapsed, dict):
        primary = str(collapsed.get("primary_intent") or "")
        if primary == "MISSION_PROFILE":
            return BrokerExecutionCategory.MISSION

    return BrokerExecutionCategory.GENERAL


def comparison_requests_recommendation(query: str) -> bool:
    return bool(_COMPARISON_BUY_RE.search(query or ""))


def executive_layer_allowed(category: BrokerExecutionCategory, query: str) -> bool:
    """Hard guard: executive recommendation must not run for these categories."""
    if category.value in _TAIL_CATEGORIES:
        return False
    if category == BrokerExecutionCategory.COMPARISON and not _COMPARISON_BUY_RE.search(
        query or ""
    ):
        return False
    if category == BrokerExecutionCategory.LISTING:
        return False
    return True


def tail_memory_isolated(category: BrokerExecutionCategory) -> bool:
    return category.value in _TAIL_CATEGORIES


def data_first_response_mode(category: BrokerExecutionCategory) -> str:
    """Observability label for response ordering policy."""
    if category.value in _TAIL_CATEGORIES:
        return "facts_first_tail"
    if category == BrokerExecutionCategory.LISTING:
        return "facts_first_listing"
    if category == BrokerExecutionCategory.COMPARISON:
        return "facts_first_comparison"
    return "standard"


def data_first_required(category: BrokerExecutionCategory) -> bool:
    return category in (
        BrokerExecutionCategory.LISTING,
        BrokerExecutionCategory.COMPARISON,
    )


def tail_registry_prepend_required(query: str, data_used: dict) -> bool:
    """Only owner/sale registry-card turns prepend the fact block onto answers."""
    try:
        from services.broker_execution.tail_depth_mode import TailDepthMode, classify_tail_depth_mode

        depth, _ = classify_tail_depth_mode(query)
        return depth in (TailDepthMode.OWNER, TailDepthMode.SALE_STATUS)
    except Exception:
        return False


def attach_broker_execution_context(
    data_used: dict,
    *,
    query: str,
) -> BrokerExecutionCategory:
    cat = classify_broker_execution_category(query, data_used=data_used)
    data_used["broker_execution_category"] = cat.value
    data_used["executive_layer_allowed"] = executive_layer_allowed(cat, query)
    data_used["tail_memory_isolated"] = tail_memory_isolated(cat)
    data_used["data_first_response_mode"] = data_first_response_mode(cat)
    data_used["data_first_required"] = data_first_required(cat)
    return cat


__all__ = [
    "BrokerExecutionCategory",
    "attach_broker_execution_context",
    "classify_broker_execution_category",
    "comparison_requests_recommendation",
    "executive_layer_allowed",
    "tail_memory_isolated",
    "data_first_response_mode",
]

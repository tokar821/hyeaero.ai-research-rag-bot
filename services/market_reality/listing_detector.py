"""Detect listing-specific and tail-specific discussion modes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ListingMode(str, Enum):
    NONE = "NONE"
    LISTING_DISCUSSION = "LISTING_DISCUSSION"
    TAIL_INVESTIGATION = "TAIL_INVESTIGATION"
    MARKET_TIMING = "MARKET_TIMING"
    LISTING_REALISM = "LISTING_REALISM"
    WHY_SO_CHEAP = "WHY_SO_CHEAP"
    BUYER_SELLER_MARKET = "BUYER_SELLER_MARKET"


_FOUND_LISTING_RE = re.compile(
    r"(?is)\b(?:saw|found|listing\s+says?|listed\s+at|asking|for\s+sale\s+at|"
    r"i\s+found|i\s+saw)\b",
)
_PRICE_RE = re.compile(
    r"(?is)(?:for|at|asking|listed)\s+\$?\s*(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)?\b|"
    r"\$\s*(?P<amt2>\d+(?:\.\d+)?)\s*(?P<unit2>m|mm|million|mil|k)?\b",
)
_WHY_CHEAP_RE = re.compile(r"(?is)\bwhy\s+is\s+(?:this|it|that)\s+(?:aircraft\s+)?so\s+cheap\b")
_REALISTIC_RE = re.compile(r"(?is)\b(?:is\s+this\s+listing\s+realistic|listing\s+realistic|realistic\s+listing)\b")
_TIMING_RE = re.compile(r"(?is)\b(?:good\s+time\s+to\s+buy)\b")
_BUYER_SELLER_RE = re.compile(r"(?is)\b(?:buyers?\s+market|sellers?\s+market)\b")
_TAIL_WORTH_RE = re.compile(
    r"(?is)\b(?:worth\s+looking\s+at|worth\s+investigat\w*|worth\s+it)\b",
)
_OWNERSHIP_RE = re.compile(
    r"(?is)\b(?:who\s+owns|who\s+is\s+the\s+owner|owner\s+of|ownership\s+of|registered\s+owner)\b",
)
_CAN_I_BUY_RE = re.compile(
    r"(?is)\bcan\s+i\s+(?:buy|afford|get|realistically\s+buy|realistically\s+afford)\b",
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


def _resolve_model(query: str) -> Optional[str]:
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        models = detect_models_from_text(query or "")
        if models:
            from services.broker_reasoning.broker_reasoning_layer import _resolve_model_name

            return _resolve_model_name(models[0])
    except Exception:
        pass
    return None


def _extract_ask_musd(query: str) -> Optional[float]:
    m = _PRICE_RE.search(query or "")
    if not m:
        return None
    amt = m.group("amt") or m.group("amt2")
    unit = m.group("unit") or m.group("unit2") or "m"
    return _to_musd(amt, unit) if amt else None


def _extract_registrations(query: str) -> List[str]:
    try:
        from rag.aviation_tail import find_strict_tail_candidates_in_text

        return list(find_strict_tail_candidates_in_text(query or "") or [])
    except Exception:
        pass
    return list(dict.fromkeys(re.findall(r"\bN[A-Z0-9]{3,6}\b", (query or "").upper())))


@dataclass
class ListingSignal:
    mode: ListingMode
    model: Optional[str] = None
    ask_musd: Optional[float] = None
    registrations: List[str] = field(default_factory=list)
    raw_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "model": self.model,
            "ask_musd": self.ask_musd,
            "registrations": list(self.registrations),
        }


def detect_listing_signal(query: str) -> ListingSignal:
    """Classify whether the turn is about a specific listing, tail, or market conditions."""
    q = (query or "").strip()
    if not q:
        return ListingSignal(mode=ListingMode.NONE, raw_query=q)

    regs = _extract_registrations(q)
    model = _resolve_model(q)
    ask = _extract_ask_musd(q)

    if regs and (
        _OWNERSHIP_RE.search(q)
        or _TAIL_WORTH_RE.search(q)
        or (not _CAN_I_BUY_RE.search(q) and not _FOUND_LISTING_RE.search(q) and ask is None)
    ):
        return ListingSignal(
            mode=ListingMode.TAIL_INVESTIGATION,
            model=model,
            ask_musd=ask,
            registrations=regs,
            raw_query=q,
        )

    if _WHY_CHEAP_RE.search(q):
        return ListingSignal(
            mode=ListingMode.WHY_SO_CHEAP,
            model=model,
            ask_musd=ask,
            registrations=regs,
            raw_query=q,
        )

    if _REALISTIC_RE.search(q):
        return ListingSignal(
            mode=ListingMode.LISTING_REALISM,
            model=model,
            ask_musd=ask,
            registrations=regs,
            raw_query=q,
        )

    if _BUYER_SELLER_RE.search(q):
        return ListingSignal(
            mode=ListingMode.BUYER_SELLER_MARKET,
            model=model,
            ask_musd=ask,
            registrations=regs,
            raw_query=q,
        )

    if _TIMING_RE.search(q):
        return ListingSignal(
            mode=ListingMode.MARKET_TIMING,
            model=model,
            ask_musd=ask,
            registrations=regs,
            raw_query=q,
        )

    if _CAN_I_BUY_RE.search(q) and model and ask is not None:
        return ListingSignal(
            mode=ListingMode.NONE,
            model=model,
            ask_musd=ask,
            registrations=regs,
            raw_query=q,
        )

    if _FOUND_LISTING_RE.search(q) or (ask is not None and model):
        return ListingSignal(
            mode=ListingMode.LISTING_DISCUSSION,
            model=model,
            ask_musd=ask,
            registrations=regs,
            raw_query=q,
        )

    return ListingSignal(mode=ListingMode.NONE, model=model, ask_musd=ask, registrations=regs, raw_query=q)


__all__ = ["ListingMode", "ListingSignal", "detect_listing_signal"]

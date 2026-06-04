"""
Tail query depth — intent-first routing when a registration is present.

Question intent wins over bare tail detection (fixes registry template hijacking).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional, Tuple


class TailDepthMode(str, Enum):
    NONE = "none"
    OWNER = "owner"
    SALE_STATUS = "sale_status"
    SUMMARY = "summary"
    DETAIL = "detail"
    ACQUISITION = "acquisition"
    ACQUISITION_RISKS = "acquisition_risks"
    ENGINE_PROGRAM = "engine_program"
    IMAGES = "images"
    COMPARISON = "comparison"
    MARKET_PRICE = "market_price"
    CONTEXT = "context"  # tail present; answer via LLM + facts, not registry-only template


_OWNER_RE = re.compile(
    r"(?is)\b(?:who\s+owns|who\s+is\s+the\s+owner|owner\s+of|ownership\s+of|registered\s+owner)\b"
)
_SALE_RE = re.compile(
    r"(?is)\b(?:for\s+sale|on\s+the\s+market|sale\s+status|currently\s+listed|listed\s+for\s+sale)\b"
)
_DETAIL_RE = re.compile(
    r"(?is)\b(?:tell\s+me\s+everything|everything\s+about|full\s+(?:detail|history|profile)|"
    r"complete\s+profile|all\s+(?:you\s+)?(?:know|have)\s+about)\b"
)
_ACQUISITION_RE = re.compile(
    r"(?is)\b(?:worth\s+buying|should\s+i\s+buy|good\s+deal|acquisition|pre-?buy|"
    r"due\s+diligence|logbooks?)\b"
)
_RISKS_RE = re.compile(
    r"(?is)\b(?:biggest\s+(?:acquisition\s+)?risks?|acquisition\s+risks?|"
    r"what\s+are\s+the\s+risks?|risks?\s+on|deal\s+risks?|red\s+flags?\s+on)\b"
)
_ENGINE_RE = re.compile(
    r"(?is)\b(?:engine\s+program|apu\s+program|propulsion\s+program|"
    r"enrolled\s+on\s+(?:an?\s+)?engine|msp\s|jssi|maintenance\s+tracking|cescom|camp)\b"
)
_IMAGE_RE = re.compile(
    r"(?is)\b(?:show\s+me|photos?|pictures?|images?|gallery|"
    r"(?:see|view)\s+(?:the\s+)?(?:aircraft|tail|cabin|cockpit|interior|exterior))\b"
)
_IMAGE_FACET_RE = re.compile(r"(?is)\b(?:cabin|cockpit|interior|exterior|salon|layout)\b")
_COMPARISON_RE = re.compile(
    r"(?is)\b(?:compare|comparison|versus|vs\.?|against\s+(?:a\s+)?(?:typical|mission))\b"
)
_MARKET_PRICE_RE = re.compile(
    r"(?is)\b(?:aggressive|fair|cheap|overpriced|underpriced|"
    r"listed\s+at|asking\s+price|would\s+you\s+consider\s+that)\b.*\$|"
    r"\$\d+.*\b(?:aggressive|fair|cheap)\b"
)
_WHAT_AIRCRAFT_RE = re.compile(
    r"(?is)\b(?:what\s+aircraft\s+is|what\s+(?:kind|type)\s+of\s+aircraft|which\s+aircraft\s+is)\b"
)
_SELLER_MOVE_RE = re.compile(
    r"(?is)\b(?:seller\s+reduced|reduced\s+from|cut\s+from|dropped\s+from|"
    r"price\s+(?:cut|drop)|lowered\s+the\s+ask)\b"
)


def _extract_tail(query: str) -> Optional[str]:
    try:
        from rag.aviation_tail import primary_registration_from_query

        return primary_registration_from_query(query or "")
    except Exception:
        return None


def classify_tail_depth_mode(query: str) -> Tuple[TailDepthMode, Optional[str]]:
    """
    Classify **question intent** when a tail is present.

    Order matters: specific intents before generic summary/default.
    """
    q = (query or "").strip()
    reg = _extract_tail(q)
    if not reg:
        return TailDepthMode.NONE, None

    # Visual intent (cabin/cockpit/exterior) before generic "show me".
    if _IMAGE_RE.search(q) or (_IMAGE_FACET_RE.search(q) and re.search(r"(?is)\b(?:show|see|photo|image)\b", q)):
        return TailDepthMode.IMAGES, reg

    if _ENGINE_RE.search(q) and not _OWNER_RE.search(q):
        return TailDepthMode.ENGINE_PROGRAM, reg

    if _COMPARISON_RE.search(q):
        return TailDepthMode.COMPARISON, reg

    if _RISKS_RE.search(q):
        return TailDepthMode.ACQUISITION_RISKS, reg

    if _MARKET_PRICE_RE.search(q) and not _SALE_RE.search(q):
        return TailDepthMode.MARKET_PRICE, reg

    if _SELLER_MOVE_RE.search(q):
        return TailDepthMode.MARKET_PRICE, reg

    if _DETAIL_RE.search(q):
        return TailDepthMode.DETAIL, reg

    if _SALE_RE.search(q):
        return TailDepthMode.SALE_STATUS, reg

    if _OWNER_RE.search(q):
        return TailDepthMode.OWNER, reg

    if _ACQUISITION_RE.search(q):
        return TailDepthMode.ACQUISITION, reg

    if _WHAT_AIRCRAFT_RE.search(q):
        return TailDepthMode.SUMMARY, reg

    # Tail present but intent is analytical — feed LLM, do not collapse to registry card.
    return TailDepthMode.CONTEXT, reg


def registry_template_depths() -> frozenset[TailDepthMode]:
    """Depths that may use the short registry fact card (owner/sale only)."""
    return frozenset((TailDepthMode.OWNER, TailDepthMode.SALE_STATUS))


def llm_required_depths() -> frozenset[TailDepthMode]:
    """Depths that must not be replaced by deterministic registry templates."""
    return frozenset(
        (
            TailDepthMode.ENGINE_PROGRAM,
            TailDepthMode.ACQUISITION,
            TailDepthMode.ACQUISITION_RISKS,
            TailDepthMode.COMPARISON,
            TailDepthMode.DETAIL,
            TailDepthMode.CONTEXT,
            TailDepthMode.MARKET_PRICE,
            TailDepthMode.IMAGES,
        )
    )


__all__ = [
    "TailDepthMode",
    "classify_tail_depth_mode",
    "llm_required_depths",
    "registry_template_depths",
]

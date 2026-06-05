"""
Source trust ranking for aircraft image verification.

Priority (highest first):
  manufacturer → operator → trusted aviation media → verified listings → generic web
Stock / unverified hosts are scored at zero and typically hard-rejected upstream.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional, Tuple
from urllib.parse import urlparse


class SourceTier(str, Enum):
    MANUFACTURER = "manufacturer"
    OPERATOR = "operator"
    TRUSTED_MEDIA = "trusted_media"
    VERIFIED_LISTING = "verified_listing"
    GENERAL_WEB = "general_web"
    STOCK_UNVERIFIED = "stock_unverified"


# Longer fragments first where order matters inside a tier tuple.
_MANUFACTURER_FRAGS: Tuple[Tuple[str, int], ...] = (
    ("dassault-aviation.com", 1000),
    ("dassaultfalcon.", 1000),
    ("falconjet.com", 990),
    ("gulfstream.com", 1000),
    ("bombardier.com", 990),
    ("businessaircraft.bombardier.com", 990),
    ("embraer.com", 980),
    ("embraerexecutivejets.com", 980),
    ("textronaviation.com", 970),
    ("cessna.txtav.com", 970),
    ("beechcraft.txtav.com", 960),
    ("pilatus-aircraft.com", 960),
    ("hondajet.com", 950),
    ("airbus.com", 940),
)

_OPERATOR_FRAGS: Tuple[Tuple[str, int], ...] = (
    ("netjets.com", 900),
    ("flexjet.com", 890),
    ("vistajet.com", 890),
    ("wheelsup.com", 880),
    ("xo.com", 870),
    ("flyxo.com", 870),
    ("claylacy.com", 880),
    ("duncanaviation.com", 870),
    ("weststaraviation", 860),
    ("signatureflight", 850),
    ("atlanticaviation", 850),
)

_TRUSTED_MEDIA_FRAGS: Tuple[Tuple[str, int], ...] = (
    ("jetphotos.", 820),
    ("planespotters.", 810),
    ("airliners.net", 800),
    ("flightglobal.com", 780),
    ("aviationweek.com", 780),
    ("ainonline.com", 770),
    ("bjtonline.com", 760),
    ("corporatejetinvestor.com", 750),
    ("nbaa.org", 740),
    ("faa.gov", 730),
    ("flightaware.com", 720),
    ("stackexchange.com", 680),
    ("sstatic.net", 660),
)

_VERIFIED_LISTING_FRAGS: Tuple[Tuple[str, int], ...] = (
    ("virtualhangar.com", 760),
    ("website-files.com", 720),
    ("flyexclusive.com", 740),
    ("phly.com", 730),
    ("controller.com", 700),
    ("aircraft.com", 695),
    ("aircraftexchange", 690),
    ("globalair.com", 680),
    ("avbuyer.", 650),
    ("trade-a-plane", 640),
    ("aso.com", 630),
)

_STOCK_FRAGS: Tuple[str, ...] = (
    "shutterstock",
    "gettyimages",
    "istockphoto",
    "dreamstime",
    "alamy",
    "depositphotos",
    "123rf",
    "stockphoto",
    "stock-photo",
    "watermark",
)


def extract_host(url: str) -> str:
    try:
        return (urlparse(url or "").netloc or "").lower()
    except Exception:
        return ""


def classify_source_tier(
    *,
    url: str = "",
    page_url: str = "",
    source_label: str = "",
    title: str = "",
) -> Tuple[SourceTier, int]:
    """
    Return ``(tier, raw_score)`` for ranking and confidence weighting.
    """
    blob = " ".join((url, page_url, source_label, title)).lower()
    host = extract_host(page_url) or extract_host(url)
    if host:
        blob = f"{host} {blob}"

    for frag in _STOCK_FRAGS:
        if frag in blob:
            return SourceTier.STOCK_UNVERIFIED, 0

    for frag, sc in _MANUFACTURER_FRAGS:
        if frag in blob:
            return SourceTier.MANUFACTURER, sc
    for frag, sc in _OPERATOR_FRAGS:
        if frag in blob:
            return SourceTier.OPERATOR, sc
    for frag, sc in _TRUSTED_MEDIA_FRAGS:
        if frag in blob:
            return SourceTier.TRUSTED_MEDIA, sc
    for frag, sc in _VERIFIED_LISTING_FRAGS:
        if frag in blob:
            return SourceTier.VERIFIED_LISTING, sc

    return SourceTier.GENERAL_WEB, 120


def source_trust_component(tier: SourceTier, raw_score: int) -> float:
    """Map tier/score to 0..1 trust component for confidence engine."""
    if tier == SourceTier.STOCK_UNVERIFIED:
        return 0.0
    caps = {
        SourceTier.MANUFACTURER: 1.0,
        SourceTier.OPERATOR: 0.92,
        SourceTier.TRUSTED_MEDIA: 0.88,
        SourceTier.VERIFIED_LISTING: 0.82,
        SourceTier.GENERAL_WEB: 0.35,
    }
    base = caps.get(tier, 0.3)
    if raw_score >= 900:
        return min(1.0, base)
    if raw_score >= 700:
        return min(1.0, base * 0.95)
    return min(1.0, base * 0.75)

"""Classify listing ask vs market band — uses existing deal-quality math, new labels only."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from services.market_intelligence.deal_quality_engine import (
    DealQualityResult,
    DealQualityVerdict,
    evaluate_deal_quality,
)
from services.market_intelligence.market_band_builder import BandConfidence, MarketBand


class ListingPriceConfidence(str, Enum):
    LIKELY_MARKET = "LIKELY_MARKET"
    UNUSUALLY_CHEAP = "UNUSUALLY_CHEAP"
    UNUSUALLY_EXPENSIVE = "UNUSUALLY_EXPENSIVE"
    POTENTIAL_DATA_ERROR = "POTENTIAL_DATA_ERROR"
    UNKNOWN = "UNKNOWN"


def analyze_listing_confidence(
    *,
    model: str,
    ask_usd: Optional[float],
    band: MarketBand,
    year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Map ask to band using ``evaluate_deal_quality`` (unchanged thresholds).
    """
    deal: Optional[DealQualityResult] = None
    if ask_usd is not None and band.mid is not None:
        deal = evaluate_deal_quality(model=model, year=year, ask_usd=ask_usd, band=band)

    if band.confidence == BandConfidence.INSUFFICIENT or band.mid is None:
        return {
            "confidence": ListingPriceConfidence.UNKNOWN.value,
            "reason": band.reason or "insufficient_market_band",
            "deal_verdict": None,
            "position_pct": None,
        }

    ask = float(ask_usd) if ask_usd else None
    if ask is None:
        return {
            "confidence": ListingPriceConfidence.UNKNOWN.value,
            "reason": "missing_ask",
            "deal_verdict": None,
            "position_pct": None,
        }

    low = float(band.low) if band.low is not None else None
    high = float(band.high) if band.high is not None else None
    mid = float(band.mid)

    if mid and ask < mid * 0.45:
        label = ListingPriceConfidence.POTENTIAL_DATA_ERROR
        reason = "Ask sits far below the verified listing band — verify the listing is real and complete."
    elif low is not None and ask < low * 0.45:
        label = ListingPriceConfidence.POTENTIAL_DATA_ERROR
        reason = "Ask sits far below the verified listing band — verify the listing is real and complete."
    elif high is not None and ask > high * 1.55:
        label = ListingPriceConfidence.POTENTIAL_DATA_ERROR
        reason = "Ask sits far above the verified listing band — confirm model year and equipment."
    elif deal and deal.verdict == DealQualityVerdict.GOOD_DEAL:
        label = ListingPriceConfidence.UNUSUALLY_CHEAP
        reason = deal.reason or "Material discount vs market median."
    elif deal and deal.verdict == DealQualityVerdict.OVERPRICED:
        label = ListingPriceConfidence.UNUSUALLY_EXPENSIVE
        reason = deal.reason or "Above typical market median."
    else:
        label = ListingPriceConfidence.LIKELY_MARKET
        reason = "Within a plausible range vs current listing-derived band."

    return {
        "confidence": label.value,
        "reason": reason,
        "deal_verdict": deal.display_verdict if deal else None,
        "position_pct": deal.position_pct if deal else (ask - mid) / mid if mid else None,
        "market_mid_usd": mid,
        "band_low_usd": low,
        "band_high_usd": high,
    }


__all__ = ["ListingPriceConfidence", "analyze_listing_confidence"]

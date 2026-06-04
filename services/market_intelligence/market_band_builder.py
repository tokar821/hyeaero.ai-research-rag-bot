"""Authoritative market bands from listing snapshots with outlier and staleness guards."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import List, Optional, Sequence

from services.market_intelligence.listing_analytics import MarketSnapshot

MIN_LISTINGS_FOR_BAND = 5
IQR_MULTIPLIER = 1.5


class BandConfidence(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class MarketBand:
    low: Optional[float]
    mid: Optional[float]
    high: Optional[float]
    confidence: BandConfidence
    listing_count: int
    rejected_outliers: int = 0
    reason: Optional[str] = None


def _parse_asks(rows: Sequence[dict]) -> List[float]:
    out: List[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            ask = float(r.get("ask_price"))
        except (TypeError, ValueError):
            continue
        if ask > 0:
            out.append(ask)
    return sorted(out)


def _reject_iqr_outliers(asks: List[float]) -> tuple[List[float], int]:
    if len(asks) < 4:
        return asks, 0
    qs = sorted(asks)
    n = len(qs)
    q1 = qs[n // 4]
    q3 = qs[(3 * n) // 4]
    iqr = q3 - q1
    if iqr <= 0:
        return asks, 0
    lo_fence = q1 - IQR_MULTIPLIER * iqr
    hi_fence = q3 + IQR_MULTIPLIER * iqr
    kept = [a for a in asks if lo_fence <= a <= hi_fence]
    rejected = len(asks) - len(kept)
    if len(kept) < MIN_LISTINGS_FOR_BAND:
        return asks, 0
    return kept, rejected


def build_market_band(
    snapshot: MarketSnapshot,
    ask_prices: Optional[Sequence[float]] = None,
    *,
    min_listings: int = MIN_LISTINGS_FOR_BAND,
) -> MarketBand:
    """
    Build low / mid / high band from snapshot aggregates and optional raw asks.

    Rejects stale snapshots and extreme IQR outliers; never invents prices.
    """
    if snapshot.insufficient_reason == "stale_market":
        return MarketBand(
            low=None,
            mid=None,
            high=None,
            confidence=BandConfidence.INSUFFICIENT,
            listing_count=snapshot.active_listing_count,
            reason="stale_market",
        )

    prices: List[float] = []
    if ask_prices:
        prices = sorted(float(p) for p in ask_prices if p and float(p) > 0)
    elif snapshot.median_ask_price is not None:
        if snapshot.low_ask_price is not None:
            prices.append(float(snapshot.low_ask_price))
        prices.append(float(snapshot.median_ask_price))
        if snapshot.high_ask_price is not None:
            prices.append(float(snapshot.high_ask_price))

    if len(prices) < min_listings:
        return MarketBand(
            low=None,
            mid=None,
            high=None,
            confidence=BandConfidence.INSUFFICIENT,
            listing_count=snapshot.active_listing_count,
            reason=snapshot.insufficient_reason or "too_few_listings",
        )

    trimmed, rejected = _reject_iqr_outliers(prices)
    if len(trimmed) < min_listings:
        return MarketBand(
            low=None,
            mid=None,
            high=None,
            confidence=BandConfidence.INSUFFICIENT,
            listing_count=len(prices),
            rejected_outliers=rejected,
            reason="too_few_listings_after_outlier_rejection",
        )

    low = float(min(trimmed))
    high = float(max(trimmed))
    mid = float(median(trimmed))

    conf = BandConfidence.HIGH
    if snapshot.active_listing_count < 10 or snapshot.stale:
        conf = BandConfidence.MODERATE
    if len(trimmed) < 8:
        conf = BandConfidence.MODERATE
    if snapshot.insufficient_reason:
        conf = BandConfidence.LOW

    return MarketBand(
        low=low,
        mid=mid,
        high=high,
        confidence=conf,
        listing_count=len(trimmed),
        rejected_outliers=rejected,
    )


def build_market_band_from_asks(
    snapshot: MarketSnapshot,
    rows: Sequence[dict],
) -> MarketBand:
    asks = _parse_asks(rows)
    return build_market_band(snapshot, ask_prices=asks)

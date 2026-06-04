"""Deterministic liquidity scoring from listing depth, DOM, and price dispersion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from services.market_intelligence.listing_analytics import MarketSnapshot


class LiquidityBand(str, Enum):
    HIGH = "HIGH"
    GOOD = "GOOD"
    MODERATE = "MODERATE"
    THIN = "THIN"


@dataclass(frozen=True)
class LiquidityScore:
    score: int
    band: LiquidityBand
    listing_points: int
    dom_points: int
    dispersion_points: int


def _dispersion_points(low: Optional[float], high: Optional[float], mid: Optional[float]) -> int:
    if mid is None or mid <= 0 or low is None or high is None:
        return 10
    spread_pct = (float(high) - float(low)) / float(mid)
    if spread_pct < 0.25:
        return 25
    if spread_pct < 0.45:
        return 15
    if spread_pct < 0.70:
        return 8
    return 3


def _dom_points(avg_dom: Optional[float]) -> int:
    if avg_dom is None:
        return 12
    if avg_dom <= 90:
        return 35
    if avg_dom <= 140:
        return 28
    if avg_dom <= 180:
        return 20
    if avg_dom <= 270:
        return 12
    if avg_dom <= 365:
        return 6
    return 3


def _listing_points(active_count: int) -> int:
    # 25 listings -> 40 pts (caps at 40)
    return min(40, max(0, int(round(active_count * 1.6))))


def _band_for_score(score: int) -> LiquidityBand:
    if score >= 80:
        return LiquidityBand.HIGH
    if score >= 60:
        return LiquidityBand.GOOD
    if score >= 40:
        return LiquidityBand.MODERATE
    return LiquidityBand.THIN


def compute_liquidity_score(snapshot: MarketSnapshot) -> LiquidityScore:
    """
    Deterministic 0–100 liquidity score.

    Weights: listings 40%, DOM 35%, dispersion 25% (each sub-score capped as designed).
    """
    lp = _listing_points(snapshot.active_listing_count)
    dp = _dom_points(snapshot.average_days_on_market)
    sp = _dispersion_points(
        snapshot.low_ask_price,
        snapshot.high_ask_price,
        snapshot.median_ask_price,
    )
    raw = lp + dp + sp
    score = max(0, min(100, int(raw)))
    return LiquidityScore(
        score=score,
        band=_band_for_score(score),
        listing_points=lp,
        dom_points=dp,
        dispersion_points=sp,
    )

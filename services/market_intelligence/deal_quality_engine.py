"""Deterministic deal-quality verdict vs market median / band."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from services.market_intelligence.market_band_builder import BandConfidence, MarketBand

GOOD_DEAL_THRESHOLD = -0.12
FAIR_UPPER = 0.15
OVERPRICED_THRESHOLD = 0.15


class DealQualityVerdict(str, Enum):
    GOOD_DEAL = "GOOD_DEAL"
    FAIR_DEAL = "FAIR_DEAL"
    OVERPRICED = "OVERPRICED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class DealQualityResult:
    verdict: DealQualityVerdict
    reason: str
    market_mid: Optional[float]
    ask_usd: Optional[float]
    position_pct: Optional[float]

    @property
    def display_verdict(self) -> str:
        return _display_verdict(self.verdict)


def _display_verdict(v: DealQualityVerdict) -> str:
    return {
        DealQualityVerdict.GOOD_DEAL: "GOOD DEAL",
        DealQualityVerdict.FAIR_DEAL: "FAIR DEAL",
        DealQualityVerdict.OVERPRICED: "OVERPRICED",
        DealQualityVerdict.INSUFFICIENT_DATA: "INSUFFICIENT_DATA",
    }.get(v, "INSUFFICIENT_DATA")


def evaluate_deal_quality(
    *,
    model: str,
    year: Optional[int],
    ask_usd: Optional[float],
    band: MarketBand,
) -> DealQualityResult:
    """
    Compare ask to market mid.

    Never fabricates market mid — requires band with usable mid and confidence.
    """
    _ = model, year  # reserved for future year-adjusted bands

    if band.confidence == BandConfidence.INSUFFICIENT or band.mid is None:
        reason = band.reason or "insufficient_listing_depth"
        return DealQualityResult(
            verdict=DealQualityVerdict.INSUFFICIENT_DATA,
            reason=reason,
            market_mid=None,
            ask_usd=ask_usd,
            position_pct=None,
        )

    if ask_usd is None or ask_usd <= 0:
        return DealQualityResult(
            verdict=DealQualityVerdict.INSUFFICIENT_DATA,
            reason="missing_ask_price",
            market_mid=band.mid,
            ask_usd=ask_usd,
            position_pct=None,
        )

    mid = float(band.mid)
    position_pct = (float(ask_usd) - mid) / mid

    if position_pct <= GOOD_DEAL_THRESHOLD:
        pct = abs(position_pct) * 100.0
        return DealQualityResult(
            verdict=DealQualityVerdict.GOOD_DEAL,
            reason=f"{pct:.1f}% below market median",
            market_mid=mid,
            ask_usd=float(ask_usd),
            position_pct=position_pct,
        )

    if position_pct >= OVERPRICED_THRESHOLD:
        pct = position_pct * 100.0
        return DealQualityResult(
            verdict=DealQualityVerdict.OVERPRICED,
            reason=f"{pct:.1f}% above market median",
            market_mid=mid,
            ask_usd=float(ask_usd),
            position_pct=position_pct,
        )

    pct = position_pct * 100.0
    direction = "above" if pct >= 0 else "below"
    return DealQualityResult(
        verdict=DealQualityVerdict.FAIR_DEAL,
        reason=f"{abs(pct):.1f}% {direction} market median",
        market_mid=mid,
        ask_usd=float(ask_usd),
        position_pct=position_pct,
    )

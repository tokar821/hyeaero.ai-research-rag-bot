"""Forward pricing band derived from current band + deterministic drift."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from services.market_intelligence.market_band_builder import MarketBand
from services.temporal_market.price_drift_analyzer import PriceDriftReport, TrendDirection

MIN_HISTORY_POINTS = 5
MAX_SHIFT_PCT = 0.12


class ForwardBandConfidence(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


@dataclass(frozen=True)
class ForwardMarketBand:
    forward_low: Optional[float]
    forward_mid: Optional[float]
    forward_high: Optional[float]
    confidence: ForwardBandConfidence
    shift_pct_applied: float = 0.0
    mirrors_current: bool = False


def _clamp_shift(raw_shift: float, volatility_index: int) -> float:
    vol_cap = MAX_SHIFT_PCT * (1.0 - min(volatility_index, 80) / 100.0)
    cap = max(0.02, vol_cap)
    return max(-cap, min(cap, raw_shift))


def build_forward_band(
    current_band: MarketBand,
    drift: PriceDriftReport,
    *,
    history_points: int,
) -> ForwardMarketBand:
    """
    Project band forward using 90d drift, clamped by volatility.

    Requires ≥5 history points; otherwise mirrors current band at LOW confidence.
    """
    if (
        history_points < MIN_HISTORY_POINTS
        or drift.insufficient_history
        or current_band.mid is None
        or current_band.low is None
        or current_band.high is None
    ):
        return ForwardMarketBand(
            forward_low=current_band.low,
            forward_mid=current_band.mid,
            forward_high=current_band.high,
            confidence=ForwardBandConfidence.LOW,
            shift_pct_applied=0.0,
            mirrors_current=True,
        )

    ref_drift = drift.drift_90d_pct
    if ref_drift is None:
        ref_drift = drift.slope_pct_per_90d
    if ref_drift is None:
        ref_drift = 0.0

    shift = _clamp_shift(ref_drift / 100.0, drift.volatility_index)
    if drift.trend_direction == TrendDirection.DOWN:
        shift = -abs(shift) if shift != 0 else -min(MAX_SHIFT_PCT, abs(ref_drift) / 100.0)
    elif drift.trend_direction == TrendDirection.UP:
        shift = abs(shift) if shift != 0 else min(MAX_SHIFT_PCT, abs(ref_drift) / 100.0)
    else:
        shift = 0.0

    mid = float(current_band.mid)
    low = float(current_band.low)
    high = float(current_band.high)
    span = high - low

    f_mid = mid * (1.0 + shift)
    if drift.trend_direction == TrendDirection.DOWN:
        f_low = low * (1.0 + shift * 0.8)
        f_high = high * (1.0 + shift * 1.1)
    elif drift.trend_direction == TrendDirection.UP:
        f_low = low * (1.0 + shift * 0.9)
        f_high = high * (1.0 + shift * 1.15)
    else:
        f_low = low
        f_high = high
        f_mid = mid

    if f_low > f_mid:
        f_low = f_mid - span * 0.45
    if f_high < f_mid:
        f_high = f_mid + span * 0.45

    conf = ForwardBandConfidence.MODERATE
    if history_points >= 12 and drift.volatility_index < 40:
        conf = ForwardBandConfidence.HIGH
    elif drift.volatility_index > 65:
        conf = ForwardBandConfidence.LOW

    return ForwardMarketBand(
        forward_low=max(0.0, f_low),
        forward_mid=max(0.0, f_mid),
        forward_high=max(0.0, f_high),
        confidence=conf,
        shift_pct_applied=shift * 100.0,
        mirrors_current=abs(shift) < 0.001,
    )

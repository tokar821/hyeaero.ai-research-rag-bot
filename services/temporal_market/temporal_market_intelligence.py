"""Temporal market orchestrator — extends market intelligence without altering core math."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from services.market_intelligence.liquidity_scoring import LiquidityBand, LiquidityScore
from services.market_intelligence.market_band_builder import BandConfidence, MarketBand
from services.market_intelligence.market_intelligence_engine import MarketIntelligenceBundle, fmt_musd
from services.temporal_market.forward_pricing_band import ForwardMarketBand, build_forward_band
from services.temporal_market.price_drift_analyzer import PriceDriftReport, TrendDirection, analyze_price_drift
from services.temporal_market.price_history import PriceHistorySeries, collect_price_history

if TYPE_CHECKING:
    from database.postgres_client import PostgresClient


class DealTimingSignal(str, Enum):
    EARLY_CYCLE = "EARLY_CYCLE"
    MID_CYCLE = "MID_CYCLE"
    LATE_CYCLE = "LATE_CYCLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class TemporalMarketExtension:
    price_history: PriceHistorySeries
    price_drift_report: PriceDriftReport
    forward_market_band: ForwardMarketBand
    time_weighted_liquidity_adjustment: int
    trend_adjusted_deal_signal: str
    deal_timing_signal: DealTimingSignal
    temporal_confidence_low: bool = False


def _liquidity_is_high(liq: Optional[LiquidityScore]) -> bool:
    if liq is None:
        return False
    return liq.band in (LiquidityBand.HIGH, LiquidityBand.GOOD) or liq.score >= 60


def _liquidity_is_thin(liq: Optional[LiquidityScore]) -> bool:
    if liq is None:
        return True
    return liq.band in (LiquidityBand.THIN, LiquidityBand.MODERATE) and liq.score < 60


def classify_deal_timing(
    drift: PriceDriftReport,
    liquidity: Optional[LiquidityScore],
) -> DealTimingSignal:
    """
    Deterministic cycle classification (augmentation only).

    UP + thin liquidity → EARLY_CYCLE
    DOWN + high liquidity → LATE_CYCLE
    else → MID_CYCLE
    """
    if drift.insufficient_history:
        return DealTimingSignal.UNKNOWN
    if drift.trend_direction == TrendDirection.UP and _liquidity_is_thin(liquidity):
        return DealTimingSignal.EARLY_CYCLE
    if drift.trend_direction == TrendDirection.DOWN and _liquidity_is_high(liquidity):
        return DealTimingSignal.LATE_CYCLE
    if drift.trend_direction == TrendDirection.FLAT:
        return DealTimingSignal.MID_CYCLE
    return DealTimingSignal.MID_CYCLE


def _trend_deal_signal(drift: PriceDriftReport, timing: DealTimingSignal) -> str:
    if drift.insufficient_history:
        return "insufficient_history_for_trend_signal"
    d90 = drift.drift_90d_pct
    pct = f"{d90:+.1f}%" if d90 is not None else "n/a"
    return (
        f"90d drift {pct}; trend {drift.trend_direction.value}; "
        f"timing {timing.value}; volatility {drift.volatility_index}/100"
    )


def _time_weighted_liquidity_adjustment(
    base_score: int,
    drift: PriceDriftReport,
) -> int:
    """Deterministic liquidity score nudge from trend (does not replace MI liquidity)."""
    adj = base_score
    if drift.trend_direction == TrendDirection.UP:
        adj -= min(8, drift.volatility_index // 15)
    elif drift.trend_direction == TrendDirection.DOWN:
        adj += min(6, drift.volatility_index // 20)
    return max(0, min(100, adj))


def build_temporal_extension(
    *,
    model: str,
    db: Any = None,
    market_bundle: Optional[MarketIntelligenceBundle] = None,
    listing_rows: Optional[tuple] = None,
) -> TemporalMarketExtension:
    """Build temporal overlay from history + existing market bundle."""
    rows = list(listing_rows or ())
    if not rows and market_bundle is not None:
        rows = list(market_bundle.listing_rows or ())

    history = collect_price_history(db, model, listing_rows=rows if rows else None)
    drift = analyze_price_drift(history)

    band = (
        market_bundle.band
        if market_bundle is not None
        else MarketBand(
            low=None,
            mid=None,
            high=None,
            confidence=BandConfidence.INSUFFICIENT,
            listing_count=0,
        )
    )
    forward = build_forward_band(band, drift, history_points=history.point_count)

    liq = market_bundle.liquidity if market_bundle else None
    base_score = liq.score if liq else 0
    tw_adj = _time_weighted_liquidity_adjustment(base_score, drift)
    timing = classify_deal_timing(drift, liq)
    signal = _trend_deal_signal(drift, timing)

    return TemporalMarketExtension(
        price_history=history,
        price_drift_report=drift,
        forward_market_band=forward,
        time_weighted_liquidity_adjustment=tw_adj,
        trend_adjusted_deal_signal=signal,
        deal_timing_signal=timing,
        temporal_confidence_low=history.insufficient_history or forward.mirrors_current,
    )


def format_temporal_buy_sections(ext: TemporalMarketExtension) -> List[str]:
    lines: List[str] = []
    drift = ext.price_drift_report
    fwd = ext.forward_market_band

    lines.append("")
    lines.append("Market Trend (temporal overlay):")
    if drift.insufficient_history:
        lines.append("- Historical depth insufficient for drift (TEMPORAL_CONFIDENCE_LOW).")
        lines.append("- Forward band mirrors current band.")
    else:
        if drift.drift_30d_pct is not None:
            lines.append(f"- 30d drift: {drift.drift_30d_pct:+.1f}%")
        if drift.drift_90d_pct is not None:
            lines.append(f"- 90d drift: {drift.drift_90d_pct:+.1f}%")
        if drift.drift_1y_pct is not None:
            lines.append(f"- 1y drift: {drift.drift_1y_pct:+.1f}%")
        lines.append(f"- Trend: {drift.trend_direction.value}")
        lines.append(f"- Volatility index: {drift.volatility_index}/100")

    if fwd.forward_low is not None and fwd.forward_high is not None:
        lines.append(
            f"- Forward Band: {fmt_musd(fwd.forward_low)}–{fmt_musd(fwd.forward_high)} "
            f"(confidence: {fwd.confidence.value})"
        )
        if fwd.forward_mid is not None:
            lines.append(f"- Forward Median: {fmt_musd(fwd.forward_mid)}")
        if not fwd.mirrors_current:
            lines.append(f"- Projected shift: {fwd.shift_pct_applied:+.1f}%")

    lines.append("")
    lines.append("Deal Timing Signal:")
    lines.append(f"- {ext.deal_timing_signal.value}")
    lines.append(f"- {ext.trend_adjusted_deal_signal}")
    return lines


def format_temporal_valuation_sections(ext: TemporalMarketExtension) -> List[str]:
    lines: List[str] = ["", "Temporal Context:"]
    drift = ext.price_drift_report
    if ext.temporal_confidence_low:
        lines.append("- TEMPORAL_CONFIDENCE_LOW (fewer than 5 historical price points).")
    if not drift.insufficient_history:
        if drift.drift_90d_pct is not None:
            lines.append(f"- 90d price movement: {drift.drift_90d_pct:+.1f}%")
        lines.append(f"- Trend direction: {drift.trend_direction.value}")
        lines.append(f"- Volatility index: {drift.volatility_index}/100")
        fwd = ext.forward_market_band
        if fwd.forward_low and fwd.forward_high:
            lines.append(
                f"- Forward band projection: {fmt_musd(fwd.forward_low)}–{fmt_musd(fwd.forward_high)} "
                f"({fwd.confidence.value})"
            )
        conf_note = "LOW" if ext.temporal_confidence_low else "MODERATE"
        lines.append(f"- Volatility-adjusted temporal confidence: {conf_note}")
    else:
        lines.append("- No sufficient listing price history for movement analysis.")
    return lines


def format_comparison_temporal_overlay(
    model_a: str,
    model_b: str,
    *,
    db: Any = None,
) -> List[str]:
    """Optional comparison overlay — does not alter verdict."""
    ext_a = build_temporal_extension(model=model_a, db=db)
    ext_b = build_temporal_extension(model=model_b, db=db)
    lines = ["", "Temporal Overlay (informational):"]
    for label, ext in ((model_a, ext_a), (model_b, ext_b)):
        d = ext.price_drift_report
        if d.insufficient_history:
            lines.append(f"- {label}: drift unavailable (thin history)")
        else:
            d90 = f"{d.drift_90d_pct:+.1f}%" if d.drift_90d_pct is not None else "n/a"
            lines.append(
                f"- {label}: trend {d.trend_direction.value}, 90d {d90}, "
                f"volatility {d.volatility_index}/100"
            )
    va = ext_a.price_drift_report.volatility_index
    vb = ext_b.price_drift_report.volatility_index
    if not ext_a.price_drift_report.insufficient_history and not ext_b.price_drift_report.insufficient_history:
        if va > vb + 15:
            lines.append(f"- Relative volatility: {model_a} higher than {model_b}")
        elif vb > va + 15:
            lines.append(f"- Relative volatility: {model_b} higher than {model_a}")
        else:
            lines.append("- Relative volatility: comparable between both models")
        fa = ext_a.forward_market_band
        fb = ext_b.forward_market_band
        if fa.forward_mid and fb.forward_mid and fa.forward_mid != fb.forward_mid:
            lines.append(
                f"- Forward band divergence: {model_a} mid {fmt_musd(fa.forward_mid)} vs "
                f"{model_b} mid {fmt_musd(fb.forward_mid)}"
            )
    return lines


def temporal_to_data_used_dict(ext: TemporalMarketExtension) -> Dict[str, Any]:
    d = ext.price_drift_report
    f = ext.forward_market_band
    return {
        "deal_timing_signal": ext.deal_timing_signal.value,
        "trend_direction": d.trend_direction.value,
        "drift_90d_pct": d.drift_90d_pct,
        "volatility_index": d.volatility_index,
        "temporal_confidence_low": ext.temporal_confidence_low,
        "forward_band": {
            "low": f.forward_low,
            "mid": f.forward_mid,
            "high": f.forward_high,
            "confidence": f.confidence.value,
        },
    }

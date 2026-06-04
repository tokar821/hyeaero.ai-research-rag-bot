"""Phase 37 — temporal market drift and forward pricing tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from services.market_intelligence.market_band_builder import BandConfidence, MarketBand
from services.temporal_market.forward_pricing_band import ForwardBandConfidence, build_forward_band
from services.temporal_market.price_drift_analyzer import TrendDirection, analyze_price_drift
from services.temporal_market.price_history import PriceHistorySeries
from services.temporal_market.temporal_market_intelligence import (
    DealTimingSignal,
    build_temporal_extension,
    classify_deal_timing,
)
from services.consistency.consistency_injection_layer import prepare_buy_decision_state, render_buy_decision_answer
from services.market_intelligence.liquidity_scoring import LiquidityBand, LiquidityScore, compute_liquidity_score
from services.market_intelligence.listing_analytics import MarketSnapshot

pytestmark = pytest.mark.deterministic


def _synthetic_series(n: int = 10, *, upward: bool = True) -> PriceHistorySeries:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = []
    prices = []
    for i in range(n):
        dt = base + timedelta(days=i * 14)
        timestamps.append(dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        p = 10_000_000.0 + (i * 200_000 if upward else -i * 150_000)
        prices.append(p)
    return PriceHistorySeries(
        model="Citation Latitude",
        timestamps=tuple(timestamps),
        prices=tuple(prices),
        listing_sources=("controller",),
        last_updated=timestamps[-1],
        point_count=n,
        insufficient_history=n < 5,
    )


def test_drift_direction_stability() -> None:
    series = _synthetic_series(12, upward=True)
    r1 = analyze_price_drift(series)
    r2 = analyze_price_drift(series)
    assert r1.trend_direction == r2.trend_direction
    assert r1.trend_direction == TrendDirection.UP
    assert r1.drift_90d_pct is not None


def test_volatility_determinism() -> None:
    series = _synthetic_series(8, upward=False)
    assert analyze_price_drift(series).volatility_index == analyze_price_drift(series).volatility_index


def test_forward_band_consistency() -> None:
    band = MarketBand(
        low=9_800_000.0,
        mid=11_700_000.0,
        high=13_500_000.0,
        confidence=BandConfidence.HIGH,
        listing_count=10,
    )
    drift = analyze_price_drift(_synthetic_series(10, upward=True))
    fwd = build_forward_band(band, drift, history_points=10)
    assert fwd.forward_mid is not None
    assert fwd.forward_mid >= band.mid  # UP trend shifts up
    assert fwd.confidence in (
        ForwardBandConfidence.HIGH,
        ForwardBandConfidence.MODERATE,
        ForwardBandConfidence.LOW,
    )


def test_forward_band_mirrors_when_insufficient_history() -> None:
    band = MarketBand(
        low=10e6,
        mid=11e6,
        high=12e6,
        confidence=BandConfidence.HIGH,
        listing_count=3,
    )
    drift = analyze_price_drift(_synthetic_series(3))
    fwd = build_forward_band(band, drift, history_points=3)
    assert fwd.mirrors_current
    assert fwd.forward_mid == band.mid


def test_deal_timing_signal_correctness() -> None:
    drift_up = analyze_price_drift(_synthetic_series(10, upward=True))
    snap = MarketSnapshot(
        model="Citation Latitude",
        active_listing_count=3,
        median_ask_price=11e6,
        low_ask_price=10e6,
        high_ask_price=12e6,
        median_year=2018,
        average_days_on_market=200.0,
        last_refresh="2026-01-01T00:00:00Z",
    )
    thin = compute_liquidity_score(snap)
    assert classify_deal_timing(drift_up, thin) == DealTimingSignal.EARLY_CYCLE

    drift_down = analyze_price_drift(_synthetic_series(10, upward=False))
    deep = MarketSnapshot(
        model="Citation Latitude",
        active_listing_count=25,
        median_ask_price=11e6,
        low_ask_price=9e6,
        high_ask_price=13e6,
        median_year=2018,
        average_days_on_market=90.0,
        last_refresh="2026-01-01T00:00:00Z",
    )
    liq = compute_liquidity_score(deep)
    if liq.band in (LiquidityBand.HIGH, LiquidityBand.GOOD):
        assert classify_deal_timing(drift_down, liq) == DealTimingSignal.LATE_CYCLE


def test_temporal_extension_present_in_buy_state() -> None:
    parsed = {"model": "Citation Latitude", "year": 2015, "ask_usd": 5_000_000.0}
    du: dict = {}
    state = prepare_buy_decision_state(
        query="Is a 2015 Citation Latitude for $5M a good deal?",
        parsed=parsed,
        db=None,
        data_used=du,
    )
    assert state.temporal is not None
    assert du.get("unified_broker_state", {}).get("temporal") is not None
    body = render_buy_decision_answer(state)
    assert "Market Trend" in body
    assert "Deal Timing Signal" in body
    assert "GOOD DEAL" in body.upper() or "FAIR DEAL" in body.upper()

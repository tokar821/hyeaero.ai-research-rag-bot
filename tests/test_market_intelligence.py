"""Phase 35 — unit tests for deterministic market intelligence."""

from __future__ import annotations

import pytest

from services.market_intelligence.deal_quality_engine import (
    DealQualityVerdict,
    evaluate_deal_quality,
)
from services.market_intelligence.liquidity_scoring import LiquidityBand, compute_liquidity_score
from services.market_intelligence.listing_analytics import MarketSnapshot, build_market_snapshot
from services.market_intelligence.market_band_builder import (
    BandConfidence,
    MarketBand,
    build_market_band,
)
from services.market_intelligence.market_intelligence_engine import analyze_market

pytestmark = pytest.mark.deterministic


def _sample_rows(model: str, n: int = 8, base_ask: float = 11_700_000.0) -> list:
    rows = []
    for i in range(n):
        rows.append(
            {
                "source_platform": "controller" if i % 2 == 0 else "aircraftexchange",
                "listing_status": "for_sale",
                "ask_price": base_ask + (i - n // 2) * 250_000,
                "days_on_market": 120 + i * 5,
                "manufacturer": "Cessna",
                "model": model,
                "manufacturer_year": 2016 + (i % 4),
                "updated_at": "2026-05-01T12:00:00Z",
            }
        )
    return rows


def test_liquidity_high_for_deep_market() -> None:
    snap = build_market_snapshot(
        "Citation Latitude",
        _sample_rows("Citation Latitude", n=25),
        aircraftpost_for_sale=25,
    )
    liq = compute_liquidity_score(snap)
    assert liq.score >= 60
    assert liq.band in (LiquidityBand.HIGH, LiquidityBand.GOOD)


def test_market_band_requires_minimum_listings() -> None:
    snap = build_market_snapshot("Citation Latitude", _sample_rows("Citation Latitude", n=3))
    band = build_market_band(snap, ask_prices=[9.8e6, 11.7e6, 13.5e6])
    assert band.confidence == BandConfidence.INSUFFICIENT


def test_market_band_from_sufficient_asks() -> None:
    asks = [9.8e6, 10.5e6, 11.0e6, 11.7e6, 12.4e6, 13.1e6, 13.5e6]
    snap = MarketSnapshot(
        model="Citation Latitude",
        active_listing_count=len(asks),
        median_ask_price=11.7e6,
        low_ask_price=min(asks),
        high_ask_price=max(asks),
        median_year=2018,
        average_days_on_market=164.0,
        last_refresh="2026-05-01T12:00:00Z",
        listing_sources=("controller",),
        stale=False,
    )
    band = build_market_band(snap, ask_prices=asks)
    assert band.confidence in (BandConfidence.HIGH, BandConfidence.MODERATE)
    assert band.low is not None and band.high is not None and band.mid is not None
    assert band.low <= band.mid <= band.high


def test_deal_quality_good_below_median() -> None:
    band = MarketBand(
        low=10.2e6,
        mid=11.8e6,
        high=13.1e6,
        confidence=BandConfidence.HIGH,
        listing_count=10,
    )
    result = evaluate_deal_quality(
        model="Citation Latitude",
        year=2018,
        ask_usd=9.5e6,
        band=band,
    )
    assert result.verdict == DealQualityVerdict.GOOD_DEAL
    assert "below" in result.reason.lower()


def test_deal_quality_overpriced() -> None:
    band = MarketBand(
        low=10.2e6,
        mid=11.8e6,
        high=13.1e6,
        confidence=BandConfidence.HIGH,
        listing_count=10,
    )
    result = evaluate_deal_quality(
        model="Citation Latitude",
        year=2018,
        ask_usd=14.0e6,
        band=band,
    )
    assert result.verdict == DealQualityVerdict.OVERPRICED


def test_analyze_market_authority_fallback_no_db() -> None:
    auth = {
        "status": "OK",
        "expected_market_band_usd": {"low": 10_500_000, "mid": 11_800_000, "high": 13_000_000},
    }
    bundle = analyze_market(None, "Citation Latitude", ask_usd=9.7e6, auth_market=auth)
    assert bundle.band.mid == 11_800_000
    assert bundle.deal_quality is not None
    assert bundle.deal_quality.verdict == DealQualityVerdict.GOOD_DEAL

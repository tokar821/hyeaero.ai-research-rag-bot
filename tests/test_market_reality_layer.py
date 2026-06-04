"""Phase 43 — market reality and listing intelligence tests."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from services.market_reality.buyer_leverage_analyzer import BuyerLeverage
from services.market_reality.listing_confidence_analyzer import ListingPriceConfidence
from services.market_reality.listing_detector import ListingMode, detect_listing_signal
from services.market_reality.market_reality_layer import apply_market_reality_layer, build_market_reality_brief
from services.market_intelligence.market_band_builder import BandConfidence, MarketBand


def _mock_band(mid: float = 25_000_000.0, low: float = 20_000_000.0, high: float = 32_000_000.0) -> MarketBand:
    return MarketBand(
        low=low,
        mid=mid,
        high=high,
        confidence=BandConfidence.HIGH,
        listing_count=12,
        reason="",
    )


def _fake_bundle(snapshot, band, liquidity):
    return SimpleNamespace(
        snapshot=snapshot,
        band=band,
        liquidity=liquidity,
        deal_quality=None,
        listing_rows=(),
    )


def test_detect_listing_discussion():
    sig = detect_listing_signal("I found a G650 for 18M")
    assert sig.mode == ListingMode.LISTING_DISCUSSION
    assert sig.ask_musd == pytest.approx(18.0)
    assert "G650" in (sig.model or "")


def test_detect_tail_investigation():
    sig = detect_listing_signal("is N719GF worth looking at")
    assert sig.mode == ListingMode.TAIL_INVESTIGATION
    assert "N719GF" in sig.registrations


def test_detect_buyer_seller_market():
    sig = detect_listing_signal("buyers market or sellers market for Longitude")
    assert sig.mode == ListingMode.BUYER_SELLER_MARKET


def test_listing_confidence_unusually_cheap(monkeypatch):
    from services.market_reality import listing_confidence_analyzer as lca

    band = _mock_band(mid=25_000_000.0)
    result = lca.analyze_listing_confidence(
        model="Gulfstream G650",
        ask_usd=18_000_000.0,
        band=band,
    )
    assert result["confidence"] == ListingPriceConfidence.UNUSUALLY_CHEAP.value


def test_market_reality_brief_g650_18m(monkeypatch):
    from services.market_intelligence import market_intelligence_engine as mie
    from services.market_intelligence.listing_analytics import MarketSnapshot
    from services.market_intelligence.liquidity_scoring import LiquidityBand, LiquidityScore

    snapshot = MarketSnapshot(
        model="Gulfstream G650",
        active_listing_count=14,
        median_ask_price=25_000_000.0,
        low_ask_price=20_000_000.0,
        high_ask_price=32_000_000.0,
        median_year=2016,
        average_days_on_market=120.0,
        last_refresh=None,
        insufficient_reason="",
    )
    band = _mock_band()
    liq = LiquidityScore(score=65, band=LiquidityBand.GOOD, listing_points=30, dom_points=20, dispersion_points=15)
    bundle = _fake_bundle(snapshot, band, liq)
    monkeypatch.setattr(mie, "analyze_market", lambda *a, **k: bundle)

    brief = build_market_reality_brief("I found a G650 for 18M", data_used={})
    assert brief
    assert "18" in brief or "below" in brief.lower()
    assert re.search(r"(?i)year|engine|damage|listing", brief)


def test_apply_layer_leads_with_deal_read(monkeypatch):
    from services.market_intelligence import market_intelligence_engine as mie
    from services.market_intelligence.listing_analytics import MarketSnapshot
    from services.market_intelligence.liquidity_scoring import LiquidityBand, LiquidityScore

    snapshot = MarketSnapshot(
        model="Gulfstream G700",
        active_listing_count=8,
        median_ask_price=40_000_000.0,
        low_ask_price=35_000_000.0,
        high_ask_price=50_000_000.0,
        median_year=2020,
        average_days_on_market=150.0,
        last_refresh=None,
        insufficient_reason="",
    )

    liq = LiquidityScore(score=50, band=LiquidityBand.MODERATE, listing_points=20, dom_points=15, dispersion_points=15)
    band = _mock_band(mid=40_000_000.0, low=35_000_000.0, high=50_000_000.0)
    monkeypatch.setattr(mie, "analyze_market", lambda *a, **k: _fake_bundle(snapshot, band, liq))

    raw = "Citation Latitude is a super-midsize aircraft."
    out = apply_market_reality_layer(raw, query="I saw a G700 for 12M", data_used={})
    assert "G700" in out
    assert "12" in out
    first = out.split("\n\n")[0].lower()
    assert any(
        kw in first
        for kw in ("below", "materially", "unusual", "does not line up", "high vs")
    )


def test_tail_brief_no_speculation():
    brief = build_market_reality_brief("is N719GF worth looking at", data_used={})
    assert brief
    assert "N719GF" in brief
    assert "verify" in brief.lower()
    assert "speculate" not in brief.lower() or "not speculate" in brief.lower()


_ACCEPTANCE = [
    "I found a G650 for 18M",
    "I saw a G700 for 12M",
    "is N719GF worth looking at",
    "good time to buy a Longitude",
    "buyers market or sellers market",
    "why is this aircraft so cheap",
    "is this listing realistic",
]


@pytest.mark.parametrize("query", _ACCEPTANCE)
def test_acceptance_produces_deal_focused_prose(query, monkeypatch):
    from services.market_intelligence import market_intelligence_engine as mie
    from services.market_intelligence.listing_analytics import MarketSnapshot
    from services.market_intelligence.liquidity_scoring import LiquidityBand, LiquidityScore

    if "N719GF" in query or "N123AB" in query:
        brief = build_market_reality_brief(query, data_used={})
        assert brief
        return

    snapshot = MarketSnapshot(
        model="Citation Longitude",
        active_listing_count=10,
        median_ask_price=22_000_000.0,
        low_ask_price=18_000_000.0,
        high_ask_price=28_000_000.0,
        median_year=2019,
        average_days_on_market=130.0,
        last_refresh=None,
        insufficient_reason="",
    )

    liq = LiquidityScore(score=55, band=LiquidityBand.MODERATE, listing_points=25, dom_points=15, dispersion_points=15)
    band = _mock_band(mid=22_000_000.0)
    monkeypatch.setattr(
        mie,
        "analyze_market",
        lambda *a, **k: _fake_bundle(snapshot, band, liq),
    )

    out = apply_market_reality_layer("generic model text", query=query, data_used={})
    assert out.strip()
    assert not out.strip().startswith("Citation Latitude is a")

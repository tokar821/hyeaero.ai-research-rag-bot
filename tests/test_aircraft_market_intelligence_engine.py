"""Phase 24 — Aircraft Market Intelligence Engine tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.deterministic

from services.market.aircraft_market_intelligence_engine import (
    analyze_inventory_trend,
    analyze_market_liquidity,
    analyze_price_trend,
    attach_market_intelligence_if_enabled,
    build_market_intelligence,
    build_market_intelligence_report,
    evaluate_market_intelligence_hooks,
    evaluate_purchase_timing,
    evaluate_replacement_risk,
    evaluate_sale_timing,
    market_intelligence_enabled,
)


def _buyer_market_snapshot():
    return {
        "controller": {"current": 18, "prior": 12},
        "aircraft_exchange": {"current": 10, "prior": 8},
        "phly": {"current": 4, "prior": 3},
        "historical": [
            {"price_usd": 12_000_000},
            {"price_usd": 11_500_000},
            {"price_usd": 11_000_000},
            {"price_usd": 10_500_000},
        ],
        "listing_velocity": 6.0,
    }


def _seller_market_snapshot():
    return {
        "controller": {"current": 6, "prior": 12},
        "aircraft_exchange": {"current": 4, "prior": 8},
        "phly": {"current": 2, "prior": 4},
        "historical": [
            {"price_usd": 10_000_000},
            {"price_usd": 10_400_000},
            {"price_usd": 10_900_000},
            {"price_usd": 11_200_000},
        ],
        "listing_velocity": 3.0,
    }


def _stable_market_snapshot():
    return {
        "controller": {"current": 10, "prior": 10},
        "aircraft_exchange": {"current": 6, "prior": 6},
        "phly": {"current": 2, "prior": 2},
        "historical": [
            {"price_usd": 11_000_000},
            {"price_usd": 11_100_000},
            {"price_usd": 10_950_000},
            {"price_usd": 11_050_000},
        ],
        "listing_velocity": 4.0,
    }


def test_buyer_market():
    snap = _buyer_market_snapshot()
    report = build_market_intelligence_report(
        "Citation Latitude",
        controller_listings=snap["controller"],
        aircraft_exchange_listings=snap["aircraft_exchange"],
        phly_listings=snap["phly"],
        historical_listing_data=snap["historical"],
        listing_velocity=snap["listing_velocity"],
    )
    assert report.market_state == "BUYER_MARKET"
    assert report.inventory_trend == "rising"
    assert report.price_trend == "depreciating"
    assert report.evidence


def test_seller_market():
    snap = _seller_market_snapshot()
    report = build_market_intelligence_report(
        "Gulfstream G650",
        controller_listings=snap["controller"],
        aircraft_exchange_listings=snap["aircraft_exchange"],
        phly_listings=snap["phly"],
        historical_listing_data=snap["historical"],
        listing_velocity=snap["listing_velocity"],
    )
    assert report.market_state == "SELLER_MARKET"
    assert report.inventory_trend == "declining"
    assert report.price_trend == "appreciating"


def test_stable_market():
    snap = _stable_market_snapshot()
    report = build_market_intelligence_report(
        "Challenger 3500",
        controller_listings=snap["controller"],
        aircraft_exchange_listings=snap["aircraft_exchange"],
        phly_listings=snap["phly"],
        historical_listing_data=snap["historical"],
        listing_velocity=snap["listing_velocity"],
    )
    assert report.inventory_trend == "stable"
    assert report.price_trend == "stable"
    assert report.market_state == "BALANCED_MARKET"


def test_rising_inventory():
    trend = analyze_inventory_trend(
        controller_listings={"current": 20, "prior": 10},
        aircraft_exchange_listings={"current": 8, "prior": 6},
        phly_listings={"current": 4, "prior": 3},
    )
    assert trend == "rising"


def test_falling_inventory():
    trend = analyze_inventory_trend(
        controller_listings={"current": 5, "prior": 12},
        aircraft_exchange_listings={"current": 3, "prior": 8},
        phly_listings={"current": 1, "prior": 3},
    )
    assert trend == "declining"


def test_appreciating_model():
    trend = analyze_price_trend(
        [
            {"price_usd": 10_000_000},
            {"price_usd": 10_200_000},
            {"price_usd": 10_800_000},
            {"price_usd": 11_500_000},
        ]
    )
    assert trend == "appreciating"


def test_depreciating_model():
    trend = analyze_price_trend(
        [
            {"price_usd": 12_000_000},
            {"price_usd": 11_800_000},
            {"price_usd": 11_200_000},
            {"price_usd": 10_500_000},
        ]
    )
    assert trend == "depreciating"


def test_replacement_risk():
    cj_risk = evaluate_replacement_risk("Citation CJ3+")
    g650_risk = evaluate_replacement_risk("Gulfstream G650")
    assert cj_risk in ("MODERATE", "HIGH")
    assert g650_risk in ("LOW", "MODERATE")


def test_buy_timing():
    timing, evidence = evaluate_purchase_timing(
        market_state="BUYER_MARKET",
        inventory_trend="rising",
        price_trend="depreciating",
        liquidity_score=65.0,
    )
    assert timing == "favorable"
    assert evidence

    unfav, _ = evaluate_purchase_timing(
        market_state="SELLER_MARKET",
        inventory_trend="declining",
        price_trend="appreciating",
        liquidity_score=35.0,
    )
    assert unfav == "unfavorable"


def test_sell_timing():
    timing, evidence = evaluate_sale_timing(
        market_state="SELLER_MARKET",
        inventory_trend="declining",
        price_trend="appreciating",
        liquidity_score=40.0,
    )
    assert timing == "favorable"
    assert evidence

    unfav, _ = evaluate_sale_timing(
        market_state="BUYER_MARKET",
        inventory_trend="rising",
        price_trend="depreciating",
        liquidity_score=70.0,
    )
    assert unfav == "unfavorable"


def test_liquidity_score_range():
    score = analyze_market_liquidity(listing_velocity=8.0, inventory_levels=12)
    assert 0 <= score <= 100


def test_report_reproducibility():
    snap = _stable_market_snapshot()
    a = build_market_intelligence_report(
        "Citation Latitude",
        controller_listings=snap["controller"],
        aircraft_exchange_listings=snap["aircraft_exchange"],
        phly_listings=snap["phly"],
        historical_listing_data=snap["historical"],
        listing_velocity=snap["listing_velocity"],
    )
    b = build_market_intelligence_report(
        "Citation Latitude",
        controller_listings=snap["controller"],
        aircraft_exchange_listings=snap["aircraft_exchange"],
        phly_listings=snap["phly"],
        historical_listing_data=snap["historical"],
        listing_velocity=snap["listing_velocity"],
    )
    assert a.report_id == b.report_id
    assert a.to_dict() == b.to_dict()


def test_evaluator_market_hooks():
    bad = {
        "answer": "This is a buyer market with appreciating prices.",
        "data_used": {
            "market_intelligence": {
                "market_state": "BUYER_MARKET",
                "inventory_trend": "declining",
                "price_trend": "appreciating",
                "liquidity_trend": "tightening",
                "confidence": 0.9,
                "evidence": ["test"],
            }
        },
    }
    failures = evaluate_market_intelligence_hooks(bad)
    assert "market_consistency" in failures


def test_build_market_intelligence_insufficient_data():
    out = build_market_intelligence(
        "market for Latitude",
        {"data_used": {"consultant_recommendations": [{"model": "Citation Latitude"}]}},
    )
    assert out["status"] == "INSUFFICIENT_DATA"
    assert out["confidence"] == 0
    assert out["trends"] == {}
    assert out["liquidity_score"] == 0.0
    assert out["market_panel"]["liquidity"] == 0.0


def test_attach_only_when_env_enabled(monkeypatch):
    monkeypatch.delenv("ENABLE_MARKET_INTELLIGENCE", raising=False)
    payload = {"answer": "x", "data_used": {}}
    assert not market_intelligence_enabled()
    out = attach_market_intelligence_if_enabled("test", payload)
    assert "market_intelligence" not in (out.get("data_used") or {})

    monkeypatch.setenv("ENABLE_MARKET_INTELLIGENCE", "1")
    out2 = attach_market_intelligence_if_enabled(
        "market for Latitude",
        {
            "data_used": {
                "consultant_recommendations": [{"model": "Citation Latitude"}],
                "market_listing_snapshot": _stable_market_snapshot(),
            }
        },
    )
    assert "market_intelligence" in (out2.get("data_used") or {})

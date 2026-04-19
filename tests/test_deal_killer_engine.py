"""Unit tests for Deal Killer Engine (verdicts, scores, mission mismatch)."""

from __future__ import annotations

import pytest

from services import deal_killer_engine as dke


def test_overpriced_above_comp_high():
    out = dke.run_deal_killer_engine(
        aircraft={
            "model": "Challenger 350",
            "year": 2015,
            "ask_price": 12_000_000,
            "total_time": 3200,
            "programs": ["MSP Gold"],
            "maintenance_tracking": "CAMP",
        },
        market_data={
            "avg_price": 8_500_000,
            "price_range_low": 7_000_000,
            "price_range_high": 9_000_000,
            "liquidity": "moderate",
            "demand_level": "balanced",
            "comp_row_count": 8,
        },
        buyer_context={"mission_profile": {}},
        peer_airframe_hours=[2800, 3000, 3100, 2900, 3050],
    )
    assert out["verdict"] == dke.VERDICT_OVERPRICED
    assert out["scores"]["price_score"] < 0.4


def test_do_not_buy_mission_mismatch():
    out = dke.run_deal_killer_engine(
        aircraft={"model": "Citation CJ4", "year": 2018, "ask_price": 6_000_000, "total_time": 1800},
        market_data={
            "avg_price": 6_200_000,
            "price_range_low": 5_500_000,
            "price_range_high": 6_800_000,
            "liquidity": "moderate",
            "demand_level": "balanced",
            "comp_row_count": 10,
        },
        buyer_context={
            "mission_profile": {"longest_leg_nm": 2800.0},
            "longest_leg_nm": 2800.0,
        },
        peer_airframe_hours=[1500, 1600, 1700],
    )
    assert out["verdict"] == dke.VERDICT_DO_NOT_BUY
    assert out["scores"]["mission_fit_score"] < 0.6


def test_good_deal_strong_programs_in_band():
    out = dke.run_deal_killer_engine(
        aircraft={
            "model": "Challenger 350",
            "year": 2016,
            "ask_price": 7_500_000,
            "total_time": 3000,
            "engines": "Honeywell HTF7350",
            "programs": ["MSP", "JSSI"],
            "maintenance_tracking": "CESCOM",
        },
        market_data={
            "avg_price": 8_200_000,
            "price_range_low": 7_000_000,
            "price_range_high": 9_500_000,
            "liquidity": "strong",
            "demand_level": "firm",
            "comp_row_count": 14,
        },
        buyer_context={"mission_profile": {"longest_leg_nm": 1800.0}},
        peer_airframe_hours=[2800, 3000, 3200, 2900, 3100, 3050, 2950, 3150, 3000, 3020],
    )
    assert out["verdict"] == dke.VERDICT_GOOD_DEAL
    assert out["scores"]["mission_fit_score"] >= 0.75


def test_suspiciously_low_price_flag():
    out = dke.run_deal_killer_engine(
        aircraft={"model": "Falcon 2000", "year": 2005, "ask_price": 2_000_000, "total_time": 4500},
        market_data={
            "avg_price": 6_000_000,
            "price_range_low": 5_000_000,
            "price_range_high": 7_000_000,
            "liquidity": "thin",
            "demand_level": "buyer's market",
            "comp_row_count": 3,
        },
        buyer_context={"mission_profile": {}},
        peer_airframe_hours=None,
    )
    assert any("below" in f.lower() or "20%" in f for f in out["red_flags"])


def test_consultant_context_returns_none_without_signals():
    assert (
        dke.run_deal_killer_from_consultant_context(
            phly_rows=None,
            primary_listing=None,
            query="hello",
            buyer_psychology=None,
            db=None,
        )
        is None
    )


def test_deal_killer_toggle_off(monkeypatch):
    monkeypatch.setenv("CONSULTANT_DEAL_KILLER", "0")
    assert dke.consultant_deal_killer_enabled() is False

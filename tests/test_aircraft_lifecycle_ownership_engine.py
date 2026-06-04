"""Phase 25 — Aircraft Lifecycle Ownership Intelligence Engine tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.deterministic

from services.ownership.aircraft_lifecycle_ownership_engine import (
    attach_ownership_intelligence_if_enabled,
    build_ownership_intelligence,
    build_ownership_report,
    compare_ownership_profiles,
    estimate_depreciation_curve,
    estimate_future_resale_value,
    evaluate_ownership_intelligence_hooks,
    evaluate_ownership_risk,
    ownership_intelligence_enabled,
)


def test_5_year_ownership():
    report = build_ownership_report("Citation Latitude")
    assert report.total_cost_5_year > 0
    assert report.projected_resale_5_year > 0
    assert report.projected_resale_5_year < report.acquisition_price
    assert report.annual_operating_cost == report.annual_fixed_cost + report.annual_variable_cost


def test_10_year_ownership():
    report = build_ownership_report("Gulfstream G650")
    assert report.total_cost_10_year > report.total_cost_5_year
    assert report.projected_resale_10_year < report.projected_resale_5_year
    assert report.depreciation_amount == report.acquisition_price - report.projected_resale_10_year


def test_depreciation_projection():
    curve = estimate_depreciation_curve(
        aircraft_age_years=8,
        production_status=False,
        liquidity_score=40.0,
        market_intelligence={"price_trend": "depreciating", "replacement_risk": "HIGH"},
        category="large-cabin",
    )
    assert 0.04 <= curve["annual_rate"] <= 0.12
    assert curve["retention_5_year"] > curve["retention_10_year"]
    assert curve["retention_10_year"] < 1.0


def test_resale_projection():
    curve = estimate_depreciation_curve(
        production_status=True,
        liquidity_score=70.0,
        category="large-cabin",
    )
    resale = estimate_future_resale_value(
        current_market_value=10_000_000,
        depreciation_curve=curve,
        years=5,
    )
    assert 0 < resale < 10_000_000


def test_ownership_comparison():
    result = compare_ownership_profiles(
        ["Gulfstream G650", "Falcon 8X", "Global 7500"],
    )
    assert result["cheapest_to_own"]
    assert result["strongest_resale"]
    assert result["lowest_risk"]
    assert len(result["reports"]) == 3
    costs = {r["aircraft"]: r["total_cost_10_year"] for r in result["reports"]}
    assert result["cheapest_to_own"] in costs


def test_ownership_risk():
    g650_risk = evaluate_ownership_risk("Gulfstream G650", liquidity_score=65.0)
    cj_risk = evaluate_ownership_risk(
        "Citation CJ3+",
        market_intelligence={"replacement_risk": "HIGH"},
        liquidity_score=40.0,
    )
    assert 0 <= g650_risk <= 100
    assert 0 <= cj_risk <= 100
    assert g650_risk >= cj_risk


def test_lifecycle_score():
    comparison = compare_ownership_profiles(["Citation Latitude", "Gulfstream G650"])
    for report in comparison["reports"]:
        assert 0 <= report["lifecycle_score"] <= 100


def test_evaluator_hooks():
    bad = {
        "data_used": {
            "ownership_intelligence": {
                "ownership_reports": [
                    {
                        "acquisition_price": 10_000_000,
                        "projected_resale_5_year": 9_000_000,
                        "projected_resale_10_year": 11_000_000,
                        "depreciation_amount": -1_000_000,
                        "total_cost_5_year": 20_000_000,
                        "total_cost_10_year": 15_000_000,
                        "lifecycle_score": 95,
                        "confidence": 0.85,
                        "annual_fixed_cost": 0,
                        "annual_variable_cost": 0,
                        "annual_operating_cost": 0,
                    }
                ]
            }
        }
    }
    failures = evaluate_ownership_intelligence_hooks(bad)
    assert "depreciation_consistency" in failures
    assert "lifecycle_score_consistency" in failures
    assert "ownership_cost_completeness" in failures


def test_reproducibility():
    a = build_ownership_report("Citation Latitude")
    b = build_ownership_report("Citation Latitude")
    assert a.report_id == b.report_id
    assert a.to_dict() == b.to_dict()


def test_env_gating(monkeypatch):
    monkeypatch.delenv("ENABLE_OWNERSHIP_INTELLIGENCE", raising=False)
    payload = {"answer": "x", "data_used": {}}
    assert not ownership_intelligence_enabled()
    out = attach_ownership_intelligence_if_enabled("test", payload)
    assert "ownership_intelligence" not in (out.get("data_used") or {})

    monkeypatch.setenv("ENABLE_OWNERSHIP_INTELLIGENCE", "1")
    out2 = attach_ownership_intelligence_if_enabled(
        "ownership cost for Latitude",
        {
            "data_used": {
                "consultant_recommendations": [
                    {"model": "Citation Latitude"},
                    {"model": "Gulfstream G650"},
                ]
            }
        },
    )
    bundle = (out2.get("data_used") or {}).get("ownership_intelligence") or {}
    assert bundle
    assert bundle.get("ownership_summary")
    assert bundle.get("ownership_panel")

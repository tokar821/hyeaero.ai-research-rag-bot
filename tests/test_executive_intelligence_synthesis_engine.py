"""Executive Intelligence Synthesis Engine tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.deterministic

from services.synthesis.executive_intelligence_synthesis_engine import (
    attach_executive_synthesis_if_enabled,
    build_executive_synthesis,
    evaluate_executive_synthesis_hooks,
    executive_synthesis_enabled,
)


def _full_data_used() -> dict:
    return {
        "ownership_intelligence": {
            "ownership_reports": [
                {
                    "aircraft": "Citation Latitude",
                    "lifecycle_score": 72.0,
                    "ownership_risk_score": 68.0,
                    "total_cost_10_year": 18_000_000,
                },
                {
                    "aircraft": "Gulfstream G650",
                    "lifecycle_score": 55.0,
                    "ownership_risk_score": 75.0,
                    "total_cost_10_year": 52_000_000,
                },
            ],
        },
        "market_intelligence": {
            "aircraft": "Gulfstream G650",
            "market_strength_score": 62.0,
            "liquidity_score": 58.0,
            "market_state": "BALANCED_MARKET",
            "sell_timing": "neutral",
        },
        "optimization_result": {
            "winner": "Citation Latitude",
            "ranked_candidates": [
                {"aircraft": "Citation Latitude", "total_score": 78.0},
                {"aircraft": "Gulfstream G650", "total_score": 65.0},
            ],
        },
        "fleet_portfolio_strategy": {
            "current_aircraft": ["Citation Latitude", "Gulfstream G650"],
            "total_fleet_efficiency_score": 68.0,
            "mission_coverage_map": {
                "transcontinental": 88.0,
                "intercontinental": 75.0,
                "regional": 100.0,
                "short_hop": 70.0,
                "charter": 82.0,
            },
            "redundancy_analysis": {"redundancy_score": 35.0, "duplicated_mission_pairs": []},
            "replacement_priority_order": ["Gulfstream G650", "Citation Latitude"],
            "optimization_recommendation": ["add short-field capable light or turboprop asset"],
            "utilization_assumptions": {"annual_utilization_hours": 280},
            "gap_analysis": ["missing short-field / short-hop access"],
        },
        "recommendation_confidence": {
            "aircraft_confidence": [
                {"aircraft": "Citation Latitude", "overall_confidence": 74.0},
                {"aircraft": "Gulfstream G650", "overall_confidence": 62.0},
            ],
        },
        "recommendation_justification": {
            "recommendations": [
                {"aircraft": "Citation Latitude", "mission_alignment_score": 80.0},
                {"aircraft": "Gulfstream G650", "mission_alignment_score": 70.0},
            ],
        },
    }


def test_build_executive_synthesis_structure():
    bundle = build_executive_synthesis(_full_data_used())
    assert "fleet_summary" in bundle
    assert "portfolio_ranking" in bundle
    assert "strategic_actions" in bundle
    assert "insights" in bundle

    summary = bundle["fleet_summary"]
    assert 0 <= summary["fleet_health_score"] <= 100
    assert summary["liquidity_position"] in ("strong", "moderate", "weak")
    assert 0 <= summary["mission_efficiency"] <= 100


def test_portfolio_ranking_global():
    bundle = build_executive_synthesis(_full_data_used())
    ranking = bundle["portfolio_ranking"]
    assert len(ranking) >= 2
    assert ranking[0]["total_score"] >= ranking[1]["total_score"]
    for row in ranking:
        assert "drivers" in row
        for key in ("ownership", "market", "mission", "optimization", "confidence"):
            assert key in row["drivers"]


def test_strategic_actions():
    bundle = build_executive_synthesis(_full_data_used())
    actions = bundle["strategic_actions"]
    assert "keep" in actions
    assert "upgrade" in actions
    assert "sell" in actions
    assert "acquire" in actions
    assert actions["upgrade"]


def test_deterministic_insights():
    bundle = build_executive_synthesis(_full_data_used())
    assert bundle["insights"]
    assert any("portfolio leader" in i for i in bundle["insights"])


def test_reproducibility():
    du = _full_data_used()
    a = build_executive_synthesis(du)
    b = build_executive_synthesis(du)
    assert a["synthesis_id"] == b["synthesis_id"]
    assert a == b


def test_empty_data_used_graceful():
    bundle = build_executive_synthesis({})
    assert bundle["fleet_summary"]["fleet_health_score"] == 50.0
    assert bundle["portfolio_ranking"] == []


def test_evaluator_hooks():
    bad = {
        "data_used": {
            "executive_synthesis": {
                "fleet_summary": {"fleet_health_score": 150.0, "liquidity_position": "invalid"},
                "portfolio_ranking": [
                    {"aircraft": "A", "total_score": 90.0},
                    {"aircraft": "B", "total_score": 95.0},
                ],
                "strategic_actions": {
                    "keep": ["A"],
                    "upgrade": [],
                    "sell": ["A"],
                    "acquire": [],
                },
            }
        }
    }
    failures = evaluate_executive_synthesis_hooks(bad)
    assert "synthesis_score_consistency" in failures
    assert "portfolio_ranking_consistency" in failures
    assert "strategic_actions_consistency" in failures


def test_attach_only_when_env_enabled(monkeypatch):
    monkeypatch.delenv("ENABLE_EXECUTIVE_SYNTHESIS", raising=False)
    payload = {"answer": "x", "data_used": _full_data_used()}
    assert not executive_synthesis_enabled()
    out = attach_executive_synthesis_if_enabled("test", payload)
    assert "executive_synthesis" not in (out.get("data_used") or {})

    monkeypatch.setenv("ENABLE_EXECUTIVE_SYNTHESIS", "1")
    out2 = attach_executive_synthesis_if_enabled("test", payload)
    assert "executive_synthesis" in (out2.get("data_used") or {})
    assert out2["data_used"]["executive_synthesis"]["executive_panel"]

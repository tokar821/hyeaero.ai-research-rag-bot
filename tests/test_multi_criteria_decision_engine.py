"""Phase 23 — Multi-Criteria Decision Optimization Engine tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.deterministic

from services.optimization.multi_criteria_decision_engine import (
    CABIN_FOCUSED,
    CORPORATE_FLIGHT_DEPARTMENT,
    COST_FOCUSED,
    RANGE_FOCUSED,
    RESALE_FOCUSED,
    STANDARD_BUYER,
    attach_optimization_result_if_enabled,
    build_optimization_result,
    build_tradeoff_analysis,
    decision_optimization_enabled,
    evaluate_optimization_hooks,
    infer_buyer_profile,
    optimize_aircraft_ranking,
)


def test_cost_focused_buyer():
    result = optimize_aircraft_ranking(
        ["Gulfstream G650", "Citation Latitude"],
        profile=COST_FOCUSED,
    )
    assert result.winner == "Citation Latitude"
    assert result.ranked_candidates[0].total_score >= result.ranked_candidates[1].total_score
    assert result.ranked_candidates[0].acquisition_score > result.ranked_candidates[1].acquisition_score


def test_range_focused_buyer():
    result = optimize_aircraft_ranking(
        ["Gulfstream G650", "Citation Latitude"],
        profile=RANGE_FOCUSED,
    )
    assert result.winner == "Gulfstream G650"
    assert result.ranked_candidates[0].range_score == 100.0
    assert result.ranked_candidates[1].range_score == 0.0


def test_cabin_focused_buyer():
    result = optimize_aircraft_ranking(
        ["Challenger 3500", "Citation Longitude", "Gulfstream G650"],
        profile=CABIN_FOCUSED,
    )
    assert result.winner == "Challenger 3500"
    assert result.ranked_candidates[0].cabin_score >= result.ranked_candidates[1].cabin_score


def test_resale_focused_buyer():
    result = optimize_aircraft_ranking(
        ["Gulfstream G650", "Citation CJ3+"],
        profile=RESALE_FOCUSED,
    )
    assert result.winner == "Gulfstream G650"
    assert result.ranked_candidates[0].resale_score >= result.ranked_candidates[1].resale_score


def test_corporate_profile():
    result = optimize_aircraft_ranking(
        ["Gulfstream G650", "Citation Latitude", "Challenger 3500"],
        profile=CORPORATE_FLIGHT_DEPARTMENT,
    )
    assert result.buyer_profile.name == "CORPORATE_FLIGHT_DEPARTMENT"
    assert result.buyer_profile.weight_total == 100
    assert result.winner
    assert len(result.ranked_candidates) == 3


def test_tradeoff_generation():
    result = optimize_aircraft_ranking(
        ["Gulfstream G650", "Citation Latitude"],
        profile=RANGE_FOCUSED,
    )
    assert result.tradeoffs
    assert any("G650" in t or "Gulfstream G650" in t for t in result.tradeoffs)
    assert any("wins" in t.lower() for t in result.tradeoffs)

    tradeoffs = build_tradeoff_analysis(result.ranked_candidates)
    assert tradeoffs


def test_ranking_reproducibility():
    candidates = ["Gulfstream G650", "Citation Latitude", "Challenger 3500"]
    a = optimize_aircraft_ranking(candidates, profile=COST_FOCUSED)
    b = optimize_aircraft_ranking(candidates, profile=COST_FOCUSED)
    assert a.optimization_id == b.optimization_id
    assert [r.aircraft for r in a.ranked_candidates] == [r.aircraft for r in b.ranked_candidates]
    assert [r.total_score for r in a.ranked_candidates] == [r.total_score for r in b.ranked_candidates]


def test_score_normalization():
    result = optimize_aircraft_ranking(
        ["Gulfstream G650", "Citation Latitude"],
        profile=STANDARD_BUYER,
    )
    scores = result.ranked_candidates
    assert all(0 <= s.acquisition_score <= 100 for s in scores)
    assert all(0 <= s.range_score <= 100 for s in scores)
    assert max(s.range_score for s in scores) == 100.0
    assert min(s.range_score for s in scores) == 0.0
    assert max(s.acquisition_score for s in scores) == 100.0


def test_different_profile_different_winner():
    candidates = ["Gulfstream G650", "Citation Latitude"]
    cost = optimize_aircraft_ranking(candidates, profile=COST_FOCUSED)
    range_r = optimize_aircraft_ranking(candidates, profile=RANGE_FOCUSED)
    assert cost.winner != range_r.winner


def test_infer_buyer_profile_from_query():
    assert infer_buyer_profile("best cabin comfort jet").name == "CABIN_FOCUSED"
    assert infer_buyer_profile("long range nonstop mission").name == "RANGE_FOCUSED"
    assert infer_buyer_profile("affordable operating cost").name == "COST_FOCUSED"


def test_build_optimization_result_from_response():
    bundle = build_optimization_result(
        "cost focused shortlist under budget",
        {
            "data_used": {
                "consultant_recommendations": [
                    {"model": "Gulfstream G650"},
                    {"model": "Citation Latitude"},
                ]
            }
        },
    )
    assert bundle["winner"]
    assert bundle["optimization_panel"]["ranking_table"]
    assert bundle["buyer_profile"]["name"] == "COST_FOCUSED"


def test_evaluator_optimization_hooks():
    bad = {
        "data_used": {
            "optimization_result": {
                "optimization_id": "abc123",
                "ranked_candidates": [
                    {"aircraft": "A", "total_score": 50},
                    {"aircraft": "B", "total_score": 80},
                ],
                "buyer_profile": {"acquisition_cost_weight": 50},
            }
        }
    }
    failures = evaluate_optimization_hooks(bad)
    assert "ranking_stability" in failures
    assert "profile_consistency" in failures


def test_attach_only_when_env_enabled(monkeypatch):
    monkeypatch.delenv("ENABLE_DECISION_OPTIMIZATION", raising=False)
    payload = {"answer": "x", "data_used": {}}
    assert not decision_optimization_enabled()
    out = attach_optimization_result_if_enabled("test", payload)
    assert "optimization_result" not in (out.get("data_used") or {})

    monkeypatch.setenv("ENABLE_DECISION_OPTIMIZATION", "1")
    out2 = attach_optimization_result_if_enabled(
        "compare G650 vs Latitude",
        {
            "data_used": {
                "consultant_recommendations": [
                    {"model": "Gulfstream G650"},
                    {"model": "Citation Latitude"},
                ]
            }
        },
    )
    assert "optimization_result" in (out2.get("data_used") or {})


def test_build_optimization_result_insufficient_data():
    out = build_optimization_result("compare jets", {"data_used": {}})
    assert out["status"] == "INSUFFICIENT_DATA"
    assert out["confidence"] == 0
    assert out["winner"] == ""

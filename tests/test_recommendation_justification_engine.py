"""Phase 21 — Recommendation Justification Engine tests."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.recommendation.recommendation_justification_engine import (
    attach_recommendation_justification_if_enabled,
    build_buy_decision_justification,
    build_comparison_justification,
    build_mission_justification,
    build_recommendation_justification,
    build_rejection_analysis,
    recommendation_justification_enabled,
)


def _mission(pax: int = 8, budget_m: float = 20.0) -> MissionState:
    return MissionState(
        passenger_count=pax,
        routes=["LA to Miami"],
        budget_usd=budget_m * 1_000_000,
        nonstop_requirement=True,
    )


def test_mission_recommendation_justification():
    query = "8 passengers LA to Miami under $20M"
    response = {
        "answer": "Recommendation brief.",
        "data_used": {
            "consultant_recommendations": [
                {"model": "Citation Latitude", "fit": "Good Fit"},
                {"model": "Praetor 600", "fit": "Good Fit"},
            ]
        },
    }
    bundle = build_recommendation_justification(query, response)
    assert bundle["recommendations"]
    top = bundle["recommendations"][0]
    assert top["aircraft"] == "Citation Latitude"
    assert top["mission_alignment_score"] > 0
    assert "great aircraft" not in top["recommendation_reason"].lower()
    assert bundle["mission"]["mission_constraints"]


def test_budget_rejection():
    mission = _mission(budget_m=8.0)
    rejections = build_rejection_analysis(
        ["Gulfstream G650"],
        "Citation Latitude",
        mission=mission,
    )
    assert rejections
    g650_rej = rejections[0]
    assert "G650" in g650_rej.aircraft
    assert "budget" in g650_rej.rejection_reason.lower() or "exceeds" in g650_rej.rejection_reason.lower()


def test_range_rejection():
    mission = MissionState(passenger_count=6, routes=["New York to Tokyo"], nonstop_requirement=True)
    rejections = build_rejection_analysis(
        ["Citation CJ3+", "Gulfstream G650"],
        "Gulfstream G650",
        mission=mission,
    )
    cj = next((r for r in rejections if "CJ3" in r.aircraft), None)
    assert cj is not None
    assert "range" in cj.rejection_reason.lower() or "insufficient" in cj.rejection_reason.lower()


def test_comparison_winner_justification():
    dataset = {
        "status": "OK",
        "aircraft": [
            {
                "canonical_name": "Gulfstream G650",
                "range_nm": 6000,
                "speed_ktas": 488,
                "baggage": 0.9,
            },
            {
                "canonical_name": "Falcon 8X",
                "range_nm": 5600,
                "speed_ktas": 470,
                "baggage": 0.88,
            },
        ],
    }
    decision = build_comparison_justification(dataset)
    assert decision is not None
    assert decision.winner == "Gulfstream G650"
    assert decision.factors
    assert "range" in decision.factors[0].lower()


def test_buy_decision_justification():
    buy = build_buy_decision_justification(
        market_context={
            "expected_market_band_usd": {"low": 8_000_000, "mid": 10_000_000, "high": 12_000_000},
            "ask_position": "in_band",
            "depreciation_band": "moderate",
            "age_position": "mid_life",
        },
        deal_killer={"verdict": "Fair Deal"},
    )
    assert buy["why_deal_is_good"]
    assert buy["valuation_drivers"]


def test_multiple_alternatives_rejection():
    mission = _mission()
    rejections = build_rejection_analysis(
        ["Praetor 600", "Challenger 350", "Gulfstream G280"],
        "Citation Latitude",
        mission=mission,
    )
    assert len(rejections) == 3
    for r in rejections:
        assert r.rejection_reason
        assert "excellent choice" not in r.rejection_reason.lower()


def test_deterministic_replay_consistency():
    query = "Alternatives to Longitude"
    response = {
        "answer": "Alternatives list.",
        "data_used": {
            "alternative_execution": {
                "target": "Citation Longitude",
                "candidates": ["Praetor 600", "Challenger 350"],
            }
        },
    }
    a = build_recommendation_justification(query, response)
    b = build_recommendation_justification(query, response)
    assert a["recommendations"] == b["recommendations"]
    assert a["rejections"] == b["rejections"]
    assert a["mission"]["primary_constraint"] == b["mission"]["primary_constraint"]


def test_mission_justification_block():
    block = build_mission_justification(
        _mission(),
        selected_aircraft="Citation Latitude",
    )
    assert block["why_aircraft_fits"]
    assert block["mission_constraints"]


def test_attach_only_when_env_enabled(monkeypatch):
    monkeypatch.delenv("ENABLE_RECOMMENDATION_JUSTIFICATION", raising=False)
    payload = {"answer": "x", "data_used": {}}
    assert not recommendation_justification_enabled()
    out = attach_recommendation_justification_if_enabled("test", payload)
    assert "recommendation_justification" not in (out.get("data_used") or {})

    monkeypatch.setenv("ENABLE_RECOMMENDATION_JUSTIFICATION", "1")
    out2 = attach_recommendation_justification_if_enabled("test", payload)
    assert "recommendation_justification" in (out2.get("data_used") or {})

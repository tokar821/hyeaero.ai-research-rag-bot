"""
Mission-fit recommendation ranker — operational realism and overbuying penalties.
"""

from __future__ import annotations

from services.consultant.mission_state import MissionState, build_mission_from_current_turn
from services.consultant.recommendation_engine import rank_aircraft_recommendations
from services.recommendation.mission_ranker import (
    MissionCategory,
    classify_mission_category,
    rank_missions,
)


def _top_models(query: str, *, max_results: int = 8) -> list[str]:
    mission = build_mission_from_current_turn(query)
    recs = rank_aircraft_recommendations(mission, max_results=max_results)
    return [r.model for r in recs if not r.avoid]


def _ranked(query: str, *, max_results: int = 8):
    mission = build_mission_from_current_turn(query)
    return rank_aircraft_recommendations(mission, max_results=max_results)


def test_miami_caribbean_does_not_recommend_falcon_8x():
    models = _top_models("8 passengers Miami to Caribbean, short runway focus")
    assert models
    top5 = models[:5]
    assert "Falcon 8X" not in top5
    assert "Gulfstream G650" not in top5
    cat, _, _, _ = rank_missions(build_mission_from_current_turn("8 passengers Miami to Caribbean"))
    assert cat in (MissionCategory.REGIONAL_UTILITY, MissionCategory.MOUNTAIN_AIRPORT)


def test_tokyo_mission_prefers_ultra_long_range():
    models = _top_models(
        "6 executives from San Francisco to Tokyo nonstop westbound in winter"
    )
    top3 = models[:3]
    assert any(m in top3 for m in ("Gulfstream G650", "Falcon 8X", "Global 7500"))
    cat, _, _, _ = rank_missions(
        build_mission_from_current_turn(
            "6 executives from San Francisco to Tokyo nonstop westbound in winter"
        )
    )
    assert cat == MissionCategory.ULTRA_LONG_RANGE


def test_runway_mission_prefers_flexible_aircraft():
    models = _top_models("6 passengers Dallas to Aspen hot and high short runway")
    cat, recs, _, _ = rank_missions(build_mission_from_current_turn("Dallas to Aspen hot and high 6 pax"))
    assert cat == MissionCategory.MOUNTAIN_AIRPORT
    top3 = [r.model for r in recs if not r.avoid][:3]
    assert any(
        m in top3
        for m in (
            "Citation Latitude",
            "Praetor 600",
            "Challenger 350",
            "Gulfstream G280",
            "Challenger Longitude",
            "Challenger 650",
            "Pilatus PC-24",
        )
    )
    cj2 = next((r for r in recs if r.model == "Citation CJ2"), None)
    assert "Citation CJ2" not in top3 or (
        cj2 is not None and cj2.fit in ("Partial Fit", "Not Recommended", "Good Fit")
    )


def test_each_recommendation_has_tradeoff_fields():
    recs = _ranked("8 passengers LA to Miami nonstop around $10M budget", max_results=4)
    top = recs[0]
    assert top.explanation
    assert top.explanation.why_it_fits or top.explanation.strengths
    assert top.explanation.operational_compromises or top.explanation.operational_caveats
    dims = {s.dimension for s in top.scores}
    assert "route_realism" in dims
    assert "overbuying_penalty" in dims
    assert "operating_economics" in dims


def test_ulr_not_top_on_la_miami():
    models = _top_models("8 passengers LA to Miami nonstop around $10M budget")
    assert models[0] not in ("Falcon 8X", "Gulfstream G650", "Global 7500")


def test_classify_coast_to_coast():
    mission = build_mission_from_current_turn("8 passengers LA to Miami nonstop")
    assert classify_mission_category(mission) == MissionCategory.COAST_TO_COAST


def test_falcon_8x_penalized_on_regional():
    from services.pipeline.run_pipeline import run_advisory_pipeline

    result = run_advisory_pipeline(
        "8 passengers Miami to Caribbean",
        max_results=8,
    )
    models = [r.model for r in result.recommendations]
    assert "Falcon 8X" not in models
    assert "Falcon 8X" in result.eliminated_models
    assert not result.feasibility_map["Falcon 8X"].feasible

"""Senior-advisor recommendation framing — category before model."""

import re

from services.consultant.mission_state import MissionState, build_mission_from_current_turn
from services.consultant.recommendation_engine import rank_aircraft_recommendations
from services.consultant.response_formatter import format_consultant_response
from services.consultant.recommendation_framing import (
    mission_is_complex,
    should_anchor_single_model,
    use_tiered_advisor_framing,
)
from services.recommendation.mission_ranker import MissionCategory, classify_mission_category


def _format(query: str) -> str:
    mission = build_mission_from_current_turn(query)
    recs = rank_aircraft_recommendations(mission, max_results=3)
    return format_consultant_response(
        mission=mission,
        recommendations=recs,
        route_assessments=[],
        query=query,
        turn_seed=query,
    )


def test_simple_regional_mission_anchors_single_model():
    mission = build_mission_from_current_turn("6 pax Boston to Miami nonstop recommend")
    recs = rank_aircraft_recommendations(mission, max_results=3)
    cat = classify_mission_category(mission)
    assert recs
    assert recs[0].fit_verdict in (
        "PRIMARY RECOMMENDATION",
        "VIABLE WITH COMPROMISES",
        "BEST FIT",
        "CONDITIONAL FIT",
    )
    if len(recs) > 1:
        gap = recs[0].total_score - recs[1].total_score
        if gap >= 0.08:
            assert should_anchor_single_model(mission, recs, cat)
    assert not use_tiered_advisor_framing(mission, recs, [], mission_category=cat)
    text = _format("6 pax Boston to Miami nonstop recommend")
    assert "Mission Fit:" in text
    assert recs[0].model in text


def test_transpacific_complex_uses_broker_territory_framing():
    mission = build_mission_from_current_turn(
        "12 passengers San Francisco to Tokyo nonstop westbound recommend"
    )
    recs = rank_aircraft_recommendations(mission, max_results=3)
    cat = classify_mission_category(mission)
    assert cat in (MissionCategory.ULTRA_LONG_RANGE, MissionCategory.TRANSATLANTIC_EXECUTIVE)
    assert mission_is_complex(mission, cat, recs, [])
    if recs:
        text = _format("12 passengers San Francisco to Tokyo nonstop westbound recommend")
        assert "Mission Fit:" in text
        assert any(
            tok in text
            for tok in ("Global 7500", "G650", "Gulfstream", "Falcon")
        )
        assert "Verdict:" in text
        assert any(
            tag in text
            for tag in (
                "PRIMARY RECOMMENDATION:",
                "VIABLE WITH COMPROMISES:",
                "MISSION-RISKY:",
                "BEST FIT:",
                "CONDITIONAL FIT:",
            )
        )


def test_tiered_response_orders_category_before_bullets():
    mission = MissionState(
        routes=["San Francisco -> Tokyo"],
        passenger_count=12,
        nonstop_requirement=True,
        westbound=True,
    )
    recs = rank_aircraft_recommendations(mission, max_results=3)
    if len(recs) < 2:
        return
    text = format_consultant_response(
        mission=mission,
        recommendations=recs,
        route_assessments=[],
        query="12 pax SFO to Tokyo westbound nonstop",
        turn_seed="tiered-order",
    )
    assert "Mission Fit:" in text
    assert "Aircraft Options:" in text
    assert "Verdict:" in text

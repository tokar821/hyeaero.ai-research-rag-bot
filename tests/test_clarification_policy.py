"""Decisive recommendations vs targeted clarification."""

from services.consultant.mission_state import MissionState, build_mission_from_current_turn
from services.consultant.recommendation_engine import rank_aircraft_recommendations
from services.consultant.response_formatter import format_consultant_response
from services.recommendation.fit_policy import (
    mission_clarification_needs,
    mission_maps_to_category,
    mission_well_defined,
)


def test_route_missing_only_triggers_route_clarification():
    mission = build_mission_from_current_turn("recommend a business jet")
    needs = mission_clarification_needs(mission, "recommend a business jet")
    assert needs.needs_route
    assert not needs.needs_passenger_count
    assert not mission_well_defined(mission, "recommend a business jet")


def test_transpacific_route_maps_without_pax():
    mission = build_mission_from_current_turn("San Francisco to Tokyo nonstop westbound recommend")
    assert mission_maps_to_category(mission)
    needs = mission_clarification_needs(mission, "San Francisco to Tokyo nonstop westbound recommend")
    assert not needs.needs_route
    assert not needs.needs_passenger_count


def test_la_miami_with_pax_is_decisive():
    q = "8 pax LA to Miami nonstop recommend"
    mission = build_mission_from_current_turn(q)
    assert mission_well_defined(mission, q)
    needs = mission_clarification_needs(mission, q)
    assert not needs.any


def test_route_only_response_asks_route_not_hedge():
    mission = build_mission_from_current_turn("recommend a business jet")
    text = format_consultant_response(
        mission=mission,
        recommendations=[],
        route_assessments=[],
        query="recommend a business jet",
    )
    assert "city pair" in text.lower() or "origin and destination" in text.lower()
    assert "probably" not in text.lower()
    assert "before we lock" not in text.lower()


def test_transpacific_response_decisive_without_pax_question():
    q = "San Francisco to Tokyo nonstop westbound recommend"
    mission = build_mission_from_current_turn(q)
    recs = rank_aircraft_recommendations(mission, max_results=3)
    text = format_consultant_response(
        mission=mission,
        recommendations=recs,
        route_assessments=[],
        query=q,
    )
    assert len(recs) >= 1
    assert recs[0].model in text
    assert "how many passengers" not in text.lower()


def test_ambiguous_budget_triggers_budget_clarification():
    mission = MissionState(
        routes=["Los Angeles -> Miami"],
        passenger_count=8,
        budget_usd=10_000_000,
    )
    needs = mission_clarification_needs(mission, "8 pax LA to Miami $10M recommend")
    assert not needs.needs_route
    assert needs.needs_budget or mission_maps_to_category(mission)


def test_multi_city_international_does_not_ask_route_or_pax():
    q = "Dallas, New York, London, 15 passengers recommend"
    mission = build_mission_from_current_turn(q)
    assert mission.routes
    assert mission.passenger_count == 15
    needs = mission_clarification_needs(mission, q)
    assert not needs.needs_route
    assert not needs.needs_passenger_count
    assert not needs.needs_category_usage
    assert mission_well_defined(mission, q)


def test_multi_city_international_does_not_append_clarifiers():
    q = "Dallas, New York, London, 15 passengers recommend"
    mission = build_mission_from_current_turn(q)
    needs = mission_clarification_needs(mission, q, recommendations=[])
    assert not needs.any
    text = format_consultant_response(
        mission=mission,
        recommendations=[],
        route_assessments=[],
        query=q,
    )
    assert "city pair" not in text.lower()
    assert "how many passengers" not in text.lower()
    assert "domestic" not in text.lower()
    assert "transoceanic" not in text.lower()


def test_us_only_multi_city_asks_domestic_vs_transoceanic():
    q = "Dallas, New York, Chicago, 6 passengers recommend"
    mission = build_mission_from_current_turn(q)
    needs = mission_clarification_needs(mission, q)
    assert not needs.needs_route
    assert needs.needs_category_usage
    recs = rank_aircraft_recommendations(mission, max_results=3)
    text = format_consultant_response(
        mission=mission,
        recommendations=recs,
        route_assessments=[],
        query=q,
    )
    assert "domestic" in text.lower() and "transoceanic" in text.lower()

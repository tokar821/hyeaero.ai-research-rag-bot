"""Qualitative fit labels — no numeric scores in exports."""

from services.consultant.mission_state import MissionState, build_mission_from_current_turn
from services.consultant.recommendation_engine import rank_aircraft_recommendations
from services.consultant.response_formatter import format_consultant_response, sanitize_advisor_output
from services.recommendation.fit_policy import (
    EXTENDED_RECOMMENDATION_LIMIT,
    FIT_GOOD,
    FIT_NOT_RECOMMENDED,
    FIT_PARTIAL,
    FIT_STRONG,
    STANDARD_RECOMMENDATION_LIMIT,
    assign_fit_tiers,
    mission_clarification_needs,
    mission_maps_to_category,
    mission_well_defined,
    normalize_fit_label,
    recommendation_limit_from_query,
    score_to_fit_label,
)


def test_recommendation_limit_default_three():
    assert recommendation_limit_from_query("recommend a jet for LA to Miami") == STANDARD_RECOMMENDATION_LIMIT


def test_recommendation_limit_five_when_explicit():
    assert recommendation_limit_from_query("give me top 5 aircraft options") == EXTENDED_RECOMMENDATION_LIMIT


def test_score_to_fit_label_buckets():
    assert score_to_fit_label(0.8) == FIT_STRONG
    assert score_to_fit_label(0.6) == FIT_GOOD
    assert score_to_fit_label(0.45) == FIT_PARTIAL
    assert score_to_fit_label(0.3) == FIT_NOT_RECOMMENDED
    assert score_to_fit_label(0.9, avoid=True) == FIT_NOT_RECOMMENDED


def test_rank_returns_qualitative_fit_only():
    mission = build_mission_from_current_turn("8 pax LA to Miami $10M nonstop recommend")
    recs = rank_aircraft_recommendations(mission)
    assert len(recs) <= 3
    assert all(r.fit in (FIT_STRONG, FIT_GOOD, FIT_PARTIAL, FIT_NOT_RECOMMENDED) for r in recs)
    d = recs[0].to_dict()
    assert "total_score" not in d
    assert "rank" not in d
    assert "confidence" not in d


def test_to_dict_uses_fit_not_numeric_scores():
    mission = build_mission_from_current_turn("8 pax LA to Miami nonstop")
    recs = rank_aircraft_recommendations(mission, max_results=3)
    d = recs[0].to_dict()
    assert d["fit"] in (FIT_STRONG, FIT_GOOD, FIT_PARTIAL, FIT_NOT_RECOMMENDED)
    if d.get("scores"):
        assert "score" not in d["scores"][0]
        assert "fit" in d["scores"][0]


def test_mission_well_defined_requires_route_and_pax():
    assert mission_well_defined(build_mission_from_current_turn("8 pax LA to Miami nonstop"))
    assert not mission_well_defined(MissionState())
    assert not mission_well_defined(build_mission_from_current_turn("recommend a business jet"))


def test_mission_maps_to_category_on_long_route():
    mission = build_mission_from_current_turn("San Francisco to Tokyo nonstop westbound")
    assert mission_maps_to_category(mission)


def test_clarification_needs_route_only_when_missing():
    mission = build_mission_from_current_turn("recommend a business jet")
    needs = mission_clarification_needs(mission, "recommend a business jet")
    assert needs.needs_route
    assert not needs.needs_passenger_count


def test_under_defined_asks_route_not_hedge():
    mission = build_mission_from_current_turn("recommend a business jet")
    text = format_consultant_response(
        mission=mission,
        recommendations=[],
        route_assessments=[],
        query="recommend a business jet",
    )
    assert "Conditional options" not in text
    assert "conditional paths" not in text.lower()
    assert "city pair" in text.lower() or "origin and destination" in text.lower()
    assert "probably" not in text.lower()


def test_assign_fit_tiers_leader_is_strong_fit():
    mission = build_mission_from_current_turn("8 pax LA to Miami nonstop")
    recs = rank_aircraft_recommendations(mission, max_results=3)
    assign_fit_tiers(recs)
    assert recs[0].fit == FIT_STRONG


def test_normalize_legacy_high_medium_low():
    assert normalize_fit_label("High") == FIT_STRONG
    assert normalize_fit_label("Medium") == FIT_GOOD
    assert normalize_fit_label("Low") == FIT_PARTIAL


def test_sanitize_strips_confidence_and_numeric_scores():
    dirty = "G650 — mission-fit score 0.91 (confidence 85%) rank #1 Fit: High"
    clean = sanitize_advisor_output(dirty)
    assert "0.91" not in clean
    assert "85%" not in clean
    assert "rank" not in clean.lower() or "#1" not in clean

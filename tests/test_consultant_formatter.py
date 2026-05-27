"""Consultant response formatter — advisor tone and structure."""

import re

from services.consultant.mission_state import MissionState, build_mission_from_current_turn
from services.consultant.recommendation_engine import rank_aircraft_recommendations
from services.consultant.response_formatter import (
    format_consultant_response,
    sanitize_advisor_output,
    should_use_structured_formatter,
)
from services.consultant.route_feasibility import assess_mission_routes
from services.consultant.recommendation_engine import _AIRCRAFT_PROFILES

_FORBIDDEN_HEADERS = (
    "Mission Summary",
    "Best Fit Aircraft",
    "Why They Fit",
    "Operational Tradeoffs",
    "Why Alternatives Ranked Lower",
    "Alternatives scored lower",
    "Bottom-Line Recommendation",
    "Mission type:",
    "Route(s):",
    "The tradeoffs to keep in view",
)

_ROBOTIC_PHRASES = (
    "conditional options",
    "conditional paths",
    "mission firms up",
    "lock a single winner",
    "until then",
    "clearest fit",
)


def test_no_internal_headers_in_formatted_output():
    mission = build_mission_from_current_turn("8 pax LA to Miami $10M nonstop recommend")
    recs = rank_aircraft_recommendations(mission, max_results=4)
    prof = _AIRCRAFT_PROFILES[recs[0].model]
    routes = assess_mission_routes(
        mission,
        aircraft_practical_nm=float(prof["practical_nm"]),
        aircraft_brochure_nm=float(prof["brochure_nm"]),
    )
    text = format_consultant_response(
        mission=mission,
        recommendations=recs,
        route_assessments=routes,
    )
    for header in _FORBIDDEN_HEADERS:
        assert header not in text
    for phrase in _ROBOTIC_PHRASES:
        assert phrase not in text.lower()
    assert not re.search(r"mission[- ]?fit score", text, re.I)


def test_advisor_tone_structure():
    mission = build_mission_from_current_turn("8 pax LA to Miami $10M nonstop recommend")
    recs = rank_aircraft_recommendations(mission, max_results=4)
    text = format_consultant_response(
        mission=mission,
        recommendations=recs,
        route_assessments=[],
        query="8 pax LA to Miami nonstop recommend",
    )
    assert "PRIMARY RECOMMENDATION:" in text
    assert "Mission Fit:" in text
    assert "Aircraft Options:" in text
    assert "Mission Summary" not in text


def test_advisor_bullet_conversational():
    mission = build_mission_from_current_turn("8 pax LA to Miami $10M nonstop recommend")
    recs = rank_aircraft_recommendations(mission, max_results=4)
    text = format_consultant_response(mission=mission, recommendations=recs, route_assessments=[])
    # Broker layer uses prose + fit footer, not middleware bullets
    assert "PRIMARY RECOMMENDATION:" in text
    assert "mission profile" not in text.lower()
    assert "strong alternative with" not in text.lower()


def test_partial_mission_avoids_robotic_phrasing():
    mission = build_mission_from_current_turn("recommend a business jet")
    recs = rank_aircraft_recommendations(mission, max_results=3)
    text = format_consultant_response(
        mission=mission,
        recommendations=recs,
        route_assessments=[],
        query="recommend a business jet",
    )
    for phrase in _ROBOTIC_PHRASES:
        assert phrase not in text.lower()
    assert "Conditional options" not in text
    assert "city pair" in text.lower() or "origin and destination" in text.lower()
    assert "probably" not in text.lower()


def test_route_line_not_char_spaced():
    mission = build_mission_from_current_turn(
        "6 executives San Francisco to Tokyo and Seoul nonstop westbound"
    )
    recs = rank_aircraft_recommendations(mission, max_results=3)
    text = format_consultant_response(mission=mission, recommendations=recs, route_assessments=[])
    assert "Route(s):" not in text


def test_sanitize_strips_robotic_phrases():
    dirty = (
        "Conditional options\n"
        "I would not lock a single winner yet — conditional paths until the mission firms up.\n"
        "Gulfstream G650 — clearest fit (confidence 85%)\n"
    )
    clean = sanitize_advisor_output(dirty)
    assert "Conditional options" not in clean
    assert "conditional paths" not in clean.lower()
    assert "mission firms up" not in clean.lower()
    assert "confidence" not in clean.lower() or "85%" not in clean


def test_should_use_formatter_for_advisory_mode():
    assert should_use_structured_formatter(
        {"consultant_response_mode": "mission_advisory"},
        MissionState(),
        "recommend a jet",
    )

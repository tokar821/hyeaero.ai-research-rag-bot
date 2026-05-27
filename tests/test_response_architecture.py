"""Fixed response architecture — recommendations and comparisons."""

import re

from services.consultant.mission_state import build_mission_from_current_turn
from services.consultant.recommendation_engine import rank_aircraft_recommendations
from services.consultant.response_architecture import (
    format_comparison_architecture,
    format_recommendation_architecture,
)
from services.consultant.response_formatter import format_consultant_response
from services.consultant.comparison_engine import build_structured_comparison


def test_recommendation_has_three_sections():
    mission = build_mission_from_current_turn("6 pax Boston to Miami nonstop")
    recs = rank_aircraft_recommendations(mission, max_results=3)
    text = format_recommendation_architecture(mission, recs)
    assert "Mission Fit:" in text
    assert "* Route:" in text
    assert "* Pax:" in text
    assert "* Priorities:" in text
    assert "Aircraft Options:" in text
    assert "Why it fits:" in text
    assert "Key compromise:" in text
    assert "Verdict:" in text
    assert re.search(r"\* PRIMARY RECOMMENDATION:", text)


def test_no_territory_opener():
    mission = build_mission_from_current_turn("6 pax Boston to Miami nonstop recommend")
    recs = rank_aircraft_recommendations(mission, max_results=3)
    text = format_consultant_response(
        mission=mission,
        recommendations=recs,
        route_assessments=[],
        query="6 pax Boston to Miami nonstop recommend",
    )
    assert "territory" not in text.lower()
    assert "Mission Fit:" in text


def test_comparison_only_five_dimensions():
    text = format_comparison_architecture(["Challenger 350", "Citation Latitude"])
    assert "Comparison:" in text
    assert "* Range:" in text or "* range:" in text.lower()
    assert "Operating cost:" in text
    assert "Runway capability:" in text
    assert "Liquidity:" in text
    assert "mission fit" not in text.lower()
    assert "pros" not in text.lower()


def test_compare_query_uses_comparison_architecture():
    mission = build_mission_from_current_turn(
        "Compare Challenger 350 vs Praetor 600 for 8 pax New York to London"
    )
    comp = build_structured_comparison(
        ["Challenger 350", "Praetor 600"],
        mission,
    )
    text = format_consultant_response(
        mission=mission,
        recommendations=[],
        route_assessments=[],
        comparison=comp,
        query="Compare Challenger 350 vs Praetor 600",
    )
    assert "Comparison:" in text
    assert "Liquidity:" in text
    assert "Mission Fit:" not in text

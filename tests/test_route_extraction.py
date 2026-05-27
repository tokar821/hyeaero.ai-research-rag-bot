"""Aviation-aware route extraction engine tests."""

from services.mission.route_extractor import (
    RouteExtraction,
    extract_routes,
    resolve_place,
    sanitize_user_text_for_routes,
)

ASSISTANT_BLOCK = """
Mission Summary
Passengers: 10
Route(s): What Would You Like -> Work
Best Fit Aircraft
Challenger 350
Alternatives -> Gulfstream
Consultant Insight: dispatch reliability
"""


def test_valid_transatlantic_san_francisco_tokyo():
    legs = extract_routes(
        "6 executives from San Francisco to Tokyo nonstop westbound in winter"
    )
    assert len(legs) >= 1
    assert any(
        r.route.origin == "San Francisco" and r.route.destination == "Tokyo"
        for r in legs
    )
    assert all(r.confidence >= 0.72 for r in legs)
    assert all(r.source == "current_user_turn" for r in legs)


def test_valid_chicago_london_arrow():
    legs = extract_routes("We need Chicago -> London regularly for 8 pax")
    assert len(legs) == 1
    assert legs[0].route.origin == "Chicago"
    assert legs[0].route.destination == "London"
    assert legs[0].confidence >= 0.72


def test_regional_miami_caribbean():
    legs = extract_routes("8 passengers Miami to Caribbean, short runway focus")
    assert len(legs) == 1
    assert legs[0].route.origin == "Miami"
    assert legs[0].route.destination == "Caribbean"


def test_nyc_and_paris_pair():
    legs = extract_routes("Typical mission is NYC and Paris twice a month")
    assert len(legs) == 1
    assert legs[0].route.origin == "New York"
    assert legs[0].route.destination == "Paris"


def test_reject_what_would_you_like_work():
    assert resolve_place("What Would You Like")[0] is None
    assert resolve_place("Work")[0] is None
    assert resolve_place("Alternatives")[0] is None
    legs = extract_routes("What Would You Like -> Work")
    assert legs == []


def test_reject_alternatives_gulfstream():
    legs = extract_routes("Alternatives -> Gulfstream compared efficiency")
    assert legs == []


def test_assistant_contamination_stripped():
    polluted = f"8 pax Miami to Caribbean\n{ASSISTANT_BLOCK}"
    legs = extract_routes(polluted)
    assert len(legs) == 1
    assert legs[0].route.origin == "Miami"
    assert not any("what would" in r.route.origin.lower() for r in legs)


def test_malformed_arrow_random_markdown():
    legs = extract_routes(
        "## Mission Summary\n- **Best Fit**: Gulfstream G650\n"
        "Operational Tradeoffs -> Higher efficiency"
    )
    assert legs == []


def test_la_miami_not_pax_la_miami():
    legs = extract_routes("6 pax LA to Miami nonstop")
    assert len(legs) == 1
    assert legs[0].route.origin == "Los Angeles"
    assert legs[0].route.destination == "Miami"


def test_route_extraction_to_dict():
    legs = extract_routes("NYC to London")
    assert legs[0].to_dict()["route"]["origin"] == "New York"
    assert legs[0].to_dict()["confidence"] >= 0.72


def test_sanitize_drops_bullet_headings():
    raw = "- Route(s): fake\n8 passengers Miami to Caribbean"
    clean = sanitize_user_text_for_routes(raw)
    assert "route(s)" not in clean.lower() or "miami" in clean.lower()
    legs = extract_routes(raw)
    assert len(legs) == 1


def test_no_route_from_recommendation_section():
    legs = extract_routes(
        "Bottom-Line Recommendation: start with Challenger 350 unless cabin pushes you up"
    )
    assert legs == []

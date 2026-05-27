"""Mission extraction (turn-isolated) tests."""

from services.consultant.mission_state import build_mission_from_current_turn
from services.consultant.mission_state import format_routes_for_display, normalize_routes
from services.mission import extract_mission


def test_extract_passengers_and_route_la_miami():
    ms = build_mission_from_current_turn("8 passengers LA to Miami nonstop under $10M budget")
    assert ms.passenger_count == 8
    assert any("Miami" in r for r in ms.routes)
    assert ms.budget_usd and ms.budget_usd >= 9_000_000
    assert ms.nonstop_requirement is True


def test_westbound_winter_mission_signals():
    p = extract_mission("West Coast to Europe westbound in winter — need reliable nonstop for 10 pax")
    assert p.westbound_sensitive is True
    assert p.seasonal_note == "winter_headwinds"
    assert p.passengers == 10


def test_turn_isolated_no_incremental_merge():
    m1 = build_mission_from_current_turn("We need 8 passengers")
    m2 = build_mission_from_current_turn("Budget around $12M, cabin must feel premium")
    assert m1.passenger_count == 8
    assert m2.passenger_count is None
    assert m2.cabin_priority == "high"
    assert m2.budget_usd and m2.budget_usd >= 11_000_000


def test_fractional_vs_ownership():
    p = extract_mission("Should I go fractional with NetJets or buy outright?")
    assert p.ownership_interest is not None
    assert p.ownership_interest.value == "fractional"


def test_normalize_routes_string_not_char_joined():
    assert normalize_routes("San Francisco -> Tokyo") == ["San Francisco -> Tokyo"]
    assert format_routes_for_display("San Francisco -> Tokyo") == "San Francisco -> Tokyo"
    assert ", a, n" not in format_routes_for_display("San Francisco -> Tokyo")


def test_join_on_string_route_produces_garbage_detected():
    bad = ", ".join("San Francisco -> Tokyo")
    assert ", a, n" in bad
    assert format_routes_for_display(bad) == ""


def test_sf_tokyo_seoul_routes():
    p = extract_mission(
        "6 executives from San Francisco to Tokyo and Seoul, nonstop westbound winter"
    )
    routes = " ".join(r.label() for r in p.routes).lower()
    assert "san francisco" in routes and "tokyo" in routes

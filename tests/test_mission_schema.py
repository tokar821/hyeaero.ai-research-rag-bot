"""Typed mission schema, normalization, and validation tests."""

import pytest

from services.mission import extract_mission
from services.mission.models import MissionProfile, OwnershipMode, Route
from services.mission.normalization import normalize_place, normalize_ownership
from services.mission.validators import (
    validate_route_candidate,
    validate_route_label,
    validate_profile,
    validate_passengers,
)


def test_valid_route_typed():
    r = validate_route_candidate("nyc", "London")
    assert r is not None
    assert r.origin == "New York"
    assert r.destination == "London"
    assert r.label() == "New York -> London"


def test_invalid_route_ui_phrase_rejected():
    assert validate_route_candidate("What Would You Like", "Work") is None
    assert validate_route_candidate("Full Ownership", "Higher") is None
    assert validate_route_label("What Would You Like -> Work") is None


def test_invalid_route_generic_nouns():
    assert validate_route_candidate("mission", "route") is None


def test_normalize_city_aliases():
    assert normalize_place("nyc") == "New York"
    assert normalize_place("SFO") == "San Francisco"
    assert normalize_ownership("NetJets fractional share") == OwnershipMode.FRACTIONAL


def test_passenger_sanity():
    assert validate_passengers(8) == 8
    assert validate_passengers(0) is None
    assert validate_passengers(25) is None


def test_extract_mission_returns_typed_routes():
    p = extract_mission("8 passengers Miami to Caribbean, operating cost priority")
    assert p.passengers == 8
    assert len(p.routes) == 1
    assert isinstance(p.routes[0], Route)
    assert p.routes[0].origin == "Miami"
    assert p.routes[0].destination == "Caribbean"
    assert p.operating_cost_priority.value == "high"


def test_malformed_extraction_rejected_in_profile():
    p = extract_mission(
        "What would you like to explore full ownership work higher efficiency"
    )
    assert p.routes == []
    d = p.to_dict()
    assert d["schema_version"] == 3
    assert isinstance(d["routes"], list)
    assert all("origin" in r and "destination" in r for r in d["routes"])


def test_to_dict_no_stringly_route_blob():
    p = extract_mission("6 pax LA to Miami nonstop")
    d = p.to_dict()
    assert d["routes"] == [{"origin": "Los Angeles", "destination": "Miami"}]
    assert "priorities" not in d
    assert "field_confidence" not in d


def test_duplicate_route_cleanup():
    profile = MissionProfile(
        routes=[
            Route(origin="Miami", destination="Caribbean"),
            Route(origin="Miami", destination="Caribbean"),
            Route(origin="Los Angeles", destination="Miami"),
        ]
    )
    out = validate_profile(profile)
    assert len(out.routes) == 2


def test_route_frozen_immutable():
    r = Route(origin="Dallas", destination="Aspen")
    with pytest.raises(Exception):
        r.origin = "Houston"  # type: ignore[misc]

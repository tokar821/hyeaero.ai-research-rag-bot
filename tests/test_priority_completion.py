"""Completion tests for priorities 6–9 gaps."""

from __future__ import annotations

from services.comparison.aircraft_registry_lock import lock_comparison_aircraft
from services.conversation_continuity.mission_evolution import (
    format_mission_evolution_response,
    is_mission_evolution_query,
)
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES


def test_citation_longitude_legacy_600_comparison_registry():
    locked = lock_comparison_aircraft(
        ["Cessna Citation Longitude", "Embraer Legacy 600"]
    )
    assert "Citation Longitude" in locked.canonical
    assert "Legacy 600" in locked.canonical
    assert len(locked.canonical) == 2


def test_profiles_exist():
    assert "Citation Longitude" in AIRCRAFT_PROFILES
    assert "Legacy 600" in AIRCRAFT_PROFILES


def test_mission_evolution_aspen_follow_up():
    q = "Would your recommendation change if Aspen winter operations become frequent?"
    assert is_mission_evolution_query(q)
    text = format_mission_evolution_response(q, None, data_used={})
    assert "aspen" in text.lower()
    assert "change" in text.lower() or "frequent" in text.lower()


def test_survival_filter_query_detection():
    from services.recommendation.survival_filter_shortlist import is_survival_filter_query

    assert is_survival_filter_query(
        "What realistically survives once winter Pacific reserve margins are applied?"
    )


def test_image_exterior_only_query():
    from services.orchestration.query_archetype import is_image_request_query

    assert is_image_request_query(
        "Show verified exterior-only images of the Praetor 600."
    )


def test_hong_kong_route_geodesic():
    from services.mission.route_distance_authority import resolve_route_distance

    res = resolve_route_distance("Seattle -> Hong Kong")
    assert res.distance_nm > 4500

"""Route directionality — continuation flow and extractor leak rejection."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.mission.models import MissionProfile, Route
from services.mission.route_directionality import (
    enforce_route_directionality,
    literal_direction_in_text,
    validate_route_direction,
)


def test_literal_direction_forward():
    assert literal_direction_in_text("San Francisco", "Doha", "nonstop from San Francisco to Doha") is True


def test_literal_direction_backward():
    assert literal_direction_in_text("San Francisco", "Doha", "Doha to San Francisco only") is False


def test_swap_me_to_us_inversion():
    profile = MissionProfile(
        routes=[Route(origin="Doha", destination="San Francisco")]
    )
    mission = MissionState(routes=["Doha -> San Francisco"])
    report = enforce_route_directionality(
        "California domestic plus Doha nonstop from San Francisco",
        profile,
        mission,
    )
    labels = " ".join(r.label().lower() for r in profile.routes)
    assert "san francisco -> doha" in labels
    assert report.swapped or "doha -> san francisco" not in labels


def test_remove_singapore_aspen_leak():
    route = Route(origin="Singapore", destination="Aspen")
    corrected, action = validate_route_direction(
        route,
        "LA ski, Tokyo, and Aspen winter ops — no direct Singapore ski link",
    )
    assert action == "remove"
    assert corrected is None

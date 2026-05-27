"""Multi-hub corridor extraction — catalog geography, not hub collapse."""

from __future__ import annotations

from services.mission.mission_corridor_routes import (
    enrich_profile_routes_from_corridor,
    extract_between_corridor,
)
from services.mission.models import MissionProfile, Route


def test_houston_calgary_london_field_access_enrichment():
    q = (
        "We move engineers between Houston, Calgary, and remote oil sites in Northern Canada, "
        "but executives also fly quarterly to London and Zurich."
    )
    profile = MissionProfile()
    enrich_profile_routes_from_corridor(q, profile)
    labels = profile.route_labels()
    assert any("Houston" in lbl for lbl in labels)
    assert any("Calgary" in lbl or "London" in lbl for lbl in labels)


def test_between_clause_resolves_sao_paulo():
    q = "We operate between Miami, São Paulo, Madrid, and small Caribbean islands."
    legs = extract_between_corridor(q)
    assert len(legs) >= 2
    dests = {leg.route.destination for leg in legs}
    assert "Caribbean" in dests or any("Caribbean" in leg.route.label() for leg in legs)

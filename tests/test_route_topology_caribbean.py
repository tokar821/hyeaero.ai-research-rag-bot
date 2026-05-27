"""Topology — Pacific/ME to Caribbean and executive overlay preservation."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.mission.models import MissionProfile, Route
from services.mission.route_topology_validator import validate_route_edge


def test_tokyo_to_caribbean_rejected():
    route = Route(origin="Tokyo", destination="Caribbean")
    edge = validate_route_edge(
        route,
        query="Los Angeles, Tokyo, Singapore, Dubai, and Caribbean islands",
        authority="explicit",
    )
    assert edge.structurally_valid is False
    assert "caribbean" in edge.rejection_reason


def test_miami_to_caribbean_allowed():
    route = Route(origin="Miami", destination="Caribbean")
    edge = validate_route_edge(
        route,
        query="Caribbean island hops from Miami",
        authority="hub_inferred",
    )
    assert edge.structurally_valid is True


def test_dubai_to_caribbean_rejected():
    route = Route(origin="Dubai", destination="Caribbean")
    edge = validate_route_edge(
        route,
        query="Dubai and Caribbean islands",
        authority="continuation",
    )
    assert edge.structurally_valid is False

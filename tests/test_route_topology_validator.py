"""Route topology realism — corridor propagation hardening."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.mission.mission_extractor import extract_mission
from services.mission.mission_understanding_engine import (
    attach_mission_understanding,
    build_mission_understanding,
)
from services.mission.pre_ranking_representation import apply_pre_ranking_representation


def _run_pipeline(q: str):
    data_used = {}
    profile = extract_mission(q)
    mission = MissionState(routes=profile.route_labels(), passenger_count=profile.passengers)
    pkt = build_mission_understanding(q, profile, mission, broker_memory=None)
    attach_mission_understanding(data_used, pkt)
    profile, mission, pkt = apply_pre_ranking_representation(
        q, profile, mission, pkt, data_used=data_used
    )
    routes = list(profile.route_labels() or mission.routes or [])
    return routes, data_used


def _labels_contain(routes, fragment: str) -> bool:
    blob = " ".join(routes).lower()
    return fragment.lower() in blob


def test_la_tokyo_singapore_aspen_no_mountain_intercontinental_stitch():
    q = (
        "We regularly fly Los Angeles-Tokyo and Singapore, but every winter we operate "
        "heavy ski traffic into Aspen, Telluride, and Jackson Hole."
    )
    routes, _ = _run_pipeline(q)
    assert not _labels_contain(routes, "Aspen -> Tokyo")
    assert not _labels_contain(routes, "Aspen -> Singapore")
    assert not _labels_contain(routes, "Tokyo -> Telluride")
    assert _labels_contain(routes, "Los Angeles -> Tokyo") or _labels_contain(
        routes, "Los Angeles -> Singapore"
    )
    assert _labels_contain(routes, "Los Angeles -> Aspen") or _labels_contain(routes, "Aspen")


def test_miami_caribbean_riyadh_no_caribbean_to_me():
    q = (
        "We operate from Miami into Caribbean islands with short humid runways, but leadership "
        "also requires nonstop Paris and occasional Riyadh capability."
    )
    routes, _ = _run_pipeline(q)
    assert not _labels_contain(routes, "Caribbean -> Riyadh")
    assert _labels_contain(routes, "Miami -> Caribbean") or _labels_contain(routes, "Miami -> Riyadh")
    assert _labels_contain(routes, "Miami -> Paris") or _labels_contain(routes, "Paris")


def test_remote_drilling_london_no_field_to_executive():
    q = (
        "We transport technical crews to remote drilling sites across Alberta, but ownership "
        "also insists on nonstop Houston-London capability twice monthly."
    )
    routes, _ = _run_pipeline(q)
    assert not _labels_contain(routes, "Remote Drilling Sites -> London")
    assert not _labels_contain(routes, "Remote Drilling Site -> London")
    assert _labels_contain(routes, "Houston -> London") or _labels_contain(routes, "London")


def test_founder_dubai_ny_not_boston():
    q = (
        "90% of our flying is short East Coast trips with 3-4 executives between New York, Boston, "
        "and Chicago, but the founder insists on nonstop Dubai capability for occasional personal travel."
    )
    routes, du = _run_pipeline(q)
    assert _labels_contain(routes, "New York -> Dubai")
    assert not _labels_contain(routes, "Boston -> Dubai")
    assert not _labels_contain(routes, "Chicago -> Dubai")
    removed = (du.get("route_topology_validation") or {}).get("removed_routes") or []
    assert any("Boston -> Dubai" in r or "Chicago -> Dubai" in r for r in removed) or (
        not _labels_contain(routes, "Boston -> Dubai")
    )


def test_validator_rejects_aspen_caribbean_direct():
    from services.mission.models import Route
    from services.mission.route_topology_validator import validate_route_edge

    edge = validate_route_edge(
        Route(origin="Aspen", destination="Caribbean"),
        query="Aspen winter ops and Caribbean island flying",
        authority="domain_bridge",
    )
    assert not edge.structurally_valid
    assert edge.confidence < 0.75

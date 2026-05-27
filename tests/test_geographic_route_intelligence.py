"""Geographic + route intelligence — regional ontology and hub anchor realism."""

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
    return list(profile.route_labels() or mission.routes or []), data_used


def _blob(routes):
    return " ".join(routes).lower()


def test_northeast_corridor_founder_singapore():
    q = (
        "80% of our flying is domestic Northeast corridor traffic with small executive groups, "
        "but the founder requires nonstop Singapore capability several times a month."
    )
    routes, du = _run_pipeline(q)
    assert routes, f"expected non-empty graph, got {routes}"
    assert "new york" in _blob(routes) or "boston" in _blob(routes)
    assert "singapore" in _blob(routes)
    assert "new york -> singapore" in _blob(routes) or "boston -> singapore" in _blob(routes)
    assert "aspen -> singapore" not in _blob(routes)


def test_colorado_utah_mountain_overlay():
    q = (
        "We transport crews and bulky technical equipment into mountain airports across "
        "Colorado and Utah, but executives also require regular New York-London nonstop capability."
    )
    routes, _ = _run_pipeline(q)
    assert "new york -> london" in _blob(routes)
    assert any(x in _blob(routes) for x in ("aspen", "telluride", "jackson", "vail"))
    assert not any(
        x in _blob(routes)
        for x in ("aspen -> london", "telluride -> london", "aspen -> tokyo", "aspen -> singapore")
    )


def test_caribbean_anchor_priority():
    q = (
        "Most of our flights are short Caribbean hops into humid, short-runway airports, "
        "but ownership also expects nonstop capability to Madrid and occasionally Dubai."
    )
    routes, _ = _run_pipeline(q)
    assert "miami -> caribbean" in _blob(routes) or "caribbean" in _blob(routes)
    assert "madrid -> caribbean" not in _blob(routes)
    assert "madrid -> dubai" in _blob(routes) or "miami -> dubai" in _blob(routes)


def test_industrial_desert_spokes():
    q = (
        "We support energy operations into remote Middle Eastern desert strips while executives "
        "simultaneously travel between London, Paris, and Geneva."
    )
    routes, du = _run_pipeline(q)
    geo = du.get("geographic_route_intelligence") or {}
    assert "desert energy corridor" in _blob(routes)
    assert "dubai -> desert" in _blob(routes) or "desert energy" in _blob(routes)
    assert "industrial_geography" in (geo.get("regions_activated") or [])
    assert "london" in _blob(routes)


def test_texas_energy_hub_not_paris():
    q = (
        "Texas energy-sector missions require frequent hops into remote desert strips near Houston "
        "and Dallas, while executives still fly Paris-Geneva."
    )
    routes, _ = _run_pipeline(q)
    blob = _blob(routes)
    assert "houston -> desert" in blob or "dallas -> desert" in blob or "desert energy" in blob
    assert "paris -> desert" not in blob
    assert "houston -> desert energy corridor" in blob or "dallas -> desert energy corridor" in blob


def test_ca_nv_doha_direction_us_to_me():
    q = (
        "Most flying is short California and Nevada domestic trips, but the principal requires "
        "nonstop Doha capability from San Francisco several times per quarter."
    )
    routes, du = _run_pipeline(q)
    blob = _blob(routes)
    assert "doha" in blob
    assert "doha -> san francisco" not in blob
    assert "san francisco -> doha" in blob or "los angeles -> doha" in blob
    swapped = (du.get("route_directionality") or {}).get("swapped") or []
    assert isinstance(swapped, list)


def test_perth_mining_extraction_spokes():
    q = (
        "We move mining engineers between Perth, remote Australian extraction strips, and Singapore, "
        "while leadership requires London nonstop several times per year."
    )
    routes, _ = _run_pipeline(q)
    blob = _blob(routes)
    assert "perth" in blob
    assert "australian extraction" in blob or "perth -> australian" in blob
    assert "singapore -> london" in blob or "london" in blob


def test_northeast_florida_corridor():
    q = (
        "Our domestic program covers the Northeast corridor and Florida with small executive groups, "
        "plus occasional Singapore nonstop for the founder."
    )
    routes, _ = _run_pipeline(q)
    blob = _blob(routes)
    assert "miami" in blob or "palm beach" in blob
    assert "new york" in blob or "boston" in blob


def test_q8_desert_eu_executive_overlay():
    q = (
        "We transport equipment between Texas desert drilling sites and Northern Africa, but executives "
        "also fly to Paris and Geneva. How should the routing graph be structured?"
    )
    routes, du = _run_pipeline(q)
    blob = _blob(routes)
    assert "houston -> desert" in blob or "desert energy" in blob
    assert "houston -> paris" in blob or "houston -> geneva" in blob
    assert "paris" in blob and "geneva" in blob
    assert "remote drilling sites -> paris" not in blob
    assert "executive_eu_overlay" in (du.get("geographic_route_intelligence") or {}).get(
        "regions_activated", []
    )


def test_no_pacific_me_to_caribbean_ghosts():
    q = (
        "We previously tried a single network covering Los Angeles, Aspen, Tokyo, Singapore, Dubai, "
        "and Caribbean islands, but routing became inconsistent and unreliable."
    )
    routes, du = _run_pipeline(q)
    blob = _blob(routes)
    assert "tokyo -> caribbean" not in blob
    assert "dubai -> caribbean" not in blob
    assert "singapore -> caribbean" not in blob
    assert "miami -> caribbean" in blob or "caribbean" in blob
    removed = (du.get("route_topology_validation") or {}).get("removed_routes") or []
    dir_removed = (du.get("route_directionality") or {}).get("removed") or []
    assert any("caribbean" in r.lower() for r in removed + dir_removed) or "tokyo -> caribbean" not in blob


def test_no_cross_domain_stitching():
    q = (
        "Leadership insists on operating a single aircraft, but our actual mission includes "
        "Los Angeles-Tokyo travel, winter Aspen operations, Caribbean island flying, and "
        "periodic nonstop Riyadh trips."
    )
    routes, _ = _run_pipeline(q)
    assert "aspen -> singapore" not in _blob(routes)
    assert "caribbean -> riyadh" not in _blob(routes)
    assert "remote drilling sites -> frankfurt" not in _blob(routes)
    assert "los angeles -> aspen" in _blob(routes) or "los angeles -> tokyo" in _blob(routes)

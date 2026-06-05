"""Route distance authority — no silent fallbacks."""

from services.mission.route_distance_authority import (
    mission_route_blocks_ranking,
    resolve_route_distance,
)


def test_teterboro_london_operational_override():
    r = resolve_route_distance("Teterboro -> London")
    assert r.source == "operational_override"
    assert r.distance_nm >= 3000
    assert r.is_verified


def test_unknown_route_unresolved():
    r = resolve_route_distance("Fictional City A -> Fictional City B")
    assert r.source == "unresolved"
    assert r.distance_nm == 0.0
    assert r.blocks_ranking


def test_no_1800_heuristic():
    r = resolve_route_distance("Randomville -> Anotherplace")
    assert r.distance_nm != 1800.0


def test_geodesic_dallas_aspen():
    r = resolve_route_distance("Dallas -> Aspen")
    assert r.is_verified
    assert 500 < r.distance_nm < 800


def test_dallas_not_resolved_as_los_angeles():
    from services.mission.route_distance_authority import _resolve_place_icao

    assert _resolve_place_icao("dallas") == "KDFW"
    assert _resolve_place_icao("dallas") != "KLAX"


def test_sfo_tokyo_london_does_not_block_ranking():
    blocks, resolutions = mission_route_blocks_ranking(
        ["San Francisco -> Tokyo", "San Francisco -> London"]
    )
    assert not blocks
    assert any(r.is_verified for r in resolutions)
    assert any(r.distance_nm >= 4000 for r in resolutions)


def test_dallas_london_geodesic():
    r = resolve_route_distance("Dallas -> London")
    assert r.source == "geodesic"
    assert 4000 <= r.distance_nm <= 4800


def test_denver_reykjavik_geodesic_not_catalog():
    r = resolve_route_distance("Denver -> Reykjavik")
    assert r.source == "geodesic"
    assert r.distance_nm >= 2800

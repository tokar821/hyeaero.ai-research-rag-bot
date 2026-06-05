"""Route realism validator and ultra-long corridor detection."""

from services.aircraft_feasibility.mission_context import mission_context_from_json
from services.aircraft_feasibility.route_realism_validator import (
    match_ultra_long_corridor,
    validate_route_realism,
)
from services.aircraft_feasibility import (
    VERDICT_NOT_A_FIT,
    assess_aircraft_hard_feasibility,
)


def test_nyc_dubai_ultra_long_corridor():
    cid = match_ultra_long_corridor("New York -> Dubai", 6200.0)
    assert cid == "nyc_dubai"


def test_la_london_corridor():
    cid = match_ultra_long_corridor("Los Angeles -> London", 5450.0)
    assert cid == "la_london"


def test_sfo_tokyo_corridor():
    cid = match_ultra_long_corridor("San Francisco -> Tokyo", 5100.0)
    assert cid == "sfo_tokyo"


def test_light_jet_rejected_nyc_dubai_nonstop():
    mission = {
        "origin": "New York",
        "destination": ["Dubai"],
        "nonstop_required": True,
        "passengers": 6,
        "international_ops": True,
    }
    assessment = assess_aircraft_hard_feasibility(mission, "Citation CJ2")
    assert not assessment.feasible
    assert assessment.fit_verdict == VERDICT_NOT_A_FIT
    assert any("light-jet" in r.lower() for r in assessment.verdict.rejection_reasons)


def test_light_jet_allowed_with_stop_required():
    mission = {
        "origin": "New York",
        "destination": ["Dubai"],
        "nonstop_required": False,
        "stop_required": True,
        "passengers": 6,
    }
    ctx = mission_context_from_json(mission)
    assert ctx.stop_required
    assessment = assess_aircraft_hard_feasibility(mission, "Citation CJ2")
    assert not any("light-jet" in r.lower() for r in assessment.verdict.rejection_reasons)


def test_practical_less_than_required_not_a_fit():
    mission = {
        "origin": "Los Angeles",
        "destination": ["London"],
        "nonstop_required": True,
        "transatlantic": True,
    }
    assessment = assess_aircraft_hard_feasibility(mission, "Challenger 350")
    assert not assessment.feasible
    assert assessment.fit_verdict == VERDICT_NOT_A_FIT


def test_route_realism_catalog_source():
    ctx = mission_context_from_json(
        {"origin": "Los Angeles", "destination": ["Miami"], "passengers": 6}
    )
    result = validate_route_realism(ctx)
    assert result.realistic
    assert result.distance_source in ("geodesic", "operational_override", "catalog")
    assert result.stage_distance_nm >= 1900

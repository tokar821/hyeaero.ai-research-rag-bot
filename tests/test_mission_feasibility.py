"""
Hard-feasibility mission evaluation — elimination before ranking.
"""

from __future__ import annotations

import pytest

from services.mission.feasibility_engine import (
    FeasibilityResult,
    compute_practical_range,
    filter_feasible_aircraft,
    peak_required_route_nm,
)
from services.mission.models import MissionProfile, PriorityLevel, Route
from services.recommendation.mission_ranker import rank_missions
from services.mission.adapters import mission_profile_to_state


def _profile(**kwargs) -> MissionProfile:
    return MissionProfile(**kwargs)


def test_compute_practical_range_uses_practical_not_brochure():
    from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

    spec = AIRCRAFT_PROFILES["Gulfstream G650"]
    available = compute_practical_range(spec, passengers=12, westbound=True, nbaa_reserves=True)
    assert available < float(spec["practical_nm"])
    assert available < float(spec["brochure_nm"])


def test_case1_15_pax_transatlantic_eliminates_super_midsize():
    profile = _profile(
        passengers=15,
        routes=[
            Route(origin="Dallas", destination="New York"),
            Route(origin="New York", destination="London"),
        ],
        nonstop_required=True,
        nbaa_reserve_required=True,
        international_ops=True,
    )
    results = filter_feasible_aircraft(
        profile,
        ["Challenger 350", "Gulfstream G280", "Praetor 600", "Global 7500", "Falcon 8X"],
    )
    assert not results["Challenger 350"].feasible
    assert not results["Gulfstream G280"].feasible
    assert not results["Praetor 600"].feasible
    assert any("Passenger" in r for r in results["Challenger 350"].elimination_reasons)


def test_case2_sfo_tokyo_winter_eliminates_longitude_and_praetor():
    profile = _profile(
        passengers=6,
        routes=[Route(origin="San Francisco", destination="Tokyo")],
        nonstop_required=True,
        westbound_sensitive=True,
        seasonal_note="winter_headwinds",
        nbaa_reserve_required=True,
    )
    assert peak_required_route_nm(profile) >= 5000
    results = filter_feasible_aircraft(
        profile,
        ["Challenger Longitude", "Praetor 600", "Global 7500", "Falcon 8X", "Gulfstream G650"],
    )
    assert not results["Challenger Longitude"].feasible
    assert not results["Praetor 600"].feasible
    assert results["Global 7500"].feasible


def test_case3_caribbean_short_field_eliminates_global_7500():
    profile = _profile(
        passengers=6,
        routes=[Route(origin="Miami", destination="Caribbean")],
        operating_cost_priority=PriorityLevel.HIGH,
        short_field_priority=PriorityLevel.HIGH,
        runway_priority=PriorityLevel.HIGH,
        international_ops=True,
    )
    results = filter_feasible_aircraft(
        profile,
        ["Pilatus PC-24", "Praetor 600", "Citation Latitude", "Global 7500"],
    )
    assert results["Pilatus PC-24"].feasible
    assert results["Citation Latitude"].feasible
    assert not results["Global 7500"].feasible
    assert any(
        "Short-field" in r or "runway" in r.lower()
        for r in results["Global 7500"].elimination_reasons
    )


def test_infeasible_never_top_recommendation():
    from services.pipeline.run_pipeline import run_advisory_pipeline

    result = run_advisory_pipeline(
        "8 passengers Miami to Caribbean short runway",
        mission_profile=_profile(
            passengers=8,
            routes=[Route(origin="Miami", destination="Caribbean")],
            short_field_priority=PriorityLevel.HIGH,
            operating_cost_priority=PriorityLevel.HIGH,
        ),
        max_results=8,
    )
    assert result.recommendations
    models = [r.model for r in result.recommendations]
    assert "Global 7500" not in models
    assert all(not r.avoid for r in result.recommendations)


def test_rank_missions_returns_feasibility_map():
    profile = _profile(
        passengers=6,
        routes=[Route(origin="San Francisco", destination="Tokyo")],
        westbound_sensitive=True,
        seasonal_note="winter_headwinds",
        nonstop_required=True,
    )
    mission = mission_profile_to_state(profile)
    _cat, recs, feas, _ = rank_missions(mission, mission_profile=profile)
    assert isinstance(feas, dict)
    assert len(feas) > 0
    assert not feas["Praetor 600"].feasible


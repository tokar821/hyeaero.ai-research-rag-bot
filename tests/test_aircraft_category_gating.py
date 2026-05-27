"""Mission category gating — pool restriction before scoring."""

from services.consultant.mission_state import MissionState
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.recommendation.aircraft_category_gating import (
    GatedMissionCategory,
    aircraft_catalog_category,
    apply_mission_category_gating,
    determine_gated_mission_category,
)
from services.recommendation.mission_ranker import rank_missions


def test_la_miami_light_jet_band():
    mission = MissionState(routes=["Los Angeles -> Miami"], passenger_count=6)
    gate = determine_gated_mission_category(mission)
    assert gate.category == GatedMissionCategory.LIGHT_JET
    assert gate.max_leg_nm < 2000


def test_eight_pax_bumps_to_super_midsize_on_short_leg():
    mission = MissionState(routes=["Los Angeles -> Miami"], passenger_count=8)
    gate = determine_gated_mission_category(mission)
    assert gate.category == GatedMissionCategory.SUPER_MIDSIZE


def test_eight_pax_caribbean_executive_super_mid_floor():
    mission = MissionState(routes=["Miami -> Caribbean"], passenger_count=8)
    gate = determine_gated_mission_category(mission)
    assert gate.category == GatedMissionCategory.SUPER_MIDSIZE


def test_nyc_london_large_cabin():
    mission = MissionState(routes=["New York -> London"], passenger_count=8, nonstop_requirement=True)
    gate = determine_gated_mission_category(mission)
    assert gate.category == GatedMissionCategory.LARGE_CABIN


def test_sfo_tokyo_ultra_long_range():
    mission = MissionState(
        routes=["San Francisco -> Tokyo"],
        passenger_count=6,
        nonstop_requirement=True,
        westbound=True,
    )
    gate = determine_gated_mission_category(mission)
    assert gate.category == GatedMissionCategory.ULTRA_LONG_RANGE


def test_nyc_dubai_ultra_long_range():
    mission = MissionState(routes=["New York -> Dubai"], passenger_count=6, nonstop_requirement=True)
    gate = determine_gated_mission_category(mission)
    assert gate.category == GatedMissionCategory.ULTRA_LONG_RANGE


def test_turboprop_excluded_from_ulr_pool():
    mission = MissionState(
        routes=["San Francisco -> Tokyo"],
        passenger_count=6,
        nonstop_requirement=True,
        westbound=True,
    )
    all_models = list(AIRCRAFT_PROFILES.keys())
    gate = apply_mission_category_gating(mission, all_models)
    assert "Pilatus PC-12" not in gate.candidate_models
    assert any("Pilatus" in e["aircraft_name"] for e in gate.exclusion_log)


def test_ulr_pool_only_ultra_long_catalog():
    mission = MissionState(
        routes=["San Francisco -> Tokyo"],
        passenger_count=6,
        nonstop_requirement=True,
        westbound=True,
    )
    gate = apply_mission_category_gating(mission, list(AIRCRAFT_PROFILES.keys()))
    for model in gate.candidate_models:
        assert aircraft_catalog_category(model) == "ultra-long"


def test_coast_to_coast_super_midsize_pool():
    mission = MissionState(routes=["New York -> Los Angeles"], passenger_count=7)
    gate = determine_gated_mission_category(mission)
    assert gate.category == GatedMissionCategory.SUPER_MIDSIZE
    gated = apply_mission_category_gating(mission, ["Challenger 350", "Citation CJ2", "Global 7500"])
    assert "Challenger 350" in gated.candidate_models
    assert "Citation CJ2" not in gated.candidate_models
    assert "Global 7500" not in gated.candidate_models


def test_rank_missions_only_scores_in_band():
    mission = MissionState(routes=["Boston -> Miami"], passenger_count=6, nonstop_requirement=True)
    category, recs, _, _audit = rank_missions(mission, max_results=3)
    assert recs
    for r in recs:
        if r.avoid:
            continue
        assert aircraft_catalog_category(r.model) == "light"
    assert category.value == "regional_utility"


def test_twelve_pax_bumps_large_cabin():
    mission = MissionState(
        routes=["New York -> London"],
        passenger_count=12,
        nonstop_requirement=True,
    )
    gate = determine_gated_mission_category(mission)
    assert gate.category == GatedMissionCategory.LARGE_CABIN

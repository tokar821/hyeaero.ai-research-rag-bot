"""Weighted aircraft scoring — dimensions, penalties, fit verdicts."""

from services.consultant.mission_state import MissionState
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.recommendation.mission_ranker import MissionCategory, score_aircraft_for_mission_ranked
from services.recommendation.weighted_aircraft_scoring import (
    VERDICT_BEST_FIT,
    VERDICT_CONDITIONAL_FIT,
    VERDICT_NOT_A_FIT,
    score_aircraft_weighted,
)


def test_nine_scoring_dimensions_present():
    mission = MissionState(routes=["Boston -> Miami"], passenger_count=6)
    result = score_aircraft_weighted(
        "Citation CJ4",
        AIRCRAFT_PROFILES["Citation CJ4"],
        mission,
        mission_category=MissionCategory.REGIONAL_UTILITY,
    )
    dims = {s.dimension for s in result.dimension_scores}
    assert dims == {
        "range_realism",
        "runway_performance",
        "dispatch_reliability",
        "operating_economics",
        "cabin_comfort",
        "baggage",
        "resale_liquidity",
        "maintenance_network",
        "passenger_count_fit",
    }


def test_ulr_beats_light_on_sfo_tokyo():
    mission = MissionState(
        routes=["San Francisco -> Tokyo"],
        passenger_count=6,
        nonstop_requirement=True,
        westbound=True,
    )
    ulr = score_aircraft_weighted(
        "Global 7500",
        AIRCRAFT_PROFILES["Global 7500"],
        mission,
        mission_category=MissionCategory.ULTRA_LONG_RANGE,
    )
    light = score_aircraft_weighted(
        "Citation CJ2",
        AIRCRAFT_PROFILES["Citation CJ2"],
        mission,
        mission_category=MissionCategory.ULTRA_LONG_RANGE,
    )
    assert ulr.total_score > light.total_score
    assert any(p.key == "insufficient_range" for p in light.penalties)


def test_overkill_penalty_on_regional_ulr():
    mission = MissionState(routes=["Miami -> Caribbean"], passenger_count=6)
    falcon = score_aircraft_weighted(
        "Falcon 8X",
        AIRCRAFT_PROFILES["Falcon 8X"],
        mission,
        mission_category=MissionCategory.REGIONAL_UTILITY,
    )
    assert any(p.key == "overkill_aircraft" for p in falcon.penalties)
    assert falcon.fit_verdict in (VERDICT_CONDITIONAL_FIT, VERDICT_NOT_A_FIT)


def test_fit_explanation_and_tradeoffs_populated():
    mission = MissionState(routes=["New York -> Los Angeles"], passenger_count=7)
    result = score_aircraft_weighted(
        "Challenger 350",
        AIRCRAFT_PROFILES["Challenger 350"],
        mission,
        mission_category=MissionCategory.COAST_TO_COAST,
    )
    assert result.fit_explanation
    assert result.total_score > 0
    assert result.fit_verdict in (VERDICT_BEST_FIT, VERDICT_CONDITIONAL_FIT, VERDICT_NOT_A_FIT)


def test_ranker_emits_fit_verdict_and_legacy_dimensions():
    mission = MissionState(routes=["Boston -> Miami"], passenger_count=6, nonstop_requirement=True)
    rec = score_aircraft_for_mission_ranked(
        "Pilatus PC-24",
        AIRCRAFT_PROFILES["Pilatus PC-24"],
        mission,
        mission_category=MissionCategory.REGIONAL_UTILITY,
    )
    assert rec.fit_verdict in (VERDICT_BEST_FIT, VERDICT_CONDITIONAL_FIT, VERDICT_NOT_A_FIT)
    dims = {s.dimension for s in rec.scores}
    assert "route_realism" in dims
    assert "overbuying_penalty" in dims
    assert rec.explanation
    assert rec.explanation.summary

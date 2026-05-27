"""Modern operator archetype weighting in mission ranking."""

from services.consultant.mission_state import build_mission_from_current_turn
from services.consultant.recommendation_engine import rank_aircraft_recommendations
from services.recommendation.aircraft_archetype_weighting import (
    OperatorArchetype,
    modern_operational_fit_score,
    operator_archetype_for_model,
)
from services.recommendation.mission_ranker import rank_missions


def test_archetype_classification():
    assert operator_archetype_for_model("Challenger 350") == OperatorArchetype.MODERN_OPERATOR_PREFERRED
    assert operator_archetype_for_model("Falcon 2000") == OperatorArchetype.LEGACY_COMMON
    assert operator_archetype_for_model("Gulfstream G650") == OperatorArchetype.MODERN_FLAGSHIP


def test_modern_scores_above_legacy():
    assert modern_operational_fit_score("Challenger 350") > modern_operational_fit_score("Falcon 2000")
    assert modern_operational_fit_score("Praetor 600") > modern_operational_fit_score("Citation CJ2")
    assert modern_operational_fit_score("Learjet 75") < 0.55


def test_la_miami_prefers_modern_super_midsize_over_falcon_2000():
    mission = build_mission_from_current_turn("8 pax LA to Miami nonstop $10M recommend")
    recs = rank_aircraft_recommendations(mission, max_results=8)
    models = [r.model for r in recs if not r.avoid]
    assert models[0] in (
        "Challenger 350",
        "Praetor 600",
        "Citation Latitude",
        "Gulfstream G280",
        "Challenger Longitude",
        "Challenger 650",
    )
    assert "Falcon 2000" not in models[:3]
    top = recs[0]
    dims = {s.dimension for s in top.scores}
    assert "modern_operational_fit" in dims


def test_tokyo_still_surfaces_ulr_flagships():
    mission = build_mission_from_current_turn(
        "6 executives San Francisco to Tokyo nonstop westbound"
    )
    recs = rank_aircraft_recommendations(mission, max_results=5)
    top3 = [r.model for r in recs if not r.avoid][:3]
    assert any(m in top3 for m in ("Gulfstream G650", "Falcon 8X", "Global 7500"))


def test_rank_missions_includes_modern_dimension_on_legacy():
    mission = build_mission_from_current_turn("6 pax Boston to Miami nonstop")
    _, recs, _, _ = rank_missions(mission, max_results=12)
    legacy = next((r for r in recs if r.model == "Learjet 75"), None)
    assert legacy is not None
    mod = next((s for s in legacy.scores if s.dimension == "modern_operational_fit"), None)
    assert mod is not None
    assert mod.score < 0.6

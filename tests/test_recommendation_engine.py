"""Aircraft recommendation engine tests."""

from services.consultant.mission_state import build_mission_from_current_turn
from services.consultant.recommendation_engine import rank_aircraft_recommendations


def test_la_miami_10m_budget_ranks_super_midsize_not_only_challenger():
    mission = build_mission_from_current_turn(
        "8 passengers LA to Miami nonstop around $10M budget",
    )
    recs = rank_aircraft_recommendations(mission, max_results=6)
    models = [r.model for r in recs if not r.avoid]
    assert len(models) >= 3
    assert any(m in models for m in ("Citation Latitude", "Praetor 600", "Gulfstream G280"))
    # Challenger 350 may rank well but must not be the only type returned
    assert models[0]  # has a leader
    assert len(set(models)) >= 3


def test_europe_west_coast_favors_long_range():
    mission = build_mission_from_current_turn(
        "West coast to Europe westbound winter, 12 passengers, nonstop required",
    )
    recs = rank_aircraft_recommendations(mission, max_results=5)
    top_models = [r.model for r in recs[:3]]
    assert any(m in top_models for m in ("Falcon 8X", "Gulfstream G650", "Global 7500", "Falcon 7X"))


def test_avoid_light_jet_on_long_westbound():
    mission = build_mission_from_current_turn("SFO to Paris nonstop westbound 8 pax")
    recs = rank_aircraft_recommendations(mission, max_results=8)
    top_models = [r.model for r in recs if not r.avoid]
    assert "Citation CJ2" not in top_models
    cj2 = next((r for r in recs if r.model == "Citation CJ2"), None)
    if cj2 is not None:
        assert cj2.avoid or cj2.fit in ("Partial Fit", "Not Recommended", "Good Fit")


def test_recommendation_has_scoring_dimensions():
    mission = build_mission_from_current_turn("Recommend best jet for 6 pax transcon")
    recs = rank_aircraft_recommendations(mission, max_results=3)
    assert recs[0].scores
    dims = {s.dimension for s in recs[0].scores}
    assert "route_realism" in dims
    assert "overbuying_penalty" in dims
    assert recs[0].fit in ("Strong Fit", "Good Fit", "Partial Fit", "Not Recommended")

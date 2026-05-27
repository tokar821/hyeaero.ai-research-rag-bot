"""Comparative analysis engine tests."""

from services.consultant.comparison_engine import build_structured_comparison
from services.consultant.mission_state import build_mission_from_current_turn
from services.consultant.recommendation_engine import rank_aircraft_recommendations


def test_falcon_vs_challenger_comparison_table():
    mission = build_mission_from_current_turn("Falcon 2000 vs Challenger 350 for US missions")
    recs = rank_aircraft_recommendations(mission, max_results=4)
    cmp = build_structured_comparison(
        ["Falcon 2000", "Challenger 350"],
        mission,
        recommendations=recs,
    )
    assert len(cmp.rows) == 2
    assert "| Model |" in cmp.markdown_table
    assert "Mission fit" in cmp.markdown_table
    assert "mission_score" not in cmp.markdown_table.lower()
    assert cmp.rows[0].mission_fit in ("Strong Fit", "Good Fit", "Partial Fit", "Not Recommended")
    cards = cmp.visual_normalized.get("comparison_cards") or []
    assert cards and "mission_fit" in cards[0]
    assert "headline_score" not in cards[0]


def test_cabin_comparison_includes_tradeoffs():
    mission = build_mission_from_current_turn(
        "Compare cabin comfort Gulfstream G650 vs Global 7500",
    )
    cmp = build_structured_comparison(["Gulfstream G650", "Global 7500"], mission)
    assert cmp.operational_tradeoffs or cmp.acquisition_vs_operating
    assert cmp.json_schema["comparison_type"] == "mission_fit_table"


def test_fractional_ownership_note():
    mission = build_mission_from_current_turn("Fractional vs full ownership for 300 hours")
    mission.acquisition_strategy = "fractional"
    cmp = build_structured_comparison(["Challenger 350", "Citation Latitude"], mission)
    assert "fractional" in cmp.acquisition_vs_operating.lower()

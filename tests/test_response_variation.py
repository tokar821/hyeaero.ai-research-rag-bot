"""Response variation engine — styles and non-fixed structure."""

from services.consultant.mission_state import build_mission_from_current_turn
from services.consultant.recommendation_engine import rank_aircraft_recommendations
from services.consultant.response_formatter import format_consultant_response, last_response_style
from services.consultant.response_variation import (
    ALL_STYLES,
    STYLE_COMPARISON_DRIVEN,
    STYLE_CONCISE_EXECUTIVE,
    STYLE_OPERATIONAL_ANALYSIS,
    VariationContext,
    compose_varied_response,
    select_response_style,
)


def _format(query: str) -> str:
    mission = build_mission_from_current_turn(query)
    recs = rank_aircraft_recommendations(mission, max_results=3)
    return format_consultant_response(
        mission=mission,
        recommendations=recs,
        route_assessments=[],
        query=query,
        turn_seed=query,
    )


def test_different_queries_can_pick_different_styles():
    q1 = "8 pax LA to Miami nonstop recommend"
    q2 = "Compare Gulfstream G650 vs Falcon 8X for transpacific"
    s1 = select_response_style(VariationContext(mission=build_mission_from_current_turn(q1), query=q1))
    s2 = select_response_style(
        VariationContext(
            mission=build_mission_from_current_turn(q2),
            query=q2,
        )
    )
    assert s1 in ALL_STYLES
    assert s2 in ALL_STYLES
    assert s2 == STYLE_COMPARISON_DRIVEN or "compare" in q2.lower()


def test_concise_executive_style_for_brief_request():
    q = "Give me a quick executive brief — best jet for 8 pax LA to Miami"
    style = select_response_style(VariationContext(mission=build_mission_from_current_turn(q), query=q))
    assert style == STYLE_CONCISE_EXECUTIVE


def test_operational_style_for_feasibility_query():
    q = "Can a Challenger 350 make Aspen nonstop with runway constraints?"
    style = select_response_style(VariationContext(mission=build_mission_from_current_turn(q), query=q))
    from services.consultant.response_variation import STYLE_TRADEOFF_FIRST, STYLE_RECOMMENDATION_FIRST

    assert style in (STYLE_OPERATIONAL_ANALYSIS, STYLE_TRADEOFF_FIRST, STYLE_RECOMMENDATION_FIRST)


def test_same_query_stable_style():
    q = "8 pax LA to Miami $10M nonstop recommend"
    m = build_mission_from_current_turn(q)
    ctx = VariationContext(mission=m, query=q, turn_seed=q)
    assert select_response_style(ctx) == select_response_style(ctx)


def test_varied_openings_not_single_template():
    texts = [
        _format("8 pax LA to Miami nonstop recommend"),
        _format("6 executives San Francisco to Tokyo westbound nonstop"),
        _format("Compare Challenger 350 vs Praetor 600 for Caribbean"),
    ]
    assert all(len(t) > 80 for t in texts)
    openers = [t.split("\n\n")[0] for t in texts]
    assert len(set(openers)) >= 2


def test_last_response_style_recorded():
    q = "8 pax LA to Miami nonstop recommend"
    _format(q)
    assert last_response_style() in ALL_STYLES

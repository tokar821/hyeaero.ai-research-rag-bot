"""Visualization turns render from entities — no spurious follow-ups."""

from services.consultant.intelligence_engine import run_consultant_intelligence_layer
from services.consultant.mission_state import MissionState
from services.consultant.visualization_handler import (
    VisualizationKind,
    classify_visualization_kind,
    extract_visualization_entities,
    run_visualization_turn,
)
from services.recommendation.clarification_decision import mission_clarification_needs
from services.recommendation.query_recommendation_intent import (
    QueryRecommendationIntent,
    classify_query_recommendation_intent,
    is_visualization_query,
)


def test_visualization_query_patterns():
    for q in (
        "Visualize range from SFO for Gulfstream G650",
        "Show me a map from New York to London for Falcon 8X",
        "reachable cities from Miami for Praetor 600",
        "range map SFO to Paris Falcon 8X",
        "compare layouts Challenger 350 vs Praetor 600",
        "cabin graphic for Global 7500",
    ):
        assert is_visualization_query(q), q
        assert classify_query_recommendation_intent(q).intent == QueryRecommendationIntent.VISUALIZATION_REQUEST


def test_range_map_with_route_and_aircraft_no_followup():
    q = "Show range map SFO to Paris for Falcon 8X"
    result = run_visualization_turn(q, mission=MissionState())
    assert not result.followup_needed
    assert result.kind == VisualizationKind.RANGE_MAP
    assert result.entities.origin_label
    assert result.bundle.range_maps or result.bundle.mission_reachability


def test_reachable_cities_from_origin_no_passenger_followup():
    q = "Visualize reachable cities from Dallas for Challenger 350"
    needs = mission_clarification_needs(MissionState(), q)
    assert not needs.any
    result = run_visualization_turn(q, mission=MissionState())
    assert not result.followup_needed
    assert result.entities.origin_label


def test_cabin_graphic_missing_aircraft_asks_once():
    q = "Show cabin graphic"
    result = run_visualization_turn(q, mission=MissionState())
    assert result.followup_needed
    assert "aircraft" in result.followup_message.lower()


def test_cabin_graphic_with_aircraft_direct():
    q = "cabin graphic for Global 7500"
    result = run_visualization_turn(q, mission=MissionState())
    assert not result.followup_needed
    assert result.bundle.cabin_layouts


def test_range_map_missing_origin_asks_once():
    q = "Show range map for Gulfstream G650"
    result = run_visualization_turn(q, mission=MissionState())
    assert result.followup_needed
    assert "anchor" in result.followup_message.lower() or "from" in result.followup_message.lower()


def test_entities_from_history():
    history = [{"role": "user", "content": "Tell me about the Falcon 8X"}]
    entities = extract_visualization_entities(
        "show cabin graphic",
        MissionState(),
        history=history,
    )
    assert "Falcon 8X" in entities.aircraft_models or entities.aircraft_models


def test_visualization_render_includes_summary_and_svg():
    from services.consultant.visualization_render import format_visualization_user_response

    q = "range map SFO to Paris for Falcon 8X"
    viz = run_visualization_turn(q, mission=MissionState())
    text, patch = format_visualization_user_response(viz)
    assert not viz.followup_needed
    assert "Range" in text or "Falcon" in text
    assert patch.get("consultant_visualization_rendered") == 1
    assert patch.get("consultant_visualization_svg") or patch.get("consultant_visual_models")


def test_intelligence_layer_visualization_direct():
    result = run_consultant_intelligence_layer(
        answer="Draft.",
        query="range map SFO to Paris for Falcon 8X",
        history=None,
        data_used={},
    )
    assert result.data_used_patch.get("consultant_structured_formatter") == "visualization_direct"
    visuals = result.data_used_patch.get("consultant_visual_models") or {}
    assert visuals.get("range_maps") or visuals.get("mission_reachability")
    assert "passenger" not in result.answer.lower()
    assert "budget" not in result.answer.lower()
    assert result.data_used_patch.get("consultant_visualization_rendered") == 1

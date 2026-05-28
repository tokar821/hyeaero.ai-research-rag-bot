"""Final orchestration response mode + recommendation gate regression tests."""

from __future__ import annotations

import re

import pytest

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.orchestration.hierarchy_weighting import detect_dominant_mission
from services.orchestration.recommendation_gate import (
    evaluate_recommendation_gate,
    render_interpretation_first_response,
    strip_aircraft_from_response,
)
from services.orchestration.response_mode_classifier import (
    OrchestrationResponseMode,
    classify_orchestration_response_mode,
    explicit_aircraft_request,
)

_AIRCRAFT_RE = re.compile(
    r"\b(?:gulfstream|global\s+\d+|falcon\s+\d+|g650|g700|citation|phenom)\b",
    re.I,
)

_INTERPRETATION_QUERIES = [
    "How should this network be interpreted?",
    "What structure fits this mission?",
    "Is this coherent?",
    "How should continuation hubs be represented?",
    "What operational domains exist?",
    "What actually dominates utilization?",
]

_RECOMMENDATION_QUERIES = [
    "Which aircraft should we buy for this mission?",
    "Recommend a jet that fits NYC to Singapore nonstop.",
    "What jet fits our West Coast corridor?",
]


@pytest.mark.parametrize("query", _INTERPRETATION_QUERIES)
def test_interpretation_queries_classify_without_aircraft_mode(query: str):
    result = classify_orchestration_response_mode(query)
    assert result.suppresses_aircraft_recommendations
    assert result.mode in (
        OrchestrationResponseMode.INTERPRETATION_MODE,
        OrchestrationResponseMode.STRUCTURE_MODE,
    )
    assert not explicit_aircraft_request(query)


@pytest.mark.parametrize("query", _RECOMMENDATION_QUERIES)
def test_explicit_recommendation_queries_allow_aircraft(query: str):
    result = classify_orchestration_response_mode(query)
    assert not result.suppresses_aircraft_recommendations
    assert result.mode in (
        OrchestrationResponseMode.RECOMMENDATION_MODE,
        OrchestrationResponseMode.BUY_DECISION_MODE,
    )


def test_hierarchy_weighting_domestic_dominates_ulr_overlay():
    q = (
        "70% of annual hours are West Coast corridor flights. "
        "Executives occasionally fly to Singapore via Dubai."
    )
    pkt = MissionUnderstandingPacket(
        inferred_constraints={
            "domestic_utilization_dominates_except_founder_ulr": True,
            "continuation_hubs_semantic_only_not_primary_origin": True,
            "operational_priority_order": [
                "domestic corridor utilization",
                "Pacific ULR overlay",
                "Dubai continuation constraint",
            ],
        },
        travel_pattern="domestic corridor",
    )
    hierarchy = detect_dominant_mission(pkt, query=q)
    assert "domestic" in hierarchy.dominant_utilization.lower() or "corridor" in hierarchy.dominant_utilization.lower()
    assert hierarchy.continuation_hub_discipline
    assert any("70%" in n for n in hierarchy.weighting_notes)


def test_hierarchy_ny_boston_chicago_domestic_executive():
    q = "Most flying is NY/Boston/Chicago. Executives occasionally continue to Singapore."
    pkt = MissionUnderstandingPacket(
        inferred_constraints={"domestic_utilization_dominant": True},
        travel_pattern="domestic executive network",
    )
    hierarchy = detect_dominant_mission(pkt, query=q)
    assert "domestic" in hierarchy.dominant_utilization.lower()
    assert hierarchy.executive_exceptions


def test_recommendation_gate_suppresses_generic_dump():
    recs = [
        AircraftRecommendation(model="Global 7500", category="ulr", total_score=0.9, confidence=0.8, rank=1),
        AircraftRecommendation(model="G650ER", category="ulr", total_score=0.85, confidence=0.75, rank=2),
        AircraftRecommendation(model="Falcon 8X", category="ulr", total_score=0.8, confidence=0.7, rank=3),
    ]
    gate = evaluate_recommendation_gate(
        "How should this network be interpreted?",
        recs,
        packet=MissionUnderstandingPacket(recommend_aircraft=True),
    )
    assert gate.suppress_aircraft
    assert gate.render_interpretation_only
    assert len(gate.filtered_recommendations) == 0


def test_render_interpretation_first_no_aircraft_models():
    mission = MissionState(
        routes=["New York -> Chicago", "New York -> Dubai"],
        passenger_count=4,
    )
    pkt = MissionUnderstandingPacket(
        operational_synthesis="Domestic corridor dominates; Dubai is continuation only.",
        inferred_constraints={
            "domestic_utilization_dominant": True,
            "incompatible_mission_bands": True,
            "defer_global_shortlist": True,
        },
        fallback_operational_band=[
            "Domestic field-access executive band",
            "Middle East ULR continuation band",
        ],
        recommend_aircraft=False,
    )
    out = render_interpretation_first_response(
        mission,
        pkt,
        query="How should this network be interpreted?",
    )
    assert not _AIRCRAFT_RE.search(out)
    assert "Operational Interpretation" in out or "Operational Structure" in out
    assert "Utilization Hierarchy" in out
    assert "structurally" in out.lower() or "incompatible" in out.lower() or "decomposition" in out.lower()


def test_strip_aircraft_from_response_removes_models():
    text = "You should consider a Global 7500 or G650 for this route."
    cleaned = strip_aircraft_from_response(text)
    assert "Global 7500" not in cleaned
    assert "G650" not in cleaned


def test_structural_coherence_query_is_structure_mode():
    result = classify_orchestration_response_mode("Is this structurally coherent?")
    assert result.suppresses_aircraft_recommendations
    assert result.mode == OrchestrationResponseMode.STRUCTURE_MODE

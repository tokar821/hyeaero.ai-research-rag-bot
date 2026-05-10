from __future__ import annotations

from rag.consultant_conversation_state import (
    finalize_consultant_conversation_state,
    merge_consultant_conversation_state,
)


def test_deictic_carries_aircraft_from_prior_state():
    prev = {
        "user_style": None,
        "current_aircraft_reference": "Gulfstream G650",
        "current_visual_intent": "cabin interior",
        "current_budget": None,
        "current_mission": None,
        "current_passenger_count": None,
        "current_cabin_preference": None,
        "conversation_mode": "browsing",
    }
    out = merge_consultant_conversation_state(
        prev,
        query="something nicer and more modern — like a hotel",
        history=None,
        entity_models=[],
        hybrid_kind="aviation_mission",
        fine_intent="aircraft_recommendation",
        response_mode="visual_mode",
        user_wants_gallery=True,
    )
    assert out.get("current_aircraft_reference") == "Gulfstream G650"
    assert out.get("current_visual_intent")


def test_explicit_reset_clears_state():
    prev = {
        "current_aircraft_reference": "Citation X",
        "current_budget": "$8M",
        "user_style": "luxury-focused",
        "current_visual_intent": None,
        "current_mission": None,
        "current_passenger_count": None,
        "current_cabin_preference": None,
        "conversation_mode": "shopping",
    }
    out = merge_consultant_conversation_state(
        prev,
        query="New question — forget that. What is a turboprop?",
        history=None,
        entity_models=[],
    )
    assert out.get("current_aircraft_reference") is None
    assert out.get("current_budget") is None


def test_finalize_mutates_data_used():
    du: dict = {"tavily_results": 0}
    st = finalize_consultant_conversation_state(
        du,
        None,
        query="Show me Falcon 7X cockpit",
        history=None,
        entity_models=["Falcon 7X"],
        hybrid_kind="aviation_mission",
        fine_intent="aircraft_specs",
        response_mode="visual_mode",
        user_wants_gallery=True,
    )
    assert du.get("consultant_conversation_state") == st
    assert st.get("current_aircraft_reference") == "Falcon 7X"
    assert st.get("conversation_mode") in ("browsing", "comparing", "shopping", "aspirational", "deal-analysis")

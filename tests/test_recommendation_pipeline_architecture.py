"""Deterministic pipeline is decision authority; LLM does not invent aircraft."""

from services.consultant.intelligence_engine import run_consultant_intelligence_layer
from services.consultant.llm_explanation_layer import (
    build_pipeline_authority_block,
    intents_requiring_deterministic_pipeline,
)
from services.consultant.mission_state import build_mission_from_current_turn
from services.consultant.pre_llm_recommendation import should_run_pre_llm_pipeline
from services.consultant.recommendation_authority import (
    detect_unauthorized_aircraft,
    reconcile_answer_with_pipeline,
)
from services.consultant.recommendation_engine import rank_aircraft_recommendations
from services.recommendation.recommendation_pipeline import (
    DECISION_SOURCE,
    PIPELINE_STAGES,
    run_recommendation_pipeline,
)


def test_pipeline_stages_and_decision_source():
    mission_q = "8 pax LA to Miami nonstop $10M recommend"
    result, trace = run_recommendation_pipeline(mission_q)
    assert trace.decision_source == DECISION_SOURCE
    assert "mission_extraction" in PIPELINE_STAGES
    assert len(result.recommendations) >= 1
    assert result.recommendations[0].model


def test_authority_block_lists_ranked_only():
    result, _ = run_recommendation_pipeline("8 pax LA to Miami nonstop recommend")
    block = build_pipeline_authority_block(result)
    assert "BROKER ADVISORY CONTEXT" in block
    assert "FEASIBLE AIRCRAFT" in block
    assert result.recommendations[0].model in block
    assert "total_score" not in block.lower()
    assert "deterministic" in block.lower()


def test_reconcile_strips_llm_invented_models():
    mission = build_mission_from_current_turn("8 pax LA to Miami nonstop")
    recs = rank_aircraft_recommendations(mission, max_results=3)
    allowed = {r.model for r in recs}
    bad_llm = (
        "You should buy a Citation CJ2 for this trip.\n\n"
        "Also consider a Learjet 75."
    )
    assert detect_unauthorized_aircraft(bad_llm, allowed)
    fixed, regen = reconcile_answer_with_pipeline(
        bad_llm,
        mission=mission,
        recommendations=recs,
        query="8 pax LA to Miami",
    )
    assert regen
    assert "Citation CJ2" not in fixed or fixed.index("Challenger") >= 0


def test_intelligence_enforces_pipeline_body():
    import os

    os.environ["CONSULTANT_INTELLIGENCE_LAYER"] = "1"
    llm_draft = (
        "The Citation CJ2 is perfect for LA to Miami with 8 passengers. "
        "Also try Learjet 75 and Phenom 300."
    )
    from services.state.mission_state import sync_persistent_mission_state

    history = [{"role": "user", "content": "8 passengers LA to Miami $10M nonstop"}]
    data_used = {"consultant_response_mode": "mission_advisory"}
    sync_persistent_mission_state(history[0]["content"], data_used=data_used)
    out = run_consultant_intelligence_layer(
        answer=llm_draft,
        query="What aircraft do you recommend?",
        history=history,
        data_used=data_used,
    )
    assert out.data_used_patch.get("recommendation_decision_source") == DECISION_SOURCE
    assert out.data_used_patch.get("pipeline_authority_enforced")
    assert "Citation CJ2" not in out.answer
    assert any(m in out.answer for m in ("Challenger", "Praetor", "G280", "Latitude"))


def test_pre_llm_intent_gate():
    assert should_run_pre_llm_pipeline("aircraft_recommendation", "recommend a jet")
    assert not should_run_pre_llm_pipeline(
        "aircraft_recommendation",
        "What aircraft would you avoid?",
        query_intent="aircraft_critique",
    )
    assert intents_requiring_deterministic_pipeline("aviation_mission")

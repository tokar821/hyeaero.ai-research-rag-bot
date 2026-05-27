"""Consultant orchestration — stage order, tracing, fail-safe."""

import os

from services.orchestration.constants import (
    DECISION_SOURCE,
    LOW_CONFIDENCE_GUIDANCE_PREFIX,
    ORCHESTRATION_STAGES,
)
from services.orchestration.fail_safe import apply_low_confidence_guidance
from services.orchestration.modes import OrchestrationMode, orchestration_mode
from services.orchestration.pipeline_orchestrator import (
    run_consultant_orchestration,
    run_deterministic_stages,
)


def test_orchestration_stage_order():
    assert ORCHESTRATION_STAGES[0] == "mission_extraction"
    assert ORCHESTRATION_STAGES[-1] == "final_response_formatting"
    assert len(ORCHESTRATION_STAGES) == 7


def test_deterministic_stages_produce_ranked_models():
    result, trace = run_deterministic_stages("8 pax LA to Miami nonstop recommend")
    assert trace.decision_source == DECISION_SOURCE
    assert "mission_extraction" in trace.completed_stage_names()
    assert "recommendation_ranking" in trace.completed_stage_names()
    assert len(result.recommendations) >= 1


def test_full_orchestration_mission_advisory():
    orch = run_consultant_orchestration("8 pax LA to Miami nonstop recommend")
    assert orch.answer.strip()
    assert len(orch.recommendations) >= 1
    assert orch.recommendations[0].model in orch.answer
    completed = orch.trace.completed_stage_names()
    assert "broker_narrative_generation" in completed
    assert "final_response_formatting" in completed
    assert orch.data_used_patch.get("recommendation_decision_source") == DECISION_SOURCE


def test_route_missing_skips_ranking_stages():
    result, trace = run_deterministic_stages("recommend a business jet")
    assert not result.recommendations
    names = trace.completed_stage_names()
    assert "mission_extraction" in names
    assert "recommendation_ranking" not in names


def test_low_confidence_prefix():
    text, low = apply_low_confidence_guidance("Short answer.", 0.4)
    assert low
    assert LOW_CONFIDENCE_GUIDANCE_PREFIX in text


def test_debug_mode_env(monkeypatch):
    monkeypatch.setenv("CONSULTANT_ORCHESTRATION_MODE", "debug")
    assert orchestration_mode() == OrchestrationMode.DEBUG


def test_intelligence_uses_orchestration(monkeypatch):
    monkeypatch.setenv("CONSULTANT_INTELLIGENCE_LAYER", "1")
    monkeypatch.setenv("CONSULTANT_ORCHESTRATION", "1")
    from services.consultant.intelligence_engine import run_consultant_intelligence_layer
    from services.state.mission_state import sync_persistent_mission_state

    history = [{"role": "user", "content": "8 passengers LA to Miami $10M nonstop"}]
    data_used = {"consultant_response_mode": "mission_advisory"}
    sync_persistent_mission_state(history[0]["content"], data_used=data_used)
    llm_draft = "Citation CJ2 is perfect for this trip."
    out = run_consultant_intelligence_layer(
        answer=llm_draft,
        query="What aircraft do you recommend?",
        history=history,
        data_used=data_used,
    )
    assert out.data_used_patch.get("orchestration")
    assert "Citation CJ2" not in out.answer
    assert out.data_used_patch.get("pipeline_authority_enforced")

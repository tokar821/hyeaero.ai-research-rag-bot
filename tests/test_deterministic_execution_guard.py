"""Phase 15 — deterministic execution guard tests."""

from __future__ import annotations

import re

import pytest

from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.authority_dispatch import consult_authority_dispatch
from services.routing.deterministic_execution_assertion import (
    DeterministicExecutionViolation,
    assert_no_llm_leak,
    deterministic_assertion_enabled,
)
from services.routing.deterministic_execution_guard import (
    build_deterministic_guard_context,
    query_requires_hard_deterministic_pipeline,
    requires_hard_deterministic_responder,
    resolve_deterministic_bypass_response,
    should_bypass_llm_execution,
)
from services.routing.unified_intent_router import classify_unified_intent

pytestmark = pytest.mark.deterministic

_KERNEL = re.compile(r"\boperational\s+synthesis\b", re.I)


def _ctx(query: str, **overrides):
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    patch = dict(overrides.pop("pre_llm_pipeline_patch", {}))
    dispatch = consult_authority_dispatch(
        query,
        qri=qri,
        unified_route=route,
        context={"db": None, "pre_llm_pipeline_patch": patch},
    )
    base = build_deterministic_guard_context(
        query=query,
        qri=qri,
        unified_route=route,
        authority_dispatch_result=dispatch,
        pre_llm_pipeline_patch=patch,
        pipeline_authority_block=overrides.pop("pipeline_authority_block", ""),
    )
    base.update(overrides)
    return base, dispatch


def test_comparison_defers_to_llm_when_narration_enabled():
    ctx, dispatch = _ctx("G650 vs Falcon 8X")
    assert dispatch is None
    assert ctx["pre_llm_pipeline_patch"].get("comparison_deferred_llm") == 1
    assert should_bypass_llm_execution(ctx) is False


def test_comparison_bypasses_llm_when_narration_disabled(monkeypatch):
    monkeypatch.setenv("CONSULTANT_FORCE_LLM", "0")
    monkeypatch.setenv("CONSULTANT_LLM_NARRATE_STRUCTURED", "0")
    ctx, dispatch = _ctx("Compare G650 vs Falcon 8X")
    assert dispatch is not None
    assert should_bypass_llm_execution(ctx) is True
    bypass = resolve_deterministic_bypass_response(ctx)
    assert bypass is not None
    kind, payload = bypass
    assert kind == "professional"
    assert payload["answer"]
    assert not _KERNEL.search(payload["answer"])
    assert payload["data_used"]["deterministic_execution"]["bypassed_llm"] is True
    assert payload["data_used"]["deterministic_execution"]["final_responder"] == "respond_aircraft_comparison"


def test_alternative_defers_to_llm_when_narration_enabled():
    ctx, dispatch = _ctx("Alternatives to Longitude")
    assert dispatch is None
    assert ctx["pre_llm_pipeline_patch"].get("alternative_deferred_llm") == 1
    assert should_bypass_llm_execution(ctx) is False


def test_alternative_bypasses_llm_when_narration_disabled(monkeypatch):
    monkeypatch.setenv("CONSULTANT_FORCE_LLM", "0")
    monkeypatch.setenv("CONSULTANT_LLM_NARRATE_STRUCTURED", "0")
    ctx, dispatch = _ctx("Alternatives to Longitude")
    assert dispatch is not None
    assert should_bypass_llm_execution(ctx) is True
    bypass = resolve_deterministic_bypass_response(ctx)
    assert bypass is not None
    assert "tier-peer" in bypass[1]["answer"].lower() or "alternatives" in bypass[1]["answer"].lower()
    assert not _KERNEL.search(bypass[1]["answer"])


def test_buy_decision_uses_deal_engine_responder():
    ctx, dispatch = _ctx("2016 Latitude $10M good deal?")
    assert dispatch is not None
    assert dispatch.dispatch_kind == "buy_decision"
    assert should_bypass_llm_execution(ctx) is True
    bypass = resolve_deterministic_bypass_response(ctx)
    assert bypass is not None
    assert "Verdict" in bypass[1]["answer"] or "Market Reality" in bypass[1]["answer"]
    assert bypass[1]["data_used"]["deterministic_execution"]["final_responder"] == "respond_buy_decision"


def test_incomplete_mission_allowed_to_reach_llm():
    ctx = build_deterministic_guard_context(
        query="I need a jet for long trips, what should I buy?",
        qri=classify_query_recommendation_intent("I need a jet for long trips, what should I buy?", []),
        unified_route=classify_unified_intent("I need a jet for long trips, what should I buy?"),
        authority_dispatch_result=None,
        pre_llm_pipeline_patch={},
        pipeline_authority_block="",
    )
    assert should_bypass_llm_execution(ctx) is False


def test_complete_mission_bypasses_llm_when_pre_llm_produced_block():
    ctx = build_deterministic_guard_context(
        query="Need Challenger for TEB-LAX 6 pax nonstop",
        qri=classify_query_recommendation_intent("Need Challenger for TEB-LAX 6 pax nonstop", []),
        unified_route=None,
        authority_dispatch_result=None,
        pre_llm_pipeline_patch={
            "recommendation_pipeline": {"ranked_models": ["Challenger 350"]},
            "mission_preprocessing": {"routes": ["TEB-LAX"], "passenger_count": 6},
            "query_recommendation_requires_pipeline": True,
        },
        pipeline_authority_block="Mission Interpretation\n- Primary stage: TEB-LAX with 6 passengers.\n\nRanked Aircraft Shortlist\n* Challenger 350",
    )
    assert should_bypass_llm_execution(ctx) is True
    bypass = resolve_deterministic_bypass_response(ctx)
    assert bypass is not None
    assert "Challenger" in bypass[1]["answer"]


def test_mixed_signals_comparison_defers_to_llm():
    patch: dict = {}
    q = "Compare G650 vs Falcon 8X for my TEB-LAX mission"
    qri = classify_query_recommendation_intent(q, [])
    route = classify_unified_intent(q)
    dispatch = consult_authority_dispatch(
        q,
        qri=qri,
        unified_route=route,
        context={"db": None, "pre_llm_pipeline_patch": patch},
    )
    ctx = build_deterministic_guard_context(
        query=q,
        qri=qri,
        unified_route=route,
        authority_dispatch_result=dispatch,
        pre_llm_pipeline_patch=patch,
        pipeline_authority_block="",
    )
    assert ctx["deterministic_intent"] == "comparison"
    assert dispatch is None
    assert should_bypass_llm_execution(ctx) is False


def test_hard_deterministic_blocks_pre_llm_requirement():
    ctx, _ = _ctx("Alternatives to Praetor 600")
    assert requires_hard_deterministic_responder(ctx) is True


def test_query_requires_hard_deterministic_pipeline():
    assert query_requires_hard_deterministic_pipeline("G650 vs Falcon 8X") is True
    assert query_requires_hard_deterministic_pipeline("Alternatives to Longitude") is True
    assert query_requires_hard_deterministic_pipeline("2016 Latitude $10M good deal?") is True
    assert query_requires_hard_deterministic_pipeline("What is a Citation Longitude worth?") is True
    assert query_requires_hard_deterministic_pipeline("fleet portfolio strategy") is True
    assert query_requires_hard_deterministic_pipeline("multi-criteria decision ranking") is True
    assert query_requires_hard_deterministic_pipeline("Tell me about business aviation trends") is False


def test_assert_no_llm_leak_debug_mode(monkeypatch):
    monkeypatch.setenv("CONSULTANT_DETERMINISTIC_ASSERT", "1")
    assert deterministic_assertion_enabled() is True
    with pytest.raises(DeterministicExecutionViolation):
        assert_no_llm_leak(
            {
                "deterministic_intent": "comparison",
                "llm_executed": True,
            }
        )


def test_assert_no_leak_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("CONSULTANT_DETERMINISTIC_ASSERT", raising=False)
    monkeypatch.delenv("DETERMINISTIC_EXECUTION_ASSERT", raising=False)
    assert_no_llm_leak({"deterministic_intent": "comparison", "llm_executed": True})

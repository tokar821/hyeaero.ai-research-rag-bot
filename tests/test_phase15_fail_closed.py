"""Phase 15.1 — fail-closed hard deterministic enforcement tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.deterministic

from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.authority_dispatch import consult_authority_dispatch
from services.routing.deterministic_execution_guard import (
    build_deterministic_guard_context,
    requires_hard_deterministic_responder,
    resolve_deterministic_bypass_response,
    should_bypass_llm_execution,
)
from services.routing.unified_intent_router import classify_unified_intent
from rag.consultant_retrieval import _build_hard_deterministic_safety_fallback


def _unresolved_comparison_context():
    query = "G650 vs UnknownJetXYZ"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    dispatch = consult_authority_dispatch(query, qri=qri, unified_route=route, context={"db": None})
    ctx = build_deterministic_guard_context(
        query=query,
        qri=qri,
        unified_route=route,
        authority_dispatch_result=dispatch,
    )
    return query, ctx


def test_unresolved_comparison_bypass_without_resolve():
    _, ctx = _unresolved_comparison_context()
    dispatch = consult_authority_dispatch(
        "G650 vs UnknownJetXYZ",
        qri=ctx["qri"],
        unified_route=ctx["unified_route"],
        context={"db": None},
    )
    assert dispatch is not None
    assert dispatch.dispatch_kind == "comparison"
    assert "Insufficient verified data" in dispatch.answer
    assert dispatch.data_used.get("authority_dispatch_safety_fallback") == "comparison"
    assert should_bypass_llm_execution(ctx) is True
    assert requires_hard_deterministic_responder(ctx) is True
    ctx_with_dispatch = dict(ctx)
    ctx_with_dispatch["authority_dispatch_result"] = dispatch
    bypass = resolve_deterministic_bypass_response(ctx_with_dispatch)
    assert bypass is not None


def test_unresolved_comparison_safety_fallback_never_llm():
    _, ctx = _unresolved_comparison_context()
    kind, payload = _build_hard_deterministic_safety_fallback(ctx)
    assert kind == "professional"
    assert kind != "llm"
    meta = payload["data_used"]["deterministic_execution"]
    assert meta["trigger_reason"] == "hard_intent_insufficient_resolution"
    assert meta["final_responder"] == "deterministic_safety_fallback"
    assert meta["bypassed_llm"] is True
    assert meta["deterministic_intent"] == "comparison"
    assert "Insufficient verified data" in payload["answer"]


def test_mission_incomplete_does_not_use_safety_fallback_helper():
    ctx = build_deterministic_guard_context(
        query="I need a jet for long trips, what should I buy?",
        qri=classify_query_recommendation_intent(
            "I need a jet for long trips, what should I buy?", []
        ),
        unified_route=classify_unified_intent("I need a jet for long trips, what should I buy?"),
        authority_dispatch_result=None,
    )
    assert should_bypass_llm_execution(ctx) is False
    assert requires_hard_deterministic_responder(ctx) is False

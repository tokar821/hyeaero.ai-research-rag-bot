"""Phase 29 — Fail-closed matrix tests."""

from __future__ import annotations

import pytest

from services.core.semantic_intent_lock_engine import (
    bind_dispatch_authority,
    build_intent_lock,
    enforce_intent_lock_at_guard,
)
from services.fleet.fleet_portfolio_strategy_engine import build_fleet_portfolio_strategy
from services.market.aircraft_market_intelligence_engine import build_market_intelligence
from services.optimization.multi_criteria_decision_engine import build_optimization_result
from services.ownership.aircraft_lifecycle_ownership_engine import build_ownership_intelligence
from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.authority_dispatch import consult_authority_dispatch
from services.routing.deterministic_execution_guard import (
    build_deterministic_guard_context,
    resolve_deterministic_bypass_response,
    should_bypass_llm_execution,
)
from services.routing.unified_intent_router import classify_unified_intent
from rag.consultant_retrieval import _build_hard_deterministic_safety_fallback
from tests.conftest import run_retrieval

pytestmark = pytest.mark.deterministic


FAIL_CLOSED_QUERIES = [
    ("G650 vs UnknownJetXYZ", "comparison"),
    ("Compare FakeJet9000 vs AnotherFakeJet", "comparison"),
    ("G650 vs", "comparison"),
    ("G650 vs Falcon 8X vs Global 7500 under $5M", "comparison"),
]


@pytest.fixture(autouse=True)
def _deterministic_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard):
    pass


@pytest.mark.parametrize("query,expected_kind", FAIL_CLOSED_QUERIES)
def test_fail_closed_dispatch_safety_fallback(mock_svc, query, expected_kind):
    kind, payload = run_retrieval(query, svc=mock_svc)
    du = payload.get("data_used") or {}
    assert kind == "professional"
    assert du.get("authority_dispatch_safety_fallback") == expected_kind or (
        expected_kind == "buy_decision" and du.get("authority_dispatch_kind") == "buy_decision"
    )
    trace = du.get("intent_execution_trace") or {}
    assert trace.get("llm_invoked") is False
    assert "Insufficient verified data" in payload.get("answer", "") or "Verdict" in payload.get("answer", "")


def test_zero_models_comparison_fail_closed():
    query = "Compare something for me"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    result = consult_authority_dispatch(query, qri=qri, unified_route=route, context={"db": None})
    if result is not None:
        assert result.data_used.get("authority_dispatch_safety_fallback") == "comparison"


def test_one_verified_one_fake_fail_closed(mock_svc):
    kind, payload = run_retrieval("G650 vs FakeJet9000", svc=mock_svc)
    du = payload.get("data_used") or {}
    assert kind == "professional"
    assert du.get("authority_dispatch_safety_fallback") == "comparison"
    assert (du.get("intent_execution_trace") or {}).get("llm_invoked") is False


def test_missing_lock_at_guard_fail_closed():
    query = "G650 vs UnknownJetXYZ"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    dispatch = consult_authority_dispatch(query, qri=qri, unified_route=route, context={"db": None})
    ctx = build_deterministic_guard_context(
        query=query,
        qri=qri,
        unified_route=route,
        authority_dispatch_result=dispatch,
        pre_llm_pipeline_patch={},
    )
    out = enforce_intent_lock_at_guard(ctx)
    assert out is not None
    assert out[0] == "professional"
    assert out[1]["data_used"].get("intent_lock_validation_failed") == 1


def test_budget_removes_all_candidates_fail_closed():
    query = "G650 vs Falcon 8X vs Global 7500 under $5M"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    result = consult_authority_dispatch(
        query,
        qri=qri,
        unified_route=route,
        context={"db": None},
    )
    assert result is not None
    assert result.data_used.get("authority_dispatch_safety_fallback") == "comparison"


def test_optimization_empty_returns_insufficient_data():
    result = build_optimization_result("generic query", {"data_used": {}})
    assert result.get("status") == "INSUFFICIENT_DATA"


def test_ownership_empty_returns_insufficient_data():
    result = build_ownership_intelligence("ownership", {"data_used": {}})
    assert result.get("status") == "INSUFFICIENT_DATA"


def test_fleet_empty_returns_insufficient_data():
    result = build_fleet_portfolio_strategy("fleet strategy", {"data_used": {}})
    assert result.get("status") == "INSUFFICIENT_DATA"


def test_market_empty_returns_insufficient_data():
    result = build_market_intelligence("market trends", {"data_used": {}})
    assert result.get("status") == "INSUFFICIENT_DATA"


def test_dispatch_mismatch_triggers_guard_fallback():
    query = "G650 vs Falcon 8X"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    from services.routing.authority_dispatch import AuthorityDispatchResult

    bad = AuthorityDispatchResult(
        answer="x",
        dispatch_kind="comparison",
        progress_step="test",
        data_used={"authority_dispatch_models": ["FakeJet9000"]},
    )
    lock = bind_dispatch_authority(lock, bad)
    ctx = build_deterministic_guard_context(
        query=query,
        qri=qri,
        unified_route=route,
        authority_dispatch_result=bad,
        pre_llm_pipeline_patch={"intent_lock": lock.to_dict()},
    )
    out = enforce_intent_lock_at_guard(ctx)
    assert out is not None
    assert "Insufficient verified data" in out[1].get("answer", "")


def test_hard_deterministic_safety_fallback_never_llm():
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
    kind, payload = _build_hard_deterministic_safety_fallback(ctx)
    assert kind != "llm"
    assert payload["data_used"]["deterministic_execution"]["bypassed_llm"] is True


def test_unresolved_comparison_bypasses_llm():
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
    assert should_bypass_llm_execution(ctx) is True
    bypass = resolve_deterministic_bypass_response(ctx)
    assert bypass is not None
    assert bypass[0] == "professional"


def test_icrl_does_not_override_dispatch(mock_svc):
    query = "G650 vs Falcon 8X vs Global 7500 under $30M"
    kind, payload = run_retrieval(query, svc=mock_svc)
    du = payload.get("data_used") or {}
    trace = du.get("intent_execution_trace") or {}
    assert trace.get("final_execution_path") == "authority_dispatch"
    assert trace.get("llm_invoked") is False
    assert kind == "professional"


def test_valuation_without_price_fail_closed(mock_svc):
    kind, payload = run_retrieval("What is a Citation Longitude worth?", svc=mock_svc)
    du = payload.get("data_used") or {}
    assert kind == "professional"
    assert (du.get("intent_execution_trace") or {}).get("llm_invoked") is False


def test_alternative_fake_target_fail_closed_dispatch():
    query = "Alternatives to FakeJet9000"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    result = consult_authority_dispatch(query, qri=qri, unified_route=route, context={"db": None})
    assert result is not None
    assert result.data_used.get("authority_dispatch_safety_fallback") == "alternative"


def test_buy_malformed_listing_fail_closed():
    query = "good deal?"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    result = consult_authority_dispatch(query, qri=qri, unified_route=route, context={"db": None})
    assert result is None or result.data_used.get("authority_dispatch_safety_fallback")

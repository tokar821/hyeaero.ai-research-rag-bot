"""Phase 29 — IntentLock guard integration tests."""

from __future__ import annotations

import pytest

from services.core.semantic_intent_lock_engine import (
    IntentLock,
    bind_dispatch_authority,
    build_intent_lock,
    enforce_intent_lock_at_guard,
    intent_lock_failures_require_safety_fallback,
    validate_intent_lock_consistency,
)
from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.authority_dispatch import AuthorityDispatchResult, consult_authority_dispatch
from services.routing.deterministic_execution_guard import build_deterministic_guard_context
from services.routing.intent_conflict_resolution import resolve_intent_conflicts
from services.routing.unified_intent_router import classify_unified_intent

pytestmark = pytest.mark.deterministic


def _guard_ctx(query: str, *, pre_llm_patch: dict | None = None, dispatch=None):
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    if dispatch is None:
        dispatch = consult_authority_dispatch(
            query, qri=qri, unified_route=route, context={"db": None}
        )
    patch = dict(pre_llm_patch or {})
    return build_deterministic_guard_context(
        query=query,
        qri=qri,
        unified_route=route,
        authority_dispatch_result=dispatch,
        pre_llm_pipeline_patch=patch,
    ), dispatch, qri, route


@pytest.fixture(autouse=True)
def _enable_lock(monkeypatch):
    monkeypatch.setenv("ENABLE_INTENT_LOCK", "1")


def test_missing_lock_triggers_safety_fallback():
    ctx, _, _, _ = _guard_ctx("G650 vs UnknownJetXYZ", pre_llm_patch={})
    out = enforce_intent_lock_at_guard(ctx)
    assert out is not None
    kind, payload = out
    assert kind == "professional"
    du = payload.get("data_used") or {}
    assert du.get("intent_lock_validation_failed") == 1
    assert du.get("deterministic_execution", {}).get("bypassed_llm") is True
    assert "Insufficient verified data" in payload.get("answer", "")


def test_malformed_lock_triggers_safety_fallback():
    ctx, _, _, _ = _guard_ctx(
        "G650 vs Falcon 8X",
        pre_llm_patch={"intent_lock": "not-a-dict"},
    )
    out = enforce_intent_lock_at_guard(ctx)
    assert out is not None
    assert out[1]["data_used"].get("intent_lock_validation_failed") == 1


def test_dispatch_intent_mismatch_detected():
    query = "G650 vs Falcon 8X"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    dispatch = AuthorityDispatchResult(
        answer="test",
        dispatch_kind="alternative",
        progress_step="test",
        data_used={"authority_dispatch_models": list(lock.canonical_models)},
    )
    lock = bind_dispatch_authority(lock, lock.__class__(
        intent_type="comparison",
        canonical_models=lock.canonical_models,
        constraints=lock.constraints,
        origin_query_hash=lock.origin_query_hash,
        deterministic_flags=dict(lock.deterministic_flags),
        dispatch_authority_id=lock.dispatch_authority_id,
        timestamp=lock.timestamp,
    ))
    failures = validate_intent_lock_consistency(lock, dispatch_result=dispatch)
    assert "dispatch_intent_mismatch" in failures
    assert intent_lock_failures_require_safety_fallback(failures)


def test_dispatch_model_not_in_lock_detected():
    query = "G650 vs Falcon 8X"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    dispatch = AuthorityDispatchResult(
        answer="test",
        dispatch_kind="comparison",
        progress_step="test",
        data_used={"authority_dispatch_models": ["FakeJet9000"]},
    )
    failures = validate_intent_lock_consistency(lock, dispatch_result=dispatch)
    assert "dispatch_model_not_in_lock" in failures


def test_missing_dispatch_authority_id_detected():
    query = "G650 vs Falcon 8X"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    dispatch = consult_authority_dispatch(
        query, qri=qri, unified_route=route, context={"db": None, "intent_lock": lock}
    )
    failures = validate_intent_lock_consistency(lock, dispatch_result=dispatch)
    assert "missing_dispatch_authority_id" in failures


def test_icrl_drift_detected_with_resolution_object():
    query = "Tell me about business aviation trends"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    resolution = resolve_intent_conflicts(
        {"query": query, "qri": qri, "unified_route": route, "db": None}
    )
    failures = validate_intent_lock_consistency(
        lock,
        icrl_resolution=resolution,
    )
    if resolution.plan.primary_mode == "comparison" and lock.intent_type == "mission":
        assert "icrl_intent_type_drift" in failures


def test_icrl_dict_drift_detected():
    """Guard passes ICRL as dict — drift must still fire."""
    query = "Tell me about business aviation trends"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    resolution = resolve_intent_conflicts(
        {"query": query, "qri": qri, "unified_route": route, "db": None}
    )
    failures = validate_intent_lock_consistency(
        lock,
        icrl_resolution=resolution.to_dict(),
    )
    if resolution.plan.primary_mode == "comparison" and lock.intent_type == "mission":
        assert "icrl_intent_type_drift" in failures


def test_icrl_dict_no_drift_when_intents_align():
    query = "G650 vs Falcon 8X"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    dispatch = consult_authority_dispatch(
        query, qri=qri, unified_route=route, context={"db": None, "intent_lock": lock}
    )
    resolution = resolve_intent_conflicts(
        {
            "query": query,
            "qri": qri,
            "unified_route": route,
            "authority_dispatch_result": dispatch,
            "intent_lock": lock,
        }
    )
    failures = validate_intent_lock_consistency(
        lock,
        dispatch_result=dispatch,
        icrl_resolution=resolution.to_dict(),
    )
    assert "icrl_intent_type_drift" not in failures


def test_optimization_model_outside_lock_detected():
    query = "G650 vs Falcon 8X"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    failures = validate_intent_lock_consistency(
        lock,
        data_used={
            "optimization_result": {
                "status": "OK",
                "ranked_candidates": [{"aircraft": "FakeJet9000", "total_score": 90}],
            }
        },
    )
    assert "optimization_model_not_in_lock" in failures


def test_fleet_constraint_override_detected():
    query = "G650 vs Falcon 8X under $30M"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    failures = validate_intent_lock_consistency(
        lock,
        data_used={
            "fleet_portfolio_strategy": {
                "status": "OK",
                "fleet_input": {"budget_constraints": {"budget_m": 99.0}},
            }
        },
    )
    assert "fleet_constraint_override" in failures


def test_enforce_guard_returns_fallback_on_model_mismatch():
    query = "G650 vs Falcon 8X"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    bad_dispatch = AuthorityDispatchResult(
        answer="x",
        dispatch_kind="comparison",
        progress_step="test",
        data_used={"authority_dispatch_models": ["Nonexistent Aircraft XYZ"]},
    )
    lock = bind_dispatch_authority(lock, bad_dispatch)
    ctx = build_deterministic_guard_context(
        query=query,
        qri=qri,
        unified_route=route,
        authority_dispatch_result=bad_dispatch,
        pre_llm_pipeline_patch={"intent_lock": lock.to_dict()},
    )
    out = enforce_intent_lock_at_guard(ctx)
    assert out is not None
    failures = out[1]["data_used"].get("intent_lock_consistency_failures") or []
    assert "dispatch_model_not_in_lock" in failures


def test_consistent_lock_passes_guard():
    query = "G650 vs Falcon 8X"
    ctx, dispatch, qri, route = _guard_ctx(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    lock = bind_dispatch_authority(lock, dispatch)
    ctx["pre_llm_pipeline_patch"] = {"intent_lock": lock.to_dict()}
    ctx["authority_dispatch_result"] = dispatch
    assert enforce_intent_lock_at_guard(ctx) is None


def test_valuation_lock_buy_dispatch_allowed():
    query = "What is a Citation Longitude worth?"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    dispatch = consult_authority_dispatch(
        query, qri=qri, unified_route=route, context={"db": None, "intent_lock": lock}
    )
    lock = bind_dispatch_authority(lock, dispatch)
    failures = validate_intent_lock_consistency(lock, dispatch_result=dispatch)
    assert "dispatch_intent_mismatch" not in failures


def test_intent_lock_disabled_skips_enforcement(monkeypatch):
    monkeypatch.setenv("ENABLE_INTENT_LOCK", "0")
    ctx, _, _, _ = _guard_ctx("G650 vs UnknownJetXYZ", pre_llm_patch={})
    assert enforce_intent_lock_at_guard(ctx) is None

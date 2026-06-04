"""Phase 28 — Semantic Intent Lock Engine tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.deterministic

from services.core.semantic_intent_lock_engine import (
    IntentLock,
    bind_dispatch_authority,
    build_execution_trace_v2,
    build_intent_lock,
    compute_deterministic_evaluation_id,
    compute_origin_query_hash,
    enforce_intent_lock_at_guard,
    validate_intent_lock_consistency,
)
from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.authority_dispatch import AuthorityDispatchResult, consult_authority_dispatch
from services.routing.unified_intent_router import classify_unified_intent


def test_build_intent_lock_comparison():
    query = "G650 vs Falcon 8X"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    assert lock.intent_type == "comparison"
    assert len(lock.canonical_models) >= 2
    assert lock.origin_query_hash == compute_origin_query_hash(query)
    assert lock.semantic_version == "v1"
    assert lock.timestamp.startswith("lock-")


def test_intent_lock_deterministic_replay():
    query = "G650 vs Falcon 8X under $70M"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    a = build_intent_lock(query, qri=qri, unified_route=route)
    b = build_intent_lock(query, qri=qri, unified_route=route)
    assert a.to_dict() == b.to_dict()


def test_bind_dispatch_authority_id():
    query = "G650 vs Falcon 8X"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    dispatch = consult_authority_dispatch(
        query,
        qri=qri,
        unified_route=route,
        context={"db": None, "intent_lock": lock},
    )
    assert dispatch is not None
    bound = bind_dispatch_authority(lock, dispatch)
    assert bound.dispatch_authority_id
    assert bound.deterministic_flags.get("dispatch_kind") == "comparison"


def test_validate_intent_lock_consistency_ok():
    query = "G650 vs Falcon 8X"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    dispatch = consult_authority_dispatch(
        query,
        qri=qri,
        unified_route=route,
        context={"db": None, "intent_lock": lock},
    )
    bound = bind_dispatch_authority(lock, dispatch)
    failures = validate_intent_lock_consistency(
        bound,
        data_used={"authority_dispatch_kind": "comparison"},
        dispatch_result=dispatch,
    )
    assert failures == []


def test_validate_missing_intent_lock():
    assert validate_intent_lock_consistency(None) == ["missing_intent_lock"]


def test_enforce_intent_lock_at_guard_missing_lock():
    from services.routing.deterministic_execution_guard import build_deterministic_guard_context

    ctx = build_deterministic_guard_context(
        query="G650 vs UnknownJetXYZ",
        qri=classify_query_recommendation_intent("G650 vs UnknownJetXYZ", []),
        unified_route=classify_unified_intent("G650 vs UnknownJetXYZ"),
        pre_llm_pipeline_patch={},
    )
    out = enforce_intent_lock_at_guard(ctx)
    assert out is not None
    kind, payload = out
    assert kind == "professional"
    assert payload["data_used"].get("intent_lock_validation_failed") == 1


def test_deterministic_evaluation_id():
    query = "G650 vs Falcon 8X"
    lock = build_intent_lock(
        query,
        qri=classify_query_recommendation_intent(query, []),
        unified_route=classify_unified_intent(query),
    )
    a = compute_deterministic_evaluation_id(query, intent_lock=lock, answer="Verified catalog comparison")
    b = compute_deterministic_evaluation_id(query, intent_lock=lock, answer="Verified catalog comparison")
    assert a == b
    assert len(a) == 24


def test_execution_trace_v2_shape():
    query = "G650 vs Falcon 8X"
    lock = build_intent_lock(
        query,
        qri=classify_query_recommendation_intent(query, []),
        unified_route=classify_unified_intent(query),
    )
    trace = build_execution_trace_v2(
        intent_lock=lock,
        data_used={"intent_lock": lock.to_dict()},
        final_answer="test answer",
    )
    assert trace["semantic_version"] == "v1"
    assert trace["intent_lock_snapshot"]["intent_type"] == "comparison"
    assert trace["akal_version"] == "akal-v1"
    assert trace["final_output_hash"]
    assert trace["trace_id"]

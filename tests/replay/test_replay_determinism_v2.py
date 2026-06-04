"""Phase 30 Track C — Replay determinism v2."""

from __future__ import annotations

import pytest

from services.core.semantic_intent_lock_engine import (
    build_execution_trace_v2,
    build_intent_lock,
    compute_deterministic_evaluation_id,
    compute_origin_query_hash,
)
from services.evaluation.consultant_evaluator import attach_consultant_evaluation_if_enabled
from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.authority_dispatch import consult_authority_dispatch
from services.routing.unified_intent_router import classify_unified_intent
from tests.conftest import run_full_pipeline_snapshot, run_retrieval

pytestmark = pytest.mark.deterministic

REPLAY_QUERIES = [
    "G650 vs Falcon 8X",
    "Compare G650 vs Falcon 8X",
    "Alternatives to Longitude",
    "2016 Latitude $10M good deal?",
    "G650 vs UnknownJetXYZ",
    "G650 vs Falcon 8X under $70M",
    "Longitude vs Challenger 3500",
]

ALIAS_EQUIVALENTS = [
    ("G650 vs Falcon 8X", "Compare Gulfstream G650 vs Dassault Falcon 8X"),
    ("Alternatives to Longitude", "Show alternatives to Citation Longitude"),
]

WHITESPACE_EQUIVALENTS = [
    ("G650 vs Falcon 8X", "  G650   vs   Falcon 8X  "),
    ("G650 vs Falcon 8X", "G650 vs Falcon 8X\n"),
]


@pytest.fixture(autouse=True)
def _replay_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard, monkeypatch):
    monkeypatch.setenv("ENABLE_CONSULTANT_EVALUATION", "1")


def _lock_snapshot(query: str) -> dict:
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    return build_intent_lock(query, qri=qri, unified_route=route).to_dict()


@pytest.mark.parametrize("query", REPLAY_QUERIES)
def test_intent_lock_identical_across_runs(mock_svc, query):
    a = _lock_snapshot(query)
    b = _lock_snapshot(query)
    assert a == b


@pytest.mark.parametrize("query", REPLAY_QUERIES)
def test_dispatch_authority_id_stable(mock_svc, query):
    _, a = run_retrieval(query, svc=mock_svc)
    _, b = run_retrieval(query, svc=mock_svc)
    lock_a = (a.get("data_used") or {}).get("intent_lock") or {}
    lock_b = (b.get("data_used") or {}).get("intent_lock") or {}
    assert lock_a.get("dispatch_authority_id") == lock_b.get("dispatch_authority_id")
    assert lock_a.get("dispatch_authority_id")


@pytest.mark.parametrize("query", REPLAY_QUERIES)
def test_execution_trace_v2_trace_id_stable(mock_svc, query):
    _, a = run_retrieval(query, svc=mock_svc)
    _, b = run_retrieval(query, svc=mock_svc)
    t_a = (a.get("data_used") or {}).get("execution_trace_v2") or {}
    t_b = (b.get("data_used") or {}).get("execution_trace_v2") or {}
    assert t_a.get("trace_id") == t_b.get("trace_id")
    assert t_a.get("trace_id")


@pytest.mark.parametrize("query", REPLAY_QUERIES)
def test_evaluation_id_stable(mock_svc, query):
    snap_a = run_full_pipeline_snapshot(query, svc=mock_svc)
    snap_b = run_full_pipeline_snapshot(query, svc=mock_svc)
    assert snap_a.get("evaluation_id") == snap_b.get("evaluation_id")
    assert snap_a.get("evaluation_id")


@pytest.mark.stress
def test_intent_lock_100x_identical(mock_svc):
    query = "G650 vs Falcon 8X"
    baseline = _lock_snapshot(query)
    for _ in range(100):
        assert _lock_snapshot(query) == baseline


@pytest.mark.stress
def test_trace_id_100x_identical(mock_svc):
    query = "G650 vs Falcon 8X"
    _, first = run_retrieval(query, svc=mock_svc)
    trace_id = ((first.get("data_used") or {}).get("execution_trace_v2") or {}).get("trace_id")
    assert trace_id
    for _ in range(100):
        _, payload = run_retrieval(query, svc=mock_svc)
        tid = ((payload.get("data_used") or {}).get("execution_trace_v2") or {}).get("trace_id")
        assert tid == trace_id


@pytest.mark.stress
def test_evaluation_id_100x_identical(mock_svc):
    query = "G650 vs Falcon 8X"
    first = run_full_pipeline_snapshot(query, svc=mock_svc)
    eval_id = first["evaluation_id"]
    assert eval_id
    for _ in range(100):
        snap = run_full_pipeline_snapshot(query, svc=mock_svc)
        assert snap["evaluation_id"] == eval_id


@pytest.mark.parametrize("a,b", WHITESPACE_EQUIVALENTS)
def test_whitespace_normalization_stable_hash(a, b):
    assert compute_origin_query_hash(a) == compute_origin_query_hash(b)


@pytest.mark.parametrize("query", REPLAY_QUERIES[:4])
def test_final_output_hash_stable_when_answer_identical(mock_svc, query):
    _, a = run_retrieval(query, svc=mock_svc)
    _, b = run_retrieval(query, svc=mock_svc)
    assert a.get("answer") == b.get("answer")
    h_a = ((a.get("data_used") or {}).get("execution_trace_v2") or {}).get("final_output_hash")
    h_b = ((b.get("data_used") or {}).get("execution_trace_v2") or {}).get("final_output_hash")
    assert h_a == h_b


def test_deterministic_evaluation_id_module_level():
    query = "G650 vs Falcon 8X"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    dispatch = consult_authority_dispatch(
        query, qri=qri, unified_route=route, context={"db": None, "intent_lock": lock}
    )
    answer = dispatch.answer if dispatch else ""
    a = compute_deterministic_evaluation_id(query, intent_lock=lock, answer=answer)
    b = compute_deterministic_evaluation_id(query, intent_lock=lock, answer=answer)
    assert a == b


def test_execution_trace_v2_module_level_deterministic():
    query = "G650 vs Falcon 8X"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    dispatch = consult_authority_dispatch(
        query, qri=qri, unified_route=route, context={"db": None, "intent_lock": lock}
    )
    answer = dispatch.answer if dispatch else "test"
    a = build_execution_trace_v2(intent_lock=lock, dispatch_result=dispatch, final_answer=answer)
    b = build_execution_trace_v2(intent_lock=lock, dispatch_result=dispatch, final_answer=answer)
    assert a == b


def test_evaluator_attach_idempotent(mock_svc):
    query = "G650 vs Falcon 8X"
    _, payload = run_retrieval(query, svc=mock_svc)
    once = attach_consultant_evaluation_if_enabled(query, payload)
    twice = attach_consultant_evaluation_if_enabled(query, once)
    e1 = (once.get("data_used") or {}).get("consultant_evaluation") or {}
    e2 = (twice.get("data_used") or {}).get("consultant_evaluation") or {}
    assert e1.get("evaluation_id") == e2.get("evaluation_id")


def test_origin_query_hash_deterministic():
    q = "G650 vs Falcon 8X"
    assert compute_origin_query_hash(q) == compute_origin_query_hash(q)


def test_lock_timestamp_deterministic_not_wall_clock():
    q = "G650 vs Falcon 8X"
    lock = _lock_snapshot(q)
    assert lock["timestamp"].startswith("lock-")
    assert lock["timestamp"] == f"lock-{compute_origin_query_hash(q)[:16]}"

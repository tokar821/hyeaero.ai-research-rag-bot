"""Phase 29 — Retrieval E2E tests for IntentLock + authority dispatch pipeline."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from services.core.semantic_intent_lock_engine import compute_deterministic_evaluation_id
from services.evaluation.consultant_evaluator import attach_consultant_evaluation_if_enabled
from tests.conftest import run_retrieval



pytestmark = pytest.mark.deterministic
HARD_INTENT_QUERIES = [
    ("G650 vs Falcon 8X", "comparison"),
    ("Compare G650 vs Falcon 8X", "comparison"),
    ("Alternatives to Longitude", "alternative"),
    ("2016 Latitude $10M good deal?", "buy_decision"),
    ("What is a Citation Longitude worth?", "valuation"),
    ("G650 vs UnknownJetXYZ", "comparison"),
    ("G650 vs Falcon 8X vs Global 7500 under $30M", "comparison"),
    ("Show me alternatives to Global 7500", "alternative"),
    ("G650 vs Falcon 8X under $70M", "comparison"),
    ("compare maybe perhaps G650 and Falcon 8X", "comparison"),
    ("Compare G650 vs Falcon 8X for my TEB-LAX mission", "comparison"),
    ("G650 vs Falcon 8X vs Global 7500", "comparison"),
    ("Alternatives to Praetor 600", "alternative"),
    ("2018 Challenger 350 $8M fair deal?", "buy_decision"),
    ("Longitude vs Challenger 3500", "comparison"),
]


@pytest.fixture(autouse=True)
def deterministic_e2e_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard, monkeypatch):
    monkeypatch.setenv("ENABLE_CONSULTANT_EVALUATION", "1")


def _assert_hard_intent_payload(kind: str, payload: dict, *, expected_dispatch_kind: str) -> None:
    assert kind == "professional"
    du = payload.get("data_used") or {}
    assert isinstance(du, dict)

    lock = du.get("intent_lock")
    assert isinstance(lock, dict), "intent_lock must exist in data_used"
    assert lock.get("semantic_version") == "v1"
    assert lock.get("origin_query_hash")

    trace = du.get("intent_execution_trace") or {}
    assert trace.get("llm_invoked") is False
    assert trace.get("final_execution_path") == "authority_dispatch"

    trace_v2 = du.get("execution_trace_v2")
    assert isinstance(trace_v2, dict), "execution_trace_v2 must exist"
    assert trace_v2.get("semantic_version") == "v1"
    assert trace_v2.get("intent_lock_snapshot")

    assert du.get("authority_dispatch_kind") == expected_dispatch_kind
    if du.get("authority_dispatch_safety_fallback"):
        ans = str(payload.get("answer") or "").lower()
        assert ans.strip()
        broker_fallback = any(
            phrase in ans
            for phrase in (
                "insufficient verified",
                "comparing against",
                "reliable recommendation",
                "enough information",
                "need both models",
            )
        )
        assert broker_fallback or len(str(payload.get("answer") or "")) > 80
    assert lock.get("dispatch_authority_id"), "dispatch_authority_id must be bound after dispatch"

    icrl = du.get("intent_conflict_resolution") or {}
    if expected_dispatch_kind in ("comparison", "alternative", "buy_decision"):
        assert icrl.get("handled_by_icrl") is False or du.get("authority_dispatch_kind")

    det = du.get("deterministic_execution") or {}
    assert det.get("bypassed_llm") is True
    assert payload.get("answer")


@pytest.mark.parametrize("query,expected_kind", HARD_INTENT_QUERIES)
def test_hard_intent_e2e_pipeline(mock_svc, query, expected_kind):
    kind, payload = run_retrieval(query, svc=mock_svc)
    _assert_hard_intent_payload(kind, payload, expected_dispatch_kind=expected_kind)


def test_intent_lock_deterministic_across_runs(mock_svc):
    query = "G650 vs Falcon 8X"
    _, a = run_retrieval(query, svc=mock_svc)
    _, b = run_retrieval(query, svc=mock_svc)
    lock_a = (a.get("data_used") or {}).get("intent_lock")
    lock_b = (b.get("data_used") or {}).get("intent_lock")
    assert lock_a == lock_b

    trace_a = (a.get("data_used") or {}).get("execution_trace_v2") or {}
    trace_b = (b.get("data_used") or {}).get("execution_trace_v2") or {}
    assert trace_a.get("trace_id") == trace_b.get("trace_id")


def test_evaluation_id_deterministic(mock_svc):
    query = "G650 vs Falcon 8X"
    _, payload = run_retrieval(query, svc=mock_svc)
    du = payload.get("data_used") or {}
    lock = du.get("intent_lock")
    answer = payload.get("answer") or ""

    eval_a = compute_deterministic_evaluation_id(query, intent_lock=lock, answer=answer)
    eval_b = compute_deterministic_evaluation_id(query, intent_lock=lock, answer=answer)
    assert eval_a == eval_b
    assert len(eval_a) == 24


def test_evaluation_id_stable_after_evaluator_attach(mock_svc):
    query = "G650 vs Falcon 8X"
    _, payload = run_retrieval(query, svc=mock_svc)
    v1 = attach_consultant_evaluation_if_enabled(query, payload)
    _, payload2 = run_retrieval(query, svc=mock_svc)
    v2 = attach_consultant_evaluation_if_enabled(query, payload2)
    eval1 = (v1.get("data_used") or {}).get("consultant_evaluation") or {}
    eval2 = (v2.get("data_used") or {}).get("consultant_evaluation") or {}
    assert eval1.get("evaluation_id")
    assert eval1.get("evaluation_id") == eval2.get("evaluation_id")


def test_fine_intent_llm_never_called_on_hard_query(mock_svc):
    query = "G650 vs Falcon 8X"

    def _fail_llm(*_args, **_kwargs):
        raise AssertionError("fine-intent LLM must not run on hard deterministic pipeline")

    with patch("rag.consultant_fine_intent.classify_consultant_fine_intent_llm", side_effect=_fail_llm):
        kind, payload = run_retrieval(query, svc=mock_svc)
    _assert_hard_intent_payload(kind, payload, expected_dispatch_kind="comparison")


def test_conversation_guard_does_not_short_circuit_hard_comparison(mock_svc):
    query = "G650 vs Falcon 8X"
    kind, payload = run_retrieval(query, svc=mock_svc)
    du = payload.get("data_used") or {}
    assert du.get("consultant_conversation_guard") != 1
    assert du.get("conversation_message_type") != "small_talk"
    _assert_hard_intent_payload(kind, payload, expected_dispatch_kind="comparison")


def test_dispatch_result_survives_to_final_payload(mock_svc):
    query = "G650 vs Falcon 8X"
    kind, payload = run_retrieval(query, svc=mock_svc)
    du = payload.get("data_used") or {}
    assert du.get("authority_dispatch_kind") == "comparison"
    models = du.get("authority_dispatch_models") or []
    assert len(models) >= 2
    assert du.get("intent_lock", {}).get("deterministic_flags", {}).get("dispatch_kind") == "comparison"
    assert "G650" in payload.get("answer") or "Gulfstream" in payload.get("answer")


def test_icrl_does_not_override_dispatch_on_budget_fail_closed(mock_svc):
    query = "G650 vs Falcon 8X vs Global 7500 under $30M"
    kind, payload = run_retrieval(query, svc=mock_svc)
    du = payload.get("data_used") or {}
    trace = du.get("intent_execution_trace") or {}
    assert trace.get("final_execution_path") == "authority_dispatch"
    assert du.get("authority_dispatch_safety_fallback") == "comparison"
    assert kind == "professional"
    assert trace.get("llm_invoked") is False


def test_safety_fallback_still_has_intent_lock(mock_svc):
    query = "G650 vs UnknownJetXYZ"
    kind, payload = run_retrieval(query, svc=mock_svc)
    du = payload.get("data_used") or {}
    assert isinstance(du.get("intent_lock"), dict)
    assert du.get("authority_dispatch_safety_fallback") == "comparison"
    trace_v2 = du.get("execution_trace_v2")
    assert trace_v2
    assert trace_v2.get("intent_lock_snapshot")
    assert kind == "professional"
    assert (du.get("intent_execution_trace") or {}).get("llm_invoked") is False

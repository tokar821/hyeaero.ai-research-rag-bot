"""Shared fixtures for Phase 29+ deterministic regression suites."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.authority_dispatch import consult_authority_dispatch
from services.routing.unified_intent_router import classify_unified_intent


def pytest_collection_modifyitems(config, items):
    from tests.ci.tier_registry import apply_tier_markers

    apply_tier_markers(items)


@pytest.fixture
def mock_svc():
    """Minimal service object for consultant retrieval bundle tests."""
    return SimpleNamespace(db=None, openai_api_key="", chat_model="gpt-4o-mini")


@pytest.fixture
def enable_intent_lock(monkeypatch):
    monkeypatch.setenv("ENABLE_INTENT_LOCK", "1")


@pytest.fixture
def aviation_conversation_guard(monkeypatch):
    """Bypass conversation guard so E2E tests exercise IntentLock pipeline."""
    from rag.conversation_guard import ConversationGuardResult, ConversationMessageType

    def _always_aviation(*_args, **_kwargs):
        return ConversationGuardResult(
            message_type=ConversationMessageType.AVIATION_QUERY,
            reply=None,
        )

    monkeypatch.setattr("rag.conversation_guard.evaluate_conversation_guard", _always_aviation)


@pytest.fixture
def disable_fine_intent_llm(monkeypatch):
    """Ensure fine-intent LLM is not invoked during deterministic tests."""
    monkeypatch.setenv("CONSULTANT_FINE_INTENT_LLM", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def enable_all_advisory_layers(monkeypatch):
    """Enable every advisory attach layer for isolation tests."""
    for key in (
        "ENABLE_DECISION_OPTIMIZATION",
        "ENABLE_MARKET_INTELLIGENCE",
        "ENABLE_OWNERSHIP_INTELLIGENCE",
        "ENABLE_FLEET_PORTFOLIO_STRATEGY",
        "ENABLE_EXECUTIVE_SYNTHESIS",
        "ENABLE_CONSULTANT_EVALUATION",
        "ENABLE_RECOMMENDATION_JUSTIFICATION",
        "ENABLE_RECOMMENDATION_CONFIDENCE",
    ):
        monkeypatch.setenv(key, "1")


def run_retrieval(
    query: str,
    *,
    svc: Any = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    from rag.consultant_retrieval import run_consultant_retrieval_bundle

    service = svc or SimpleNamespace(db=None, openai_api_key="", chat_model="gpt-4o-mini")
    kind, payload = run_consultant_retrieval_bundle(
        service,
        query,
        top_k=5,
        max_context_chars=8000,
        score_threshold=None,
        history=history or [],
        progress=None,
        client_conversation_state=None,
    )
    assert isinstance(payload, dict)
    return kind, payload


def dispatch_for_query(query: str):
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    return consult_authority_dispatch(query, qri=qri, unified_route=route, context={"db": None})


def routing_authority_snapshot(data_used: Dict[str, Any]) -> Dict[str, Any]:
    """Extract routing authority fields that advisory layers must not mutate."""
    lock = data_used.get("intent_lock") or {}
    trace = data_used.get("intent_execution_trace") or {}
    return {
        "intent_lock": dict(lock) if isinstance(lock, dict) else lock,
        "dispatch_authority_id": (lock.get("dispatch_authority_id") if isinstance(lock, dict) else ""),
        "authority_dispatch_kind": data_used.get("authority_dispatch_kind"),
        "authority_dispatch_models": list(data_used.get("authority_dispatch_models") or []),
        "final_execution_path": trace.get("final_execution_path"),
        "authority_dispatch_safety_fallback": data_used.get("authority_dispatch_safety_fallback"),
    }


def build_comparison_payload(query: str = "G650 vs Falcon 8X") -> Dict[str, Any]:
    """Build a realistic post-dispatch payload for advisory isolation tests."""
    from services.core.semantic_intent_lock_engine import bind_dispatch_authority, build_intent_lock
    from services.routing.intent_execution_trace import (
        IntentExecutionTraceCapture,
        attach_intent_execution_trace,
    )

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
    lock = bind_dispatch_authority(lock, dispatch)
    du: Dict[str, Any] = {
        "intent_lock": lock.to_dict(),
        "authority_dispatch_kind": dispatch.dispatch_kind,
    }
    du.update(dispatch.data_used)
    capture = IntentExecutionTraceCapture(raw_query=query, request_id="isolation-test")
    capture.capture_qri_unified(qri, route)
    capture.capture_intent_lock(lock)
    capture.capture_authority_dispatch(dispatch)
    capture.capture_deterministic_guard(should_bypass=True, resolve_hit=True)
    payload = attach_intent_execution_trace(
        capture,
        "professional",
        {
            "answer": dispatch.answer,
            "sources": [],
            "data_used": du,
            "aircraft_images": [],
            "error": None,
        },
        path_override="authority_dispatch",
        llm_invoked=False,
    )
    return payload


def run_full_pipeline_snapshot(
    query: str,
    *,
    svc: Any = None,
    with_advisory: bool = True,
) -> Dict[str, Any]:
    """Capture end-to-end pipeline artifacts for reproducibility tests."""
    from rag.query_service import _apply_api_contract_versioning
    from services.evaluation.consultant_evaluator import attach_consultant_evaluation_if_enabled

    kind, payload = run_retrieval(query, svc=svc)
    out = dict(payload)
    if with_advisory:
        out = _apply_api_contract_versioning(out)
        out = attach_consultant_evaluation_if_enabled(query, out)
    du = out.get("data_used") or {}
    lock = du.get("intent_lock") or {}
    trace_v2 = du.get("execution_trace_v2") or {}
    evaluation = du.get("consultant_evaluation") or {}
    return {
        "return_kind": kind,
        "answer": out.get("answer"),
        "intent_lock": dict(lock),
        "dispatch_authority_id": lock.get("dispatch_authority_id"),
        "authority_dispatch_kind": du.get("authority_dispatch_kind"),
        "trace_id": trace_v2.get("trace_id"),
        "final_output_hash": trace_v2.get("final_output_hash"),
        "evaluation_id": evaluation.get("evaluation_id"),
        "execution_path": (du.get("intent_execution_trace") or {}).get("final_execution_path"),
        "llm_invoked": (du.get("intent_execution_trace") or {}).get("llm_invoked"),
    }

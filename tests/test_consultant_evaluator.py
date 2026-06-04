"""Phase 19 — Consultant Evaluation & Decision Scoring Framework tests."""

from __future__ import annotations

import pytest

from services.evaluation.consultant_evaluator import (
    attach_consultant_evaluation_if_enabled,
    consultant_evaluation_enabled,
    evaluate_consultant_response,
)
from services.evaluation.evaluation_analytics import aggregate_evaluations
from services.routing.authority_dispatch import consult_authority_dispatch
from services.routing.intent_execution_trace import (
    IntentExecutionTraceCapture,
    build_intent_execution_trace,
)
from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.unified_intent_router import classify_unified_intent

pytestmark = pytest.mark.deterministic


def _trace_response(
    query: str,
    answer: str,
    *,
    return_kind: str = "professional",
    path_override: str | None = None,
    dispatch_kind: str | None = None,
    llm_invoked: bool = False,
    icrl_handled: bool = False,
) -> dict:
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    dispatch = consult_authority_dispatch(query, qri=qri, unified_route=route, context={"db": None})
    capture = IntentExecutionTraceCapture(raw_query=query, request_id="eval-test")
    capture.capture_qri_unified(qri, route)
    capture.capture_authority_dispatch(dispatch)
    if icrl_handled:
        capture.icrl_handled = True
        capture.icrl_triggered = True
    if dispatch_kind is None and dispatch is not None:
        dispatch_kind = dispatch.dispatch_kind
    if llm_invoked:
        capture.capture_deterministic_guard(should_bypass=False)
    elif dispatch or icrl_handled:
        capture.capture_deterministic_guard(should_bypass=True, resolve_hit=True)

    ctx = capture.to_build_context()
    ctx.update(
        {
            "return_kind": return_kind,
            "path_override": path_override,
            "llm_invoked": llm_invoked,
        }
    )
    trace = build_intent_execution_trace(ctx)
    du: dict = {"intent_execution_trace": trace.to_dict()}
    if dispatch_kind:
        du["authority_dispatch_kind"] = dispatch_kind
    return {
        "answer": answer,
        "data_used": du,
        "query": query,
    }


def test_comparison_answer_high_routing_score():
    query = "G650 vs Falcon 8X"
    dispatch = consult_authority_dispatch(
        query,
        qri=classify_query_recommendation_intent(query, []),
        unified_route=classify_unified_intent(query),
        context={"db": None},
    )
    answer = (
        "Verified catalog comparison:\n"
        "- Gulfstream G650: ultra-long class; practical range 6000 nm; seats 14; operating cost band high.\n"
        "- Falcon 8X: large-cabin class; practical range 5800 nm; seats 14; cost band high.\n"
    )
    ev = evaluate_consultant_response(
        query,
        _trace_response(
            query,
            answer,
            path_override="authority_dispatch",
            dispatch_kind=dispatch.dispatch_kind if dispatch else "comparison",
        ),
    )
    assert ev.routing_score == 20.0
    assert "llm_leak" not in ev.failures
    assert ev.execution_path == "authority_dispatch"
    assert ev.total_score >= 60


def test_alternative_answer_routing():
    query = "Alternatives to Longitude"
    dispatch = consult_authority_dispatch(
        query,
        qri=classify_query_recommendation_intent(query, []),
        unified_route=classify_unified_intent(query),
        context={"db": None},
    )
    if dispatch is None:
        pytest.skip("alternative dispatch unavailable")
    answer = (
        "Tier-peer alternatives to Citation Longitude:\n"
        "- Praetor 600\n- Challenger 350\n"
    )
    ev = evaluate_consultant_response(
        query,
        _trace_response(
            query,
            answer,
            path_override="authority_dispatch",
            dispatch_kind="alternative",
        ),
    )
    assert ev.routing_score == 20.0
    assert ev.intent_type == "alternative"


def test_buy_decision_answer():
    query = "2016 Latitude $10M good deal?"
    dispatch = consult_authority_dispatch(
        query,
        qri=classify_query_recommendation_intent(query, []),
        unified_route=classify_unified_intent(query),
        context={"db": None},
    )
    if dispatch is None:
        pytest.skip("buy dispatch unavailable")
    answer = (
        "Aircraft: Citation Latitude\n"
        "Market Reality:\n- Typical band aligns with ask.\n"
        "Red flags:\n- None material.\n"
        "Verdict: GOOD DEAL\n"
    )
    ev = evaluate_consultant_response(
        query,
        _trace_response(
            query,
            answer,
            path_override="authority_dispatch",
            dispatch_kind="buy_decision",
        ),
    )
    assert ev.routing_score == 20.0
    assert ev.verdict_score == 20.0
    assert "missing_verdict" not in ev.failures


def test_mission_clarification_full_mission_score():
    query = "What should I buy?"
    answer = (
        "Before I recommend specific aircraft, what is your typical route, "
        "passenger count, and budget envelope?"
    )
    ev = evaluate_consultant_response(
        query,
        _trace_response(
            query,
            answer,
            return_kind="llm",
            path_override="pre_llm_mission",
            llm_invoked=True,
        ),
    )
    assert ev.mission_score == 20.0
    assert "mission_violation" not in ev.failures


def test_hallucinated_model_penalty():
    query = "Tell me about the Gulfstream G750"
    answer = "The Gulfstream G750 offers excellent range and cabin comfort for ultra-long missions."
    ev = evaluate_consultant_response(
        query,
        _trace_response(query, answer, return_kind="llm", llm_invoked=True, path_override="llm_fallback"),
    )
    assert "hallucinated_model" in ev.failures
    assert ev.factual_score == 0.0
    assert ev.passed is False


def test_kernel_leak_penalty():
    query = "G650 vs Falcon 8X"
    answer = (
        "OPERATIONAL SYNTHESIS (AUTHORITATIVE)\n"
        "VIABLE WITH COMPROMISES for both aircraft."
    )
    ev = evaluate_consultant_response(
        query,
        _trace_response(
            query,
            answer,
            path_override="authority_dispatch",
            dispatch_kind="comparison",
        ),
    )
    assert "kernel_leak" in ev.failures
    assert ev.total_score < 90


def test_missing_verdict_buy():
    query = "2016 Latitude $10M good deal?"
    answer = "Aircraft: Citation Latitude\nMarket Reality:\n- Limited data.\n"
    ev = evaluate_consultant_response(
        query,
        _trace_response(
            query,
            answer,
            path_override="authority_dispatch",
            dispatch_kind="buy_decision",
        ),
    )
    assert ev.verdict_score == 0.0
    assert "missing_verdict" in ev.failures


def test_wrong_execution_path_llm_leak():
    query = "G650 vs Falcon 8X"
    answer = "Both jets are great options depending on your mission."
    ev = evaluate_consultant_response(
        query,
        _trace_response(
            query,
            answer,
            return_kind="llm",
            path_override="llm_fallback",
            llm_invoked=True,
        ),
    )
    assert ev.routing_score == 0.0
    assert "llm_leak" in ev.failures
    assert ev.passed is False


def test_icrl_comparison_matrix_routing():
    query = "G650 vs Falcon 8X vs Global 7500 under $30M"
    answer = (
        "Verified catalog comparison:\n"
        "- G650: range 6000 nm; cabin large; cost high.\n"
        "- Falcon 8X: range 5800 nm; cabin large; cost high.\n"
        "- Global 7500: range 6600 nm; cabin large; cost ultra.\n"
        "Budget constraint filter applied.\n"
    )
    ev = evaluate_consultant_response(
        query,
        _trace_response(
            query,
            answer,
            path_override="icrl_deterministic",
            icrl_handled=True,
            dispatch_kind="comparison",
        ),
    )
    assert ev.routing_score == 20.0
    assert "llm_leak" not in ev.failures


def test_attach_only_when_env_enabled(monkeypatch):
    monkeypatch.delenv("ENABLE_CONSULTANT_EVALUATION", raising=False)
    payload = _trace_response("G650 vs Falcon 8X", "comparison text", path_override="authority_dispatch")
    assert not consultant_evaluation_enabled()
    out = attach_consultant_evaluation_if_enabled("G650 vs Falcon 8X", payload)
    assert "consultant_evaluation" not in (out.get("data_used") or {})

    monkeypatch.setenv("ENABLE_CONSULTANT_EVALUATION", "1")
    out2 = attach_consultant_evaluation_if_enabled("G650 vs Falcon 8X", payload)
    assert "consultant_evaluation" in (out2.get("data_used") or {})


def test_aggregate_evaluations_metrics():
    rows = [
        evaluate_consultant_response(
            "G650 vs Falcon 8X",
            _trace_response(
                "G650 vs Falcon 8X",
                "G650 vs Falcon 8X range nm cabin cost comparison.",
                path_override="authority_dispatch",
                dispatch_kind="comparison",
            ),
        ),
        evaluate_consultant_response(
            "What should I buy?",
            _trace_response(
                "What should I buy?",
                "What route and passengers do you need?",
                return_kind="llm",
                llm_invoked=True,
                path_override="pre_llm_mission",
            ),
        ),
    ]
    agg = aggregate_evaluations(rows)
    assert agg.count == 2
    assert agg.average_score > 0
    assert "comparison" in agg.score_by_intent or agg.count == 2
    assert agg.pass_rate >= 0

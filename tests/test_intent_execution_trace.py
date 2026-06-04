"""Phase 17 — intent execution trace observability tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.deterministic

from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.authority_dispatch import consult_authority_dispatch
from services.routing.intent_conflict_resolution import execute_icrl_plan, resolve_intent_conflicts
from services.routing.intent_execution_trace import (
    IntentExecutionTraceCapture,
    build_intent_execution_trace,
    stream_trace_events,
)
from services.routing.unified_intent_router import classify_unified_intent


def _build_trace_for_query(
    query: str,
    *,
    return_kind: str = "professional",
    icrl_handled: bool = False,
    path_override: str | None = None,
    pre_llm_executed: bool = False,
    llm_invoked: bool | None = None,
):
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    dispatch = consult_authority_dispatch(query, qri=qri, unified_route=route, context={"db": None})
    resolution = resolve_intent_conflicts(
        {
            "query": query,
            "qri": qri,
            "unified_route": route,
            "authority_dispatch_result": dispatch,
        }
    )
    capture = IntentExecutionTraceCapture(raw_query=query, request_id="test-trace")
    capture.capture_qri_unified(qri, route)
    capture.capture_authority_dispatch(dispatch)
    capture.capture_icrl(resolution)
    if icrl_handled:
        capture.icrl_handled = True
    if pre_llm_executed:
        capture.mark_pre_llm_executed()
    if dispatch is not None and return_kind == "professional" and not icrl_handled:
        capture.capture_deterministic_guard(should_bypass=True, resolve_hit=True)
    elif icrl_handled:
        capture.capture_deterministic_guard(should_bypass=True, resolve_hit=True)
    elif return_kind == "llm":
        capture.capture_deterministic_guard(should_bypass=False)

    ctx = capture.to_build_context()
    ctx.update(
        {
            "return_kind": return_kind,
            "path_override": path_override,
            "llm_invoked": llm_invoked,
        }
    )
    return build_intent_execution_trace(ctx), dispatch, resolution


def test_multi_intent_icrl_trace():
    query = "G650 vs Falcon 8X vs Global 7500 under $30M"
    dispatch = consult_authority_dispatch(
        query,
        qri=classify_query_recommendation_intent(query, []),
        unified_route=classify_unified_intent(query),
        context={"db": None},
    )
    resolution = resolve_intent_conflicts(
        {
            "query": query,
            "qri": classify_query_recommendation_intent(query, []),
            "unified_route": classify_unified_intent(query),
            "authority_dispatch_result": dispatch,
        }
    )
    assert resolution.handled_by_icrl is True
    assert dispatch is not None
    out = execute_icrl_plan(
        {"query": query, "pre_llm_pipeline_patch": {}, "authority_dispatch_result": dispatch},
        resolution,
    )
    assert out is None
    trace, _, _ = _build_trace_for_query(
        query,
        return_kind="professional",
        icrl_handled=False,
        path_override="authority_dispatch",
        llm_invoked=False,
    )
    assert trace.icrl_handled is True
    assert trace.authority_dispatch_result == "comparison"
    assert trace.final_execution_path == "authority_dispatch"
    assert trace.llm_invoked is False
    assert trace.final_execution_path != "llm_fallback"


def test_authority_dispatch_trace():
    query = "G650 vs Falcon 8X"
    trace, dispatch, _ = _build_trace_for_query(query, return_kind="professional")
    assert dispatch is not None
    assert trace.authority_dispatch_result == "comparison"
    assert trace.final_execution_path == "authority_dispatch"
    assert trace.llm_invoked is False
    assert trace.ui_intent == "comparison"


def test_mission_query_trace_path():
    query = "What should I buy for 8 pax LA to Miami"
    trace, dispatch, resolution = _build_trace_for_query(
        query,
        return_kind="llm",
        pre_llm_executed=True,
        llm_invoked=True,
    )
    assert dispatch is None
    assert resolution.handled_by_icrl is False
    assert trace.final_execution_path in ("pre_llm_mission", "hybrid_unified")
    assert trace.llm_invoked is True
    assert "deterministic_only" not in trace.bypass_reasons
    assert "authority_dispatch_hit" not in trace.bypass_reasons


def test_no_llm_fallback_when_deterministic_handled():
    cases = [
        ("G650 vs Falcon 8X vs Global 7500 under $30M", False, "authority_dispatch"),
        ("G650 vs Falcon 8X", False, "authority_dispatch"),
    ]
    for query, icrl_handled, expected_path in cases:
        trace, _, _ = _build_trace_for_query(
            query,
            return_kind="professional",
            icrl_handled=icrl_handled,
            path_override=expected_path,
            llm_invoked=False,
        )
        assert trace.final_execution_path == expected_path
        assert trace.final_execution_path != "llm_fallback"
        assert trace.llm_invoked is False


def test_stream_trace_events_shape():
    trace, _, _ = _build_trace_for_query("G650 vs Falcon 8X", return_kind="professional")
    events = stream_trace_events(trace.to_dict())
    types = {e["type"] for e in events}
    assert "trace:authority_dispatch" in types
    assert "trace:final_path" in types
    final = [e for e in events if e["type"] == "trace:final_path"][0]
    assert final["path"] == "authority_dispatch"
    assert final["llm_invoked"] is False


def test_enforcement_upgrades_invalid_llm_fallback():
    trace = build_intent_execution_trace(
        {
            "request_id": "enforce-1",
            "raw_query": "G650 vs Falcon 8X",
            "qri_intent": "aircraft_comparison",
            "authority_dispatch_result": "comparison",
            "icrl_handled": False,
            "deterministic_guard_result": "bypass",
            "return_kind": "llm",
            "llm_invoked": True,
        }
    )
    assert trace.final_execution_path == "authority_dispatch"
    assert trace.llm_invoked is True

"""Phase 18 — Execution Replay Engine tests."""

from __future__ import annotations

import os

import pytest

from services.recommendation.query_recommendation_intent import (
    apply_query_intent_metadata,
    classify_query_recommendation_intent,
)
from services.replay.execution_replay_engine import (
    attach_execution_replay_if_enabled,
    build_execution_replay,
    execution_replay_enabled,
    stream_execution_replay_events,
)
from services.routing.authority_dispatch import consult_authority_dispatch
from services.routing.intent_conflict_resolution import resolve_intent_conflicts
from services.routing.intent_execution_trace import (
    IntentExecutionTraceCapture,
    build_intent_execution_trace,
)
from services.routing.unified_intent_router import classify_unified_intent

pytestmark = pytest.mark.deterministic


def _response_for_query(
    query: str,
    *,
    return_kind: str = "professional",
    icrl_handled: bool = False,
    path_override: str | None = None,
    pre_llm_executed: bool = False,
    llm_invoked: bool | None = None,
    extra_data_used: dict | None = None,
) -> dict:
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

    capture = IntentExecutionTraceCapture(raw_query=query, request_id="replay-test")
    capture.capture_qri_unified(qri, route)
    capture.capture_authority_dispatch(dispatch)
    capture.capture_icrl(resolution)
    if icrl_handled:
        capture.icrl_handled = True
    if pre_llm_executed:
        capture.mark_pre_llm_executed()
    if dispatch and return_kind == "professional" and not icrl_handled:
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
    trace = build_intent_execution_trace(ctx)
    data_used: dict = {
        "intent_execution_trace": trace.to_dict(),
        "intent_conflict_resolution": resolution.to_dict(),
    }
    apply_query_intent_metadata(data_used, qri)
    if dispatch is not None:
        data_used["authority_dispatch_kind"] = dispatch.dispatch_kind
        data_used.update(dispatch.data_used)
    if pre_llm_executed:
        data_used["deterministic_pre_llm_executed"] = 1
    if extra_data_used:
        data_used.update(extra_data_used)

    return {
        "answer": "Test answer.",
        "sources": [],
        "data_used": data_used,
        "aircraft_images": [],
        "error": None,
        "query": query,
    }


def _stage(session, name: str):
    return next(s for s in session.replay_steps if s.stage == name)


def test_authority_dispatch_comparison_replay():
    query = "G650 vs Falcon 8X"
    session = build_execution_replay(_response_for_query(query))
    auth = _stage(session, "authority_dispatch")
    assert auth.outputs["triggered"] is True
    assert auth.outputs["responder_selected"] == "comparison_dispatch"
    assert session.llm_invoked is False
    assert session.final_execution_path == "authority_dispatch"
    assert "authority_dispatch" in session.replay_summary


def test_icrl_comparison_matrix_replay():
    query = "G650 vs Falcon 8X vs Global 7500 under $30M"
    session = build_execution_replay(
        _response_for_query(
            query,
            icrl_handled=True,
            path_override="icrl_deterministic",
            llm_invoked=False,
        )
    )
    icrl = _stage(session, "intent_conflict_resolution")
    assert icrl.outputs["handled"] is True
    assert icrl.outputs["strategy_selected"] == "comparison_matrix"
    assert session.final_execution_path == "icrl_deterministic"
    assert session.llm_invoked is False


def test_buy_decision_dispatch_replay():
    query = "2016 Latitude $10M good deal?"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    dispatch = consult_authority_dispatch(query, qri=qri, unified_route=route, context={"db": None})
    if dispatch is None:
        pytest.skip("buy dispatch unavailable in this catalog environment")
    session = build_execution_replay(_response_for_query(query))
    auth = _stage(session, "authority_dispatch")
    assert auth.outputs["dispatch_kind"] == "buy_decision"
    guard = _stage(session, "deterministic_guard")
    assert guard.outputs["bypass_llm"] is True
    assert session.llm_invoked is False


def test_mission_pre_llm_replay():
    query = "8 passengers LA to Miami under $10M"
    session = build_execution_replay(
        _response_for_query(
            query,
            return_kind="llm",
            pre_llm_executed=True,
            llm_invoked=True,
            path_override="pre_llm_mission",
            extra_data_used={"mission_clarification_status": "partial"},
        )
    )
    pre = _stage(session, "pre_llm_mission")
    assert pre.outputs["executed"] is True
    assert session.final_execution_path in ("pre_llm_mission", "llm_fallback", "hybrid_unified")


def test_general_query_llm_replay():
    query = "Tell me about business aviation trends"
    session = build_execution_replay(
        _response_for_query(
            query,
            return_kind="llm",
            llm_invoked=True,
            path_override="llm_fallback",
        )
    )
    llm = _stage(session, "llm")
    assert llm.outputs["allowed"] is True
    assert session.llm_invoked is True
    assert session.final_execution_path == "llm_fallback"


def test_no_llm_fallback_when_deterministic_handled():
    for query, icrl, path in [
        ("G650 vs Falcon 8X", False, "authority_dispatch"),
        ("G650 vs Falcon 8X vs Global 7500 under $30M", False, "authority_dispatch"),
    ]:
        session = build_execution_replay(
            _response_for_query(
                query,
                icrl_handled=icrl,
                path_override=path,
                llm_invoked=False,
            )
        )
        assert session.final_execution_path != "llm_fallback"
        assert session.llm_invoked is False


def test_attach_only_when_env_enabled(monkeypatch):
    monkeypatch.delenv("ENABLE_EXECUTION_REPLAY", raising=False)
    payload = _response_for_query("G650 vs Falcon 8X")
    assert not execution_replay_enabled()
    out = attach_execution_replay_if_enabled(payload)
    assert "execution_replay" not in (out.get("data_used") or {})

    monkeypatch.setenv("ENABLE_EXECUTION_REPLAY", "1")
    out2 = attach_execution_replay_if_enabled(payload)
    assert "execution_replay" in (out2.get("data_used") or {})


def test_stream_replay_events_opt_in():
    session = build_execution_replay(_response_for_query("G650 vs Falcon 8X"))
    assert stream_execution_replay_events(session, emit=False) == []
    events = stream_execution_replay_events(session, emit=True)
    types = [e["type"] for e in events]
    assert types[0] == "replay:start"
    assert "replay:step" in types
    assert types[-1] == "replay:complete"

"""
Execution Replay Engine (ERE) — passive reconstruction from Phase 17 traces.

Does not alter routing, dispatch, ICRL, guards, normalization, UI contract, or API versioning.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_REPLAY_ENV = "ENABLE_EXECUTION_REPLAY"

_STAGE_QRI = "qri_classification"
_STAGE_UNIFIED = "unified_intent_router"
_STAGE_AUTHORITY = "authority_dispatch"
_STAGE_ICRL = "intent_conflict_resolution"
_STAGE_GUARD = "deterministic_guard"
_STAGE_PRE_LLM = "pre_llm_mission"
_STAGE_LLM = "llm"
_STAGE_NORMALIZATION = "response_normalization"
_STAGE_UI = "ui_render_contract"
_STAGE_API = "api_versioning"

_DISPATCH_RESPONDER_LABELS = {
    "comparison": "comparison_dispatch",
    "alternative": "alternative_dispatch",
    "buy_decision": "buy_decision_dispatch",
}

_ICRL_STRATEGY_LABELS = {
    "TRIPLE_PLUS_COMPARISON": "comparison_matrix",
    "MULTI_COMPARISON": "comparison_matrix",
    "COMPARISON_PLUS_CONSTRAINT": "comparison_plus_budget",
    "COMPARISON_PLUS_BUY": "comparison_plus_buy",
    "MISSION_OVERLAY": "mission_overlay",
    "SINGLE_INTENT": "single_intent",
}


@dataclass
class ReplayStep:
    step_number: int
    stage: str
    timestamp: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    decision_reason: str = ""
    execution_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "decision_reason": self.decision_reason,
            "execution_path": self.execution_path,
        }


@dataclass
class ReplaySession:
    replay_id: str
    request_id: str
    raw_query: str
    qri_intent: str
    unified_intent: str
    final_execution_path: str
    llm_invoked: bool
    replay_steps: List[ReplayStep] = field(default_factory=list)
    replay_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "request_id": self.request_id,
            "raw_query": self.raw_query,
            "qri_intent": self.qri_intent,
            "unified_intent": self.unified_intent,
            "final_execution_path": self.final_execution_path,
            "llm_invoked": self.llm_invoked,
            "replay_steps": [s.to_dict() for s in self.replay_steps],
            "replay_summary": self.replay_summary,
        }


def execution_replay_enabled() -> bool:
    return (os.getenv(_REPLAY_ENV) or "").strip().lower() in ("1", "true", "yes")


def _step_timestamp(step_number: int) -> str:
    return f"replay-stage-{step_number:02d}"


def _data_used(response: Any) -> Dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    du = response.get("data_used")
    return dict(du) if isinstance(du, dict) else {}


def _trace(response: Any, data_used: Dict[str, Any]) -> Dict[str, Any]:
    trace = data_used.get("intent_execution_trace")
    if isinstance(trace, dict):
        return trace
    if isinstance(response, dict):
        top = response.get("intent_execution_trace")
        if isinstance(top, dict):
            return top
    return {}


def _qri_alternatives(data_used: Dict[str, Any], qri_intent: str) -> List[str]:
    signals = data_used.get("query_recommendation_intent_signals")
    if isinstance(signals, list) and signals:
        return [str(s) for s in signals[:8]]
    mission = "acquisition_recommendation"
    comparison = "aircraft_comparison"
    if qri_intent == comparison:
        return [comparison, mission, "mission_feasibility"]
    if qri_intent in ("acquisition_recommendation", "mission_feasibility"):
        return [qri_intent, comparison, "shortlist_ranking"]
    return [qri_intent, comparison, mission] if qri_intent else []


def _icrl_strategy_label(conflict_type: str, plan: Dict[str, Any]) -> str:
    if plan.get("comparison_mode") == "comparison_matrix":
        return "comparison_matrix"
    labeled = _ICRL_STRATEGY_LABELS.get(conflict_type or "")
    if labeled:
        return labeled
    layout = str(plan.get("layout_type") or "")
    if "comparison_matrix" in layout:
        return "comparison_matrix"
    return str(plan.get("execution_strategy") or "hybrid_safe")


def _dispatch_reason(dispatch_kind: str) -> str:
    label = _DISPATCH_RESPONDER_LABELS.get(dispatch_kind, dispatch_kind)
    return f"Authority dispatch selected {label} responder for hard deterministic path."


def _build_replay_steps(
    *,
    trace: Dict[str, Any],
    data_used: Dict[str, Any],
    response: Dict[str, Any],
) -> List[ReplayStep]:
    steps: List[ReplayStep] = []
    n = 0
    final_path = str(trace.get("final_execution_path") or "hybrid_unified")
    qri_intent = str(trace.get("qri_intent") or data_used.get("query_recommendation_intent") or "")
    unified_intent = str(trace.get("unified_intent") or "")
    llm_invoked = bool(trace.get("llm_invoked"))

    def add(
        stage: str,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        decision_reason: str = "",
        execution_path: str = "",
    ) -> None:
        nonlocal n
        n += 1
        steps.append(
            ReplayStep(
                step_number=n,
                stage=stage,
                timestamp=_step_timestamp(n),
                inputs=dict(inputs or {}),
                outputs=dict(outputs or {}),
                decision_reason=decision_reason,
                execution_path=execution_path or final_path,
            )
        )

    confidence = data_used.get("query_recommendation_intent_confidence")
    add(
        _STAGE_QRI,
        inputs={"raw_query": trace.get("raw_query") or ""},
        outputs={
            "detected_intent": qri_intent,
            "confidence": confidence,
            "alternatives_considered": _qri_alternatives(data_used, qri_intent),
            "source": data_used.get("query_recommendation_intent_source"),
        },
        decision_reason=f"QRI classified turn as {qri_intent or 'unknown'}.",
        execution_path="classification",
    )

    shadow = data_used.get("unified_intent_shadow")
    unified_outputs: Dict[str, Any] = {
        "execution_path": unified_intent or "none",
        "route_decision": unified_intent or "none",
    }
    if isinstance(shadow, dict):
        unified_outputs["shadow"] = {
            "secondary_intent": shadow.get("secondary_intent"),
            "execution_path": shadow.get("execution_path"),
        }
    add(
        _STAGE_UNIFIED,
        inputs={"qri_intent": qri_intent},
        outputs=unified_outputs,
        decision_reason=(
            f"Unified router selected execution path {unified_intent}."
            if unified_intent
            else "Unified router inactive or returned no path."
        ),
        execution_path=unified_intent or "none",
    )

    dispatch_kind = trace.get("authority_dispatch_result") or data_used.get("authority_dispatch_kind")
    add(
        _STAGE_AUTHORITY,
        inputs={"qri_intent": qri_intent, "unified_intent": unified_intent},
        outputs={
            "triggered": bool(dispatch_kind),
            "responder_selected": _DISPATCH_RESPONDER_LABELS.get(str(dispatch_kind or ""), dispatch_kind),
            "dispatch_kind": dispatch_kind,
        },
        decision_reason=(
            _dispatch_reason(str(dispatch_kind))
            if dispatch_kind
            else "Authority dispatch not triggered; downstream layers may handle turn."
        ),
        execution_path=str(dispatch_kind) if dispatch_kind else "skipped",
    )

    icrl = data_used.get("intent_conflict_resolution")
    icrl_dict = icrl if isinstance(icrl, dict) else {}
    icrl_graph = icrl_dict.get("graph") if isinstance(icrl_dict.get("graph"), dict) else {}
    icrl_plan = trace.get("resolved_plan") or icrl_dict.get("plan") or {}
    if not isinstance(icrl_plan, dict):
        icrl_plan = {}
    conflict_type = str(trace.get("conflict_type") or icrl_dict.get("conflict_type") or "")
    strategy = _icrl_strategy_label(conflict_type, icrl_plan)
    add(
        _STAGE_ICRL,
        inputs={"entities": icrl_graph.get("entities"), "modifiers": icrl_graph.get("modifiers")},
        outputs={
            "triggered": bool(trace.get("icrl_triggered")),
            "handled": bool(trace.get("icrl_handled")),
            "conflict_type": conflict_type or None,
            "intent_graph": icrl_graph,
            "resolved_execution_plan": icrl_plan,
            "strategy_selected": strategy,
        },
        decision_reason=(
            f"ICRL resolved {conflict_type} using strategy {strategy}."
            if trace.get("icrl_triggered")
            else "ICRL evaluated; no multi-intent override required."
        ),
        execution_path="icrl_deterministic" if trace.get("icrl_handled") else "skipped",
    )

    det_exec = data_used.get("deterministic_execution")
    if not isinstance(det_exec, dict):
        det_exec = {}
    guard_result = trace.get("deterministic_guard_result")
    bypass_llm = guard_result in ("bypass", "safety_fallback") or bool(det_exec.get("bypassed_llm"))
    hard_required = bool(
        trace.get("authority_dispatch_result")
        or trace.get("icrl_handled")
        or str(det_exec.get("deterministic_intent") or "") in ("comparison", "alternative", "buy_decision")
    )
    add(
        _STAGE_GUARD,
        inputs={
            "deterministic_intent": det_exec.get("deterministic_intent"),
            "icrl_handled": trace.get("icrl_handled"),
        },
        outputs={
            "bypass_llm": bypass_llm,
            "requires_hard_deterministic": hard_required,
            "guard_result": guard_result,
            "trigger_reason": det_exec.get("trigger_reason"),
            "final_responder": det_exec.get("final_responder"),
        },
        decision_reason=(
            f"Deterministic guard: {guard_result or 'pass'}; "
            f"{'LLM bypassed' if bypass_llm else 'LLM path allowed'}."
        ),
        execution_path=str(guard_result or "pass"),
    )

    pre_llm = bool(
        data_used.get("deterministic_pre_llm_executed")
        or trace.get("pre_llm_executed")
    )
    if pre_llm or final_path == "pre_llm_mission":
        add(
            _STAGE_PRE_LLM,
            inputs={"qri_intent": qri_intent},
            outputs={
                "executed": pre_llm,
                "mission_completeness": data_used.get("mission_completeness"),
                "clarification_status": data_used.get("mission_clarification_status"),
                "recommendation_pipeline_state": data_used.get("consultant_response_mode"),
            },
            decision_reason="Pre-LLM mission pipeline produced deterministic mission brief context.",
            execution_path="pre_llm_mission",
        )

    bypass_reasons = trace.get("bypass_reasons") or []
    if not isinstance(bypass_reasons, list):
        bypass_reasons = []
    llm_reason = bypass_reasons[0] if llm_invoked and bypass_reasons else (
        "mission_incomplete" if final_path == "pre_llm_mission" else "general_query"
    )
    add(
        _STAGE_LLM,
        inputs={"final_execution_path": final_path},
        outputs={
            "allowed": llm_invoked,
            "reason": llm_reason if llm_invoked else None,
            "model_used": data_used.get("consultant_chat_model") or data_used.get("llm_model"),
        },
        decision_reason=(
            f"LLM invoked: {llm_reason}."
            if llm_invoked
            else "LLM not invoked; deterministic or dispatch path satisfied turn."
        ),
        execution_path="llm_fallback" if llm_invoked else final_path,
    )

    norm = response.get("normalized_response") or data_used.get("normalized_response")
    if isinstance(norm, dict):
        add(
            _STAGE_NORMALIZATION,
            inputs={"answer_present": bool(str(response.get("answer") or "").strip())},
            outputs={
                "intent_type": norm.get("intent_type"),
                "verdict": norm.get("verdict"),
                "confidence": norm.get("confidence"),
            },
            decision_reason=f"Response normalized as {norm.get('intent_type') or 'structured'} output.",
            execution_path=final_path,
        )

    ui = response.get("ui_render_contract") or data_used.get("ui_render_contract")
    if isinstance(ui, dict):
        add(
            _STAGE_UI,
            inputs={"normalized_intent": (norm or {}).get("intent_type") if isinstance(norm, dict) else None},
            outputs={
                "ui_intent": ui.get("ui_intent") or trace.get("ui_intent"),
                "layout_type": ui.get("layout_type"),
            },
            decision_reason=(
                f"UI contract applied: {ui.get('ui_intent')} / {ui.get('layout_type')}."
            ),
            execution_path=final_path,
        )
    elif trace.get("ui_intent"):
        add(
            _STAGE_UI,
            outputs={
                "ui_intent": trace.get("ui_intent"),
                "layout_type": (icrl_plan or {}).get("layout_type"),
            },
            decision_reason=f"UI intent derived from execution trace: {trace.get('ui_intent')}.",
            execution_path=final_path,
        )

    contract_version = (
        response.get("api_contract_version")
        or data_used.get("api_contract_version")
        or "v3"
    )
    add(
        _STAGE_API,
        outputs={"contract_version": contract_version},
        decision_reason=f"API contract version {contract_version} applied to response envelope.",
        execution_path=final_path,
    )

    return steps


def build_replay_summary(session: ReplaySession) -> str:
    """Generate concise human-readable replay summary."""
    lines: List[str] = []
    qri = session.qri_intent or "unknown"
    lines.append(f"Query classified as {qri.upper().replace('_', ' ')}.")

    for step in session.replay_steps:
        if step.stage == _STAGE_AUTHORITY and step.outputs.get("triggered"):
            responder = step.outputs.get("responder_selected") or step.outputs.get("dispatch_kind")
            lines.append(f"Authority Dispatch selected {responder} responder.")
        if step.stage == _STAGE_ICRL and step.outputs.get("handled"):
            strategy = step.outputs.get("strategy_selected")
            lines.append(f"ICRL executed strategy {strategy}.")
        if step.stage == _STAGE_GUARD and step.outputs.get("bypass_llm"):
            lines.append("Deterministic execution bypassed LLM.")
        if step.stage == _STAGE_PRE_LLM and step.outputs.get("executed"):
            lines.append("Pre-LLM mission pipeline executed.")
        if step.stage == _STAGE_NORMALIZATION and step.outputs.get("intent_type"):
            lines.append(f"Response normalized as {step.outputs.get('intent_type')}.")
        if step.stage == _STAGE_UI and step.outputs.get("layout_type"):
            lines.append(
                f"UI rendered using {step.outputs.get('layout_type')} layout "
                f"({step.outputs.get('ui_intent')})."
            )
        if step.stage == _STAGE_LLM and step.outputs.get("allowed"):
            lines.append(f"LLM executed ({step.outputs.get('reason')}).")

    lines.append(f"Final execution path: {session.final_execution_path}.")
    return "\n".join(lines)


def build_execution_replay(response: Any) -> ReplaySession:
    """
    Reconstruct execution timeline from a consultant response payload.

    Requires ``data_used.intent_execution_trace`` (Phase 17). Missing trace yields
    minimal reconstruction from available ``data_used`` metadata only.
    """
    payload = response if isinstance(response, dict) else {}
    data_used = _data_used(payload)
    trace = _trace(payload, data_used)

    request_id = str(trace.get("request_id") or data_used.get("consultant_progress_id") or "")
    raw_query = str(trace.get("raw_query") or payload.get("query") or "")

    steps = _build_replay_steps(trace=trace, data_used=data_used, response=payload)

    session = ReplaySession(
        replay_id=uuid.uuid4().hex[:16],
        request_id=request_id,
        raw_query=raw_query,
        qri_intent=str(trace.get("qri_intent") or data_used.get("query_recommendation_intent") or ""),
        unified_intent=str(trace.get("unified_intent") or ""),
        final_execution_path=str(trace.get("final_execution_path") or "hybrid_unified"),
        llm_invoked=bool(trace.get("llm_invoked")),
        replay_steps=steps,
    )
    session.replay_summary = build_replay_summary(session)
    return session


def attach_execution_replay_if_enabled(response: Any) -> Any:
    """Attach ``data_used.execution_replay`` when ``ENABLE_EXECUTION_REPLAY=1``."""
    if not execution_replay_enabled():
        return response
    if not isinstance(response, dict):
        return response
    out = dict(response)
    du = dict(out.get("data_used") or {}) if isinstance(out.get("data_used"), dict) else {}
    session = build_execution_replay(out)
    du["execution_replay"] = session.to_dict()
    out["data_used"] = du
    return out


def stream_execution_replay_events(
    replay: Any,
    *,
    emit: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build replay SSE events for debugging UI.

    Events are only returned when *emit* is True (opt-in; not emitted by default).
    """
    if not emit:
        return []
    session_dict = replay.to_dict() if hasattr(replay, "to_dict") else replay
    if not isinstance(session_dict, dict):
        return []

    events: List[Dict[str, Any]] = [
        {
            "type": "replay:start",
            "replay_id": session_dict.get("replay_id"),
            "request_id": session_dict.get("request_id"),
            "raw_query": session_dict.get("raw_query"),
        }
    ]
    for step in session_dict.get("replay_steps") or []:
        if not isinstance(step, dict):
            continue
        events.append(
            {
                "type": "replay:step",
                "step_number": step.get("step_number"),
                "stage": step.get("stage"),
                "decision_reason": step.get("decision_reason"),
                "execution_path": step.get("execution_path"),
            }
        )
    events.append(
        {
            "type": "replay:complete",
            "final_execution_path": session_dict.get("final_execution_path"),
            "llm_invoked": session_dict.get("llm_invoked"),
            "replay_summary": session_dict.get("replay_summary"),
        }
    )
    return events


__all__ = [
    "ReplaySession",
    "ReplayStep",
    "attach_execution_replay_if_enabled",
    "build_execution_replay",
    "build_replay_summary",
    "execution_replay_enabled",
    "stream_execution_replay_events",
]

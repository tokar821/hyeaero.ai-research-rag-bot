"""
Intent execution trace — observability and determinism provenance for consultant routing.

Phase 17: instruments execution without altering routing decisions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_ALLOWED_FINAL_PATHS = frozenset(
    {
        "authority_dispatch",
        "icrl_deterministic",
        "pre_llm_mission",
        "llm_fallback",
        "hybrid_unified",
    }
)

_LLM_ALLOWED_BYPASS_REASONS = frozenset(
    {"mission_incomplete", "general_query", "fact_lookup"}
)

_DETERMINISTIC_BYPASS_REASONS = frozenset(
    {
        "deterministic_only",
        "authority_dispatch_hit",
        "icrl_deterministic_only",
        "deterministic_guard_bypass",
    }
)


@dataclass
class IntentExecutionTrace:
    request_id: str
    raw_query: str
    qri_intent: str
    unified_intent: str
    conflict_type: Optional[str]
    resolved_plan: Optional[Dict[str, Any]]
    authority_dispatch_result: Optional[str]
    deterministic_guard_result: Optional[str]
    icrl_triggered: bool
    icrl_handled: bool
    llm_invoked: bool
    final_execution_path: str
    bypass_reasons: List[str] = field(default_factory=list)
    ui_intent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "raw_query": self.raw_query,
            "qri_intent": self.qri_intent,
            "unified_intent": self.unified_intent,
            "conflict_type": self.conflict_type,
            "resolved_plan": self.resolved_plan,
            "authority_dispatch_result": self.authority_dispatch_result,
            "deterministic_guard_result": self.deterministic_guard_result,
            "icrl_triggered": self.icrl_triggered,
            "icrl_handled": self.icrl_handled,
            "llm_invoked": self.llm_invoked,
            "final_execution_path": self.final_execution_path,
            "bypass_reasons": list(self.bypass_reasons),
            "ui_intent": self.ui_intent,
        }


class IntentExecutionTraceCapture:
    """Mutable capture bag updated at pipeline instrumentation points."""

    def __init__(self, *, raw_query: str, request_id: Optional[str] = None) -> None:
        self.request_id = (request_id or uuid.uuid4().hex[:12]).strip()
        self.raw_query = raw_query or ""
        self.qri_intent = ""
        self.unified_intent = ""
        self.conflict_type: Optional[str] = None
        self.resolved_plan: Optional[Dict[str, Any]] = None
        self.authority_dispatch_result: Optional[str] = None
        self.deterministic_guard_result: Optional[str] = None
        self.icrl_triggered = False
        self.icrl_handled = False
        self.pre_llm_executed = False
        self.intent_lock_snapshot: Optional[Dict[str, Any]] = None
        self.icrl_resolution_snapshot: Optional[Dict[str, Any]] = None

    def capture_intent_lock(self, intent_lock: Any = None) -> None:
        if intent_lock is None:
            return
        if hasattr(intent_lock, "to_dict"):
            self.intent_lock_snapshot = intent_lock.to_dict()
        elif isinstance(intent_lock, dict):
            self.intent_lock_snapshot = dict(intent_lock)

    def capture_qri_unified(self, qri: Any = None, unified_route: Any = None) -> None:
        if qri is not None and getattr(qri, "intent", None) is not None:
            self.qri_intent = str(qri.intent.value)
        self.unified_intent = _unified_intent_str(unified_route)

    def capture_authority_dispatch(self, result: Any = None) -> None:
        if result is None:
            self.authority_dispatch_result = None
            return
        self.authority_dispatch_result = str(getattr(result, "dispatch_kind", "") or "") or None

    def capture_icrl(self, resolution: Any = None) -> None:
        if resolution is None:
            return
        self.icrl_triggered = True
        self.icrl_handled = bool(getattr(resolution, "handled_by_icrl", False))
        conflict = getattr(resolution, "conflict_type", None)
        if conflict is not None:
            self.conflict_type = str(getattr(conflict, "value", conflict))
        plan = getattr(resolution, "plan", None)
        if plan is not None and hasattr(plan, "to_dict"):
            self.resolved_plan = plan.to_dict()
        if hasattr(resolution, "to_dict"):
            self.icrl_resolution_snapshot = resolution.to_dict()

    def capture_deterministic_guard(
        self,
        *,
        should_bypass: bool = False,
        resolve_hit: bool = False,
        safety_fallback: bool = False,
    ) -> None:
        if safety_fallback:
            self.deterministic_guard_result = "safety_fallback"
        elif should_bypass and resolve_hit:
            self.deterministic_guard_result = "bypass"
        elif should_bypass:
            self.deterministic_guard_result = "bypass"
        else:
            self.deterministic_guard_result = "pass"

    def mark_pre_llm_executed(self) -> None:
        self.pre_llm_executed = True

    def to_build_context(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "raw_query": self.raw_query,
            "qri_intent": self.qri_intent,
            "unified_intent": self.unified_intent,
            "conflict_type": self.conflict_type,
            "resolved_plan": self.resolved_plan,
            "authority_dispatch_result": self.authority_dispatch_result,
            "deterministic_guard_result": self.deterministic_guard_result,
            "icrl_triggered": self.icrl_triggered,
            "icrl_handled": self.icrl_handled,
            "pre_llm_executed": self.pre_llm_executed,
            "intent_lock_snapshot": self.intent_lock_snapshot,
            "icrl_resolution_snapshot": self.icrl_resolution_snapshot,
        }


def _unified_intent_str(unified_route: Any) -> str:
    if unified_route is None:
        return ""
    execution_path = getattr(unified_route, "execution_path", None)
    if execution_path is not None and getattr(execution_path, "value", None):
        return str(execution_path.value)
    intent = getattr(unified_route, "intent", None)
    if intent is not None and getattr(intent, "value", None):
        return str(intent.value)
    return ""


def _extract_ui_intent(
    *,
    resolved_plan: Optional[Dict[str, Any]],
    authority_dispatch_result: Optional[str],
    data_used: Optional[Dict[str, Any]],
) -> Optional[str]:
    if isinstance(resolved_plan, dict):
        ui = str(resolved_plan.get("ui_intent") or "").strip()
        if ui:
            return ui
    if authority_dispatch_result:
        return authority_dispatch_result
    if isinstance(data_used, dict):
        icrl = data_used.get("intent_conflict_resolution") or {}
        if isinstance(icrl, dict):
            plan = icrl.get("plan") or {}
            if isinstance(plan, dict):
                ui = str(plan.get("ui_intent") or "").strip()
                if ui:
                    return ui
        norm = data_used.get("normalized_response") or {}
        if isinstance(norm, dict):
            ui = str(norm.get("intent_type") or "").strip()
            if ui:
                return ui
        du_ui = str(data_used.get("authority_dispatch_kind") or "").strip()
        if du_ui:
            return du_ui
    return None


def _infer_llm_bypass_reasons(qri_intent: str) -> List[str]:
    mission_intents = {
        "acquisition_recommendation",
        "mission_feasibility",
        "shortlist_ranking",
    }
    fact_intents = {
        "ownership_economics",
        "payload_range_analysis",
        "operational_tradeoff_analysis",
    }
    if qri_intent in mission_intents:
        return ["mission_incomplete"]
    if qri_intent in fact_intents or qri_intent == "aircraft_fact":
        return ["fact_lookup"]
    return ["general_query"]


def _resolve_final_path(
    *,
    return_kind: str,
    icrl_handled: bool,
    authority_dispatch_result: Optional[str],
    deterministic_guard_result: Optional[str],
    pre_llm_executed: bool,
    path_override: Optional[str],
    unified_enforced: bool,
) -> str:
    if path_override in _ALLOWED_FINAL_PATHS:
        return path_override

    if authority_dispatch_result and return_kind == "professional":
        return "authority_dispatch"
    if icrl_handled:
        return "icrl_deterministic"
    if deterministic_guard_result in ("bypass", "safety_fallback") and return_kind == "professional":
        return "hybrid_unified"
    if unified_enforced and return_kind == "professional":
        return "hybrid_unified"
    if return_kind == "llm":
        if pre_llm_executed:
            return "pre_llm_mission"
        return "llm_fallback"
    if return_kind == "professional":
        return "hybrid_unified"
    if return_kind in ("small_talk", "gk"):
        return "hybrid_unified"
    return "llm_fallback"


def _build_bypass_reasons(
    *,
    final_path: str,
    llm_invoked: bool,
    icrl_handled: bool,
    authority_dispatch_result: Optional[str],
    deterministic_guard_result: Optional[str],
    qri_intent: str,
) -> List[str]:
    if llm_invoked:
        return _infer_llm_bypass_reasons(qri_intent)

    reasons: List[str] = []
    if icrl_handled:
        reasons.append("icrl_deterministic_only")
    if authority_dispatch_result:
        reasons.append("authority_dispatch_hit")
    if deterministic_guard_result == "bypass":
        reasons.append("deterministic_guard_bypass")
    if deterministic_guard_result == "safety_fallback":
        reasons.append("deterministic_guard_bypass")
    if not reasons and final_path in ("icrl_deterministic", "authority_dispatch", "hybrid_unified"):
        reasons.append("deterministic_only")
    return reasons


def _enforce_trace_rules(trace: IntentExecutionTrace) -> IntentExecutionTrace:
    """Apply Phase 17 enforcement rules without changing routing."""
    deterministic_handled = bool(
        trace.icrl_handled
        or trace.authority_dispatch_result
        or trace.deterministic_guard_result in ("bypass", "safety_fallback")
    )
    if deterministic_handled and trace.final_execution_path == "llm_fallback":
        if trace.authority_dispatch_result:
            trace.final_execution_path = "authority_dispatch"
        elif trace.icrl_handled:
            trace.final_execution_path = "icrl_deterministic"
        else:
            trace.final_execution_path = "hybrid_unified"

    if trace.llm_invoked:
        trace.bypass_reasons = [
            r for r in trace.bypass_reasons if r not in _DETERMINISTIC_BYPASS_REASONS
        ]
        if not trace.bypass_reasons:
            trace.bypass_reasons = _infer_llm_bypass_reasons(trace.qri_intent)
        for reason in trace.bypass_reasons:
            if reason not in _LLM_ALLOWED_BYPASS_REASONS:
                trace.bypass_reasons = _infer_llm_bypass_reasons(trace.qri_intent)
                break
    else:
        forbidden = set(trace.bypass_reasons) & _LLM_ALLOWED_BYPASS_REASONS
        if forbidden:
            trace.bypass_reasons = [
                r for r in trace.bypass_reasons if r not in _LLM_ALLOWED_BYPASS_REASONS
            ]

    if (trace.resolved_plan or trace.authority_dispatch_result) and not trace.ui_intent:
        trace.ui_intent = _extract_ui_intent(
            resolved_plan=trace.resolved_plan,
            authority_dispatch_result=trace.authority_dispatch_result,
            data_used=None,
        )

    if trace.final_execution_path not in _ALLOWED_FINAL_PATHS:
        trace.final_execution_path = "hybrid_unified"

    return trace


def build_intent_execution_trace(context: Dict[str, Any]) -> IntentExecutionTrace:
    """Build a finalized execution trace from capture context."""
    ctx = context if isinstance(context, dict) else {}
    return_kind = str(ctx.get("return_kind") or "")
    llm_invoked = return_kind == "llm" or bool(ctx.get("llm_invoked"))

    final_path = _resolve_final_path(
        return_kind=return_kind,
        icrl_handled=bool(ctx.get("icrl_handled")),
        authority_dispatch_result=ctx.get("authority_dispatch_result"),
        deterministic_guard_result=ctx.get("deterministic_guard_result"),
        pre_llm_executed=bool(ctx.get("pre_llm_executed")),
        path_override=ctx.get("path_override"),
        unified_enforced=bool(ctx.get("unified_enforced")),
    )
    bypass_reasons = _build_bypass_reasons(
        final_path=final_path,
        llm_invoked=llm_invoked,
        icrl_handled=bool(ctx.get("icrl_handled")),
        authority_dispatch_result=ctx.get("authority_dispatch_result"),
        deterministic_guard_result=ctx.get("deterministic_guard_result"),
        qri_intent=str(ctx.get("qri_intent") or ""),
    )
    data_used = ctx.get("data_used") if isinstance(ctx.get("data_used"), dict) else None
    ui_intent = _extract_ui_intent(
        resolved_plan=ctx.get("resolved_plan"),
        authority_dispatch_result=ctx.get("authority_dispatch_result"),
        data_used=data_used,
    )

    trace = IntentExecutionTrace(
        request_id=str(ctx.get("request_id") or uuid.uuid4().hex[:12]),
        raw_query=str(ctx.get("raw_query") or ""),
        qri_intent=str(ctx.get("qri_intent") or ""),
        unified_intent=str(ctx.get("unified_intent") or ""),
        conflict_type=ctx.get("conflict_type"),
        resolved_plan=ctx.get("resolved_plan"),
        authority_dispatch_result=ctx.get("authority_dispatch_result"),
        deterministic_guard_result=ctx.get("deterministic_guard_result"),
        icrl_triggered=bool(ctx.get("icrl_triggered")),
        icrl_handled=bool(ctx.get("icrl_handled")),
        llm_invoked=llm_invoked,
        final_execution_path=final_path,
        bypass_reasons=bypass_reasons,
        ui_intent=ui_intent,
    )
    return _enforce_trace_rules(trace)


def attach_intent_execution_trace(
    capture: IntentExecutionTraceCapture,
    return_kind: str,
    payload: Any,
    *,
    path_override: Optional[str] = None,
    unified_enforced: bool = False,
    llm_invoked: Optional[bool] = None,
) -> Any:
    """Attach finalized trace to payload data_used and return payload."""
    data_used: Dict[str, Any] = {}
    if isinstance(payload, dict):
        raw_du = payload.get("data_used")
        data_used = dict(raw_du) if isinstance(raw_du, dict) else {}

    build_ctx = capture.to_build_context()
    build_ctx.update(
        {
            "return_kind": return_kind,
            "path_override": path_override,
            "unified_enforced": unified_enforced,
            "data_used": data_used,
            "llm_invoked": llm_invoked if llm_invoked is not None else return_kind == "llm",
        }
    )
    trace = build_intent_execution_trace(build_ctx)
    data_used["intent_execution_trace"] = trace.to_dict()

    try:
        from services.core.semantic_intent_lock_engine import build_execution_trace_v2

        answer_text = ""
        if isinstance(payload, dict):
            answer_text = str(payload.get("answer") or "")
        data_used["execution_trace_v2"] = build_execution_trace_v2(
            intent_lock=data_used.get("intent_lock") or capture.intent_lock_snapshot,
            icrl_resolution=capture.icrl_resolution_snapshot,
            data_used=data_used,
            final_answer=answer_text,
        )
    except Exception:
        pass

    if isinstance(payload, dict):
        payload = dict(payload)
        payload["data_used"] = data_used
    return payload


def stream_trace_events(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Lightweight delta-safe trace events for SSE streaming."""
    if not isinstance(trace, dict):
        return []

    events: List[Dict[str, Any]] = []
    if trace.get("icrl_triggered"):
        events.append(
            {
                "type": "trace:icrl",
                "handled": bool(trace.get("icrl_handled")),
                "conflict": trace.get("conflict_type"),
                "strategy": (trace.get("resolved_plan") or {}).get("execution_strategy"),
            }
        )
    if trace.get("authority_dispatch_result"):
        events.append(
            {
                "type": "trace:authority_dispatch",
                "dispatch": trace.get("authority_dispatch_result"),
            }
        )
    if trace.get("deterministic_guard_result"):
        events.append(
            {
                "type": "trace:deterministic_guard",
                "result": trace.get("deterministic_guard_result"),
            }
        )
    events.append(
        {
            "type": "trace:final_path",
            "path": trace.get("final_execution_path"),
            "llm_invoked": bool(trace.get("llm_invoked")),
            "ui_intent": trace.get("ui_intent"),
        }
    )
    return events


__all__ = [
    "IntentExecutionTrace",
    "IntentExecutionTraceCapture",
    "attach_intent_execution_trace",
    "build_intent_execution_trace",
    "stream_trace_events",
]

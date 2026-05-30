"""
Unified pipeline gate — single enforcement boundary after UnifiedIntentRouter.

Router → PipelineGate → Handler

Downstream code reads ``route.execution_path`` only; it must not reinterpret intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from services.routing.unified_intent_router import (
    UnifiedExecutionPath,
    UnifiedIntentRoute,
    get_secondary_intent_promotion_contract,
)

HandlerResult = Tuple[str, Dict[str, Any], str]


@dataclass(frozen=True)
class UnifiedPipelineGateDecision:
    execution_path: UnifiedExecutionPath
    enforce: bool
    progress_step: str


def evaluate_pipeline_gate(
    route: UnifiedIntentRoute,
    *,
    enforce_fact: bool = False,
    enforce_capability: bool = False,
    enforce_comparison: bool = False,
    enforce_alternative: bool = False,
) -> UnifiedPipelineGateDecision:
    """
    Map router authority to an enforcement decision using feature flags only.

    Does not inspect query text or re-derive intent.
    """
    path = route.execution_path
    flag_for_path = {
        UnifiedExecutionPath.AIRCRAFT_FACT: enforce_fact,
        UnifiedExecutionPath.AIRCRAFT_MARKET_FACT: enforce_fact,
        UnifiedExecutionPath.CAPABILITY: enforce_capability,
        UnifiedExecutionPath.COMPARISON: enforce_comparison,
        UnifiedExecutionPath.ALTERNATIVE: enforce_alternative,
    }
    enforce = bool(flag_for_path.get(path, False))
    step_map = {
        UnifiedExecutionPath.AIRCRAFT_FACT: "path_unified_fact_responder",
        UnifiedExecutionPath.AIRCRAFT_MARKET_FACT: "path_unified_fact_responder",
        UnifiedExecutionPath.CAPABILITY: "path_unified_capability_responder",
        UnifiedExecutionPath.COMPARISON: "path_unified_comparison_responder",
        UnifiedExecutionPath.ALTERNATIVE: "path_unified_alternative_responder",
    }
    return UnifiedPipelineGateDecision(
        execution_path=path,
        enforce=enforce,
        progress_step=step_map.get(path, ""),
    )


def execute_unified_pipeline_handler(
    route: UnifiedIntentRoute,
    decision: UnifiedPipelineGateDecision,
    query: str,
    *,
    shadow_payload: Optional[Dict[str, Any]] = None,
) -> HandlerResult:
    """
    Dispatch to the handler named by ``route.execution_path``.

    Returns (answer, data_used_patch, log_message).
    """
    path = decision.execution_path
    q = query or ""
    du: Dict[str, Any] = {
        "unified_intent": route.intent.value,
        "unified_execution_path": path.value,
        "unified_intent_promotion": get_secondary_intent_promotion_contract(route),
    }
    if shadow_payload:
        du["unified_intent_shadow"] = shadow_payload

    if path == UnifiedExecutionPath.AIRCRAFT_FACT:
        from services.fact.aircraft_fact_responder import respond_aircraft_fact

        du.update(
            {
                "unified_intent_enforced": 1,
                "unified_fact_model": route.model,
                "unified_fact_field": route.field,
            }
        )
        answer = respond_aircraft_fact(route.model or "", route.field or "")
        return answer, du, f"Unified fact path model={route.model!r} field={route.field!r}"

    if path == UnifiedExecutionPath.AIRCRAFT_MARKET_FACT:
        from services.fact.aircraft_fact_responder import respond_aircraft_fact

        du.update(
            {
                "unified_intent_enforced": 1,
                "unified_fact_model": route.model,
                "unified_fact_field": route.field,
            }
        )
        answer = respond_aircraft_fact(route.model or "", route.field or "")
        return answer, du, f"Unified market fact path model={route.model!r} field={route.field!r}"

    if path == UnifiedExecutionPath.CAPABILITY:
        from services.fact.named_aircraft_capability_responder import respond_aircraft_capability

        du.update(
            {
                "unified_capability_enforced": 1,
                "unified_capability_model": route.model,
            }
        )
        answer = respond_aircraft_capability(route.model or "", q)
        return answer, du, f"Unified capability path model={route.model!r}"

    if path == UnifiedExecutionPath.ALTERNATIVE:
        from services.comparison.alternative_pipeline_responder import respond_aircraft_alternative

        du["unified_alternative_enforced"] = 1
        answer = respond_aircraft_alternative(q)
        return answer, du, f"Unified alternative path query={q[:120]!r}"

    if path == UnifiedExecutionPath.COMPARISON:
        from services.comparison.comparison_pipeline_v2_responder import respond_aircraft_comparison

        du["unified_comparison_enforced"] = 1
        answer = respond_aircraft_comparison(q)
        return answer, du, f"Unified comparison path query={q[:120]!r}"

    raise ValueError(f"No unified handler for execution path {path.value}")


__all__ = [
    "UnifiedPipelineGateDecision",
    "evaluate_pipeline_gate",
    "execute_unified_pipeline_handler",
]

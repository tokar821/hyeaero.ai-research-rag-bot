"""
Phase 5 observability orchestration — attach drift, flags, telemetry to data_used.

Does not modify routing, execution_path, or PipelineGate behavior.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from services.routing.unified_intent_drift_monitor import detect_intent_drift
from services.routing.unified_intent_flag_validator import validate_flag_consistency
from services.routing.unified_intent_router import UnifiedIntentRoute
from services.routing.unified_pipeline_gate import UnifiedPipelineGateDecision
from services.telemetry.unified_intent_telemetry import (
    build_shadow_normalized,
    build_unified_telemetry_event,
    record_unified_intent_telemetry,
)


def attach_unified_intent_observability(
    data_used: Dict[str, Any],
    *,
    query: str,
    route: UnifiedIntentRoute,
    gate: UnifiedPipelineGateDecision,
    qri_intent: str,
    shadow_mode: bool,
    enforce_fact: bool = False,
    enforce_capability: bool = False,
    enforce_comparison: bool = False,
    enforce_alternative: bool = False,
    latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Enrich data_used with drift, flag validation, shadow normalization, and telemetry.

    Observability only — no routing side effects.
    """
    router_path = route.execution_path.value
    gate_path = gate.execution_path.value

    drift_event = detect_intent_drift(
        route,
        qri_intent=qri_intent,
        gate_execution_path=gate_path,
    )
    flag_result = validate_flag_consistency(
        route,
        enforce_fact=enforce_fact,
        enforce_capability=enforce_capability,
        enforce_comparison=enforce_comparison,
        enforce_alternative=enforce_alternative,
    )

    alignment_status = "aligned" if not drift_event.get("is_mismatch") else "diverged"
    divergence_reason = None
    if drift_event.get("mismatch_type"):
        divergence_reason = ",".join(drift_event["mismatch_type"])

    shadow_normalized = build_shadow_normalized(
        router_execution_path=router_path,
        gate_execution_path=gate_path,
        qri_intent=qri_intent,
        alignment_status=alignment_status,
        divergence_reason=divergence_reason,
    )

    telemetry = build_unified_telemetry_event(
        query=query,
        route_model=route.model,
        execution_path=router_path,
        gate_path=gate_path,
        qri_intent=qri_intent,
        router_intent=route.intent.value,
        shadow_mode=shadow_mode,
        drift_detected=bool(drift_event.get("is_mismatch")),
        latency_ms=latency_ms if latency_ms is not None else 0.0,
        drift_event=drift_event,
        flag_validation=flag_result.to_dict(),
    )
    record_unified_intent_telemetry(telemetry)

    data_used["unified_intent_drift"] = drift_event
    data_used["unified_intent_flag_validation"] = flag_result.to_dict()
    data_used["unified_intent_telemetry"] = telemetry

    shadow = data_used.get("unified_intent_shadow")
    if isinstance(shadow, dict):
        shadow["shadow_normalized"] = shadow_normalized
        data_used["unified_intent_shadow"] = shadow
    else:
        data_used["unified_intent_shadow"] = {"shadow_normalized": shadow_normalized}

    return data_used


class ObservabilityTimer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0


__all__ = ["ObservabilityTimer", "attach_unified_intent_observability"]

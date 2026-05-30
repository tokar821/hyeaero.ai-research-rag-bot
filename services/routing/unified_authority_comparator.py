"""
Unified authority comparator — structural pipeline comparison (observe-only).

Compares execution path, responder type, latency, and output length.
Does NOT compare semantic meaning or modify routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from services.routing.unified_intent_router import UnifiedExecutionPath, UnifiedIntentRoute
from services.routing.unified_pipeline_gate import UnifiedPipelineGateDecision

# Structural QRI → legacy responder category mapping (observability only).
_QRI_LEGACY_RESPONDER: Dict[str, str] = {
    "payload_range_analysis": "legacy_fact",
    "ownership_economics": "legacy_market",
    "aircraft_comparison": "legacy_comparison",
    "mission_feasibility": "legacy_mission",
    "acquisition_recommendation": "legacy_mission",
    "shortlist_ranking": "legacy_mission",
    "operational_tradeoff_analysis": "legacy_mission",
    "aircraft_critique": "legacy_critique",
    "visualization_request": "legacy_visualization",
}

# Unified execution_path → responder category.
_UNIFIED_RESPONDER: Dict[str, str] = {
    UnifiedExecutionPath.AIRCRAFT_FACT.value: "unified_fact",
    UnifiedExecutionPath.AIRCRAFT_MARKET_FACT.value: "unified_market",
    UnifiedExecutionPath.CAPABILITY.value: "unified_capability",
    UnifiedExecutionPath.COMPARISON.value: "unified_comparison",
    UnifiedExecutionPath.ALTERNATIVE.value: "unified_alternative",
    UnifiedExecutionPath.NONE.value: "unified_none",
}

# Structural alignment: unified path category compatible with legacy QRI category.
_ALIGNED_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("unified_fact", "legacy_fact"),
        ("unified_market", "legacy_market"),
        ("unified_market", "legacy_fact"),
        ("unified_capability", "legacy_mission"),
        ("unified_capability", "legacy_fact"),
        ("unified_comparison", "legacy_comparison"),
        ("unified_alternative", "legacy_mission"),
        ("unified_alternative", "legacy_comparison"),
        ("unified_none", "legacy_mission"),
        ("unified_none", "legacy_fact"),
        ("unified_none", "legacy_comparison"),
        ("unified_none", "legacy_market"),
        ("unified_none", "legacy_critique"),
        ("unified_none", "legacy_visualization"),
    }
)


@dataclass(frozen=True)
class AuthorityComparison:
    aligned: bool
    divergence_reason: Optional[str]
    unified_execution_path: str
    legacy_responder_type: str
    unified_responder_type: str
    unified_latency_ms: float
    legacy_latency_ms: float
    unified_output_length: int
    legacy_output_length: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aligned": self.aligned,
            "divergence_reason": self.divergence_reason,
            "unified_execution_path": self.unified_execution_path,
            "legacy_responder_type": self.legacy_responder_type,
            "unified_responder_type": self.unified_responder_type,
            "unified_latency_ms": round(float(self.unified_latency_ms), 2),
            "legacy_latency_ms": round(float(self.legacy_latency_ms), 2),
            "unified_output_length": int(self.unified_output_length),
            "legacy_output_length": int(self.legacy_output_length),
        }


def _legacy_responder_type(qri_intent: str) -> str:
    return _QRI_LEGACY_RESPONDER.get((qri_intent or "").strip().lower(), "legacy_general")


def _unified_responder_type(route: UnifiedIntentRoute, gate: UnifiedPipelineGateDecision) -> str:
    if gate.enforce:
        return _UNIFIED_RESPONDER.get(route.execution_path.value, "unified_handler")
    return _UNIFIED_RESPONDER.get(route.execution_path.value, "unified_none")


def compare_authority(
    route: UnifiedIntentRoute,
    gate: UnifiedPipelineGateDecision,
    *,
    qri_intent: str,
    unified_selected: bool,
    unified_latency_ms: float = 0.0,
    legacy_latency_ms: float = 0.0,
    unified_output_length: int = 0,
    legacy_output_length: int = 0,
) -> AuthorityComparison:
    """
    Structural comparison between unified and legacy pipeline metadata.

    When only one pipeline executes, missing metrics default to zero.
    """
    unified_path = route.execution_path.value
    unified_type = _unified_responder_type(route, gate)
    legacy_type = _legacy_responder_type(qri_intent)

    aligned = (unified_type, legacy_type) in _ALIGNED_PAIRS

    divergence: Optional[str] = None
    if not aligned:
        divergence = (
            f"Structural mismatch: unified={unified_type}({unified_path}) "
            f"vs legacy={legacy_type}({qri_intent}); "
            f"unified_selected={unified_selected}."
        )

    if unified_selected and gate.enforce and unified_output_length == 0:
        aligned = False
        divergence = (divergence or "") + " unified_selected but zero output length."

    if not unified_selected and route.execution_path != UnifiedExecutionPath.NONE and gate.enforce:
        aligned = False
        divergence = (divergence or "") + " gate enforce true but unified not selected."

    return AuthorityComparison(
        aligned=aligned,
        divergence_reason=divergence.strip() if divergence else None,
        unified_execution_path=unified_path,
        legacy_responder_type=legacy_type,
        unified_responder_type=unified_type,
        unified_latency_ms=unified_latency_ms,
        legacy_latency_ms=legacy_latency_ms,
        unified_output_length=unified_output_length,
        legacy_output_length=legacy_output_length,
    )


__all__ = ["AuthorityComparison", "compare_authority"]

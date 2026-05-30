"""
Unified intent drift monitor — observability-only divergence detection.

Does not modify routing, gates, or handlers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.routing.unified_intent_router import (
    UnifiedExecutionPath,
    UnifiedIntentRoute,
)

_MISSION_QRI_INTENTS = frozenset(
    {
        "mission_feasibility",
        "shortlist_ranking",
        "acquisition_recommendation",
    }
)


def _qri_vs_unified_mismatch(route: UnifiedIntentRoute, qri_intent: str) -> bool:
    qri = (qri_intent or "").strip().lower()
    path = route.execution_path
    if path in (UnifiedExecutionPath.AIRCRAFT_FACT, UnifiedExecutionPath.AIRCRAFT_MARKET_FACT):
        return qri in _MISSION_QRI_INTENTS
    if path == UnifiedExecutionPath.CAPABILITY:
        return qri == "aircraft_comparison"
    if path == UnifiedExecutionPath.COMPARISON:
        return qri in _MISSION_QRI_INTENTS
    if path == UnifiedExecutionPath.ALTERNATIVE:
        return qri in _MISSION_QRI_INTENTS
    return False


def _fact_conflict(route: UnifiedIntentRoute, qri_intent: str) -> bool:
    if route.execution_path not in (
        UnifiedExecutionPath.AIRCRAFT_FACT,
        UnifiedExecutionPath.AIRCRAFT_MARKET_FACT,
    ):
        return False
    qri = (qri_intent or "").strip().lower()
    return qri in _MISSION_QRI_INTENTS


def _capability_conflict(route: UnifiedIntentRoute, qri_intent: str) -> bool:
    if route.execution_path != UnifiedExecutionPath.CAPABILITY:
        return False
    qri = (qri_intent or "").strip().lower()
    return qri in ("aircraft_comparison", "shortlist_ranking", "acquisition_recommendation")


def _comparison_conflict(route: UnifiedIntentRoute, qri_intent: str) -> bool:
    if route.execution_path != UnifiedExecutionPath.COMPARISON:
        return False
    qri = (qri_intent or "").strip().lower()
    return qri in _MISSION_QRI_INTENTS


def detect_intent_drift(
    route: UnifiedIntentRoute,
    *,
    qri_intent: str,
    gate_execution_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build drift event comparing router authority, gate path, and legacy QRI.
    """
    router_path = route.execution_path.value
    gate_path = gate_execution_path or router_path
    mismatch_types: List[str] = []

    if _qri_vs_unified_mismatch(route, qri_intent):
        mismatch_types.append("qri_vs_unified")
    if router_path != gate_path:
        mismatch_types.append("router_vs_gate")
    if _fact_conflict(route, qri_intent):
        mismatch_types.append("fact_conflict")
    if _capability_conflict(route, qri_intent):
        mismatch_types.append("capability_conflict")
    if _comparison_conflict(route, qri_intent):
        mismatch_types.append("comparison_conflict")

    return {
        "qri_intent": qri_intent,
        "unified_intent": route.intent.value,
        "execution_path": router_path,
        "gate_path": gate_path,
        "is_mismatch": bool(mismatch_types),
        "mismatch_type": mismatch_types,
    }


def is_critical_router_gate_drift(router_path: str, gate_path: str) -> bool:
    return bool(router_path) and bool(gate_path) and router_path != gate_path


__all__ = [
    "detect_intent_drift",
    "is_critical_router_gate_drift",
]

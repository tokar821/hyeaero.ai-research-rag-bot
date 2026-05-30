"""
Unified intent execution gates — authority lock (Step 4).

Execution layer reads ``route.execution_path`` only.
Do not re-interpret query text or promotion contract here.
"""

from __future__ import annotations

from services.routing.unified_intent_router import UnifiedExecutionPath, UnifiedIntentRoute


def should_enforce_capability_path(route: UnifiedIntentRoute) -> bool:
    return route.execution_path == UnifiedExecutionPath.CAPABILITY


def should_enforce_comparison_path(route: UnifiedIntentRoute, query: str = "") -> bool:
    del query
    return route.execution_path == UnifiedExecutionPath.COMPARISON


def should_enforce_alternative_path(route: UnifiedIntentRoute, query: str = "") -> bool:
    del query
    return route.execution_path == UnifiedExecutionPath.ALTERNATIVE


def should_enforce_fact_path(route: UnifiedIntentRoute) -> bool:
    return route.execution_path in (
        UnifiedExecutionPath.AIRCRAFT_FACT,
        UnifiedExecutionPath.AIRCRAFT_MARKET_FACT,
    )


__all__ = [
    "should_enforce_alternative_path",
    "should_enforce_capability_path",
    "should_enforce_comparison_path",
    "should_enforce_fact_path",
]

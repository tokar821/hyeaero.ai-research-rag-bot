"""
Unified intent production metrics — passive counters for Phase 6 hardening observability.

Does NOT influence routing or gate decisions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional

from services.routing.unified_intent_ambiguity_classifier import AmbiguityReport
from services.routing.unified_intent_router import UnifiedExecutionPath, UnifiedIntentRoute

_COUNTERS: Dict[str, Any] = {
    "hardening_failure_count": 0,
    "execution_path_none_count": 0,
    "ambiguity_rate_by_intent": defaultdict(int),
    "legacy_fallback_rate": 0,
    "capability_without_model_rate": 0,
}


def get_production_metrics() -> Dict[str, Any]:
    """Return a snapshot of accumulated production metrics."""
    by_intent = _COUNTERS["ambiguity_rate_by_intent"]
    return {
        "hardening_failure_count": int(_COUNTERS["hardening_failure_count"]),
        "execution_path_none_count": int(_COUNTERS["execution_path_none_count"]),
        "ambiguity_rate_by_intent": dict(by_intent),
        "legacy_fallback_rate": int(_COUNTERS["legacy_fallback_rate"]),
        "capability_without_model_rate": int(_COUNTERS["capability_without_model_rate"]),
    }


def reset_production_metrics() -> None:
    """Reset counters — for tests only."""
    _COUNTERS["hardening_failure_count"] = 0
    _COUNTERS["execution_path_none_count"] = 0
    _COUNTERS["ambiguity_rate_by_intent"] = defaultdict(int)
    _COUNTERS["legacy_fallback_rate"] = 0
    _COUNTERS["capability_without_model_rate"] = 0


def record_hardening_event(
    route: UnifiedIntentRoute,
    ambiguity_report: AmbiguityReport,
    *,
    routing_failure: bool = False,
    requires_fallback_analysis: bool = False,
    capability_without_model: bool = False,
) -> Dict[str, Any]:
    """
    Record a hardening observability event and return metrics snapshot delta.
    """
    if route.execution_path == UnifiedExecutionPath.NONE:
        _COUNTERS["execution_path_none_count"] += 1

    if routing_failure:
        _COUNTERS["hardening_failure_count"] += 1

    if requires_fallback_analysis:
        _COUNTERS["legacy_fallback_rate"] += 1

    if capability_without_model:
        _COUNTERS["capability_without_model_rate"] += 1

    if ambiguity_report.is_ambiguous:
        intent_key = route.intent.value
        _COUNTERS["ambiguity_rate_by_intent"][intent_key] += 1

    return get_production_metrics()


__all__ = [
    "get_production_metrics",
    "record_hardening_event",
    "reset_production_metrics",
]

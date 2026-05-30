"""
Unified rollout telemetry — passive counters for Phase 7 production rollout.

Does NOT influence routing or gate decisions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from services.routing.unified_authority_comparator import AuthorityComparison
from services.routing.unified_rollout_controller import RolloutDecision

_COUNTERS: Dict[str, int] = {
    "unified_selected_count": 0,
    "legacy_selected_count": 0,
    "authority_divergence_count": 0,
    "rollback_trigger_count": 0,
}

_LAST_ROLLOUT_PERCENT: int = 0


def get_rollout_telemetry_snapshot() -> Dict[str, Any]:
    """Return accumulated rollout telemetry counters."""
    total = _COUNTERS["unified_selected_count"] + _COUNTERS["legacy_selected_count"]
    divergence_rate = (
        _COUNTERS["authority_divergence_count"] / total if total else 0.0
    )
    return {
        "rollout_percentage": _LAST_ROLLOUT_PERCENT,
        "unified_selected_count": _COUNTERS["unified_selected_count"],
        "legacy_selected_count": _COUNTERS["legacy_selected_count"],
        "authority_divergence_count": _COUNTERS["authority_divergence_count"],
        "rollback_trigger_count": _COUNTERS["rollback_trigger_count"],
        "authority_divergence_rate": round(divergence_rate, 4),
        "total_rollout_events": total,
    }


def reset_rollout_telemetry() -> None:
    """Reset counters — for tests only."""
    _COUNTERS["unified_selected_count"] = 0
    _COUNTERS["legacy_selected_count"] = 0
    _COUNTERS["authority_divergence_count"] = 0
    _COUNTERS["rollback_trigger_count"] = 0
    global _LAST_ROLLOUT_PERCENT
    _LAST_ROLLOUT_PERCENT = 0


def record_rollout_event(
    decision: RolloutDecision,
    *,
    comparison: Optional[AuthorityComparison] = None,
    rollback_triggered: bool = False,
) -> Dict[str, Any]:
    """Record a rollout selection event and optional authority comparison."""
    global _LAST_ROLLOUT_PERCENT
    _LAST_ROLLOUT_PERCENT = decision.rollout_percent

    if decision.enabled:
        _COUNTERS["unified_selected_count"] += 1
    else:
        _COUNTERS["legacy_selected_count"] += 1

    if comparison is not None and not comparison.aligned:
        _COUNTERS["authority_divergence_count"] += 1

    if rollback_triggered:
        _COUNTERS["rollback_trigger_count"] += 1

    return get_rollout_telemetry_snapshot()


__all__ = [
    "get_rollout_telemetry_snapshot",
    "record_rollout_event",
    "reset_rollout_telemetry",
]

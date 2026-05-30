"""
Unified emergency rollback — observe-only production safety signals.

Evaluates rollback conditions from hardening and rollout telemetry.
Does NOT automatically disable unified pipeline in Phase 7 (observe-only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from services.routing.unified_intent_production_metrics import get_production_metrics
from services.telemetry.unified_rollout_telemetry import get_rollout_telemetry_snapshot

# Observe-only thresholds — logged when exceeded; no automatic shutdown.
HARDENING_FAILURE_RATE_THRESHOLD = 0.25
AUTHORITY_DIVERGENCE_RATE_THRESHOLD = 0.30
EXECUTION_PATH_NONE_RATE_THRESHOLD = 0.40
MIN_EVENTS_FOR_RATE = 5


@dataclass(frozen=True)
class RollbackStatus:
    active: bool
    reason: str
    would_force_legacy: bool
    signals: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "reason": self.reason,
            "would_force_legacy": self.would_force_legacy,
            "signals": list(self.signals),
            "observe_only": True,
        }


def evaluate_emergency_rollback(
    *,
    production_metrics: Optional[Dict[str, Any]] = None,
    rollout_telemetry: Optional[Dict[str, Any]] = None,
) -> RollbackStatus:
    """
    Evaluate whether rollback conditions are met (observe-only).

    When active, rollout controller treats request as force_legacy.
    ``would_force_legacy`` indicates recommended action for operators.
    """
    prod = production_metrics or get_production_metrics()
    rollout = rollout_telemetry or get_rollout_telemetry_snapshot()

    signals: list[str] = []

    hardening_failures = int(prod.get("hardening_failure_count") or 0)
    path_none = int(prod.get("execution_path_none_count") or 0)
    legacy_fallback = int(prod.get("legacy_fallback_rate") or 0)
    total_hardening_events = hardening_failures + path_none + legacy_fallback
    total_hardening_events = max(total_hardening_events, hardening_failures, path_none)

    if total_hardening_events >= MIN_EVENTS_FOR_RATE:
        failure_rate = hardening_failures / total_hardening_events
        none_rate = path_none / total_hardening_events
        if failure_rate >= HARDENING_FAILURE_RATE_THRESHOLD:
            signals.append(
                f"hardening_failure_rate={failure_rate:.2f}>={HARDENING_FAILURE_RATE_THRESHOLD}"
            )
        if none_rate >= EXECUTION_PATH_NONE_RATE_THRESHOLD:
            signals.append(
                f"execution_path_none_rate={none_rate:.2f}>={EXECUTION_PATH_NONE_RATE_THRESHOLD}"
            )

    total_rollout = int(rollout.get("total_rollout_events") or 0)
    divergence_rate = float(rollout.get("authority_divergence_rate") or 0.0)
    if total_rollout >= MIN_EVENTS_FOR_RATE:
        if divergence_rate >= AUTHORITY_DIVERGENCE_RATE_THRESHOLD:
            signals.append(
                f"authority_divergence_rate={divergence_rate:.2f}"
                f">={AUTHORITY_DIVERGENCE_RATE_THRESHOLD}"
            )

    if not signals:
        return RollbackStatus(
            active=False,
            reason="All rollback thresholds within limits.",
            would_force_legacy=False,
        )

    reason = "; ".join(signals)
    return RollbackStatus(
        active=True,
        reason=reason,
        would_force_legacy=True,
        signals=tuple(signals),
    )


__all__ = [
    "AUTHORITY_DIVERGENCE_RATE_THRESHOLD",
    "EXECUTION_PATH_NONE_RATE_THRESHOLD",
    "HARDENING_FAILURE_RATE_THRESHOLD",
    "RollbackStatus",
    "evaluate_emergency_rollback",
]

"""
Production health score — composite operational health from live telemetry.

Does not modify routing, rollout, or responders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from monitoring.unified_rollout_dashboard import build_rollout_dashboard_snapshot
from services.routing.unified_emergency_rollback import evaluate_emergency_rollback


@dataclass(frozen=True)
class ProductionHealth:
    score: float
    status: str
    hardening_failure_rate: float = 0.0
    divergence_rate: float = 0.0
    rollback_active: bool = False
    unified_traffic_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(float(self.score), 4),
            "status": self.status,
            "hardening_failure_rate": round(float(self.hardening_failure_rate), 4),
            "divergence_rate": round(float(self.divergence_rate), 4),
            "rollback_active": self.rollback_active,
            "unified_traffic_percent": round(float(self.unified_traffic_percent), 2),
        }


def _status_from_score(score: float, *, rollback_active: bool) -> str:
    if rollback_active and score < 0.75:
        return "CRITICAL"
    if score < 0.50:
        return "CRITICAL"
    if score < 0.70:
        return "DEGRADED"
    if score < 0.85:
        return "WATCH"
    return "HEALTHY"


def compute_production_health(
    dashboard: Optional[Dict[str, Any]] = None,
) -> ProductionHealth:
    """
    Compute production health from dashboard snapshot inputs.
    """
    snap = dashboard or build_rollout_dashboard_snapshot()

    rollout = snap.get("rollout") or {}
    hardening = snap.get("hardening") or {}
    authority = snap.get("authority") or {}
    rollback = snap.get("rollback") or {}

    total_events = int(rollout.get("total_rollout_events") or 0)
    hardening_failures = int(hardening.get("hardening_failure_count") or 0)
    path_none = int(hardening.get("execution_path_none_count") or 0)
    denom = max(total_events, hardening_failures + path_none, 1)

    hardening_failure_rate = min(1.0, (hardening_failures + path_none) / denom)
    divergence_rate = float(authority.get("divergence_rate") or 0.0)
    rollback_active = bool(rollback.get("active"))
    unified_pct = float(rollout.get("unified_traffic_percent") or 0.0)

    live_bench = authority.get("live_benchmark") or {}
    path_agreement = float(live_bench.get("path_agreement_rate") or 1.0)

    score = (
        (1.0 - hardening_failure_rate) * 0.30
        + (1.0 - divergence_rate) * 0.25
        + path_agreement * 0.25
        + (0.0 if rollback_active else 0.15)
        + min(1.0, total_events / 100.0) * 0.05
    )
    if rollback_active:
        score *= 0.85
    score = max(0.0, min(1.0, score))

    return ProductionHealth(
        score=score,
        status=_status_from_score(score, rollback_active=rollback_active),
        hardening_failure_rate=hardening_failure_rate,
        divergence_rate=divergence_rate,
        rollback_active=rollback_active,
        unified_traffic_percent=unified_pct,
    )


__all__ = ["ProductionHealth", "compute_production_health"]

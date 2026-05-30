"""
Rollout readiness score — objective recommendation before full production authority.

Evaluation-only; does not modify rollout controller behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evaluation.legacy_unified_benchmark import benchmark_summary
from evaluation.path_accuracy_report import build_path_accuracy_report
from evaluation.aircraft_failure_report import build_aircraft_failure_report


@dataclass(frozen=True)
class RolloutReadiness:
    score: float
    recommendation: str
    routing_accuracy: float
    behavior_compliance: float
    unified_win_rate: float
    failure_concentration: float
    rollback_risk: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(float(self.score), 4),
            "recommendation": self.recommendation,
            "routing_accuracy": round(float(self.routing_accuracy), 4),
            "behavior_compliance": round(float(self.behavior_compliance), 4),
            "unified_win_rate": round(float(self.unified_win_rate), 4),
            "failure_concentration": round(float(self.failure_concentration), 4),
            "rollback_risk": round(float(self.rollback_risk), 4),
        }


def _recommendation_from_score(score: float) -> str:
    if score < 0.50:
        return "NOT_READY"
    if score < 0.65:
        return "LIMITED_ROLLOUT"
    if score < 0.75:
        return "SAFE_FOR_25_PERCENT"
    if score < 0.90:
        return "SAFE_FOR_50_PERCENT"
    return "SAFE_FOR_100_PERCENT"


def compute_rollout_readiness(
    *,
    path_report: Dict[str, Any],
    aircraft_report: Dict[str, Any],
    benchmark: Dict[str, Any],
    rollback_metrics: Optional[Dict[str, Any]] = None,
) -> RolloutReadiness:
    """
    Compute rollout readiness from evaluation artifacts and optional rollback telemetry.
    """
    overall = path_report.get("overall") or {}
    routing_accuracy = float(overall.get("pass_rate") or 0.0)

    by_cat = path_report.get("by_category") or {}
    behavior_rates = [float(v.get("behavior_pass_rate") or 0) for v in by_cat.values()]
    behavior_compliance = sum(behavior_rates) / len(behavior_rates) if behavior_rates else 0.0

    unified_win_rate = float(benchmark.get("unified_win_rate") or 0.0)

    total_cases = int(overall.get("total") or 1)
    total_failure_events = int(aircraft_report.get("total_failure_events") or 0)
    failure_concentration = min(1.0, total_failure_events / max(total_cases, 1))

    rollback = rollback_metrics or {}
    divergence_rate = float(rollback.get("authority_divergence_rate") or 0.0)
    hardening_failures = int(rollback.get("hardening_failure_count") or 0)
    rollback_risk = min(1.0, divergence_rate + hardening_failures / max(total_cases, 1))

    score = (
        routing_accuracy * 0.35
        + behavior_compliance * 0.25
        + unified_win_rate * 0.20
        + (1.0 - failure_concentration) * 0.10
        + (1.0 - rollback_risk) * 0.10
    )
    score = max(0.0, min(1.0, score))

    return RolloutReadiness(
        score=score,
        recommendation=_recommendation_from_score(score),
        routing_accuracy=routing_accuracy,
        behavior_compliance=behavior_compliance,
        unified_win_rate=unified_win_rate,
        failure_concentration=failure_concentration,
        rollback_risk=rollback_risk,
    )


def compute_rollout_readiness_from_evaluation(
    cases: List[Any],
    unified_results: List[Any],
    benchmark_results: List[Any],
    *,
    rollback_metrics: Optional[Dict[str, Any]] = None,
) -> RolloutReadiness:
    path_report = build_path_accuracy_report(cases, unified_results)
    aircraft_report = build_aircraft_failure_report(cases, unified_results)
    bench = benchmark_summary(benchmark_results)
    return compute_rollout_readiness(
        path_report=path_report,
        aircraft_report=aircraft_report,
        benchmark=bench,
        rollback_metrics=rollback_metrics,
    )


__all__ = [
    "RolloutReadiness",
    "compute_rollout_readiness",
    "compute_rollout_readiness_from_evaluation",
]

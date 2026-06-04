"""
Aggregate analytics for consultant evaluations (Phase 19).

Read-only aggregation over evaluation records — does not affect production execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class EvaluationAnalytics:
    count: int = 0
    average_score: float = 0.0
    score_by_intent: Dict[str, float] = field(default_factory=dict)
    score_by_responder: Dict[str, float] = field(default_factory=dict)
    routing_accuracy: float = 0.0
    verdict_compliance: float = 0.0
    hallucination_rate: float = 0.0
    pass_rate: float = 0.0
    failure_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "average_score": round(self.average_score, 2),
            "score_by_intent": dict(self.score_by_intent),
            "score_by_responder": dict(self.score_by_responder),
            "routing_accuracy": round(self.routing_accuracy, 4),
            "verdict_compliance": round(self.verdict_compliance, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "pass_rate": round(self.pass_rate, 4),
            "failure_counts": dict(self.failure_counts),
        }


def _coerce_evaluation(record: Any) -> Optional[Dict[str, Any]]:
    if isinstance(record, dict) and record.get("evaluation_id"):
        return record
    if hasattr(record, "to_dict"):
        return record.to_dict()
    return None


def aggregate_evaluations(
    evaluations: Sequence[Any],
) -> EvaluationAnalytics:
    """Compute aggregate metrics from consultant evaluation records."""
    rows: List[Dict[str, Any]] = []
    for item in evaluations:
        ev = _coerce_evaluation(item)
        if ev:
            rows.append(ev)

    if not rows:
        return EvaluationAnalytics()

    n = len(rows)
    total_sum = sum(float(r.get("total_score") or 0) for r in rows)

    by_intent: Dict[str, List[float]] = {}
    by_responder: Dict[str, List[float]] = {}
    failure_counts: Dict[str, int] = {}
    routing_hits = 0
    verdict_hits = 0
    hallucination_hits = 0
    pass_hits = 0

    for r in rows:
        intent = str(r.get("intent_type") or "unknown")
        path = str(r.get("execution_path") or "unknown")
        score = float(r.get("total_score") or 0)
        by_intent.setdefault(intent, []).append(score)
        by_responder.setdefault(path, []).append(score)

        failures = r.get("failures") or []
        if isinstance(failures, list):
            for f in failures:
                key = str(f)
                failure_counts[key] = failure_counts.get(key, 0) + 1

        if "llm_leak" not in failures and "wrong_execution_path" not in failures:
            routing_hits += 1
        if "missing_verdict" not in failures:
            verdict_hits += 1
        if "hallucinated_model" in failures:
            hallucination_hits += 1
        if r.get("passed"):
            pass_hits += 1

    return EvaluationAnalytics(
        count=n,
        average_score=total_sum / n,
        score_by_intent={k: sum(v) / len(v) for k, v in by_intent.items()},
        score_by_responder={k: sum(v) / len(v) for k, v in by_responder.items()},
        routing_accuracy=routing_hits / n,
        verdict_compliance=verdict_hits / n,
        hallucination_rate=hallucination_hits / n,
        pass_rate=pass_hits / n,
        failure_counts=failure_counts,
    )


__all__ = ["EvaluationAnalytics", "aggregate_evaluations"]

"""
Legacy vs Unified benchmark — structural rule-based comparison per golden case.

Does NOT invoke LLM or modify production pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evaluation.golden_dataset import GoldenTestCase
from evaluation.unified_evaluator import EvaluationResult, evaluate_unified_case
from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent

# Expected QRI intents by golden category (structural legacy proxy).
_CATEGORY_QRI_EXPECTED: Dict[str, frozenset[str]] = {
    "FACT": frozenset({"payload_range_analysis"}),
    "MARKET": frozenset({"ownership_economics", "payload_range_analysis"}),
    "CAPABILITY": frozenset({"mission_feasibility", "payload_range_analysis"}),
    "COMPARISON": frozenset({"aircraft_comparison"}),
    "ALTERNATIVE": frozenset({"aircraft_comparison", "acquisition_recommendation", "shortlist_ranking"}),
    "MISSION": frozenset(
        {
            "mission_feasibility",
            "shortlist_ranking",
            "acquisition_recommendation",
            "operational_tradeoff_analysis",
        }
    ),
    "BUY_DECISION": frozenset(
        {
            "acquisition_recommendation",
            "shortlist_ranking",
            "ownership_economics",
            "mission_feasibility",
        }
    ),
}


@dataclass(frozen=True)
class BenchmarkResult:
    case_id: str
    legacy_score: float
    unified_score: float
    winner: str
    legacy_qri_intent: str = ""
    unified_execution_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "legacy_score": round(float(self.legacy_score), 4),
            "unified_score": round(float(self.unified_score), 4),
            "winner": self.winner,
            "legacy_qri_intent": self.legacy_qri_intent,
            "unified_execution_path": self.unified_execution_path,
        }


def _legacy_structural_score(case: GoldenTestCase) -> tuple[float, str]:
    """Rule-based legacy proxy score from QRI classification."""
    qri = classify_query_recommendation_intent(case.query)
    intent = qri.intent.value
    expected = _CATEGORY_QRI_EXPECTED.get(case.category, frozenset())

    intent_match = 1.0 if intent in expected else 0.0
    confidence_factor = min(1.0, float(qri.confidence))

    # Mission/buy cases: legacy should own these (path none is correct for unified)
    category_bonus = 0.0
    if case.category in ("MISSION", "BUY_DECISION"):
        if intent in expected:
            category_bonus = 0.5
    elif intent_match:
        category_bonus = 0.0

    score = (intent_match * 0.7 + confidence_factor * 0.3) + category_bonus
    return min(1.0, score), intent


def benchmark_case(case: GoldenTestCase, *, enforce_unified: bool = True) -> BenchmarkResult:
    """Compare legacy structural score vs unified evaluation score for one case."""
    legacy_score, qri_intent = _legacy_structural_score(case)
    unified_result = evaluate_unified_case(case, enforce=enforce_unified)
    unified_score = unified_result.score

    if abs(legacy_score - unified_score) < 0.05:
        winner = "tie"
    elif unified_score > legacy_score:
        winner = "unified"
    else:
        winner = "legacy"

    return BenchmarkResult(
        case_id=case.id,
        legacy_score=legacy_score,
        unified_score=unified_score,
        winner=winner,
        legacy_qri_intent=qri_intent,
        unified_execution_path=unified_result.actual_execution_path,
    )


def benchmark_cases(
    cases: List[GoldenTestCase],
    *,
    enforce_unified: bool = True,
) -> List[BenchmarkResult]:
    return [benchmark_case(c, enforce_unified=enforce_unified) for c in cases]


def benchmark_summary(results: List[BenchmarkResult]) -> Dict[str, Any]:
    total = len(results)
    if not total:
        return {"total": 0, "legacy_wins": 0, "unified_wins": 0, "ties": 0}
    legacy_wins = sum(1 for r in results if r.winner == "legacy")
    unified_wins = sum(1 for r in results if r.winner == "unified")
    ties = sum(1 for r in results if r.winner == "tie")
    return {
        "total": total,
        "legacy_wins": legacy_wins,
        "legacy_win_rate": round(legacy_wins / total, 4),
        "unified_wins": unified_wins,
        "unified_win_rate": round(unified_wins / total, 4),
        "ties": ties,
        "tie_rate": round(ties / total, 4),
        "mean_legacy_score": round(sum(r.legacy_score for r in results) / total, 4),
        "mean_unified_score": round(sum(r.unified_score for r in results) / total, 4),
    }


__all__ = [
    "BenchmarkResult",
    "benchmark_case",
    "benchmark_cases",
    "benchmark_summary",
]

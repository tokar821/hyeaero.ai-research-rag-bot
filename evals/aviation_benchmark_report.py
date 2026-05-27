"""
Benchmark report generation — contamination, realism, diversity, precision.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

from evals.aviation_benchmark_scoring import BenchmarkCaseResult, SCORE_DIMENSIONS


def _unique_models(results: List[BenchmarkCaseResult]) -> List[str]:
    models: List[str] = []
    for r in results:
        for m in (r.metadata.get("top_recommendations") or []):
            if m and m not in models:
                models.append(m)
    return models


def build_contamination_report(results: List[BenchmarkCaseResult]) -> Dict[str, Any]:
    leak_cases = [r for r in results if any("previous_turn_leak" in f for f in r.automated_failures)]
    invalid_routes = [r for r in results if any("invalid_routes" in f for f in r.automated_failures)]
    template_hits = [
        r for r in results if any("contamination:" in i for i in r.issues)
    ]
    return {
        "turn_leak_count": len(leak_cases),
        "turn_leak_case_ids": [r.case_id for r in leak_cases],
        "invalid_route_count": len(invalid_routes),
        "invalid_route_case_ids": [r.case_id for r in invalid_routes],
        "template_contamination_count": len(template_hits),
        "mean_contamination_score": round(
            sum(r.scores.get("contamination_rate", 0) for r in results) / max(len(results), 1),
            4,
        ),
    }


def build_realism_report(results: List[BenchmarkCaseResult]) -> Dict[str, Any]:
    realism_scores = [r.scores.get("aircraft_realism", 0) for r in results]
    impossible = [
        r.case_id
        for r in results
        if any("impossible_aircraft" in f for f in r.automated_failures)
    ]
    return {
        "mean_aircraft_realism": round(sum(realism_scores) / max(len(realism_scores), 1), 4),
        "impossible_recommendation_failures": impossible,
        "cases_below_0_6_realism": [
            r.case_id for r in results if r.scores.get("aircraft_realism", 0) < 0.6
        ],
    }


def build_diversity_report(results: List[BenchmarkCaseResult]) -> Dict[str, Any]:
    counter: Counter[str] = Counter()
    for r in results:
        for m in (r.metadata.get("top_recommendations") or [])[:1]:
            if m:
                counter[m] += 1
    total = len(results) or 1
    top_model, top_count = counter.most_common(1)[0] if counter else ("none", 0)
    dominance = top_count / total
    unique_leaders = len(counter)
    return {
        "unique_top1_models": unique_leaders,
        "top_model_dominance": {top_model: round(dominance, 3)},
        "leader_distribution": dict(counter.most_common(8)),
        "diversity_score": round(min(1.0, unique_leaders / max(total * 0.35, 1)), 4),
    }


def build_precision_report(results: List[BenchmarkCaseResult]) -> Dict[str, Any]:
    hits = 0
    eligible = 0
    misses: List[str] = []
    for r in results:
        if any("expected_aircraft_missing" in i for i in r.issues):
            eligible += 1
            misses.append(r.case_id)
        elif r.scores.get("aircraft_realism", 0) >= 0.7 and not any(
            "forbidden_aircraft" in i for i in r.issues
        ):
            eligible += 1
            hits += 1
        elif not any("expected_aircraft_missing" in i for i in r.issues):
            eligible += 1
            if r.scores.get("aircraft_realism", 0) >= 0.65:
                hits += 1
    precision = hits / max(eligible, 1)
    return {
        "recommendation_precision": round(precision, 4),
        "golden_hit_cases": hits,
        "golden_eligible_cases": eligible,
        "missed_expectation_case_ids": misses,
    }


def generate_benchmark_report(
    results: List[BenchmarkCaseResult],
    *,
    suite_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Full production report payload."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    critical = sum(1 for r in results if r.critical)
    by_category: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"passed": 0, "failed": 0, "total": 0})

    for r in results:
        by_category[r.category]["total"] += 1
        if r.passed:
            by_category[r.category]["passed"] += 1
        else:
            by_category[r.category]["failed"] += 1

    dim_avg = {
        d: round(sum(x.scores.get(d, 0) for x in results) / max(total, 1), 4)
        for d in SCORE_DIMENSIONS
    }

    return {
        "schema_version": 1,
        "suite": suite_meta or {},
        "summary": {
            "total_cases": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / max(total, 1), 4),
            "critical_failures": critical,
        },
        "dimension_scores": dim_avg,
        "by_category": dict(by_category),
        "contamination_report": build_contamination_report(results),
        "realism_score": build_realism_report(results)["mean_aircraft_realism"],
        "realism_report": build_realism_report(results),
        "aircraft_diversity_score": build_diversity_report(results)["diversity_score"],
        "aircraft_diversity_report": build_diversity_report(results),
        "recommendation_precision": build_precision_report(results)["recommendation_precision"],
        "recommendation_precision_report": build_precision_report(results),
        "failed_cases": [
            {
                "id": r.case_id,
                "category": r.category,
                "automated_failures": r.automated_failures,
                "issues": r.issues,
                "scores": r.scores,
            }
            for r in results
            if not r.passed
        ],
    }

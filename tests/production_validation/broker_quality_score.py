"""Phase 32 — Broker quality scoring (read-only)."""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from tests.production_validation.validation_runner import ValidationResult


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 100.0


def score_aircraft_accuracy(results: List["ValidationResult"]) -> float:
    hard = [r for r in results if r.category in ("comparison", "alternative", "buy_decision", "valuation")]
    if not hard:
        return 100.0
    ok = sum(1 for r in hard if r.model_match and r.authority_match)
    return _pct(ok, len(hard))


def score_mission_accuracy(results: List["ValidationResult"]) -> float:
    mission = [r for r in results if r.category == "mission"]
    if not mission:
        return 100.0
    ok = sum(
        1
        for r in mission
        if r.execution_path in ("llm_fallback", "hybrid_unified", "pre_llm_mission", "authority_dispatch")
    )
    return _pct(ok, len(mission))


def score_valuation_accuracy(results: List["ValidationResult"]) -> float:
    val = [r for r in results if r.category == "valuation"]
    if not val:
        return 100.0
    ok = sum(1 for r in val if r.authority_match and r.routing_match and r.model_match and r.fail_closed_correct)
    return _pct(ok, len(val))


def score_comparison_accuracy(results: List["ValidationResult"]) -> float:
    cmp = [r for r in results if r.category == "comparison"]
    if not cmp:
        return 100.0
    ok = sum(
        1
        for r in cmp
        if r.model_match and r.routing_match and r.authority_match and r.fail_closed_correct
    )
    return _pct(ok, len(cmp))


def score_constraint_compliance(results: List["ValidationResult"]) -> float:
    ok = sum(1 for r in results if r.fail_closed_correct)
    return _pct(ok, len(results))


def score_fail_closed_correctness(results: List["ValidationResult"]) -> float:
    fc = [r for r in results if r.fail_closed or "unexpected_fail_closed" in r.issues]
    if not fc:
        return 100.0
    ok = sum(1 for r in fc if r.fail_closed_correct)
    return _pct(ok, len(fc))


def compute_broker_quality_score(results: List["ValidationResult"]) -> float:
    weights = {
        "aircraft_accuracy": 0.25,
        "mission_accuracy": 0.15,
        "valuation_accuracy": 0.15,
        "comparison_accuracy": 0.20,
        "constraint_compliance": 0.15,
        "fail_closed_correctness": 0.10,
    }
    scores = {
        "aircraft_accuracy": score_aircraft_accuracy(results),
        "mission_accuracy": score_mission_accuracy(results),
        "valuation_accuracy": score_valuation_accuracy(results),
        "comparison_accuracy": score_comparison_accuracy(results),
        "constraint_compliance": score_constraint_compliance(results),
        "fail_closed_correctness": score_fail_closed_correctness(results),
    }
    total = sum(scores[k] * weights[k] for k in weights)
    return round(total, 2)


def compute_broker_quality_report(results: List["ValidationResult"]) -> Dict[str, Any]:
    from tests.production_validation.hallucination_audit import audit_hallucinations
    from tests.production_validation.mission_fit_audit import audit_mission_fit

    routing_ok = sum(1 for r in results if r.routing_match)
    authority_ok = sum(1 for r in results if r.authority_match)
    fail_closed_ok = sum(1 for r in results if r.fail_closed_correct)
    total = len(results)

    hall = audit_hallucinations(results)
    mission = audit_mission_fit(results)

    category_breakdown: Dict[str, Dict[str, Any]] = {}
    for cat in ("comparison", "buy_decision", "mission", "alternative", "valuation"):
        subset = [r for r in results if r.category == cat]
        if not subset:
            continue
        category_breakdown[cat] = {
            "total": len(subset),
            "routing_accuracy_pct": _pct(sum(1 for r in subset if r.routing_match), len(subset)),
            "authority_accuracy_pct": _pct(sum(1 for r in subset if r.authority_match), len(subset)),
            "model_accuracy_pct": _pct(sum(1 for r in subset if r.model_match), len(subset)),
        }

    return {
        "total_queries": total,
        "routing_accuracy_pct": _pct(routing_ok, total),
        "dispatch_accuracy_pct": _pct(authority_ok, total),
        "fail_closed_accuracy_pct": _pct(fail_closed_ok, total),
        "mission_fit_accuracy_pct": mission.get("mission_fit_accuracy_pct", 100.0),
        "hallucination_rate_pct": hall.get("hallucination_rate_pct", 0.0),
        "broker_quality_score": compute_broker_quality_score(results),
        "category_scores": {
            "aircraft_accuracy": score_aircraft_accuracy(results),
            "mission_accuracy": score_mission_accuracy(results),
            "valuation_accuracy": score_valuation_accuracy(results),
            "comparison_accuracy": score_comparison_accuracy(results),
            "constraint_compliance": score_constraint_compliance(results),
            "fail_closed_correctness": score_fail_closed_correctness(results),
        },
        "category_breakdown": category_breakdown,
        "hallucination_audit": hall,
        "mission_fit_audit": mission,
        "failed_query_ids": [r.query_id for r in results if r.issues],
        "high_risk_queries": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "issues": r.issues,
                "execution_path": r.execution_path,
                "intent_lock_summary": r.intent_lock_summary,
                "recommendation": r.recommendation[:240],
                "models": r.authority_models,
            }
            for r in sorted(results, key=lambda x: len(x.issues), reverse=True)[:50]
            if r.issues
        ],
        "high_confidence_queries": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "execution_path": r.execution_path,
                "intent_lock_summary": r.intent_lock_summary,
                "recommendation": r.recommendation[:240],
                "models": r.authority_models,
            }
            for r in results
            if not r.issues and r.routing_match and r.authority_match
        ][:50],
    }

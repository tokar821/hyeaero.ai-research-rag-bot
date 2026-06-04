"""Phase 33 — Response quality scorecard (E2E final answers)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from tests.response_quality.answer_consistency_audit import audit_answer_consistency
from tests.response_quality.broker_recommendation_audit import audit_broker_recommendation
from tests.response_quality.comparison_quality_audit import audit_comparison_quality
from tests.response_quality.mission_feasibility_audit import audit_mission_feasibility
from tests.response_quality.valuation_accuracy_audit import audit_valuation_accuracy
from tests.response_quality.response_audit_runner import AuditFinding, AuditedCase, E2EResponseCase


STOP_CODES = {
    "HALLUCINATED_AIRCRAFT",
    "CROSS_MODEL_VALUATION",
    "MISSION_INFEASIBLE_RECOMMENDATION",
    "VERDICT_DRIFT",
}


def _pct(ok: int, den: int) -> float:
    return round(100.0 * ok / den, 2) if den else 100.0


def audit_case(case: E2EResponseCase) -> AuditedCase:
    findings: List[AuditFinding] = []

    broker = audit_broker_recommendation(query=case.query, answer=case.answer)
    mission = audit_mission_feasibility(query=case.query, answer=case.answer)
    consistency = audit_answer_consistency(
        answer=case.answer,
        intent_lock=case.intent_lock,
        authority_models=case.authority_models,
    )

    comparison = None
    valuation = None
    if case.category == "comparison":
        comparison = audit_comparison_quality(answer=case.answer)
    if case.category == "valuation":
        valuation = audit_valuation_accuracy(query=case.query, answer=case.answer)

    for code in broker.failures:
        findings.append(AuditFinding(code=code, message="broker recommendation audit failure"))
    for code in mission.failures:
        findings.append(AuditFinding(code=code, message="mission feasibility failure"))
    for code in consistency.failures:
        findings.append(AuditFinding(code=code, message="answer consistency failure"))
    if comparison:
        for code in comparison.failures:
            findings.append(AuditFinding(code=code, message="comparison quality failure"))
    if valuation:
        for code in valuation.failures:
            findings.append(AuditFinding(code=code, message="valuation accuracy failure"))

    # overall score: weighted average across applicable audits
    weights: List[Tuple[float, float]] = []
    weights.append((broker.score, 0.25))
    weights.append((mission.score, 0.25))
    weights.append((consistency.score, 0.30))
    if comparison:
        weights.append((comparison.score, 0.10))
    if valuation:
        weights.append((valuation.score, 0.10))

    total_w = sum(w for _, w in weights) or 1.0
    score = round(sum(s * w for s, w in weights) / total_w, 2)

    stop = any(f.code in STOP_CODES for f in findings)
    return AuditedCase(case=case, score=score, findings=findings, stop_condition_hit=stop)


def compute_scorecard(results: List[AuditedCase]) -> Dict[str, Any]:
    total = len(results)
    by_code: Dict[str, int] = {}
    for r in results:
        for f in r.findings:
            by_code[f.code] = by_code.get(f.code, 0) + 1

    broker_ok = sum(1 for r in results if all(f.code not in ("BROKER_BAD_AIRCRAFT", "BROKER_ROUTE_MISMATCH", "BROKER_BUDGET_MISMATCH", "BROKER_PAX_MISMATCH") for f in r.findings))
    mission_ok = sum(1 for r in results if all(f.code != "MISSION_INFEASIBLE_RECOMMENDATION" for f in r.findings))
    valuation_total = sum(1 for r in results if r.case.category == "valuation")
    valuation_ok = sum(1 for r in results if r.case.category == "valuation" and all(f.code not in ("CROSS_MODEL_VALUATION", "WRONG_YEAR_VALUATION", "UNKNOWN_VALUATION_SOURCE") for f in r.findings))
    comparison_total = sum(1 for r in results if r.case.category == "comparison")
    comparison_ok = sum(1 for r in results if r.case.category == "comparison" and all(f.code not in ("COMPARISON_INCOMPLETE", "COMPARISON_NO_VERDICT") for f in r.findings))
    consistency_ok = sum(1 for r in results if all(f.code not in ("VERDICT_DRIFT", "UNJUSTIFIED_MODEL_INSERTION", "HALLUCINATED_AIRCRAFT") for f in r.findings))

    overall = round(sum(r.score for r in results) / (total or 1), 2)
    stop_hits = sum(1 for r in results if r.stop_condition_hit)

    return {
        "total_audited": total,
        "overall_broker_quality_score": overall,
        "broker_recommendation_accuracy_pct": _pct(broker_ok, total),
        "mission_feasibility_accuracy_pct": _pct(mission_ok, total),
        "valuation_accuracy_pct": _pct(valuation_ok, valuation_total),
        "comparison_quality_pct": _pct(comparison_ok, comparison_total),
        "answer_consistency_pct": _pct(consistency_ok, total),
        "stop_condition_hits": stop_hits,
        "finding_counts": dict(sorted(by_code.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_risk": [
            {
                "query_id": r.case.query_id,
                "category": r.case.category,
                "query": r.case.query,
                "score": r.score,
                "stop_condition_hit": r.stop_condition_hit,
                "findings": [f.code for f in r.findings],
                "authority_models": r.case.authority_models,
            }
            for r in sorted(results, key=lambda x: (x.stop_condition_hit, len(x.findings), -x.score), reverse=True)[:50]
        ],
    }


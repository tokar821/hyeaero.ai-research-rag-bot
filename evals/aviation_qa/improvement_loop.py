"""
Auto-improvement loop — map evaluator failures to targeted fix suggestions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

from evals.aviation_qa.schemas import EvaluatorVerdict


# Failure pattern → suggested engineering actions
_FIX_CATALOG: Dict[str, List[str]] = {
    "impossible_aircraft": [
        "Verify `hard_mission_elimination` runs before ranking in `run_pipeline.py`",
        "Ensure empty `feasible_models` does not fall back to full-catalog ranking",
        "Strengthen `mission_ranker` bridge for ULR westbound gates",
    ],
    "route": [
        "Check `route_extractor` / `aviation_places` aliases for scenario cities",
        "Review `mission_validation.route_truly_missing` — avoid clarifiers when route present",
        "Audit turn-isolation in `extract_mission_with_memory`",
    ],
    "repetition": [
        "Expand `phrase_repetition_guard` banned patterns",
        "Add opening archetypes in `response_variation.py`",
        "Run `run_aviation_qa_suite` after copy changes to measure suite_repetition_score",
    ],
    "tone_robotic": [
        "Strip diagnostic headers in `sanitize_advisor_output`",
        "Disable bullet mode more often in `compose_varied_response`",
        "Update LLM system addendum in `llm_explanation_layer.build_narration_system_addendum`",
    ],
    "hallucination": [
        "Enforce `reconcile_answer_with_pipeline` on all advisory turns",
        "Add fake-model detection in `consultant_validity`",
    ],
    "fake_confidence": [
        "Prefer tiered framing over single-model anchor in `recommendation_framing`",
        "Inject confident-uncertainty phrases in `response_variation` openings",
    ],
    "weak_operational": [
        "Require elimination language in `response_formatter` for complex missions",
        "Pass route feasibility caveats into formatter bullets",
    ],
    "brochure": [
        "Ban generic spec phrases in `response_formatter._lead_reason`",
        "Add brochure-language penalties to evaluator regression tests",
    ],
    "clarification": [
        "Tighten `mission_clarification_needs` — only ask when route truly missing",
        "Skip clarifiers for visualization intent in `visualization_handler`",
    ],
    "general": [
        "Re-run `python scripts/run_aviation_qa.py` after fix and compare trust_score delta",
        "Inspect failed case answer_preview in JSON report",
    ],
}


def classify_failure_source(verdict: EvaluatorVerdict, sub_failures: List[str]) -> str:
    blob = " ".join([verdict.main_failure] + sub_failures).lower()
    if "impossible_aircraft" in blob or verdict.aircraft_realism == "FAIL":
        return "impossible_aircraft"
    if "repetitive" in blob or verdict.repetition_score >= 0.55:
        return "repetition"
    if verdict.hallucination_risk >= 0.7:
        return "hallucination"
    if verdict.fake_confidence_risk >= 0.65:
        return "fake_confidence"
    if verdict.brochure_language_risk >= 0.5:
        return "brochure"
    if verdict.route_realism == "FAIL":
        return "route"
    if verdict.operational_realism < 0.45 or verdict.missing_tradeoffs:
        return "weak_operational"
    if "clarif" in blob or "city pair" in blob:
        return "clarification"
    if verdict.tone_broker_score < 0.45:
        return "tone_robotic"
    return "general"


def suggest_fixes(failure_source: str) -> List[str]:
    return list(_FIX_CATALOG.get(failure_source, _FIX_CATALOG.get("general", ["Review case manually"])))


def build_improvement_plan(
    case_results: List[Dict[str, Any]],
    *,
    suite_repetition_score: float = 0.0,
) -> Dict[str, Any]:
    """
    Aggregate failures across a QA run and return prioritized fix suggestions.
    """
    by_source: Counter[str] = Counter()
    failed_cases: List[Dict[str, Any]] = []
    suggestions: Dict[str, List[str]] = defaultdict(list)

    for row in case_results:
        verdict_dict = row.get("evaluator") or {}
        if verdict_dict.get("passed"):
            continue
        failed_cases.append(
            {
                "id": row.get("id"),
                "category": row.get("category"),
                "main_failure": verdict_dict.get("main_failure"),
                "trust_score": verdict_dict.get("trust_score"),
            }
        )
        v = EvaluatorVerdict(
            route_realism=str(verdict_dict.get("route_realism") or "WARN"),
            aircraft_realism=str(verdict_dict.get("aircraft_realism") or "WARN"),
            hallucination_risk=float(verdict_dict.get("hallucination_risk") or 0),
            repetition_score=float(verdict_dict.get("repetition_score") or 0),
            humanness_score=float(verdict_dict.get("humanness_score") or 0),
            operational_realism=float(verdict_dict.get("operational_realism") or 0),
            tone_broker_score=float(verdict_dict.get("tone_broker_score") or 0),
            fake_confidence_risk=float(verdict_dict.get("fake_confidence_risk") or 0),
            brochure_language_risk=float(verdict_dict.get("brochure_language_risk") or 0),
            missing_tradeoffs=bool(verdict_dict.get("missing_tradeoffs")),
            main_failure=str(verdict_dict.get("main_failure") or ""),
            sub_failures=list(verdict_dict.get("sub_failures") or []),
            passed=False,
            trust_score=float(verdict_dict.get("trust_score") or 0),
        )
        src = classify_failure_source(v, v.sub_failures)
        by_source[src] += 1
        for fix in suggest_fixes(src):
            if fix not in suggestions[src]:
                suggestions[src].append(fix)

    priority = [s for s, _ in by_source.most_common()]
    if suite_repetition_score >= 0.4:
        priority = ["repetition"] + [p for p in priority if p != "repetition"]
        suggestions["repetition"].extend(_FIX_CATALOG["repetition"])

    return {
        "failed_case_count": len(failed_cases),
        "failed_cases": failed_cases[:20],
        "failure_sources": dict(by_source),
        "priority_order": priority,
        "suggested_fixes_by_source": dict(suggestions),
        "suite_repetition_score": round(suite_repetition_score, 4),
        "next_steps": _next_steps(priority, suggestions),
    }


def _next_steps(priority: List[str], suggestions: Dict[str, List[str]]) -> List[str]:
    steps: List[str] = []
    for src in priority[:4]:
        fixes = suggestions.get(src) or []
        if fixes:
            steps.append(f"[{src}] {fixes[0]}")
    if not steps:
        steps.append("All cases passed trust threshold — spot-check humanness on 3 random scenarios.")
    return steps

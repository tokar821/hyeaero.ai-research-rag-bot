"""
Evaluator agent — critiques advisor responses; does NOT answer user questions.

Produces structured JSON verdicts for QA and auto-improvement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from evals.aviation_benchmark_scoring import BenchmarkCaseResult, score_benchmark_case
from evals.aviation_qa.mission_validation import validate_mission_before_recommendation
from evals.aviation_qa.repetition_detection import score_answer_repetition
from evals.aviation_qa.schemas import EvaluatorVerdict, ScenarioQA
from evals.aviation_qa.tone_evaluation import score_operational_depth, score_tone


def evaluate_advisor_response(
    *,
    case: Dict[str, Any],
    answer: str,
    turn_profile: Dict[str, Any],
    merged_profile: Dict[str, Any],
    mission_state: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    mission_category: Optional[str] = None,
    suite_qa_defaults: Optional[Dict[str, Any]] = None,
) -> EvaluatorVerdict:
    """
    Full evaluator pass: deterministic benchmark scoring + tone/repetition/mission checks.
    """
    qa = ScenarioQA.from_case(case, suite_qa_defaults)
    bench: BenchmarkCaseResult = score_benchmark_case(
        case=case,
        turn_profile=turn_profile,
        merged_profile=merged_profile,
        mission_state=mission_state,
        recommendations=recommendations,
        mission_category=mission_category,
        answer=answer,
    )

    rep = score_answer_repetition(answer, forbidden_extra=qa.forbidden_phrases)
    humanness, broker_tone, fake_risk, brochure_risk = score_tone(answer)
    operational = score_operational_depth(answer)

    route_status = _route_status(bench)
    aircraft_status = _aircraft_status(bench)
    mission_status, mission_issues = validate_mission_before_recommendation(
        query=str(case.get("input") or ""),
        turn_profile=turn_profile,
        recommendations=recommendations,
        mission_category=mission_category,
        realism=qa.realism,
    )

    hallucination_risk = 1.0 - bench.scores.get("hallucination_rate", 1.0)
    if any("hallucination" in f for f in bench.automated_failures):
        hallucination_risk = max(hallucination_risk, 0.85)

    missing_tradeoffs = _missing_tradeoffs(answer, bench, qa)
    sub_failures = list(bench.automated_failures) + list(bench.issues) + mission_issues
    if rep.overused_phrases:
        sub_failures.append(f"repetitive_phrasing:{','.join(rep.overused_phrases[:5])}")

    main_failure = _pick_main_failure(
        bench=bench,
        route_status=route_status,
        aircraft_status=aircraft_status,
        mission_status=mission_status,
        repetition_score=rep.repetition_score,
        fake_risk=fake_risk,
        answer=answer,
        recommendations=recommendations,
    )

    trust = _trust_score(
        bench=bench,
        route_status=route_status,
        aircraft_status=aircraft_status,
        mission_status=mission_status,
        repetition_score=rep.repetition_score,
        humanness=humanness,
        operational=operational,
        fake_risk=fake_risk,
        brochure_risk=brochure_risk,
    )

    passed = (
        trust >= 0.62
        and route_status != "FAIL"
        and aircraft_status != "FAIL"
        and mission_status != "FAIL"
        and rep.repetition_score < 0.55
        and fake_risk < 0.65
        and not bench.automated_failures
    )

    return EvaluatorVerdict(
        route_realism=route_status,
        aircraft_realism=aircraft_status,
        hallucination_risk=hallucination_risk,
        repetition_score=rep.repetition_score,
        humanness_score=humanness,
        operational_realism=operational,
        tone_broker_score=broker_tone,
        fake_confidence_risk=fake_risk,
        brochure_language_risk=brochure_risk,
        missing_tradeoffs=missing_tradeoffs,
        main_failure=main_failure,
        sub_failures=sub_failures[:12],
        passed=passed,
        trust_score=trust,
    )


def _route_status(bench: BenchmarkCaseResult) -> str:
    s = bench.scores.get("route_accuracy", 0)
    if s >= 0.75:
        return "PASS"
    if s >= 0.45:
        return "WARN"
    return "FAIL"


def _aircraft_status(bench: BenchmarkCaseResult) -> str:
    if any("impossible_aircraft" in f for f in bench.automated_failures):
        return "FAIL"
    s = bench.scores.get("aircraft_realism", 0)
    if s >= 0.7:
        return "PASS"
    if s >= 0.45:
        return "WARN"
    return "FAIL"


def _missing_tradeoffs(answer: str, bench: BenchmarkCaseResult, qa: ScenarioQA) -> bool:
    if bench.scores.get("operational_reasoning", 0) >= 0.65:
        return False
    if not (answer or "").strip():
        return True
    import re

    if re.search(r"\b(tradeoff|trade-offs|margin|in practice|wouldn'?t|avoid|brochure)\b", answer, re.I):
        return False
    if qa.realism.requires_elimination_language:
        return not re.search(r"\b(wouldn'?t|avoid|steer clear|on paper)\b", answer, re.I)
    return bench.scores.get("operational_reasoning", 0) < 0.45


def _pick_main_failure(
    *,
    bench: BenchmarkCaseResult,
    route_status: str,
    aircraft_status: str,
    mission_status: str,
    repetition_score: float,
    fake_risk: float,
    answer: str,
    recommendations: List[Dict[str, Any]],
) -> str:
    if bench.automated_failures:
        f = bench.automated_failures[0]
        if f.startswith("impossible_aircraft_recommended:"):
            model = f.split(":", 1)[-1]
            return f"Unrealistic aircraft recommended for mission: {model}"
        return f.replace("_", " ")

    if mission_status == "FAIL":
        return "Mission validation failed — recommendations bypass operational realism"

    if aircraft_status == "FAIL":
        forbidden = [i for i in bench.issues if "forbidden_aircraft" in i]
        if forbidden:
            return f"Forbidden aircraft in shortlist ({forbidden[0]})"
        return "Aircraft shortlist not realistic for stated mission"

    if route_status == "FAIL":
        return "Route extraction or mission understanding failed"

    if repetition_score >= 0.55:
        return "Robotic / repetitive templated phrasing"

    if fake_risk >= 0.65:
        return "Overconfident language — fake certainty detected"

    if bench.scores.get("operational_reasoning", 0) < 0.45:
        return "Weak operational reasoning — spec-sheet tone"

    if not recommendations and answer and len(answer) > 200:
        return "Verbose answer without validated aircraft shortlist"

    if bench.passed:
        return ""
    return "Composite trust score below threshold"


def _trust_score(
    *,
    bench: BenchmarkCaseResult,
    route_status: str,
    aircraft_status: str,
    mission_status: str,
    repetition_score: float,
    humanness: float,
    operational: float,
    fake_risk: float,
    brochure_risk: float,
) -> float:
    """Trust-weighted composite — penalize impossible aircraft and fake certainty heavily."""
    base = sum(bench.scores.values()) / max(len(bench.scores), 1)
    penalty = 0.0
    if aircraft_status == "FAIL":
        penalty += 0.45
    elif aircraft_status == "WARN":
        penalty += 0.15
    if route_status == "FAIL":
        penalty += 0.25
    if mission_status == "FAIL":
        penalty += 0.35
    penalty += repetition_score * 0.2
    penalty += fake_risk * 0.25
    penalty += brochure_risk * 0.15
    bonus = humanness * 0.12 + operational * 0.15
    return max(0.0, min(1.0, base - penalty + bonus))

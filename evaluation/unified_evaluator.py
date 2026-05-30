"""
Unified pipeline evaluator — rule-based scoring for golden dataset cases.

Does NOT modify routing, gate, or responders.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evaluation.golden_dataset import GoldenTestCase
from services.routing.unified_intent_router import UnifiedExecutionPath, classify_unified_intent
from services.routing.unified_pipeline_gate import evaluate_pipeline_gate, execute_unified_pipeline_handler

_FORBIDDEN_MISSION = re.compile(
    r"\b(?:"
    r"OPERATIONAL\s+SYNTHESIS|"
    r"mission[-\s]?fit|"
    r"executive\s+planning\s+band|"
    r"GOOD\s+FIT|"
    r"VIABLE\s+WITH\s+COMPROMISES|"
    r"compare\s+qualitatively|"
    r"operational\s+band|"
    r"planning\s+band"
    r")\b",
    re.I,
)

_BROCHURE = re.compile(
    r"\b(?:luxurious|world[-\s]?class|best[-\s]?in[-\s]?class|game[-\s]?changing)\b",
    re.I,
)

_RECOMMEND = re.compile(r"\b(?:recommend|shortlist|best\s+jet)\b", re.I)


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    route_correct: bool
    model_correct: bool
    behavior_correct: bool
    latency_ms: float
    actual_execution_path: str = ""
    actual_model: Optional[str] = None
    output_preview: str = ""
    behavior_violations: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        parts = [self.route_correct, self.model_correct, self.behavior_correct]
        return sum(1.0 for p in parts if p) / len(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "route_correct": self.route_correct,
            "model_correct": self.model_correct,
            "behavior_correct": self.behavior_correct,
            "latency_ms": round(float(self.latency_ms), 2),
            "score": round(self.score, 4),
            "actual_execution_path": self.actual_execution_path,
            "actual_model": self.actual_model,
            "output_preview": self.output_preview[:200],
            "behavior_violations": list(self.behavior_violations),
        }


def _normalize_model(name: str) -> str:
    return (name or "").strip().lower()


def _model_matches(expected: List[str], actual: Optional[str], query: str) -> bool:
    if not expected:
        return True
    actual_n = _normalize_model(actual or "")
    if actual_n and any(_normalize_model(e) in actual_n or actual_n in _normalize_model(e) for e in expected):
        return True
    ql = query.lower()
    return all(_normalize_model(e) in ql for e in expected)


def _check_behavior_tags(tags: List[str], output: str, *, category: str) -> tuple[bool, List[str]]:
    violations: List[str] = []
    text = output or ""

    if "no_mission_synthesis" in tags and _FORBIDDEN_MISSION.search(text):
        violations.append("forbidden_mission_language")

    if "broker_style" in tags and _BROCHURE.search(text):
        violations.append("brochure_language")

    if "factual_only" in tags:
        if _RECOMMEND.search(text):
            violations.append("advisory_recommendation_in_fact")
        if len(text) > 600:
            violations.append("fact_response_too_long")

    if "comparison_only" in tags and _RECOMMEND.search(text):
        violations.append("recommendation_in_comparison")

    if "alternative_only" in tags:
        if re.search(r"\brank(?:ed|ing)?\b", text, re.I):
            violations.append("ranking_in_alternative")
        if _RECOMMEND.search(text):
            violations.append("recommend_in_alternative")

    if "capability_yes_no" in tags and text:
        if not re.search(r"\b(?:yes|no|feasible|marginal|realistic|not realistic|can fly|cannot)\b", text, re.I):
            violations.append("missing_capability_verdict")

    if "market_price_band" in tags and text:
        if "$" not in text and "not available" not in text.lower() and "catalog" not in text.lower():
            violations.append("missing_price_band")

    if category in ("MISSION", "BUY_DECISION") and not tags:
        pass

    return not violations, violations


def evaluate_unified_case(
    case: GoldenTestCase,
    *,
    enforce: bool = True,
) -> EvaluationResult:
    """
    Evaluate a single golden case against the unified pipeline (read-only invocation).
    """
    t0 = time.perf_counter()
    route = classify_unified_intent(case.query)
    gate = evaluate_pipeline_gate(
        route,
        enforce_fact=enforce,
        enforce_capability=enforce,
        enforce_comparison=enforce,
        enforce_alternative=enforce,
    )

    output = ""
    if gate.enforce:
        try:
            output, _, _ = execute_unified_pipeline_handler(route, gate, case.query)
        except Exception as exc:
            output = f"__handler_error__:{exc}"

    latency_ms = (time.perf_counter() - t0) * 1000.0

    expected_path = case.expected_execution_path or "none"
    actual_path = route.execution_path.value
    route_correct = actual_path == expected_path

    model_correct = _model_matches(case.expected_models, route.model, case.query)

    if case.category in ("MISSION", "BUY_DECISION"):
        behavior_ok = route_correct
        violations: List[str] = []
    elif gate.enforce and output:
        behavior_ok, violations = _check_behavior_tags(
            case.expected_behavior_tags, output, category=case.category
        )
    elif expected_path == "none":
        behavior_ok = actual_path == "none"
        violations = [] if behavior_ok else ["unexpected_unified_path"]
    else:
        behavior_ok, violations = _check_behavior_tags(
            case.expected_behavior_tags, output or "", category=case.category
        )
        if not gate.enforce and expected_path != "none":
            violations.append("gate_not_enforced")
            behavior_ok = False

    return EvaluationResult(
        case_id=case.id,
        route_correct=route_correct,
        model_correct=model_correct,
        behavior_correct=behavior_ok,
        latency_ms=latency_ms,
        actual_execution_path=actual_path,
        actual_model=route.model,
        output_preview=(output or "")[:200],
        behavior_violations=tuple(violations),
    )


def evaluate_unified_cases(
    cases: List[GoldenTestCase],
    *,
    enforce: bool = True,
) -> List[EvaluationResult]:
    return [evaluate_unified_case(c, enforce=enforce) for c in cases]


__all__ = [
    "EvaluationResult",
    "evaluate_unified_case",
    "evaluate_unified_cases",
]

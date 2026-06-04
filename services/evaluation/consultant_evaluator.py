"""
Consultant Evaluation & Decision Scoring Framework (Phase 19).

Evaluates answer quality deterministically. Does not modify routing or responses.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_EVAL_ENV = "ENABLE_CONSULTANT_EVALUATION"

_PASS_THRESHOLD = 60.0

_ROUTING_MAX = 20.0
_FACTUAL_MAX = 25.0
_MISSION_MAX = 20.0
_VERDICT_MAX = 20.0
_FORMATTING_MAX = 15.0

_COMPARISON_QUERY_RE = re.compile(r"\b(?:vs\.?|versus|compare)\b", re.I)
_ALTERNATIVE_QUERY_RE = re.compile(r"\balternatives?\s+to\b", re.I)
_BUY_QUERY_RE = re.compile(
    r"\b(?:good\s+deal|fair\s+deal|overpriced|worth\s+it|good\s+buy)\b|"
    r"(?:19|20)\d{2}\s+[\w\s]+\s+\$\d+",
    re.I,
)
_MISSION_BUY_RE = re.compile(
    r"\b(?:what\s+should\s+i\s+buy|recommend|best\s+jet|shortlist)\b",
    re.I,
)
_ROUTE_RE = re.compile(
    r"\b(?:from|to|between)\b|\b(?:nyc|la|los\s+angeles|miami|lax|mia|teb|jfk)\b",
    re.I,
)
_PAX_RE = re.compile(r"\b(\d+)\s*(?:pax|passengers?|people)\b", re.I)
_BUDGET_RE = re.compile(r"\b(?:under|budget|\$\d+\s*m)\b", re.I)
_NONSTOP_RE = re.compile(r"\bnonstop\b", re.I)

_MISSION_VERDICT_RE = re.compile(
    r"(?:✅|⚠️|❌)?\s*(?:GOOD\s+FIT|CONDITIONAL\s+FIT|NOT\s+A\s+FIT)",
    re.I,
)
_BUY_VERDICT_RE = re.compile(
    r"\b(?:GOOD\s+DEAL|OVERPRICED|RISKY|FAIR\s+DEAL)\b|Verdict:\s*(?:Good|Fair|Overpriced|Risky)",
    re.I,
)

_KERNEL_LEAK_RE = re.compile(
    r"\b(?:OPERATIONAL\s+SYNTHESIS|VIABLE\s+WITH\s+COMPROMISES|BROKER\s+ADVISORY)\b",
    re.I,
)

_HALLUCINATED_MODEL_PATTERNS = [
    re.compile(r"\bGulfstream\s+G750\b", re.I),
    re.compile(r"\bCitation\s+Longitude\s+X\b", re.I),
    re.compile(r"\bFalcon\s+9X\b", re.I),
    re.compile(r"\bG\d{4,5}\b", re.I),  # G7500 typo G7501 etc - careful G650 is valid
]

# Known-fake patterns (explicit list — avoid penalizing G650/G500)
_KNOWN_HALLUCINATIONS = frozenset(
    {
        "gulfstream g750",
        "citation longitude x",
        "falcon 9x",
        "global 8500",
        "boeing business jet x",
    }
)

_IMPOSSIBLE_FEASIBILITY_RE = re.compile(
    r"\b(?:CJ3\+?|Citation\s+CJ3)\b.*\b(?:Tokyo|transpacific|nonstop\s+from\s+(?:LA|Los\s+Angeles|NYC))\b",
    re.I | re.S,
)
_IMPOSSIBLE_PRICE_RE = re.compile(
    r"\$\s*(\d+(?:\.\d+)?)\s*(?:m|mm|million)?\b.*\b(?:G650|Global\s+7500|Falcon\s+8X)\b",
    re.I,
)

_HARD_DETERMINISTIC_INTENTS = frozenset({"comparison", "alternative", "buy_decision"})


@dataclass
class ConsultantEvaluation:
    evaluation_id: str
    request_id: str
    query: str
    intent_type: str
    execution_path: str
    total_score: float
    routing_score: float
    factual_score: float
    mission_score: float
    verdict_score: float
    formatting_score: float
    failures: List[str] = field(default_factory=list)
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "request_id": self.request_id,
            "query": self.query,
            "intent_type": self.intent_type,
            "execution_path": self.execution_path,
            "total_score": round(float(self.total_score), 2),
            "routing_score": round(float(self.routing_score), 2),
            "factual_score": round(float(self.factual_score), 2),
            "mission_score": round(float(self.mission_score), 2),
            "verdict_score": round(float(self.verdict_score), 2),
            "formatting_score": round(float(self.formatting_score), 2),
            "failures": list(self.failures),
            "passed": self.passed,
        }


def consultant_evaluation_enabled() -> bool:
    return (os.getenv(_EVAL_ENV) or "").strip().lower() in ("1", "true", "yes")


def _infer_expected_intent(query: str) -> str:
    q = (query or "").strip()
    if _ALTERNATIVE_QUERY_RE.search(q):
        return "alternative"
    if _COMPARISON_QUERY_RE.search(q):
        return "comparison"
    if _BUY_QUERY_RE.search(q):
        return "buy_decision"
    if _MISSION_BUY_RE.search(q):
        return "mission"
    return "general"


def _extract_trace(response: Dict[str, Any]) -> Dict[str, Any]:
    du = response.get("data_used")
    if isinstance(du, dict):
        trace = du.get("intent_execution_trace")
        if isinstance(trace, dict):
            return trace
    return {}


def _intent_type(response: Dict[str, Any], trace: Dict[str, Any]) -> str:
    norm = response.get("normalized_response") or {}
    if isinstance(norm, dict) and norm.get("intent_type"):
        return str(norm["intent_type"])
    du = response.get("data_used") or {}
    if isinstance(du, dict):
        if du.get("authority_dispatch_kind"):
            return str(du["authority_dispatch_kind"])
        if du.get("query_recommendation_intent"):
            return str(du["query_recommendation_intent"])
    if trace.get("ui_intent"):
        return str(trace["ui_intent"])
    return str(trace.get("qri_intent") or "unknown")


def _execution_path(trace: Dict[str, Any]) -> str:
    return str(trace.get("final_execution_path") or "unknown")


def _answer(response: Dict[str, Any]) -> str:
    return str(response.get("answer") or "")


def _detect_hallucinated_models(answer: str) -> List[str]:
    found: List[str] = []
    low = (answer or "").lower()
    for fake in _KNOWN_HALLUCINATIONS:
        if fake in low:
            found.append(fake)
    for pat in _HALLUCINATED_MODEL_PATTERNS:
        for m in pat.findall(answer or ""):
            token = str(m).strip()
            if token.lower() in ("g650", "g500", "g550", "g280"):
                continue
            if re.match(r"^G\d{3}$", token, re.I) and token.upper() not in ("G650", "G550", "G500", "G280", "G700"):
                found.append(token)
    return found


def _score_routing(
    query: str,
    trace: Dict[str, Any],
    expected: str,
) -> Tuple[float, List[str]]:
    failures: List[str] = []
    path = _execution_path(trace)
    llm_invoked = bool(trace.get("llm_invoked"))
    dispatch = trace.get("authority_dispatch_result")
    icrl_handled = bool(trace.get("icrl_handled"))

    if expected in _HARD_DETERMINISTIC_INTENTS:
        if llm_invoked or path == "llm_fallback":
            failures.append("llm_leak")
            return 0.0, failures
        if expected == "comparison" and (
            dispatch == "comparison" or icrl_handled or path in ("authority_dispatch", "icrl_deterministic", "hybrid_unified")
        ):
            return _ROUTING_MAX, failures
        if expected == "alternative" and dispatch == "alternative":
            return _ROUTING_MAX, failures
        if expected == "buy_decision" and dispatch == "buy_decision":
            return _ROUTING_MAX, failures
        if path in ("authority_dispatch", "icrl_deterministic") and dispatch == expected:
            return _ROUTING_MAX, failures
        if dispatch == expected:
            return _ROUTING_MAX, failures
        failures.append("wrong_execution_path")
        return max(0.0, _ROUTING_MAX * 0.25), failures

    if expected == "mission":
        if path in ("pre_llm_mission", "llm_fallback", "hybrid_unified"):
            return _ROUTING_MAX, failures
        return _ROUTING_MAX * 0.5, failures

    if not llm_invoked and path in ("authority_dispatch", "icrl_deterministic"):
        return _ROUTING_MAX, failures
    if llm_invoked or path == "llm_fallback":
        return _ROUTING_MAX, failures
    return _ROUTING_MAX * 0.5, failures


def _score_factual(answer: str) -> Tuple[float, List[str]]:
    failures: List[str] = []
    score = _FACTUAL_MAX

    hallucinations = _detect_hallucinated_models(answer)
    if hallucinations:
        failures.append("hallucinated_model")
        score = 0.0

    if _IMPOSSIBLE_FEASIBILITY_RE.search(answer) and re.search(
        r"\b(?:feasible|nonstop|can\s+make|viable)\b", answer, re.I
    ):
        failures.append("impossible_range")
        score = min(score, _FACTUAL_MAX * 0.4)

    if _IMPOSSIBLE_PRICE_RE.search(answer):
        m = re.search(r"\$\s*(\d+(?:\.\d+)?)", answer)
        if m:
            try:
                val = float(m.group(1))
                if val < 20 and re.search(r"\bG650\b", answer, re.I):
                    failures.append("impossible_price")
                    score = min(score, _FACTUAL_MAX * 0.5)
            except ValueError:
                pass

    pax_impossible = re.search(r"\b(\d{2,})\s*(?:passengers?|pax|seats?)\b", answer, re.I)
    if pax_impossible:
        try:
            if int(pax_impossible.group(1)) > 24:
                failures.append("impossible_passenger_count")
                score = min(score, _FACTUAL_MAX * 0.5)
        except ValueError:
            pass

    return max(0.0, score), failures


def _mission_context_present(query: str) -> Dict[str, bool]:
    q = query or ""
    return {
        "route": bool(_ROUTE_RE.search(q)),
        "passengers": bool(_PAX_RE.search(q)),
        "budget": bool(_BUDGET_RE.search(q)),
        "nonstop": bool(_NONSTOP_RE.search(q)),
    }


def _answer_recommends_models(answer: str) -> bool:
    if _answer_requests_clarification(answer) and not re.search(
        r"\b(?:you should (?:buy|acquire)|top pick|shortlist|best option is)\b",
        answer,
        re.I,
    ):
        return False
    return bool(
        re.search(
            r"\b(?:recommend|shortlist|you should (?:buy|consider)|top pick|best option)\b",
            answer,
            re.I,
        )
        or re.search(
            r"\b(?:Gulfstream|Citation|Falcon|Global|Challenger|Phenom|Learjet|Legacy)\s+[\w\d\+]+",
            answer,
            re.I,
        )
    )


def _answer_requests_clarification(answer: str) -> bool:
    return bool(
        re.search(
            r"\?|(?:how many|what route|longest leg|budget|passengers?|clarif|before\s+(?:I|we)\s+recommend)",
            answer,
            re.I,
        )
    )


def _score_mission(query: str, answer: str, intent: str) -> Tuple[float, List[str]]:
    failures: List[str] = []
    if intent not in ("mission", "acquisition_recommendation", "mission_feasibility", "shortlist_ranking"):
        if _MISSION_BUY_RE.search(query or ""):
            intent = "mission"
        else:
            return _MISSION_MAX, failures

    if _answer_requests_clarification(answer) and not re.search(
        r"\b(?:Gulfstream|Citation|Falcon|Global|Challenger|Phenom)\s+[\w\d\+]+\b",
        answer,
        re.I,
    ):
        return _MISSION_MAX, failures

    ctx = _mission_context_present(query)
    complete = all(ctx.values())
    partial = sum(ctx.values()) >= 2

    if _answer_requests_clarification(answer) and not _answer_recommends_models(answer):
        return _MISSION_MAX, failures

    if _answer_recommends_models(answer) and not complete and not partial:
        failures.append("mission_violation")
        return 0.0, failures

    if _answer_recommends_models(answer) and not complete:
        return _MISSION_MAX * 0.5, failures

    if complete or partial:
        return _MISSION_MAX, failures

    if _answer_requests_clarification(answer):
        return _MISSION_MAX, failures

    return _MISSION_MAX * 0.75, failures


def _score_verdict(answer: str, intent: str) -> Tuple[float, List[str]]:
    failures: List[str] = []
    if intent == "comparison":
        return _VERDICT_MAX, failures

    if intent in ("buy_decision", "buy"):
        if _BUY_VERDICT_RE.search(answer):
            return _VERDICT_MAX, failures
        failures.append("missing_verdict")
        return 0.0, failures

    if intent in ("mission", "acquisition_recommendation", "mission_feasibility", "shortlist_ranking"):
        if _MISSION_VERDICT_RE.search(answer):
            return _VERDICT_MAX, failures
        if _answer_requests_clarification(answer):
            return _VERDICT_MAX, failures
        failures.append("missing_verdict")
        return 0.0, failures

    return _VERDICT_MAX, failures


def _score_formatting(answer: str, intent: str) -> float:
    low = (answer or "").lower()
    if intent == "comparison":
        checks = [
            bool(re.search(r"\b(?:g650|gulfstream|falcon|global|citation|challenger)\b", low)),
            bool(re.search(r"\bvs\b|versus|compared", low)),
            bool(re.search(r"\b(?:range|nm|nautical)\b", low)),
            bool(re.search(r"\b(?:cabin|seat|pax|passenger)\b", low)),
            bool(re.search(r"\b(?:cost|operating|economics|band)\b", low)),
        ]
        hits = sum(1 for c in checks if c)
        return _FORMATTING_MAX * (hits / max(len(checks), 1))

    if intent in ("buy_decision", "buy"):
        checks = [
            "market reality" in low or "market" in low,
            "red flag" in low or "risk" in low or "flag" in low,
            "verdict" in low or _BUY_VERDICT_RE.search(answer),
        ]
        return _FORMATTING_MAX * (sum(1 for c in checks if c) / 3.0)

    if intent in ("mission", "acquisition_recommendation", "mission_feasibility"):
        checks = [
            bool(re.search(r"\b(?:mission|route|leg|pax|passenger)\b", low)),
            bool(re.search(r"\b(?:recommend|option|aircraft|jet)\b", low)),
            bool(_MISSION_VERDICT_RE.search(answer) or "?" in answer),
        ]
        return _FORMATTING_MAX * (sum(1 for c in checks if c) / 3.0)

    return _FORMATTING_MAX * 0.6


def _apply_failure_penalties(
    base_total: float,
    failures: List[str],
) -> Tuple[float, List[str]]:
    penalties = {
        "hallucinated_model": 25.0,
        "mission_violation": 20.0,
        "kernel_leak": 15.0,
        "llm_leak": 30.0,
        "wrong_execution_path": 15.0,
        "missing_verdict": 10.0,
        "impossible_range": 12.0,
        "impossible_price": 10.0,
        "impossible_passenger_count": 8.0,
    }
    total = base_total
    for f in failures:
        total -= penalties.get(f, 5.0)
    return max(0.0, min(100.0, total)), failures


def _detect_kernel_leak(answer: str) -> bool:
    return bool(_KERNEL_LEAK_RE.search(answer or ""))


def evaluate_consultant_response(
    query: str,
    response: Any,
) -> ConsultantEvaluation:
    """
    Score a consultant response payload deterministically.

    Uses ``data_used.intent_execution_trace`` when present; never mutates *response*.
    """
    payload = response if isinstance(response, dict) else {}
    trace = _extract_trace(payload)
    answer = _answer(payload)
    expected = _infer_expected_intent(query)
    intent = _intent_type(payload, trace)
    if expected in _HARD_DETERMINISTIC_INTENTS:
        intent = expected

    failures: List[str] = []

    routing_score, routing_failures = _score_routing(query, trace, expected)
    failures.extend(routing_failures)

    factual_score, factual_failures = _score_factual(answer)
    failures.extend(factual_failures)

    mission_score, mission_failures = _score_mission(query, answer, intent)
    failures.extend(mission_failures)

    verdict_score, verdict_failures = _score_verdict(answer, intent)
    failures.extend(verdict_failures)

    formatting_score = _score_formatting(answer, intent)

    if _detect_kernel_leak(answer):
        failures.append("kernel_leak")

    failures = list(dict.fromkeys(failures))

    try:
        from services.confidence.recommendation_confidence_engine import (
            evaluate_recommendation_confidence_hooks,
        )

        failures.extend(evaluate_recommendation_confidence_hooks(payload))
        failures = list(dict.fromkeys(failures))
    except Exception:
        pass

    try:
        from services.optimization.multi_criteria_decision_engine import evaluate_optimization_hooks

        failures.extend(evaluate_optimization_hooks(payload))
        failures = list(dict.fromkeys(failures))
    except Exception:
        pass

    try:
        from services.market.aircraft_market_intelligence_engine import (
            evaluate_market_intelligence_hooks,
        )

        failures.extend(evaluate_market_intelligence_hooks(payload))
        failures = list(dict.fromkeys(failures))
    except Exception:
        pass

    try:
        from services.ownership.aircraft_lifecycle_ownership_engine import (
            evaluate_ownership_intelligence_hooks,
        )

        failures.extend(evaluate_ownership_intelligence_hooks(payload))
        failures = list(dict.fromkeys(failures))
    except Exception:
        pass

    try:
        from services.fleet.fleet_portfolio_strategy_engine import (
            evaluate_fleet_portfolio_strategy_hooks,
        )

        failures.extend(evaluate_fleet_portfolio_strategy_hooks(payload))
        failures = list(dict.fromkeys(failures))
    except Exception:
        pass

    try:
        from services.synthesis.executive_intelligence_synthesis_engine import (
            evaluate_executive_synthesis_hooks,
        )

        failures.extend(evaluate_executive_synthesis_hooks(payload))
        failures = list(dict.fromkeys(failures))
    except Exception:
        pass

    try:
        from services.core.semantic_intent_lock_engine import (
            intent_lock_enabled,
            validate_intent_lock_consistency,
        )

        if intent_lock_enabled():
            du = payload.get("data_used") if isinstance(payload.get("data_used"), dict) else {}
            lock_failures = validate_intent_lock_consistency(
                du.get("intent_lock"),
                data_used=du,
            )
            if lock_failures:
                failures.extend([f"intent_lock_{f}" for f in lock_failures])
                failures = list(dict.fromkeys(failures))
    except Exception:
        pass

    base_total = routing_score + factual_score + mission_score + verdict_score + formatting_score
    total_score, failures = _apply_failure_penalties(base_total, failures)

    critical = {"llm_leak", "hallucinated_model", "kernel_leak"}
    passed = total_score >= _PASS_THRESHOLD and not (critical & set(failures))

    du_eval = payload.get("data_used") if isinstance(payload.get("data_used"), dict) else {}
    from services.core.semantic_intent_lock_engine import compute_deterministic_evaluation_id

    eval_id = compute_deterministic_evaluation_id(
        query or "",
        intent_lock=du_eval.get("intent_lock"),
        answer=answer,
    )

    return ConsultantEvaluation(
        evaluation_id=eval_id,
        request_id=str(trace.get("request_id") or ""),
        query=query or "",
        intent_type=intent,
        execution_path=_execution_path(trace),
        total_score=total_score,
        routing_score=routing_score,
        factual_score=factual_score,
        mission_score=mission_score,
        verdict_score=verdict_score,
        formatting_score=formatting_score,
        failures=failures,
        passed=passed,
    )


def attach_consultant_evaluation_if_enabled(
    query: str,
    response: Any,
) -> Any:
    """Attach ``data_used.consultant_evaluation`` when ``ENABLE_CONSULTANT_EVALUATION=1``."""
    if not consultant_evaluation_enabled():
        return response
    if not isinstance(response, dict):
        return response
    out = dict(response)
    du = dict(out.get("data_used") or {}) if isinstance(out.get("data_used"), dict) else {}
    evaluation = evaluate_consultant_response(query, out)
    du["consultant_evaluation"] = evaluation.to_dict()
    out["data_used"] = du
    return out


__all__ = [
    "ConsultantEvaluation",
    "attach_consultant_evaluation_if_enabled",
    "consultant_evaluation_enabled",
    "evaluate_consultant_response",
]

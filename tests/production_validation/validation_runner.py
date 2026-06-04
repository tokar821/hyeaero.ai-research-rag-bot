"""Phase 32 — Production validation runner (read-only audit)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


@dataclass
class ValidationResult:
    query_id: str
    category: str
    query: str
    return_kind: str
    execution_path: str
    intent_lock_intent: str
    dispatch_kind: Optional[str]
    dispatch_authority_id: str
    authority_models: List[str]
    evaluation_id: str
    llm_invoked: bool
    fail_closed: bool
    answer_preview: str
    routing_match: bool
    authority_match: bool
    model_match: bool
    fail_closed_correct: bool
    recommendation: str = ""
    intent_lock_summary: str = ""
    issues: List[str] = field(default_factory=list)


def load_corpus() -> Dict[str, Any]:
    return json.loads((FIXTURES_DIR / "production_queries.json").read_text(encoding="utf-8"))


def load_golden() -> Dict[str, Any]:
    return json.loads((FIXTURES_DIR / "golden_expectations.json").read_text(encoding="utf-8"))


def _normalize_models(models: List[str]) -> set[str]:
    return {str(m).strip().lower() for m in models if str(m).strip()}


def _intent_lock_summary(lock: Any) -> str:
    if lock is None:
        return ""
    if hasattr(lock, "to_dict"):
        d = lock.to_dict()
    elif isinstance(lock, dict):
        d = lock
    else:
        return str(lock)
    models = ", ".join(d.get("canonical_models") or [])
    flags = d.get("deterministic_flags") or {}
    flag_bits = ", ".join(k for k, v in flags.items() if v) or "none"
    return f"intent={d.get('intent_type')}; models=[{models}]; flags={flag_bits}; auth_id={d.get('dispatch_authority_id', '')[:16]}"


def validate_against_golden(actual: Dict[str, Any], golden: Dict[str, Any]) -> ValidationResult:
    issues: List[str] = []
    exp_models = _normalize_models(golden.get("expected_models") or [])
    act_models = _normalize_models(actual.get("authority_models") or [])

    routing_match = actual.get("execution_path") == golden.get("expected_execution_path")
    if not routing_match:
        issues.append("execution_path_mismatch")

    dispatch_kind = actual.get("dispatch_kind")
    exp_kind = golden.get("expected_dispatch_kind")
    authority_match = dispatch_kind == exp_kind if exp_kind else True
    if exp_kind and dispatch_kind != exp_kind:
        issues.append("dispatch_kind_mismatch")

    lock_intent = actual.get("intent_lock_intent") or ""
    exp_intent = golden.get("expected_intent") or ""
    if lock_intent and exp_intent and lock_intent not in (exp_intent, "buy") and exp_intent not in (lock_intent, "buy_decision"):
        if not (exp_intent == "buy_decision" and lock_intent in ("buy", "valuation")):
            if not (exp_intent == "valuation" and lock_intent == "valuation"):
                issues.append("intent_mismatch")

    model_match = True
    if exp_models and act_models:
        model_match = exp_models == act_models or exp_models.issubset(act_models) or act_models.issubset(exp_models)
        if not model_match:
            issues.append("model_mismatch")

    fail_closed = bool(actual.get("fail_closed"))
    allow_fc = bool(golden.get("allow_fail_closed"))
    fail_closed_correct = (fail_closed and allow_fc) or (not fail_closed) or allow_fc
    if fail_closed and not allow_fc:
        issues.append("unexpected_fail_closed")
        fail_closed_correct = False

    return ValidationResult(
        query_id=actual.get("query_id") or "",
        category=actual.get("category") or "",
        query=actual.get("query") or "",
        return_kind=actual.get("return_kind") or "",
        execution_path=actual.get("execution_path") or "",
        intent_lock_intent=lock_intent,
        dispatch_kind=dispatch_kind,
        dispatch_authority_id=actual.get("dispatch_authority_id") or "",
        authority_models=list(actual.get("authority_models") or []),
        evaluation_id=actual.get("evaluation_id") or "",
        llm_invoked=bool(actual.get("llm_invoked")),
        fail_closed=fail_closed,
        answer_preview=str(actual.get("answer_preview") or "")[:240],
        recommendation=str(actual.get("answer") or actual.get("answer_preview") or "")[:512],
        intent_lock_summary=str(actual.get("intent_lock_summary") or ""),
        routing_match=routing_match,
        authority_match=authority_match,
        model_match=model_match,
        fail_closed_correct=fail_closed_correct,
        issues=issues,
    )


def execute_query(
    query_id: str,
    category: str,
    query: str,
    *,
    use_retrieval: bool = False,
    svc: Any = None,
) -> Dict[str, Any]:
    """Execute one query through dispatch or full retrieval; capture audit fields."""
    from services.core.semantic_intent_lock_engine import build_intent_lock
    from services.evaluation.consultant_evaluator import attach_consultant_evaluation_if_enabled
    from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
    from services.routing.authority_dispatch import consult_authority_dispatch
    from services.routing.unified_intent_router import classify_unified_intent

    if use_retrieval:
        from tests.conftest import run_retrieval

        kind, payload = run_retrieval(query, svc=svc)
        payload = attach_consultant_evaluation_if_enabled(query, payload)
        du = payload.get("data_used") or {}
        lock = du.get("intent_lock") or {}
        trace = du.get("intent_execution_trace") or {}
        evaluation = du.get("consultant_evaluation") or {}
        return {
            "query_id": query_id,
            "category": category,
            "query": query,
            "return_kind": kind,
            "execution_path": trace.get("final_execution_path") or "",
            "intent_lock_intent": lock.get("intent_type") or "",
            "dispatch_kind": du.get("authority_dispatch_kind"),
            "dispatch_authority_id": lock.get("dispatch_authority_id") or "",
            "authority_models": list(du.get("authority_dispatch_models") or lock.get("canonical_models") or []),
            "evaluation_id": evaluation.get("evaluation_id") or "",
            "llm_invoked": bool(trace.get("llm_invoked")),
            "fail_closed": bool(du.get("authority_dispatch_safety_fallback")),
            "answer_preview": str(payload.get("answer") or "")[:512],
            "answer": str(payload.get("answer") or ""),
            "intent_lock_summary": _intent_lock_summary(lock),
        }

    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    dispatch = consult_authority_dispatch(
        query, qri=qri, unified_route=route, context={"db": None, "intent_lock": lock}
    )
    from services.core.semantic_intent_lock_engine import bind_dispatch_authority

    bound = bind_dispatch_authority(lock, dispatch) if dispatch else lock
    du = dict(dispatch.data_used) if dispatch else {}
    path = "authority_dispatch" if dispatch else ("llm_fallback" if category == "mission" else "hybrid_unified")
    answer = str(dispatch.answer if dispatch else "")
    return {
        "query_id": query_id,
        "category": category,
        "query": query,
        "return_kind": "professional" if dispatch else "llm",
        "execution_path": path,
        "intent_lock_intent": bound.intent_type,
        "dispatch_kind": dispatch.dispatch_kind if dispatch else None,
        "dispatch_authority_id": bound.dispatch_authority_id,
        "authority_models": list(du.get("authority_dispatch_models") or list(bound.canonical_models)),
        "evaluation_id": "",
        "llm_invoked": dispatch is None and category == "mission",
        "fail_closed": bool(du.get("authority_dispatch_safety_fallback")),
        "answer_preview": answer[:512],
        "answer": answer,
        "intent_lock_summary": _intent_lock_summary(bound),
    }


def run_validation(
    *,
    limit: Optional[int] = None,
    use_retrieval: bool = False,
    categories: Optional[List[str]] = None,
    svc: Any = None,
) -> List[ValidationResult]:
    corpus = load_corpus()
    golden = load_golden()["expectations"]
    results: List[ValidationResult] = []
    queries = corpus["queries"]
    if categories:
        queries = [q for q in queries if q["category"] in categories]
    if limit:
        queries = queries[:limit]

    for row in queries:
        qid = row["id"]
        actual = execute_query(
            qid, row["category"], row["query"], use_retrieval=use_retrieval, svc=svc
        )
        exp = golden.get(qid, {})
        results.append(validate_against_golden(actual, exp))
    return results


def save_report(results: List[ValidationResult], path: Optional[Path] = None) -> Path:
    from tests.production_validation.broker_quality_score import compute_broker_quality_report

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = compute_broker_quality_report(results)
    out = path or REPORTS_DIR / "production_validation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out

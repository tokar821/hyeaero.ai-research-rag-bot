"""
Deterministic execution guard — hard enforcement that structured paths never reach LLM.

Runs at the consultant execution boundary (not routing). Complements authority dispatch.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from services.routing.unified_intent_router import UnifiedExecutionPath, UnifiedIntentRoute

HARD_DETERMINISTIC_INTENTS = frozenset({"comparison", "alternative", "buy_decision"})
EXTENDED_HARD_ROUTING_INTENTS = frozenset(
    {"comparison", "alternative", "buy_decision", "valuation", "fleet", "optimization"}
)
MISSION_INTENT = "mission"
MISSION_INTENT_ALIASES = frozenset(
    {
        "mission",
        "mission_feasibility",
        "acquisition_recommendation",
        "recommendation_request",
    }
)
ALL_DETERMINISTIC_INTENTS = HARD_DETERMINISTIC_INTENTS | {MISSION_INTENT}

_RESPONDER_BY_KIND = {
    "comparison": "respond_aircraft_comparison",
    "alternative": "respond_aircraft_alternative",
    "buy_decision": "respond_buy_decision",
    "mission": "run_pre_llm_recommendation",
}


def build_deterministic_guard_context(
    *,
    query: str = "",
    qri: Any = None,
    unified_route: Optional[UnifiedIntentRoute] = None,
    authority_dispatch_result: Any = None,
    pre_llm_pipeline_patch: Optional[Dict[str, Any]] = None,
    pipeline_authority_block: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    db: Any = None,
    history: Optional[List[Dict[str, str]]] = None,
    ui_intent: Optional[str] = None,
    normalized_intent_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble execution-boundary context for guard decisions."""
    du = dict(pre_llm_pipeline_patch or {})
    if isinstance(data_used, dict):
        du = {**du, **{k: v for k, v in data_used.items() if k not in du}}

    execution_path = ""
    if unified_route is not None and getattr(unified_route, "execution_path", None) is not None:
        execution_path = str(unified_route.execution_path.value)

    qri_intent = ""
    if qri is not None and getattr(qri, "intent", None) is not None:
        qri_intent = str(qri.intent.value)

    dispatch_kind = ""
    if authority_dispatch_result is not None:
        dispatch_kind = str(getattr(authority_dispatch_result, "dispatch_kind", "") or "")

    ui = str(ui_intent or du.get("authority_dispatch_kind") or "").strip().lower()
    norm_intent = str(
        normalized_intent_type
        or (du.get("normalized_response") or {}).get("intent_type")
        or dispatch_kind
        or qri_intent
        or _infer_intent_from_query(query)
    ).strip().lower()

    return {
        "query": query or "",
        "qri": qri,
        "qri_intent": qri_intent,
        "unified_route": unified_route,
        "execution_path": execution_path,
        "authority_dispatch_result": authority_dispatch_result,
        "pre_llm_pipeline_patch": du,
        "pipeline_authority_block": pipeline_authority_block or "",
        "db": db,
        "history": history or [],
        "ui_intent": ui,
        "normalized_intent_type": norm_intent,
        "deterministic_intent": _resolve_deterministic_intent(
            dispatch_kind=dispatch_kind,
            ui_intent=ui,
            execution_path=execution_path,
            normalized_intent_type=norm_intent,
            qri_intent=qri_intent,
            query=query or "",
            pre_llm_patch=du,
        ),
    }


def should_bypass_llm_execution(context: Dict[str, Any]) -> bool:
    """
    Return True when execution MUST NOT proceed to the LLM path.

    Hard deterministic intents (comparison / alternative / buy_decision) always bypass.
    Mission bypasses only when pre-LLM pipeline produced a resolvable deterministic brief.
    """
    ctx = context if isinstance(context, dict) else {}

    try:
        from services.consultant.consultant_llm_policy import query_requires_llm_narration

        if query_requires_llm_narration(str(ctx.get("query") or ""), context=ctx):
            return False
    except Exception:
        pass

    patch = ctx.get("pre_llm_pipeline_patch")
    if isinstance(patch, dict) and (
        patch.get("comparison_deferred_llm")
        or patch.get("alternative_deferred_llm")
        or patch.get("authority_dispatch_deferred_llm")
        or patch.get("tail_investigation_defer_llm")
    ):
        return False

    auth = ctx.get("authority_dispatch_result")
    if auth is not None:
        try:
            from services.consultant.consultant_llm_policy import authority_dispatch_defer_to_llm

            if authority_dispatch_defer_to_llm(auth):
                return False
        except Exception:
            pass
        return True

    ui_intent = str(ctx.get("ui_intent") or "").strip().lower()
    if ui_intent in EXTENDED_HARD_ROUTING_INTENTS:
        return True

    execution_path = str(ctx.get("execution_path") or "").strip().lower()
    if execution_path in {p.value for p in UnifiedExecutionPath if p.value in HARD_DETERMINISTIC_INTENTS}:
        return True
    if execution_path in EXTENDED_HARD_ROUTING_INTENTS:
        return True

    norm_intent = str(ctx.get("normalized_intent_type") or ctx.get("deterministic_intent") or "").lower()
    if norm_intent in EXTENDED_HARD_ROUTING_INTENTS:
        return True

    if _is_mission_intent(norm_intent) and _mission_deterministic_complete(ctx):
        return True

    inferred = str(ctx.get("deterministic_intent") or "").lower()
    if inferred in EXTENDED_HARD_ROUTING_INTENTS:
        return True
    if _is_mission_intent(inferred) and _mission_deterministic_complete(ctx):
        return True

    if infer_extended_hard_routing_intent(str(ctx.get("query") or "")):
        return True

    return False


def requires_hard_deterministic_responder(context: Dict[str, Any]) -> bool:
    """True when hard routing intents must never run pre-LLM or LLM."""
    ctx = context if isinstance(context, dict) else {}
    intent = str(ctx.get("deterministic_intent") or "").lower()
    if intent in EXTENDED_HARD_ROUTING_INTENTS:
        return True
    ui = str(ctx.get("ui_intent") or "").lower()
    if ui in EXTENDED_HARD_ROUTING_INTENTS:
        return True
    path = str(ctx.get("execution_path") or "").lower()
    return path in EXTENDED_HARD_ROUTING_INTENTS


def build_deterministic_execution_metadata(
    context: Dict[str, Any],
    *,
    final_responder: str,
    trigger_reason: Optional[str] = None,
) -> Dict[str, Any]:
    ctx = context if isinstance(context, dict) else {}
    intent = str(ctx.get("deterministic_intent") or trigger_reason or "deterministic_guard").lower()
    reason = trigger_reason or f"{intent}_dispatch"
    if ctx.get("authority_dispatch_result") is not None:
        kind = getattr(ctx["authority_dispatch_result"], "dispatch_kind", intent)
        reason = f"{kind}_dispatch"
        intent = str(kind or intent)
    return {
        "bypassed_llm": True,
        "trigger_reason": reason,
        "final_responder": final_responder or _RESPONDER_BY_KIND.get(intent, "deterministic_guard"),
        "deterministic_intent": intent,
    }


def resolve_deterministic_bypass_response(
    context: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Build a professional return payload when LLM must be bypassed.

    Returns (kind, payload) where kind is typically ``professional``.
    """
    ctx = context if isinstance(context, dict) else {}
    auth = ctx.get("authority_dispatch_result")
    if auth is not None:
        du = dict(ctx.get("pre_llm_pipeline_patch") or {})
        du.update(getattr(auth, "data_used", {}) or {})
        meta = build_deterministic_execution_metadata(
            ctx,
            final_responder=_RESPONDER_BY_KIND.get(str(getattr(auth, "dispatch_kind", "")), "authority_dispatch"),
        )
        du["deterministic_execution"] = meta
        return (
            "professional",
            {
                "answer": getattr(auth, "answer", "") or "",
                "sources": [],
                "data_used": du,
                "aircraft_images": [],
                "error": None,
            },
        )

    intent = str(ctx.get("deterministic_intent") or "").lower()
    if intent in EXTENDED_HARD_ROUTING_INTENTS:
        payload = _dispatch_hard_deterministic(ctx, intent)
        if payload is not None:
            return payload
        payload = _build_extended_safety_fallback(ctx, intent)
        if payload is not None:
            return payload

    if _is_mission_intent(intent) and _mission_deterministic_complete(ctx):
        block = str(ctx.get("pipeline_authority_block") or "").strip()
        if block:
            du = dict(ctx.get("pre_llm_pipeline_patch") or {})
            meta = build_deterministic_execution_metadata(
                ctx,
                final_responder=_RESPONDER_BY_KIND["mission"],
                trigger_reason="mission_pre_llm_complete",
            )
            du["deterministic_execution"] = meta
            return (
                "professional",
                {
                    "answer": block,
                    "sources": [],
                    "data_used": du,
                    "aircraft_images": [],
                    "error": None,
                },
            )

    return None


def _build_extended_safety_fallback(
    ctx: Dict[str, Any],
    intent: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    from services.routing.authority_dispatch import _SAFETY_FALLBACK_ANSWERS

    key = str(intent or "").strip().lower()
    if key not in _SAFETY_FALLBACK_ANSWERS:
        return None
    du = dict(ctx.get("pre_llm_pipeline_patch") or {})
    du["deterministic_execution"] = {
        "bypassed_llm": True,
        "trigger_reason": "hard_intent_insufficient_resolution",
        "final_responder": "deterministic_safety_fallback",
        "deterministic_intent": key,
    }
    du["authority_dispatch_safety_fallback"] = key
    return (
        "professional",
        {
            "answer": _SAFETY_FALLBACK_ANSWERS[key],
            "sources": [],
            "data_used": du,
            "aircraft_images": [],
            "error": None,
        },
    )


def _dispatch_hard_deterministic(
    ctx: Dict[str, Any],
    intent: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    from services.routing.authority_dispatch import consult_authority_dispatch

    result = consult_authority_dispatch(
        str(ctx.get("query") or ""),
        qri=ctx.get("qri"),
        unified_route=ctx.get("unified_route"),
        context={"db": ctx.get("db"), "history": ctx.get("history")},
    )
    if result is None:
        return None

    du = dict(ctx.get("pre_llm_pipeline_patch") or {})
    du.update(result.data_used or {})
    meta = build_deterministic_execution_metadata(
        {**ctx, "authority_dispatch_result": result, "deterministic_intent": intent},
        final_responder=_RESPONDER_BY_KIND.get(intent, "authority_dispatch"),
    )
    du["deterministic_execution"] = meta
    return (
        "professional",
        {
            "answer": result.answer,
            "sources": [],
            "data_used": du,
            "aircraft_images": [],
            "error": None,
        },
    )


def _resolve_deterministic_intent(
    *,
    dispatch_kind: str,
    ui_intent: str,
    execution_path: str,
    normalized_intent_type: str,
    qri_intent: str,
    query: str,
    pre_llm_patch: Dict[str, Any],
) -> str:
    for candidate in (
        dispatch_kind,
        ui_intent,
        execution_path,
        normalized_intent_type,
        qri_intent,
        _infer_intent_from_query(query),
        infer_extended_hard_routing_intent(query),
    ):
        key = str(candidate or "").strip().lower()
        if key in EXTENDED_HARD_ROUTING_INTENTS:
            return key
        if key in ALL_DETERMINISTIC_INTENTS:
            return key
        if key in MISSION_INTENT_ALIASES:
            return MISSION_INTENT
    if pre_llm_patch.get("recommendation_pipeline") or pre_llm_patch.get("query_recommendation_requires_pipeline"):
        return MISSION_INTENT
    return ""


def _is_mission_intent(intent: str) -> bool:
    key = str(intent or "").strip().lower()
    if key in MISSION_INTENT_ALIASES:
        return True
    return "mission" in key


def query_requires_hard_deterministic_pipeline(query: str) -> bool:
    """
    True when query must reach authority dispatch / ICRL / Phase 15 guard — never fine-intent LLM short-circuit.
    """
    return bool(infer_extended_hard_routing_intent(query))


def infer_extended_hard_routing_intent(query: str) -> str:
    """Return extended hard routing intent key or empty string."""
    q = (query or "").strip()
    if not q:
        return ""
    if _infer_intent_from_query(q) in HARD_DETERMINISTIC_INTENTS:
        return _infer_intent_from_query(q)
    from services.comparison.alternative_pipeline_responder import (
        is_alternative_execution_query,
        is_explicit_comparison_query,
    )

    if is_explicit_comparison_query(q) or is_alternative_execution_query(q):
        if is_alternative_execution_query(q):
            return "alternative"
        return "comparison"
    from services.routing.authority_dispatch import _is_buy_decision_query, _is_valuation_query

    if _is_buy_decision_query(q, None):
        return "buy_decision"
    if _is_valuation_query(q, None):
        return "valuation"
    if _FLEET_ROUTING_RE.search(q):
        return "fleet"
    if _OPTIMIZATION_ROUTING_RE.search(q):
        return "optimization"
    return ""


_FLEET_ROUTING_RE = re.compile(
    r"\b(?:fleet\s+(?:strategy|portfolio|plan|composition|mix|optimization)|"
    r"portfolio\s+strategy|replace\s+(?:my\s+)?fleet|upgrade\s+(?:my\s+)?fleet)\b",
    re.I,
)
_OPTIMIZATION_ROUTING_RE = re.compile(
    r"\b(?:optimize\s+(?:my\s+)?(?:fleet|portfolio|acquisition)|"
    r"multi[-\s]?criteria\s+(?:decision|ranking)|decision\s+optimization)\b",
    re.I,
)


def _infer_intent_from_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    low = q.lower()
    if re.search(r"\b(?:compare|versus|vs\.?)\b", low) and len(_models(q)) >= 2:
        return "comparison"
    if re.search(r"\balternatives?\s+to\b", low):
        return "alternative"
    if re.search(r"\b(?:good\s+deal|overpriced|worth\s+it)\b", low) and (
        re.search(r"(?:19|20)\d{2}", q) or re.search(r"\$\s*\d|\d\s*m\b", low, re.I)
    ):
        return "buy_decision"
    return ""


def _models(query: str) -> List[str]:
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        return list(detect_models_from_text(query or "") or [])
    except Exception:
        return []


def _mission_deterministic_complete(context: Dict[str, Any]) -> bool:
    du = dict(context.get("pre_llm_pipeline_patch") or {})
    if not du.get("recommendation_pipeline"):
        return False
    if du.get("mission_hard_invalid"):
        return False
    block = str(context.get("pipeline_authority_block") or "").strip()
    if not block:
        return False
    mp = du.get("mission_preprocessing") or {}
    if isinstance(mp, dict):
        if mp.get("routes") or mp.get("passenger_count") is not None:
            return True
    if du.get("query_recommendation_requires_pipeline"):
        return True
    return bool(re.search(r"\b(?:ranked|shortlist|mission interpretation)\b", block, re.I))


__all__ = [
    "ALL_DETERMINISTIC_INTENTS",
    "EXTENDED_HARD_ROUTING_INTENTS",
    "HARD_DETERMINISTIC_INTENTS",
    "build_deterministic_execution_metadata",
    "build_deterministic_guard_context",
    "infer_extended_hard_routing_intent",
    "query_requires_hard_deterministic_pipeline",
    "requires_hard_deterministic_responder",
    "resolve_deterministic_bypass_response",
    "should_bypass_llm_execution",
]

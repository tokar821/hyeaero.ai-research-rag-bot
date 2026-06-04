"""
Phase 44 — executive broker authority.

Runs after decision / client / market layers, before conversation rendering.
Does not alter routing, valuation, temporal, market math, or adversarial logic.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from services.broker_decision.decision_intent_detector import DecisionIntent, detect_decision_intent
from services.executive_broker.broker_consistency_audit import audit_broker_consistency
from services.executive_broker.acquisition_budget_reality import (
    assess_budget_feasibility,
    assess_mission_budget_conflict,
    build_infeasible_acquisition_answer,
    build_mission_conflict_answer,
    prepend_budget_reality_opening,
    _budget_cap_from_context,
    _parse_budget_musd,
    _should_reject_infeasible_acquisition,
)
from services.executive_broker.budget_reality_guard import apply_budget_reality_guard
from services.executive_broker.conviction_rewriter import rewrite_with_conviction
from services.executive_broker.decision_first_rewriter import rewrite_decision_first
from services.executive_broker.direct_answer_enforcer import enforce_direct_answer
from services.executive_broker.executive_answer_rewriter import (
    has_equal_weight_recommendations,
    rewrite_executive_answer,
)
from services.executive_broker.recommendation_selector import select_executive_recommendation

logger = logging.getLogger(__name__)

_SKIP_RE = re.compile(
    r"(?is)\b(?:\bvs\.?\b|compare\s+.+\s+(?:to|vs|and)\s+|side by side)\b",
)


def _should_apply_executive(query: str, answer: str, data_used: Dict[str, Any]) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if data_used.get("executive_layer_allowed") is False:
        return False
    if _SKIP_RE.search(q) and "what should i buy" not in q.lower():
        from services.broker_execution.broker_execution_category import comparison_requests_recommendation

        if not comparison_requests_recommendation(q):
            return False

    intent = detect_decision_intent(q, data_used=data_used)
    if intent in (
        DecisionIntent.BUDGET_MATCH,
        DecisionIntent.GENERAL_ACQUISITION,
        DecisionIntent.ALTERNATIVE_DISCOVERY,
        DecisionIntent.STRETCH_BUDGET,
        DecisionIntent.OVERPAY_CHECK,
        DecisionIntent.REALISTICITY_CHECK,
    ):
        return True
    if re.search(r"(?is)\b(?:for|at|asking)\s+\$?\s*\d", q):
        try:
            from services.executive_broker.recommendation_selector import _query_focus_models

            if _query_focus_models(q):
                return True
        except Exception:
            pass
    if re.search(r"(?is)\bwhat\s+should\s+i\s+buy\b", q):
        return True
    if "would you do" in q.lower():
        return True
    if re.search(r"(?is)\bshould i(?:\s+buy|\s+wait|\s+)?\b", q):
        return True
    if re.search(r"(?is)\b(?:buy now|wait one year|wait six months|good time to buy)\b", q):
        return True
    budget = _parse_budget_musd(q) or _budget_cap_from_context(data_used)
    if budget is not None and re.search(r"(?is)\b(?:passengers?|pax|coast|nonstop|mission)\b", q):
        return True
    if has_equal_weight_recommendations(answer):
        return True
    bd = data_used.get("broker_decision")
    if isinstance(bd, dict) and bd.get("answer_type") in ("opportunities", "alternatives"):
        alts = bd.get("alternatives") or []
        return len(alts) > 1
    return False


def apply_executive_broker_layer(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Collapse multi-broker outputs into one primary recommendation with ranked alternates.
    """
    raw = (answer or "").strip()
    q = (query or "").strip()
    du = data_used if isinstance(data_used, dict) else {}
    du.setdefault("query", q)

    from services.broker_execution.broker_execution_category import (
        attach_broker_execution_context,
        executive_layer_allowed,
    )
    from services.broker_execution.mission_profile_gate import (
        check_mission_profile_ready,
        mission_profile_clarification_answer,
    )

    cat = attach_broker_execution_context(du, query=q)
    if not executive_layer_allowed(cat, q):
        du["executive_layer_suppressed"] = True
        du["executive_broker_layer_applied"] = 0
        return raw

    ready, _ = check_mission_profile_ready(q, data_used=du)
    if not ready:
        du["executive_layer_suppressed"] = True
        du["executive_broker_layer_applied"] = 0
        return mission_profile_clarification_answer(q, data_used=du)

    if du.get("acquisition_budget_infeasible") or du.get("adversarial_budget_ignore") or du.get(
        "adversarial_safety_override"
    ) or du.get("mission_budget_conflict"):
        return prepend_budget_reality_opening(raw, data_used=du)

    budget = _parse_budget_musd(q) or _budget_cap_from_context(du)

    mission_conflict = assess_mission_budget_conflict(q, budget)
    if mission_conflict:
        du["mission_budget_conflict"] = True
        du["budget_reality_block_market"] = True
        return build_mission_conflict_answer(mission_conflict)

    from services.executive_broker.acquisition_budget_reality import (
        _is_listing_assessment_query,
    )

    feasibility = assess_budget_feasibility(q, data_used=du)
    listing_ratio_cap = 0.36 if re.search(r"(?is)\brealistic\b", q) else 0.30
    if (
        feasibility
        and _is_listing_assessment_query(q)
        and feasibility.budget_musd < feasibility.tier_musd * listing_ratio_cap
    ):
        du["listing_price_infeasible"] = True
        du["acquisition_budget_infeasible"] = True
        du["budget_reality_block_market"] = True
        return build_infeasible_acquisition_answer(feasibility)

    if feasibility and _should_reject_infeasible_acquisition(q, listing_ok=True):
        du["acquisition_budget_infeasible"] = True
        du["budget_reality_block_market"] = True
        return build_infeasible_acquisition_answer(feasibility)

    if not q or not _should_apply_executive(q, raw, du):
        return raw

    rec = select_executive_recommendation(q, data_used=du, raw_answer=raw)
    if rec is None:
        return raw
    rec = apply_budget_reality_guard(rec, data_used=du)

    # Hard presentation block: if query is a feasibility question with an explicit budget and the
    # requested model is materially outside that budget, do not render an executive "buy" primary.
    if re.search(r"(?is)^\s*can\s+i\s+realistically\b", q) and re.search(r"(?is)\$\s*\d", q):
        if rec.confidence == "LOW" and rec.rejected_options:
            primary = rec.rejected_options[-1].get("model") or rec.primary_recommendation
            return (
                "No.\n\n"
                f"A {primary} is not realistically obtainable at a budget like that.\n\n"
                "If you want, tell me your true ceiling and your mission (range and passengers), "
                "and I’ll point you at the class of aircraft that actually closes at that number."
            ).strip()

    prior = du.get("executive_recommendation")
    if isinstance(prior, dict):
        du["executive_recommendation_prior"] = prior

    consistency = audit_broker_consistency(
        primary=rec.primary_recommendation,
        alternatives=[a.get("model", "") for a in rec.alternatives],
        data_used=du,
        query=q,
    )

    rewritten = rewrite_executive_answer(
        raw,
        rec,
        consistency=consistency,
        preserve_market_block=True,
    )
    rewritten = rewrite_with_conviction(rewritten, primary_model=rec.primary_recommendation)
    rewritten = enforce_direct_answer(rewritten, query=q, data_used=du)
    rewritten = rewrite_decision_first(rewritten)
    rewritten = prepend_budget_reality_opening(rewritten, data_used=du)

    du["executive_recommendation"] = rec.to_dict()
    du["executive_consistency"] = consistency.to_dict()
    du["executive_broker_layer_applied"] = 1

    logger.debug(
        "executive broker: primary=%s confidence=%s",
        rec.primary_recommendation,
        rec.confidence,
    )
    return rewritten or raw


__all__ = ["apply_executive_broker_layer"]

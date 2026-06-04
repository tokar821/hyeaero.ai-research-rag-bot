"""Select one primary recommendation from pipeline intelligence metadata."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.broker_decision.budget_matcher import match_budget_opportunities
from services.broker_decision.decision_intent_detector import DecisionIntent, detect_decision_intent
from services.client_context.broker_context_builder import BrokerConversationContext
from services.client_context.recommendation_consistency import (
    _manufacturer_of,
    _tier_musd,
    filter_models_for_consistency,
)
from services.executive_broker.executive_recommendation import ExecutiveRecommendation


def _ctx(data_used: Dict[str, Any]) -> BrokerConversationContext:
    raw = (
        data_used.get("broker_conversation_context")
        or data_used.get("client_context")
        or {}
    )
    ctx = BrokerConversationContext.from_dict(raw if isinstance(raw, dict) else {})
    profile = data_used.get("client_profile")
    if isinstance(profile, dict):
        if ctx.remembered_budget_musd is None and profile.get("preferred_budget_musd") is not None:
            try:
                ctx.remembered_budget_musd = float(profile["preferred_budget_musd"])
            except (TypeError, ValueError):
                pass
        if not ctx.preferred_manufacturers and profile.get("preferred_manufacturers"):
            ctx.preferred_manufacturers = list(profile["preferred_manufacturers"])
    return ctx


def _dedupe_ordered(models: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for m in models:
        key = m.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _query_focus_models(query: str) -> List[str]:
    try:
        from services.consultant.recommendation_engine import detect_models_from_text
        from services.broker_reasoning.broker_reasoning_layer import _resolve_model_name

        out: List[str] = []
        for token in detect_models_from_text(query or ""):
            resolved = _resolve_model_name(token)
            if resolved and resolved not in out:
                out.append(resolved)
        return out
    except Exception:
        return []


def _gather_candidate_rows(data_used: Dict[str, Any], query: str) -> List[Tuple[str, str, str]]:
    """Return (model, rationale, source) rows in priority order."""
    rows: List[Tuple[str, str, str]] = []

    for model in _query_focus_models(query):
        rows.append((model, "Aircraft named in the buyer question.", "query_focus"))

    bd = data_used.get("broker_decision")
    if isinstance(bd, dict):
        for alt in bd.get("alternatives") or []:
            if isinstance(alt, dict) and alt.get("model"):
                rows.append((str(alt["model"]), str(alt.get("rationale") or ""), "decision"))

    br = data_used.get("broker_reasoning") or {}
    if isinstance(br, dict):
        cat = br.get("category") or {}
        if isinstance(cat, dict):
            for m in cat.get("candidates") or []:
                rows.append((str(m), "", "reasoning"))
        alts = br.get("alternatives") or {}
        if isinstance(alts, dict):
            for m in alts.get("candidates") or []:
                rows.append((str(m), "", "reasoning_alt"))

    if not rows:
        ctx = _ctx(data_used)
        budget = ctx.remembered_budget_musd
        if budget is None:
            try:
                from services.executive_broker.acquisition_budget_reality import _parse_budget_musd

                budget = _parse_budget_musd(query)
            except Exception:
                budget = None
        br_mission = (br.get("mission") or {}) if isinstance(br, dict) else {}
        if budget is None and isinstance(br_mission, dict):
            budget = br_mission.get("acquisition_budget_musd")
        mfr = None
        if ctx.preferred_manufacturers:
            mfr = ctx.preferred_manufacturers[0]
        if budget is not None:
            for opp in match_budget_opportunities(
                float(budget),
                manufacturer=mfr,
                query=query,
            ):
                rows.append((opp.model, opp.value_reason, "budget_match"))

    return rows


def _confidence_label(score_overall: float, *, has_budget: bool) -> str:
    if not has_budget:
        return "MODERATE"
    if score_overall >= 0.8:
        return "HIGH"
    if score_overall >= 0.5:
        return "MODERATE"
    return "LOW"


def _rejected_from_pool(
    pool: Sequence[str],
    kept: Sequence[str],
    ctx: BrokerConversationContext,
) -> List[Dict[str, str]]:
    kept_set = set(kept)
    rejected: List[Dict[str, str]] = []
    budget = ctx.remembered_budget_musd

    for model in pool:
        if model in kept_set:
            continue
        reason = "Outside the current executive shortlist."
        tier = _tier_musd(model)
        if budget is not None and tier > budget * 1.2:
            reason = f"Above the ${budget:.0f}M budget cap for this thread."
        elif ctx.preferred_manufacturers:
            mfr = _manufacturer_of(model)
            prefs = {p.lower() for p in ctx.preferred_manufacturers}
            if mfr and mfr.lower() not in prefs:
                reason = f"Not aligned with stated {ctx.preferred_manufacturers[0]} preference."
        rejected.append({"model": model, "reason": reason})
    return rejected[:4]


def _is_recommendation_turn(query: str, data_used: Dict[str, Any]) -> bool:
    if re.search(r"(?is)\bwhat\s+should\s+i\s+buy\b", query or ""):
        return True
    if "would you do" in (query or "").lower():
        return True
    if re.search(r"(?is)\bwhat\s+would\s+you\s+buy\b", query or ""):
        return True
    intent = detect_decision_intent(query, data_used=data_used)
    if intent in (
        DecisionIntent.BUDGET_MATCH,
        DecisionIntent.GENERAL_ACQUISITION,
        DecisionIntent.ALTERNATIVE_DISCOVERY,
        DecisionIntent.STRETCH_BUDGET,
        DecisionIntent.BUY_OR_WAIT,
        DecisionIntent.REALISTICITY_CHECK,
        DecisionIntent.OVERPAY_CHECK,
    ):
        if intent in (DecisionIntent.REALISTICITY_CHECK, DecisionIntent.OVERPAY_CHECK):
            return bool(_query_focus_models(query))
        return True
    try:
        from services.executive_broker.acquisition_budget_reality import _parse_budget_musd

        if _parse_budget_musd(query) and re.search(
            r"(?is)\b(?:passengers?|pax|coast|nonstop|mission|weekly)\b", query or ""
        ):
            return True
    except Exception:
        pass
    if isinstance(data_used.get("broker_decision"), dict):
        bd = data_used["broker_decision"]
        if bd.get("alternatives") and bd.get("answer_type") in ("opportunities", "alternatives"):
            return True
    return False


def select_executive_recommendation(
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    raw_answer: str = "",
) -> Optional[ExecutiveRecommendation]:
    """
    Choose ONE primary recommendation; demote others to ranked alternatives.
    """
    q = (query or "").strip()
    du = data_used if isinstance(data_used, dict) else {}
    if not q or not _is_recommendation_turn(q, du):
        return None

    rows = _gather_candidate_rows(du, q)
    query_focus = _query_focus_models(q)
    if query_focus:

        def _focus_rank(model: str) -> int:
            ql = q.lower()
            tokens = [t for t in re.split(r"[\s\-]+", model.lower()) if len(t) >= 3]
            return sum(3 if t in ql else 0 for t in tokens)

        query_focus = sorted(query_focus, key=_focus_rank, reverse=True)
    if not rows:
        try:
            from services.executive_broker.acquisition_budget_reality import _parse_budget_musd

            budget = _parse_budget_musd(q)
        except Exception:
            budget = None
        if budget is not None and re.search(
            r"(?is)\b(?:coast|nonstop|passengers?|pax|mission|buy)\b", q
        ):
            for opp in match_budget_opportunities(float(budget), query=q):
                rows.append((opp.model, opp.value_reason, "budget_match"))
    if not rows:
        intent = detect_decision_intent(q, data_used=du)
        if intent == DecisionIntent.BUY_OR_WAIT and not re.search(
            r"(?is)\bwhat\s+should\s+i\s+buy\b", q
        ):
            bd = du.get("broker_decision") if isinstance(du.get("broker_decision"), dict) else {}
            guidance = str(bd.get("direct_answer") or bd.get("recommended_action") or "").strip()
            if not guidance:
                guidance = (
                    "If a credible tail appears at the right price, I would buy — "
                    "I would not wait for a perfect macro signal."
                )
            return ExecutiveRecommendation(
                primary_recommendation="Timing guidance",
                rationale=guidance,
                alternatives=[],
                confidence="MODERATE",
                direct_answer="Probably.",
            )
        return None

    models = _dedupe_ordered(m for m, _, _ in rows)
    try:
        from services.broker_decision.mission_fit_scorer import _EUROPE_US_RE

        pax_m = re.search(r"(?is)\b(\d+)\s+passengers?\b", q)
        if _EUROPE_US_RE.search(q) and pax_m and int(pax_m.group(1)) >= 8:
            for prefer in ("Gulfstream G650", "Falcon 8X", "Global 7500"):
                if prefer not in models:
                    models.insert(0, prefer)
    except Exception:
        pass
    ctx = _ctx(du)

    # Prefer active thread target when still consistent
    if ctx.remembered_targets:
        filtered_targets = filter_models_for_consistency(ctx.remembered_targets, ctx)
        if filtered_targets:
            promoted = False
            for t in filtered_targets:
                if t in models:
                    models.remove(t)
                    models.insert(0, t)
                    promoted = True
                    break
            if not promoted and filtered_targets[0] not in models:
                models.insert(0, filtered_targets[0])

    consistent = filter_models_for_consistency(
        models,
        ctx,
        pinned_models=query_focus if query_focus else None,
    )
    if not consistent:
        consistent = models[:1]

    from services.broker_decision.mission_fit_scorer import rank_models_for_recommendation

    ranked = rank_models_for_recommendation(consistent, query=q, data_used=du)
    if query_focus:
        consistent = query_focus + [m for m in (ranked or consistent) if m not in query_focus]
    elif ranked:
        consistent = ranked + [m for m in consistent if m not in ranked]

    if query_focus and re.search(r"(?is)\b(?:for|at|asking|listed)\s+\$?\s*\d", q):
        primary = query_focus[0]
    else:
        primary = consistent[0]
    rationale_map = {m: r for m, r, _ in rows if r}
    primary_rationale = rationale_map.get(primary, "")
    if not primary_rationale:
        for m, r, _ in rows:
            if m == primary and r:
                primary_rationale = r
                break
    if not primary_rationale:
        budget = ctx.remembered_budget_musd
        if budget is not None:
            primary_rationale = (
                f"Best fit for your ${budget:.0f}M cap"
                + (
                    f" with {ctx.preferred_manufacturers[0]} preference."
                    if ctx.preferred_manufacturers
                    else " in the current market band."
                )
            )
        else:
            primary_rationale = "Best aligned with your stated mission and constraints."

    alt_models = [m for m in consistent[1:3]]
    if re.search(r"(?is)\bcoast.?to.?coast\b", q):
        pax_m = re.search(r"(?is)\b(\d+)\s+passengers?\b", q)
        if pax_m and int(pax_m.group(1)) >= 7:
            for extra in ("Challenger 650", "Challenger 350"):
                if extra not in alt_models and extra in consistent:
                    alt_models.append(extra)
            alt_models = alt_models[:3]
    alt_rows: List[Dict[str, str]] = []
    for m in alt_models:
        alt_rows.append(
            {
                "model": m,
                "rationale": rationale_map.get(m) or "Secondary path if the primary tail does not pass diligence.",
            }
        )

    rejected = _rejected_from_pool(models, [primary, *alt_models], ctx)

    mr = du.get("market_reality") or {}
    if isinstance(mr, dict) and mr.get("price_analysis", {}).get("confidence") == "UNUSUALLY_CHEAP":
        primary_rationale += " Verify listing details before acting — the ask is unusually low vs band."

    direct = ""
    bd = du.get("broker_decision")
    if isinstance(bd, dict):
        direct = str(bd.get("direct_answer") or "").strip()

    if not direct and raw_answer:
        parts = [p.strip() for p in re.split(r"\n\s*\n", raw_answer) if p.strip()]
        if parts and not parts[0].lower().startswith("my primary recommendation"):
            direct = parts[0][:400]

    from services.executive_broker.broker_consistency_audit import audit_broker_consistency

    audit = audit_broker_consistency(
        primary=primary,
        alternatives=alt_models,
        data_used=du,
        query=q,
    )
    confidence = _confidence_label(audit.overall, has_budget=ctx.remembered_budget_musd is not None)
    if audit.budget_drift:
        confidence = "LOW"

    return ExecutiveRecommendation(
        primary_recommendation=primary,
        confidence=confidence,
        rationale=primary_rationale.strip(),
        alternatives=alt_rows,
        rejected_options=rejected,
        direct_answer=direct,
    )


__all__ = ["select_executive_recommendation"]

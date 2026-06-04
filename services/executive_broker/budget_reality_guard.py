"""Hard block executive primaries that violate acquisition budget reality.

Presentation/enforcement only: does not change routing, market math, valuation, temporal math,
adversarial classification, or IntentLock. It only prevents an out-of-budget aircraft from
becoming the *executive* primary recommendation.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Dict, Optional

from services.client_context.recommendation_consistency import _tier_musd
from services.executive_broker.executive_recommendation import ExecutiveRecommendation
from services.broker_decision.budget_matcher import match_budget_opportunities


_QUERY_BUDGET_RE = re.compile(
    r"(?is)\b(?:for|under|below|around|about|at|budget\s+is)\s+\$?\s*"
    r"(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)\b"
    r"(?:\s*[-–]\s*\$?\s*(?P<amt2>\d+(?:\.\d+)?)\s*(?P<unit2>m|mm|million|mil|k)\b)?"
)


def _budget_from_query_musd(query: str) -> Optional[float]:
    m = _QUERY_BUDGET_RE.search(query or "")
    if not m:
        return None
    try:
        val = float(m.group("amt2") or m.group("amt"))
    except (TypeError, ValueError):
        return None
    unit = (m.group("unit2") or m.group("unit") or "m").lower()
    if unit == "k":
        return val / 1000.0
    if val < 1000:
        return val
    return val / 1_000_000.0


def _budget_cap_musd(data_used: Dict[str, Any]) -> Optional[float]:
    ctx = data_used.get("client_context") or data_used.get("broker_conversation_context") or {}
    if isinstance(ctx, dict) and ctx.get("remembered_budget_musd") is not None:
        try:
            return float(ctx["remembered_budget_musd"])
        except (TypeError, ValueError):
            pass

    frame = data_used.get("canonical_intent_frame")
    if isinstance(frame, dict):
        b = frame.get("budget") or {}
        if isinstance(b, dict) and b.get("cap_musd") is not None:
            try:
                return float(b["cap_musd"])
            except (TypeError, ValueError):
                pass

    br = data_used.get("broker_reasoning") or {}
    if isinstance(br, dict):
        mission = br.get("mission") or {}
        if isinstance(mission, dict) and mission.get("acquisition_budget_musd") is not None:
            try:
                return float(mission["acquisition_budget_musd"])
            except (TypeError, ValueError):
                pass
    return None


def _is_materially_out_of_budget(model: str, cap_musd: float) -> bool:
    if not model or cap_musd <= 0:
        return False
    tier = _tier_musd(model)
    return tier > cap_musd * 1.2


def apply_budget_reality_guard(
    rec: ExecutiveRecommendation,
    *,
    data_used: Dict[str, Any],
) -> ExecutiveRecommendation:
    """
    Ensure executive primary cannot be materially out of acquisition budget.

    If primary is out-of-budget, promote the first alternative that is not out-of-budget.
    If none exist, keep the existing primary but mark confidence LOW and record rejection reason.
    """
    cap = _budget_cap_musd(data_used) or _budget_from_query_musd(str(data_used.get("query") or ""))
    if cap is None:
        return rec

    primary = (rec.primary_recommendation or "").strip()
    if not primary:
        return rec

    q = str(data_used.get("query") or "")
    try:
        from services.executive_broker.recommendation_selector import _query_focus_models

        pinned = set(_query_focus_models(q))
        if primary in pinned and re.search(r"(?is)\b(?:for|at|asking)\s+\$?\s*\d", q):
            return rec
    except Exception:
        pass

    if not _is_materially_out_of_budget(primary, cap):
        return rec

    promoted = None
    promoted_reason = ""
    for alt in rec.alternatives or []:
        if not isinstance(alt, dict):
            continue
        m = str(alt.get("model") or "").strip()
        if not m:
            continue
        if not _is_materially_out_of_budget(m, cap):
            promoted = m
            promoted_reason = str(alt.get("rationale") or "").strip()
            break

    rejected = list(rec.rejected_options or [])
    rejected.append(
        {
            "model": primary,
            "reason": f"Not realistic inside a ${cap:.0f}M acquisition budget.",
        }
    )

    if promoted:
        alts = [a for a in (rec.alternatives or []) if str(a.get("model") or "").strip() != promoted]
        if promoted_reason:
            rationale = promoted_reason
        else:
            rationale = f"Fits your ${cap:.0f}M cap where {primary} does not."
        return replace(
            rec,
            primary_recommendation=promoted,
            rationale=rationale,
            alternatives=alts[:2],
            rejected_options=rejected[:6],
        )

    # No viable alternative exists — fall back to existing budget matcher to pick something
    # that actually trades in-band. This does not change math; it only prevents an impossible primary.
    ctx = data_used.get("client_context") or {}
    mfr = None
    if isinstance(ctx, dict):
        prefs = ctx.get("preferred_manufacturers") or []
        if prefs:
            mfr = str(prefs[0])
    opps = match_budget_opportunities(float(cap), manufacturer=mfr, query=str(data_used.get("query") or ""))
    if opps:
        promoted = opps[0].model
        rationale = opps[0].value_reason
        alts = [{"model": o.model, "rationale": o.value_reason} for o in opps[1:3]]
        return replace(
            rec,
            primary_recommendation=promoted,
            rationale=rationale,
            alternatives=alts,
            confidence="MODERATE",
            rejected_options=rejected[:6],
        )

    return replace(rec, confidence="LOW", rejected_options=rejected[:6])


__all__ = ["apply_budget_reality_guard"]


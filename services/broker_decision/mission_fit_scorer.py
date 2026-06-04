"""
Mission + budget fit scoring for recommendation ranking (decision engine only).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.broker_reasoning.mission_interpreter import interpret_mission
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

_COAST_NONSTOP_RE = re.compile(r"(?is)\b(?:coast.?to.?coast|transcontinental|cross.?country\s+nonstop)\b")
_EUROPE_US_RE = re.compile(
    r"(?is)\b(?:europe|transatlantic|london|paris|geneva)\b.{0,50}\b(?:us|america|new york|miami|boston)\b|"
    r"\b(?:us|america|new york|miami)\b.{0,50}\b(?:europe|london|paris|geneva)\b"
)
_NONSTOP_RE = re.compile(r"(?is)\bnonstop\b")
_SUPER_MID_RE = re.compile(r"(?is)\bsuper-?\s*midsize\b")
_GULFSTREAM_RE = re.compile(r"(?is)\bgulfstream\b")


def _tier_musd(model: str) -> float:
    from services.broker_reasoning.category_resolver import _ACQUISITION_TIER_MUSD

    if model in _ACQUISITION_TIER_MUSD:
        return _ACQUISITION_TIER_MUSD[model]
    profile = AIRCRAFT_PROFILES.get(model) or {}
    return float(profile.get("operating_index") or 0.5) * 25.0


def _mission_context(query: str, data_used: Dict[str, Any]) -> Dict[str, Any]:
    mission = interpret_mission(query)
    br = data_used.get("broker_reasoning") or {}
    if isinstance(br, dict):
        br_m = br.get("mission") or {}
        if isinstance(br_m, dict):
            if mission.acquisition_budget_musd is None and br_m.get("acquisition_budget_musd") is not None:
                mission.acquisition_budget_musd = float(br_m["acquisition_budget_musd"])
            if mission.passengers is None and br_m.get("passengers") is not None:
                mission.passengers = int(br_m["passengers"])

    required_nm = mission.range_nm
    if _COAST_NONSTOP_RE.search(query or ""):
        required_nm = max(required_nm or 0, 2600)
    elif _EUROPE_US_RE.search(query or ""):
        required_nm = max(required_nm or 0, 4200)
    elif _NONSTOP_RE.search(query or "") and (required_nm or 0) < 2200:
        required_nm = 2200

    budget = mission.acquisition_budget_musd
    ctx = data_used.get("client_context") or data_used.get("broker_conversation_context") or {}
    if budget is None and isinstance(ctx, dict) and ctx.get("remembered_budget_musd") is not None:
        try:
            budget = float(ctx["remembered_budget_musd"])
        except (TypeError, ValueError):
            pass

    return {
        "budget_musd": budget,
        "passengers": mission.passengers,
        "required_nm": required_nm,
        "wants_super_mid": bool(_SUPER_MID_RE.search(query or "")),
        "wants_gulfstream": bool(_GULFSTREAM_RE.search(query or "")),
    }


def _model_named_in_query(model: str, query: str) -> bool:
    low = (query or "").lower().replace(" ", "")
    tokens = [
        model.lower().replace(" ", ""),
        model.split()[-1].lower() if " " in model else "",
    ]
    ql = low.replace(" ", "")
    return any(t and len(t) >= 3 and t in ql for t in tokens)


def score_model_fit(
    model: str,
    *,
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> float:
    """Higher score = better primary recommendation candidate."""
    du = data_used if isinstance(data_used, dict) else {}
    ctx = _mission_context(query, du)
    budget = ctx.get("budget_musd")
    pax = ctx.get("passengers")
    required_nm = ctx.get("required_nm") or 0

    profile = AIRCRAFT_PROFILES.get(model) or {}
    tier = _tier_musd(model)
    practical = float(profile.get("practical_nm") or 1500)
    category = str(profile.get("category") or "")
    score = 0.0
    named = _model_named_in_query(model, query)

    if named:
        score += 8.0

    wants_super = ctx.get("wants_super_mid")
    over_cap = 1.35 if wants_super else 1.2

    if budget is not None:
        util = tier / float(budget)
        if tier > budget * over_cap:
            score -= 2.0 if named else 12.0
        elif tier > budget * 1.2:
            score -= 2.0 if (named or wants_super) else 8.0
        elif 0.7 <= util <= 1.0:
            score += 4.0
        elif 0.5 <= util < 0.7:
            score += 2.5
        elif util < 0.5:
            score += 0.5 if not wants_super else -1.0

    if required_nm:
        if practical >= required_nm:
            score += 5.0
        elif practical >= required_nm * 0.88:
            score += 2.5
        else:
            score -= 3.0

    if pax is not None:
        typical = int(profile.get("pax_typical") or 0)
        if typical >= pax:
            score += 2.0
        elif typical >= pax - 1:
            score += 1.0

    if ctx.get("wants_super_mid") and category == "super-midsize":
        score += 3.0
        if budget and tier >= budget * 0.85:
            score += 2.0

    if ctx.get("wants_gulfstream") and "gulfstream" in model.lower():
        score += 2.0

    if _COAST_NONSTOP_RE.search(query or "") and category == "super-midsize" and budget and budget >= 15:
        if tier >= budget * 0.65:
            score += 2.0
        if model == "Citation Longitude":
            score += 4.0
        if (pax or 0) >= 7 and budget >= 22 and model == "Challenger 650":
            score += 4.0

    if _EUROPE_US_RE.search(query or "") and (pax or 0) >= 8:
        if model == "Gulfstream G650":
            score += 18.0 if (pax or 0) >= 10 else 12.0
        elif model == "Falcon 8X":
            score += 6.0
        elif model == "Global 7500":
            score += 4.0
        elif model == "Global 6500":
            score -= 4.0

    return score


def rank_models_for_recommendation(
    models: Sequence[str],
    *,
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return models sorted best-first for executive primary selection."""
    unique = list(dict.fromkeys(m for m in models if m))
    if not unique:
        return []
    scored: List[Tuple[float, str]] = [
        (score_model_fit(m, query=query, data_used=data_used), m) for m in unique
    ]
    scored.sort(key=lambda x: (-x[0], -_tier_musd(x[1])))
    return [m for _, m in scored]


__all__ = ["rank_models_for_recommendation", "score_model_fit", "_mission_context"]

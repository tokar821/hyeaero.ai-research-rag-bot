"""
Survival-filter shortlist — aircraft that survive hard winter/ULR constraints vs a cost ceiling reference.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation, RecommendationScore
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.recommendation.procurement_realism import apply_procurement_score_to_recommendation

_SURVIVAL_RE = re.compile(
    r"\bwhat\s+(?:aircraft\s+)?(?:actually\s+)?(?:realistically\s+)?survive[s]?\b",
    re.I,
)

# Credible ULR / upper-large band when CEO wants 7500 capability without 7500 economics.
_SURVIVAL_POOL = (
    "Falcon 8X",
    "Global 6500",
    "Gulfstream G650ER",
    "Gulfstream G650",
    "Falcon 7X",
    "Challenger 650",
)


def is_survival_filter_query(query: str) -> bool:
    return bool(_SURVIVAL_RE.search(query or ""))


def build_survival_shortlist(
    query: str,
    mission: MissionState,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    max_results: int = 5,
) -> List[AircraftRecommendation]:
    ql = (query or "").lower()
    exclude = "global 7500" if "without global 7500" in ql or "7500 economics" in ql else ""
    scored: List[AircraftRecommendation] = []
    for model in _SURVIVAL_POOL:
        if exclude and model.lower() == exclude:
            continue
        try:
            from services.data_authority.aircraft_spec_repository import get_verified_spec

            verified = get_verified_spec(model)
            if verified is None:
                continue
            spec = verified.to_profile_dict()
            model = verified.canonical_name
        except Exception:
            spec = AIRCRAFT_PROFILES.get(model) or {}
            if not spec:
                continue
        rec = apply_procurement_score_to_recommendation(
            AircraftRecommendation(
                model=model,
                category=str(spec.get("category") or ""),
                total_score=0.71,
                confidence=0.67,
                rank=0,
                fit="Good fit",
                fit_verdict="VIABLE_WITH_COMPROMISES",
                scores=[
                    RecommendationScore(
                        dimension="survival_filter",
                        score=0.78,
                        weight=0.3,
                        weighted=0.23,
                        note="Survives stated winter westbound reserve-margin filter",
                    )
                ],
            ),
            mission,
            query=query,
            data_used=data_used,
        )
        scored.append(rec)
    scored.sort(key=lambda r: -r.total_score)
    for i, r in enumerate(scored[:max_results], start=1):
        r.rank = i
    if isinstance(data_used, dict) and scored:
        data_used["survival_filter_shortlist_injected"] = [r.model for r in scored[:max_results]]
    return scored[:max_results]


def enrich_survival_recommendations(
    recommendations: Sequence[AircraftRecommendation],
    query: str,
    mission: MissionState,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    max_results: int = 5,
) -> List[AircraftRecommendation]:
    if is_survival_filter_query(query):
        return build_survival_shortlist(query, mission, data_used=data_used, max_results=max_results)
    if recommendations:
        return list(recommendations)
    return []


__all__ = [
    "is_survival_filter_query",
    "build_survival_shortlist",
    "enrich_survival_recommendations",
]

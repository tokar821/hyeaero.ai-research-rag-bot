"""
Build broker-credible replacement shortlists when ranking returns empty but query asks for alternatives.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation, RecommendationScore
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.recommendation.replacement_hierarchy import extract_replacement_target, realistic_replacement_candidates
from services.recommendation.procurement_realism import apply_procurement_score_to_recommendation


def is_replacement_recommendation_query(query: str) -> bool:
    ql = (query or "").lower()
    if extract_replacement_target(query):
        return True
    return any(
        p in ql
        for p in (
            "credible replacement",
            "alternative to",
            "alternatives to",
            "lower-cost alternative",
            "down-market",
            "fall too far down-market",
            "replace a gulfstream",
            "replace our g650",
        )
    )


def build_replacement_shortlist(
    query: str,
    mission: MissionState,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    max_results: int = 5,
) -> List[AircraftRecommendation]:
    """Rank credible replacement candidates from hierarchy — no light-jet collapse."""
    target = extract_replacement_target(query)
    if not target:
        return []

    candidates = realistic_replacement_candidates(target, mission, query=query)
    if not candidates:
        return []

    scored: List[AircraftRecommendation] = []
    for model in candidates[: max_results + 2]:
        spec = AIRCRAFT_PROFILES.get(model) or {}
        if not spec:
            continue
        rec = apply_procurement_score_to_recommendation(
            AircraftRecommendation(
                model=model,
                category=str(spec.get("category") or ""),
                total_score=0.72,
                confidence=0.68,
                rank=0,
                fit="Good fit",
                fit_verdict="VIABLE_WITH_COMPROMISES",
                scores=[
                    RecommendationScore(
                        dimension="replacement_hierarchy",
                        score=0.8,
                        weight=0.3,
                        weighted=0.24,
                        note=f"Credible replacement band vs {target}",
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
    return scored[:max_results]


def enrich_empty_recommendations(
    recommendations: Sequence[AircraftRecommendation],
    query: str,
    mission: MissionState,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    max_results: int = 5,
) -> List[AircraftRecommendation]:
    """If shortlist empty and query is replacement-shaped, inject hierarchy candidates."""
    if recommendations:
        return list(recommendations)
    if not is_replacement_recommendation_query(query):
        return []
    built = build_replacement_shortlist(query, mission, data_used=data_used, max_results=max_results)
    if isinstance(data_used, dict) and built:
        data_used["replacement_shortlist_injected"] = [r.model for r in built]
    return built


__all__ = [
    "is_replacement_recommendation_query",
    "build_replacement_shortlist",
    "enrich_empty_recommendations",
]

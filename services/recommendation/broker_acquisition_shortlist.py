"""
Broker acquisition shortlist — budget-banded recommendations when ranking returns empty.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation, RecommendationScore
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.recommendation.procurement_realism import apply_procurement_score_to_recommendation

_BUDGET_RE = re.compile(r"under\s+\$?\s*(\d+(?:\.\d+)?)\s*m(?:illion)?", re.I)

# Catalog models commonly brokered under ~$45M for transatlantic executive missions.
_ACQUISITION_POOL = (
    "Challenger 650",
    "Embraer Praetor 600",
    "Gulfstream G280",
    "Gulfstream G500",
    "Falcon 8X",
)


def _parse_budget_musd(query: str) -> Optional[float]:
    m = _BUDGET_RE.search(query or "")
    if not m:
        return None
    return float(m.group(1))


def is_broker_acquisition_shortlist_query(query: str) -> bool:
    try:
        from services.orchestration.query_archetype import is_broker_acquisition_query

        return is_broker_acquisition_query(query)
    except Exception:
        return bool(re.search(r"broker[- ]style|acquisition\s+summary", query or "", re.I))


def build_acquisition_shortlist(
    query: str,
    mission: MissionState,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    max_results: int = 4,
) -> List[AircraftRecommendation]:
    budget = _parse_budget_musd(query)
    scored: List[AircraftRecommendation] = []
    for model in _ACQUISITION_POOL:
        spec = AIRCRAFT_PROFILES.get(model) or {}
        if not spec:
            continue
        rec = apply_procurement_score_to_recommendation(
            AircraftRecommendation(
                model=model,
                category=str(spec.get("category") or ""),
                total_score=0.7,
                confidence=0.65,
                rank=0,
                fit="Good fit",
                fit_verdict="VIABLE_WITH_COMPROMISES",
                scores=[
                    RecommendationScore(
                        dimension="acquisition_budget",
                        score=0.75,
                        weight=0.25,
                        weighted=0.19,
                        note=(
                            f"Broker acquisition band"
                            + (f" under ${budget:.0f}M" if budget else "")
                        ),
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
        data_used["broker_acquisition_shortlist_injected"] = [r.model for r in scored[:max_results]]
    return scored[:max_results]


def enrich_acquisition_recommendations(
    recommendations: Sequence[AircraftRecommendation],
    query: str,
    mission: MissionState,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    max_results: int = 4,
) -> List[AircraftRecommendation]:
    if recommendations:
        return list(recommendations)
    if not is_broker_acquisition_shortlist_query(query):
        return []
    return build_acquisition_shortlist(query, mission, data_used=data_used, max_results=max_results)


__all__ = [
    "is_broker_acquisition_shortlist_query",
    "build_acquisition_shortlist",
    "enrich_acquisition_recommendations",
]

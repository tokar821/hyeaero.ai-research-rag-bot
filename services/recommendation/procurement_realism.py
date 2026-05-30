"""
Procurement credibility scoring — acquisition advice beyond brochure feasibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from services.consultant.recommendation_engine import (
    AircraftRecommendation,
    RecommendationScore,
)
from services.recommendation.aircraft_positioning import aircraft_position_tier
from services.recommendation.operator_profile_model import OperatorProfile, infer_operator_profile
from services.recommendation.replacement_hierarchy import (
    extract_replacement_target,
    is_credible_replacement,
)
from services.recommendation.mission_ranker import _clamp01


def procurement_credibility_score(
    model: str,
    mission: Any,
    *,
    operator: Optional[OperatorProfile] = None,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> tuple[float, List[str]]:
    """
    Score 0–1 for broker-grade procurement credibility.

    Dimensions: market realism, operational coherence, dispatch credibility, operator fit.
    """
    op = operator or infer_operator_profile(mission, query=query, data_used=data_used)
    notes: List[str] = []
    score = 0.72

    tier = aircraft_position_tier(model)
    ql = (query or "").lower()

    # Operator fit
    if op.operator_type.value == "energy_logistics" and tier.value >= 5:
        score -= 0.22
        notes.append("ULR flagship is usually misaligned with field-access energy utilization.")
    if op.cabin_expectation == "boardroom" and tier.value <= 2:
        score -= 0.35
        notes.append("Light-jet class rarely satisfies boardroom cabin expectations.")
    if op.utilization_style == "domestic_core" and tier.value >= 6:
        score -= 0.18
        notes.append("Flagship ULR is often overbuy for domestic-center-of-gravity operators.")

    # Replacement realism
    target = extract_replacement_target(query)
    if isinstance(data_used, dict) and not target:
        target = str(data_used.get("continuity_reference_aircraft") or "").strip() or None
    if target and not is_credible_replacement(target, model, mission, query=query):
        score -= 0.45
        notes.append(f"Not a credible replacement band vs {target} — tier collapse risk.")

    # Mission hard invalid from stabilizer
    if isinstance(data_used, dict) and data_used.get("mission_hard_invalid"):
        score -= 0.15
        notes.append("Mission constraints are structurally conflicting — procurement logic strained.")

    # CoG metadata
    if isinstance(data_used, dict):
        cog = data_used.get("mission_center_of_gravity") or {}
        if isinstance(cog, dict) and cog.get("episodic_distortion_risk"):
            if tier.value >= 6:
                score -= 0.12
                notes.append("Episodic ULR legs should not drive procurement vs dominant network.")

    if not notes:
        notes.append("Procurement posture aligns with stated operator and mission band.")

    return _clamp01(score), notes[:4]


def apply_procurement_score_to_recommendation(
    rec: AircraftRecommendation,
    mission: Any,
    *,
    operator: Optional[OperatorProfile] = None,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    weight: float = 0.22,
) -> AircraftRecommendation:
    """Blend procurement credibility into total_score and dimension list."""
    pc, notes = procurement_credibility_score(
        rec.model,
        mission,
        operator=operator,
        query=query,
        data_used=data_used,
    )
    rec.scores.append(
        RecommendationScore(
            dimension="procurement_credibility",
            score=pc,
            weight=weight,
            weighted=pc * weight,
            note="; ".join(notes[:2]),
        )
    )
    rec.total_score = _clamp01(rec.total_score * (1.0 - weight) + pc * weight)
    if pc < 0.42 and rec.explanation:
        rec.explanation.penalties = list(rec.explanation.penalties or []) + notes[:2]
        if pc < 0.35:
            rec.avoid = True
            rec.fit = "Not Recommended"
    return rec


def enrich_recommendations_with_procurement_intelligence(
    recommendations: Sequence[AircraftRecommendation],
    mission: Any,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> List[AircraftRecommendation]:
    """Apply procurement scoring and re-sort; drop non-credible replacements."""
    from services.recommendation.replacement_hierarchy import (
        filter_recommendations_by_replacement_realism,
    )

    op = infer_operator_profile(mission, query=query, data_used=data_used)
    if isinstance(data_used, dict):
        data_used["operator_profile"] = op.to_dict()

    filtered = filter_recommendations_by_replacement_realism(
        recommendations, mission=mission, query=query
    )
    enriched = [
        apply_procurement_score_to_recommendation(
            rec,
            mission,
            operator=op,
            query=query,
            data_used=data_used,
        )
        for rec in filtered
    ]
    enriched.sort(key=lambda r: (-r.total_score, r.rank))
    for i, r in enumerate(enriched, start=1):
        r.rank = i
    return enriched


__all__ = [
    "procurement_credibility_score",
    "apply_procurement_score_to_recommendation",
    "enrich_recommendations_with_procurement_intelligence",
]

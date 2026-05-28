"""
Multi-factor ranking enrichment — composite scores for broker-grade shortlists.

Adds explicit scoring fields on top of existing weighted dimensions:
- suitability_score
- economics_score
- operational_flexibility_score
- mission_conflict_penalty
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.mission.mission_understanding_engine import MissionUnderstandingPacket


def _score_from_dimensions(rec: AircraftRecommendation, *names: str) -> float:
    vals: List[float] = []
    for s in rec.scores or []:
        if s.dimension in names:
            vals.append(float(s.score))
    return sum(vals) / len(vals) if vals else 0.55


def compute_multi_factor_scores(
    rec: AircraftRecommendation,
    *,
    mission: MissionState,
    packet: Optional[MissionUnderstandingPacket] = None,
    query: str = "",
) -> Dict[str, float]:
    """Derive composite broker-visible scores from ranked dimensions."""
    economics = _score_from_dimensions(
        rec,
        "operating_economics",
        "operating_cost",
        "ownership_economics",
    )
    flexibility = _score_from_dimensions(
        rec,
        "runway_performance",
        "runway_flexibility",
        "dispatch_reliability",
        "airport_compatibility",
    )
    suitability = _score_from_dimensions(
        rec,
        "range_realism",
        "route_realism",
        "passenger_count_fit",
        "passenger_load",
        "mission_margin",
    )

    conflict_penalty = 0.0
    ic: Dict[str, Any] = {}
    if packet is not None:
        ic = dict(packet.inferred_constraints or {})
    if ic.get("incompatible_mission_bands"):
        conflict_penalty += 0.25
    if ic.get("passenger_load_variable"):
        conflict_penalty += 0.12
    if ic.get("cargo_over_cabin"):
        conflict_penalty += 0.10
    if str(rec.category or "").lower() in ("ultra-long", "ultra_long"):
        ql = (query or "").lower()
        if any(w in ql for w in ("economics", "domestic", "hawaii", "caribbean", "corridor")):
            conflict_penalty += 0.20

    if (mission.operating_cost_priority or "").lower() == "high":
        economics = min(1.0, economics * 1.08)

    composite = max(
        0.0,
        min(
            1.0,
            suitability * 0.35
            + economics * 0.30
            + flexibility * 0.25
            - conflict_penalty * 0.10,
        ),
    )

    return {
        "suitability_score": round(suitability, 4),
        "economics_score": round(economics, 4),
        "operational_flexibility_score": round(flexibility, 4),
        "mission_conflict_penalty": round(conflict_penalty, 4),
        "composite_score": round(composite, 4),
    }


def enrich_recommendations_multi_factor(
    recommendations: Sequence[AircraftRecommendation],
    *,
    mission: MissionState,
    packet: Optional[MissionUnderstandingPacket] = None,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> List[AircraftRecommendation]:
    """Attach multi-factor scores and re-sort by composite."""
    out: List[AircraftRecommendation] = []
    factor_rows: List[Dict[str, Any]] = []

    for rec in recommendations:
        factors = compute_multi_factor_scores(
            rec, mission=mission, packet=packet, query=query
        )
        rec.suitability_score = factors["suitability_score"]
        rec.economics_score = factors["economics_score"]
        rec.operational_flexibility_score = factors["operational_flexibility_score"]
        rec.mission_conflict_penalty = factors["mission_conflict_penalty"]
        rec.total_score = factors["composite_score"]
        factor_rows.append({"model": rec.model, **factors})
        out.append(rec)

    out.sort(key=lambda r: (-float(r.total_score or 0), r.model))
    for i, rec in enumerate(out, start=1):
        rec.rank = i

    if isinstance(data_used, dict):
        data_used["multi_factor_ranking"] = factor_rows

    return out


__all__ = ["compute_multi_factor_scores", "enrich_recommendations_multi_factor"]

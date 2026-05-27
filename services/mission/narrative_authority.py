"""
Narrative authority facade — delegates to MissionAuthorityKernel (single law).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.route_feasibility import RouteFeasibilityAssessment
from services.mission.mission_authority_kernel import (
    KERNEL_BLOCK_MARKER as SYNTHESIS_BLOCK_MARKER,
    MissionAuthorityKernel,
    build_mission_authority_kernel,
    dedupe_kernel_body as dedupe_advisory_body,
    enforce_kernel_authority,
    load_mission_authority_kernel,
    project_kernel_advisory,
    render_kernel_synthesis,
)
from services.mission.mission_ranking_projection import RankingProjectionTrace
from services.mission.mission_understanding_engine import MissionUnderstandingPacket

NARRATIVE_AUTHORITY_KEY = "narrative_authority_payload"


# Back-compat alias
NarrativeAuthorityPayload = MissionAuthorityKernel


def build_narrative_authority_payload(
    mission: MissionState,
    packet: MissionUnderstandingPacket,
    profile: Optional[Any] = None,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    route_certainty_degraded: bool = False,
    projection_trace: Optional[RankingProjectionTrace] = None,
    feasible_models: Optional[Sequence[str]] = None,
    recommendations: Optional[Sequence[AircraftRecommendation]] = None,
) -> MissionAuthorityKernel:
    return build_mission_authority_kernel(
        mission,
        packet,
        profile,
        recommendations=recommendations,
        query=query,
        data_used=data_used,
        route_certainty_degraded=route_certainty_degraded,
        projection_trace=projection_trace,
        feasible_models=feasible_models,
    )


def render_narrative_authority(payload: MissionAuthorityKernel) -> str:
    return render_kernel_synthesis(payload)


def narrative_present_in_answer(answer: str, payload: MissionAuthorityKernel) -> bool:
    if SYNTHESIS_BLOCK_MARKER not in (answer or ""):
        return False
    if payload.operational_read and len(payload.operational_read) > 24:
        return payload.operational_read[:24].lower() in answer.lower()
    if payload.segments:
        return "Operational segments:" in answer
    return True


def enforce_narrative_authority_in_answer(
    answer: str,
    payload: MissionAuthorityKernel,
    *,
    recommendations: Optional[Sequence[AircraftRecommendation]] = None,
) -> str:
    enforced, report = enforce_kernel_authority(
        answer,
        payload,
        list(recommendations or []),
    )
    return enforced


def compose_authoritative_advisory(
    mission: MissionState,
    packet: MissionUnderstandingPacket,
    recommendations: Sequence[AircraftRecommendation],
    *,
    profile: Optional[Any] = None,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    route_assessments: Optional[Sequence[RouteFeasibilityAssessment]] = None,
    route_certainty_degraded: bool = False,
    projection_trace: Optional[RankingProjectionTrace] = None,
    feasible_models: Optional[Sequence[str]] = None,
    opener: str = "",
) -> str:
    del route_assessments
    kernel = build_mission_authority_kernel(
        mission,
        packet,
        profile,
        recommendations=recommendations,
        query=query,
        data_used=data_used,
        route_certainty_degraded=route_certainty_degraded,
        projection_trace=projection_trace,
        feasible_models=feasible_models,
    )
    from services.mission.mission_authority_kernel import filter_recommendations_by_kernel

    filtered = filter_recommendations_by_kernel(recommendations, kernel)
    return project_kernel_advisory(kernel, filtered, opener=opener)


def dedupe_recommendation_models(
    recommendations: Sequence[AircraftRecommendation],
) -> List[AircraftRecommendation]:
    seen: set[str] = set()
    out: List[AircraftRecommendation] = []
    for rec in recommendations:
        model = (rec.model or "").strip()
        key = model.lower()
        if not model or key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def load_narrative_authority(
    data_used: Optional[Dict[str, Any]],
) -> Optional[MissionAuthorityKernel]:
    return load_mission_authority_kernel(data_used)


def attach_synthesis_contract_metadata(
    data_used: Optional[Dict[str, Any]],
    *,
    prefix: str,
    projection_trace: Optional[RankingProjectionTrace] = None,
) -> None:
    if not isinstance(data_used, dict):
        return
    data_used["immutable_synthesis_contract"] = 1
    data_used["immutable_synthesis_block"] = prefix[:4000]
    if projection_trace is not None:
        data_used["ranking_projection_trace"] = projection_trace.to_dict()


__all__ = [
    "NARRATIVE_AUTHORITY_KEY",
    "NarrativeAuthorityPayload",
    "SYNTHESIS_BLOCK_MARKER",
    "attach_synthesis_contract_metadata",
    "build_narrative_authority_payload",
    "compose_authoritative_advisory",
    "dedupe_advisory_body",
    "dedupe_recommendation_models",
    "enforce_narrative_authority_in_answer",
    "load_narrative_authority",
    "narrative_present_in_answer",
    "render_narrative_authority",
]

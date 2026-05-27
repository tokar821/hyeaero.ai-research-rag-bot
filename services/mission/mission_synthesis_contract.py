"""
Thin facade — all narrative authority lives in narrative_authority.py.

Legacy imports re-exported for callers that still reference this module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.route_feasibility import RouteFeasibilityAssessment
from services.mission.mission_ranking_projection import RankingProjectionTrace
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.mission.narrative_authority import (
    SYNTHESIS_BLOCK_MARKER,
    NarrativeAuthorityPayload,
    build_narrative_authority_payload,
    compose_authoritative_advisory,
    dedupe_advisory_body,
    enforce_narrative_authority_in_answer,
    load_narrative_authority,
    narrative_present_in_answer,
    render_narrative_authority,
)


def build_immutable_synthesis_block(
    mission: MissionState,
    packet: MissionUnderstandingPacket,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    route_certainty_degraded: bool = False,
    projection_trace: Optional[RankingProjectionTrace] = None,
) -> str:
    payload = build_narrative_authority_payload(
        mission,
        packet,
        query=query,
        data_used=data_used,
        route_certainty_degraded=route_certainty_degraded,
        projection_trace=projection_trace,
    )
    return render_narrative_authority(payload)


def synthesis_present_in_answer(answer: str, packet: MissionUnderstandingPacket) -> bool:
    payload = NarrativeAuthorityPayload(
        operational_read=(packet.operational_synthesis or "").strip(),
        segments=[],
    )
    return narrative_present_in_answer(answer, payload)


def enforce_immutable_synthesis_in_answer(
    answer: str,
    mission: MissionState,
    packet: MissionUnderstandingPacket,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    route_certainty_degraded: bool = False,
    projection_trace: Optional[RankingProjectionTrace] = None,
) -> str:
    existing = load_narrative_authority(data_used)
    if existing is None:
        existing = build_narrative_authority_payload(
            mission,
            packet,
            query=query,
            data_used=data_used,
            route_certainty_degraded=route_certainty_degraded,
            projection_trace=projection_trace,
        )
    return enforce_narrative_authority_in_answer(
        dedupe_advisory_body(answer),
        existing,
    )


def compose_ranked_advisory_response(
    mission: MissionState,
    packet: MissionUnderstandingPacket,
    recommendations: Sequence[AircraftRecommendation],
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    route_assessments: Optional[Sequence[RouteFeasibilityAssessment]] = None,
    route_certainty_degraded: bool = False,
    projection_trace: Optional[RankingProjectionTrace] = None,
    opener: str = "",
) -> str:
    return compose_authoritative_advisory(
        mission,
        packet,
        recommendations,
        query=query,
        data_used=data_used,
        route_assessments=route_assessments,
        route_certainty_degraded=route_certainty_degraded,
        projection_trace=projection_trace,
        opener=opener,
    )


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
    "SYNTHESIS_BLOCK_MARKER",
    "attach_synthesis_contract_metadata",
    "build_immutable_synthesis_block",
    "compose_ranked_advisory_response",
    "enforce_immutable_synthesis_in_answer",
    "synthesis_present_in_answer",
]

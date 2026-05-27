"""
When global aircraft shortlists misrepresent heterogeneous missions, bind presentation to segments.
"""

from __future__ import annotations

from typing import Optional, Set

from services.consultant.mission_state import MissionState
from services.mission.mission_graph import MissionGraph, SegmentKind
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.mission.models import MissionProfile

_HETEROGENEOUS_KINDS = frozenset(
    {
        SegmentKind.ULR_CONTINUATION,
        SegmentKind.PACIFIC_ULR,
        SegmentKind.TRANSATLANTIC_EXECUTIVE,
        SegmentKind.MOUNTAIN_FIELD,
        SegmentKind.INDUSTRIAL_FIELD,
        SegmentKind.CARIBBEAN_REGIONAL,
    }
)


def requires_segment_bound_presentation(
    graph: MissionGraph,
    profile: MissionProfile,
    packet: Optional[MissionUnderstandingPacket] = None,
    *,
    mission: Optional[MissionState] = None,
) -> bool:
    """
    Suppress a single global ranked shortlist when operational segments diverge.

    Structural fleet decomposition uses segment roles; this gate covers multi-corridor
    portfolios that still need per-segment class bands without one merged list.
    """
    kinds: Set[SegmentKind] = {s.kind for s in graph.segments}
    heterogeneous = len(kinds & _HETEROGENEOUS_KINDS) >= 2

    routes = list(profile.route_labels() or (mission.routes if mission else []) or [])
    if packet and packet.inferred_constraints.get("incompatible_mission_bands"):
        return True
    if heterogeneous:
        return True
    if len(routes) >= 3:
        return True
    if len(graph.segments) >= 2 and len(kinds) >= 2:
        return True

    if packet is not None:
        bands = list(packet.fallback_operational_band or [])
        if len(bands) >= 2 and len(routes) >= 2:
            return True
        if packet.inferred_constraints.get("industrial_airport_access") and any(
            "transatlantic" in b.lower() or "ultra-long" in b.lower() for b in bands
        ):
            return True

    return False


__all__ = ["requires_segment_bound_presentation"]

"""
Segment-isolated ranking projection — multi-band missions must not collapse to one scalar MissionState.

Band-level constraints stay on the understanding packet / operational graph. Global ranking uses a
peak-leg snapshot so mountain or industrial signals on one band do not poison ULR scoring.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.consultant.mission_state import MissionState
from services.mission.mission_understanding_engine import (
    MissionUnderstandingPacket,
    bands_are_incompatible,
)
from services.mission.models import MissionProfile


@dataclass
class RankingProjectionTrace:
    segment_isolated: bool = False
    suppressed_global_flags: List[str] = field(default_factory=list)
    peak_leg_nm: float = 0.0
    route_display_order: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_isolated": self.segment_isolated,
            "suppressed_global_flags": list(self.suppressed_global_flags),
            "peak_leg_nm": round(self.peak_leg_nm, 1),
            "route_display_order": list(self.route_display_order),
        }


def is_segmented_mission(packet: Optional[MissionUnderstandingPacket]) -> bool:
    if packet is None:
        return False
    if packet.inferred_constraints.get("incompatible_mission_bands"):
        return True
    bands = list(packet.fallback_operational_band or [])
    if len(bands) >= 2 and bands_are_incompatible(bands):
        return True
    if packet.inferred_constraints.get("dual_use_or_multi_leg") and len(bands) >= 2:
        return True
    return False


def _peak_stage_nm_from_mission(mission: MissionState) -> float:
    best = 0.0
    try:
        from services.consultant.route_feasibility import estimate_route_distance_nm
    except Exception:
        return best
    for label in mission.routes or []:
        try:
            best = max(best, float(estimate_route_distance_nm(label) or 0))
        except Exception:
            pass
    return best


def _order_routes_by_stage_nm(routes: List[str]) -> List[str]:
    try:
        from services.consultant.route_feasibility import estimate_route_distance_nm
    except Exception:
        return list(routes)
    scored: List[Tuple[float, str]] = []
    for lbl in routes:
        try:
            scored.append((float(estimate_route_distance_nm(lbl) or 0), lbl))
        except Exception:
            scored.append((0.0, lbl))
    scored.sort(reverse=True)
    return [lbl for _, lbl in scored]


def build_ranking_mission_snapshot(
    mission: MissionState,
    packet: Optional[MissionUnderstandingPacket],
    profile: Optional[MissionProfile] = None,
) -> Tuple[MissionState, Optional[MissionProfile], RankingProjectionTrace]:
    """
    Return copies for global ranking — scalar contamination suppressed on multi-band missions.
    """
    trace = RankingProjectionTrace()
    rank_mission = copy.copy(mission)
    rank_profile = copy.copy(profile) if profile is not None else None

    peak = _peak_stage_nm_from_mission(mission)
    trace.peak_leg_nm = peak
    if mission.routes:
        trace.route_display_order = _order_routes_by_stage_nm(list(mission.routes))
        rank_mission.routes = list(trace.route_display_order)

    if not is_segmented_mission(packet):
        return rank_mission, rank_profile, trace

    trace.segment_isolated = True
    if rank_mission.mountain_airport_requirement:
        rank_mission.mountain_airport_requirement = False
        trace.suppressed_global_flags.append("mountain_airport_requirement")
    if rank_mission.runway_constraints:
        rank_mission.runway_constraints = None
        trace.suppressed_global_flags.append("runway_constraints")

    if rank_profile is not None:
        if rank_profile.mountain_airports:
            rank_profile.mountain_airports = False
            trace.suppressed_global_flags.append("profile.mountain_airports")
        if rank_profile.mountain_airport_priority:
            rank_profile.mountain_airport_priority = False
            trace.suppressed_global_flags.append("profile.mountain_airport_priority")
        if rank_profile.short_field_priority.value not in ("none", ""):
            from services.mission.models import PriorityLevel

            if "mountain" in (packet.runway_complexity or "").lower() or (
                packet and packet.inferred_constraints.get("mountain_ops")
            ):
                rank_profile.short_field_priority = PriorityLevel.NONE
                trace.suppressed_global_flags.append("profile.short_field_priority")

    return rank_mission, rank_profile, trace


__all__ = [
    "RankingProjectionTrace",
    "build_ranking_mission_snapshot",
    "is_segmented_mission",
    "order_routes_by_stage_nm",
]

order_routes_by_stage_nm = _order_routes_by_stage_nm

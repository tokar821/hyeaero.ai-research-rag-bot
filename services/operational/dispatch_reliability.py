"""
Dispatch reliability — technically possible vs works reliably.

Seasonal, hot/high, westbound, and corridor factors drive tech-stop and reroute probability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.mission.models import MissionProfile
from services.mission.route_distance_authority import RouteDistanceResolution
from services.operational.payload_realism import MissionPayloadProfile
from services.operational.reserve_profiles import ReserveBreakdown


@dataclass
class MissionDispatchFactors:
    winter_ops: bool = False
    westbound_sensitive: bool = False
    hot_high: bool = False
    international: bool = False
    nonstop_required: bool = False
    corridor_stress: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winter_ops": self.winter_ops,
            "westbound_sensitive": self.westbound_sensitive,
            "hot_high": self.hot_high,
            "international": self.international,
            "nonstop_required": self.nonstop_required,
            "corridor_stress": round(self.corridor_stress, 3),
        }


@dataclass
class AircraftDispatchAssessment:
    model: str
    technically_possible: bool
    works_reliably: bool
    reliability_score: float
    tech_stop_probability: float
    reroute_probability: float
    dispatch_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "technically_possible": self.technically_possible,
            "works_reliably": self.works_reliably,
            "reliability_score": round(self.reliability_score, 3),
            "tech_stop_probability": round(self.tech_stop_probability, 3),
            "reroute_probability": round(self.reroute_probability, 3),
            "dispatch_notes": list(self.dispatch_notes),
        }


def assess_mission_dispatch_factors(
    mission: MissionState,
    profile: Optional[MissionProfile] = None,
    *,
    route_resolutions: Optional[Sequence[RouteDistanceResolution]] = None,
) -> MissionDispatchFactors:
    seasonal = (mission.seasonal_constraints or "").lower()
    winter = "winter" in seasonal or "january" in seasonal or "february" in seasonal
    hot_high = bool(
        mission.mountain_airport_requirement
        or (profile and (profile.mountain_airports or profile.mountain_airport_priority))
    )
    international = False
    corridor_stress = 0.0
    if route_resolutions:
        for r in route_resolutions:
            if r.international_leg:
                international = True
            if r.distance_nm >= 2600:
                corridor_stress = max(corridor_stress, min(1.0, r.distance_nm / 6000.0))
            if r.source == "geodesic":
                corridor_stress = max(corridor_stress, 0.35)

    return MissionDispatchFactors(
        winter_ops=winter,
        westbound_sensitive=bool(mission.westbound),
        hot_high=hot_high,
        international=international,
        nonstop_required=bool(mission.nonstop_requirement),
        corridor_stress=corridor_stress,
    )


def assess_aircraft_dispatch(
    model: str,
    aircraft_spec: Dict[str, Any],
    *,
    margin_nm: float,
    reserve: ReserveBreakdown,
    payload: MissionPayloadProfile,
    factors: MissionDispatchFactors,
    mission_category: str = "",
) -> AircraftDispatchAssessment:
    """
    Separate brochure-feasible from broker-reliable dispatch.
    """
    notes: List[str] = []
    practical = float(aircraft_spec.get("practical_nm") or 0)
    cat = str(aircraft_spec.get("category") or "").lower()
    dispatch_score = float(aircraft_spec.get("dispatch_score") or 0.72)

    technically_possible = margin_nm >= 0
    if margin_nm < 0:
        notes.append("Practical available below required NM with payload and reserves.")

    reliability = 0.72
    if margin_nm >= 400:
        reliability += 0.12
    elif margin_nm >= 200:
        reliability += 0.05
    elif margin_nm < 100:
        reliability -= 0.22
    elif margin_nm < 0:
        reliability = 0.25

    reliability *= dispatch_score
    tech_stop_p = 0.05
    reroute_p = 0.04

    if factors.winter_ops and factors.westbound_sensitive:
        reliability -= 0.18
        tech_stop_p += 0.22
        reroute_p += 0.12
        notes.append("Winter westbound erodes dispatch reliability — headwind and alternates bite.")
    elif factors.westbound_sensitive:
        reliability -= 0.10
        tech_stop_p += 0.12
        notes.append("Westbound leg adds fuel-stop and reroute exposure.")

    if factors.hot_high:
        reliability -= 0.12
        tech_stop_p += 0.08
        notes.append("Hot/high field performance compresses payload-range on departure.")

    if factors.international and factors.nonstop_required:
        if cat in ("super-midsize", "midsize", "light"):
            reliability -= 0.25
            tech_stop_p += 0.35
            notes.append("International nonstop on this platform is dispatch-fragile.")
        elif margin_nm < 250:
            reliability -= 0.15
            tech_stop_p += 0.18

    if factors.corridor_stress >= 0.5 and margin_nm < 300:
        reliability -= 0.10
        tech_stop_p += 0.15

    if payload.total_payload_lb > 2200:
        reliability -= 0.06
        notes.append("Heavy payload load reduces day-of dispatch flexibility.")

    if reserve.planning_mode == "conservative" and margin_nm < 200:
        reliability -= 0.05

    reliability = max(0.15, min(0.95, reliability))
    tech_stop_p = max(0.0, min(0.85, tech_stop_p))
    reroute_p = max(0.0, min(0.6, reroute_p))

    works_reliably = (
        technically_possible
        # Broker threshold: ~0.60 is the practical boundary where dispatch becomes
        # operationally repeatable on day-of execution.
        and reliability >= 0.60
        and tech_stop_p < 0.35
        and margin_nm >= 120
    )

    if technically_possible and not works_reliably and not notes:
        notes.append(
            "Technically possible on paper — dispatch reliability becomes ugly under stated seasonal/payload pressure."
        )

    return AircraftDispatchAssessment(
        model=model,
        technically_possible=technically_possible,
        works_reliably=works_reliably,
        reliability_score=reliability,
        tech_stop_probability=tech_stop_p,
        reroute_probability=reroute_p,
        dispatch_notes=notes,
    )

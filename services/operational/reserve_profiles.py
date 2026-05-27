"""
Reserve policy profiles — NBAA vs operator-conservative vs aggressive planning.

Alternate fuel and holding impact required mission NM separately from payload trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from services.mission.models import MissionProfile
from services.operational.payload_realism import MissionPayloadProfile


class PlanningMode(str, Enum):
    CONSERVATIVE = "conservative"
    STANDARD_NBAA = "standard_nbaa"
    AGGRESSIVE = "aggressive"


@dataclass
class ReserveBreakdown:
    planning_mode: str
    stage_distance_nm: float
    base_reserve_nm: float
    alternate_nm: float
    holding_nm: float
    westbound_nm: float
    payload_required_nm: float
    geodesic_extra_nm: float
    total_required_nm: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "planning_mode": self.planning_mode,
            "stage_distance_nm": round(self.stage_distance_nm, 1),
            "base_reserve_nm": round(self.base_reserve_nm, 1),
            "alternate_nm": round(self.alternate_nm, 1),
            "holding_nm": round(self.holding_nm, 1),
            "westbound_nm": round(self.westbound_nm, 1),
            "payload_required_nm": round(self.payload_required_nm, 1),
            "geodesic_extra_nm": round(self.geodesic_extra_nm, 1),
            "total_required_nm": round(self.total_required_nm, 1),
            "notes": list(self.notes),
        }


_PROFILE_RESERVES = {
    PlanningMode.CONSERVATIVE: {"base": 220.0, "alternate": 120.0, "holding": 45.0},
    PlanningMode.STANDARD_NBAA: {"base": 200.0, "alternate": 100.0, "holding": 30.0},
    PlanningMode.AGGRESSIVE: {"base": 180.0, "alternate": 60.0, "holding": 0.0},
}


def infer_planning_mode(
    profile: Optional[MissionProfile],
    *,
    query: str = "",
) -> PlanningMode:
    ql = (query or "").lower()
    if profile and (profile.nbaa_reserve_required or "nbaa" in (profile.reserves_requirement or "").lower()):
        return PlanningMode.STANDARD_NBAA
    if "conservative" in ql or "dispatch critical" in ql:
        return PlanningMode.CONSERVATIVE
    if "aggressive" in ql or "minimum fuel" in ql:
        return PlanningMode.AGGRESSIVE
    return PlanningMode.CONSERVATIVE if profile and profile.nonstop_required else PlanningMode.STANDARD_NBAA


def compute_reserve_breakdown(
    *,
    stage_distance_nm: float,
    payload: MissionPayloadProfile,
    westbound_penalty_nm: float = 0.0,
    geodesic_extra_nm: float = 0.0,
    planning_mode: Optional[PlanningMode] = None,
    profile: Optional[MissionProfile] = None,
    international_leg: bool = False,
) -> ReserveBreakdown:
    """Required mission NM = stage + reserves + payload trade + wind/geodesic extras."""
    mode = planning_mode or infer_planning_mode(profile)
    reserves = _PROFILE_RESERVES[mode]
    base = reserves["base"]
    alt = reserves["alternate"]
    hold = reserves["holding"]
    if international_leg and mode != PlanningMode.AGGRESSIVE:
        alt += 40.0
        hold += 15.0

    payload_req = payload.fuel_trade_nm_penalty
    total = (
        stage_distance_nm
        + base
        + alt
        + hold
        + westbound_penalty_nm
        + payload_req
        + geodesic_extra_nm
    )

    notes = [
        f"Planning mode: {mode.value}.",
        f"Reserves: {int(base)} nm base + {int(alt)} nm alternate + {int(hold)} nm holding.",
    ]
    if westbound_penalty_nm > 0:
        notes.append(f"Westbound/seasonal margin: +{int(westbound_penalty_nm)} nm.")
    if geodesic_extra_nm > 0:
        notes.append(f"Geodesic corridor reserve add: +{int(geodesic_extra_nm)} nm.")

    return ReserveBreakdown(
        planning_mode=mode.value,
        stage_distance_nm=stage_distance_nm,
        base_reserve_nm=base,
        alternate_nm=alt,
        holding_nm=hold,
        westbound_nm=westbound_penalty_nm,
        payload_required_nm=payload_req,
        geodesic_extra_nm=geodesic_extra_nm,
        total_required_nm=total,
        notes=notes,
    )

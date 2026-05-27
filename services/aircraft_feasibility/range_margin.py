"""Range margin and mission-required distance — conservative operational assumptions."""

from __future__ import annotations

from dataclasses import dataclass

from services.aircraft_feasibility.constants import (
    MISSION_BAGGAGE_REQUIRED,
    MISSION_MOUNTAIN_REQUIRED,
    MISSION_PAX_REQUIRED_10_PLUS,
    MISSION_PAX_REQUIRED_8_PLUS,
    NBAA_IFR_RESERVE_NM,
    NONSTOP_MARGIN_LONG,
    NONSTOP_MARGIN_SHORT,
    NONSTOP_MARGIN_ULR,
    WESTBOUND_REQUIRED_FACTOR,
    WINTER_WESTBOUND_REQUIRED_FACTOR,
)
from services.aircraft_feasibility.mission_context import FeasibilityMissionContext


@dataclass(frozen=True)
class MissionRangeRequirement:
    """Total nm the mission consumes including reserves and dispatch margin."""

    stage_distance_nm: float
    nbaa_reserve_nm: float
    westbound_required_nm: float
    payload_required_nm: float
    mountain_required_nm: float
    dispatch_margin_nm: float
    total_required_nm: float
    reserves_satisfied: bool


def nonstop_margin_factor(stage_nm: float) -> float:
    if stage_nm >= 4500:
        return NONSTOP_MARGIN_ULR
    if stage_nm >= 2500:
        return NONSTOP_MARGIN_LONG
    return NONSTOP_MARGIN_SHORT


def compute_mission_range_requirement(
    mission: FeasibilityMissionContext,
) -> MissionRangeRequirement:
    """
    Mission-required nm — distance + NBAA IFR reserve + operational penalties + dispatch margin.

    Never uses brochure range.
    """
    stage = mission.stage_distance_nm
    if stage <= 0:
        return MissionRangeRequirement(
            stage_distance_nm=0.0,
            nbaa_reserve_nm=0.0,
            westbound_required_nm=0.0,
            payload_required_nm=0.0,
            mountain_required_nm=0.0,
            dispatch_margin_nm=0.0,
            total_required_nm=0.0,
            reserves_satisfied=True,
        )

    nbaa = NBAA_IFR_RESERVE_NM if mission.nbaa_reserves else NBAA_IFR_RESERVE_NM * 0.85

    westbound_nm = 0.0
    if mission.westbound_sensitive:
        factor = (
            0.10
            if mission.winter_westbound_transpacific
            else WESTBOUND_REQUIRED_FACTOR
        )
        westbound_nm = stage * factor

    payload_req = 0.0
    if mission.passengers >= 10:
        payload_req += MISSION_PAX_REQUIRED_10_PLUS
    elif mission.passengers >= 8:
        payload_req += MISSION_PAX_REQUIRED_8_PLUS
    if mission.baggage_high:
        payload_req += MISSION_BAGGAGE_REQUIRED

    mountain_req = MISSION_MOUNTAIN_REQUIRED if mission.mountain_airports else 0.0

    base = stage + nbaa + westbound_nm + payload_req + mountain_req

    dispatch_extra = 0.0
    if mission.nonstop_required:
        margin_factor = nonstop_margin_factor(stage)
        total = base * margin_factor
        dispatch_extra = total - base
    else:
        total = base

    return MissionRangeRequirement(
        stage_distance_nm=stage,
        nbaa_reserve_nm=nbaa,
        westbound_required_nm=westbound_nm,
        payload_required_nm=payload_req,
        mountain_required_nm=mountain_req,
        dispatch_margin_nm=dispatch_extra,
        total_required_nm=total,
        reserves_satisfied=mission.nbaa_reserves and nbaa >= NBAA_IFR_RESERVE_NM * 0.99,
    )

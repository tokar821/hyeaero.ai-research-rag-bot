"""
Mission operational assessment — integrates payload, reserves, and dispatch for P1 depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from services.broker.broker_verdicts import BrokerVerdict
from services.consultant.mission_state import MissionState
from services.mission.models import MissionProfile
from services.mission.route_distance_authority import (
    RouteDistanceResolution,
    peak_catalog_stage_nm,
    peak_verified_stage_nm,
    resolve_mission_route_authority,
)
from services.operational.dispatch_reliability import (
    AircraftDispatchAssessment,
    MissionDispatchFactors,
    assess_aircraft_dispatch,
    assess_mission_dispatch_factors,
)
from services.operational.payload_realism import (
    MissionPayloadProfile,
    build_mission_payload_profile,
    effective_practical_nm,
)
from services.operational.reserve_profiles import (
    ReserveBreakdown,
    compute_reserve_breakdown,
    infer_planning_mode,
)
from services.operational.wind_realism import WindAdjustment, compute_wind_adjustment


@dataclass
class MissionOperationalContext:
    payload: MissionPayloadProfile
    reserve: ReserveBreakdown
    dispatch_factors: MissionDispatchFactors
    route_resolutions: List[RouteDistanceResolution] = field(default_factory=list)
    peak_stage_nm: float = 0.0
    catalog_peak_nm: float = 0.0
    planning_mode: str = "standard_nbaa"
    corridor_id: Optional[str] = None
    wind: Optional[WindAdjustment] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "payload": self.payload.to_dict(),
            "reserve": self.reserve.to_dict(),
            "wind": self.wind.to_dict() if self.wind else None,
            "dispatch_factors": self.dispatch_factors.to_dict(),
            "route_resolutions": [r.to_dict() for r in self.route_resolutions],
            "peak_stage_nm": round(self.peak_stage_nm, 1),
            "catalog_peak_nm": round(self.catalog_peak_nm, 1),
            "planning_mode": self.planning_mode,
            "corridor_id": self.corridor_id,
        }


@dataclass
class AircraftOperationalAssessment:
    model: str
    effective_practical_nm: float
    required_nm: float
    margin_nm: float
    dispatch: AircraftDispatchAssessment
    recommended_verdict_cap: Optional[str] = None
    operational_caveats: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "effective_practical_nm": round(self.effective_practical_nm, 1),
            "required_nm": round(self.required_nm, 1),
            "margin_nm": round(self.margin_nm, 1),
            "dispatch": self.dispatch.to_dict(),
            "recommended_verdict_cap": self.recommended_verdict_cap,
            "operational_caveats": list(self.operational_caveats),
        }


def build_mission_operational_context(
    mission: MissionState,
    profile: MissionProfile,
    *,
    query: str = "",
    route_resolutions: Optional[Sequence[RouteDistanceResolution]] = None,
) -> MissionOperationalContext:
    """Build mission-level P1 context once per ranking pass."""
    labels = list(mission.routes or []) or profile.route_labels()
    resolutions = list(route_resolutions or resolve_mission_route_authority(labels))
    peak = peak_verified_stage_nm(resolutions)
    catalog_peak = peak_catalog_stage_nm(resolutions)

    payload = build_mission_payload_profile(
        mission,
        profile=profile,
        query=query,
        stage_distance_nm=peak or catalog_peak,
    )

    route_label = labels[0] if labels else ""
    wind = compute_wind_adjustment(
        mission,
        profile=profile,
        stage_distance_nm=peak or catalog_peak,
        route_label=route_label,
    )
    west_pen = wind.total_penalty_nm
    if not west_pen and mission.westbound and peak > 0:
        west_pen = peak * 0.08

    geo_extra = sum(r.extra_reserve_nm for r in resolutions if r.is_verified)
    intl = any(r.international_leg for r in resolutions)

    planning = infer_planning_mode(profile, query=query)
    reserve = compute_reserve_breakdown(
        stage_distance_nm=peak,
        payload=payload,
        westbound_penalty_nm=west_pen,
        geodesic_extra_nm=geo_extra,
        planning_mode=planning,
        profile=profile,
        international_leg=intl,
    )

    factors = assess_mission_dispatch_factors(mission, profile, route_resolutions=resolutions)

    corridor_id = None
    try:
        from services.aircraft_feasibility.route_realism_validator import match_ultra_long_corridor

        if labels:
            corridor_id = match_ultra_long_corridor(labels[0], peak)
    except Exception:
        pass

    return MissionOperationalContext(
        payload=payload,
        reserve=reserve,
        dispatch_factors=factors,
        route_resolutions=resolutions,
        peak_stage_nm=peak,
        catalog_peak_nm=catalog_peak,
        planning_mode=planning.value,
        corridor_id=corridor_id,
        wind=wind,
    )


def assess_aircraft_operational(
    model: str,
    aircraft_spec: Dict[str, Any],
    ctx: MissionOperationalContext,
) -> AircraftOperationalAssessment:
    """Per-aircraft P1 assessment — margin and dispatch reliability."""
    practical_nm = float(aircraft_spec.get("practical_nm") or 0)
    eff = effective_practical_nm(practical_nm, ctx.payload)
    required = ctx.reserve.total_required_nm
    margin = eff - required

    dispatch = assess_aircraft_dispatch(
        model,
        aircraft_spec,
        margin_nm=margin,
        reserve=ctx.reserve,
        payload=ctx.payload,
        factors=ctx.dispatch_factors,
    )

    caveats: List[str] = list(dispatch.dispatch_notes)
    if ctx.wind and ctx.wind.notes:
        caveats.extend(ctx.wind.notes[:2])
    verdict_cap: Optional[str] = None

    if not dispatch.technically_possible:
        verdict_cap = BrokerVerdict.NOT_OPERATIONALLY_CREDIBLE.value
        caveats.append("This mission exceeds realistic payload-range margins.")
    elif not dispatch.works_reliably:
        verdict_cap = BrokerVerdict.MISSION_RISKY.value
        if dispatch.tech_stop_probability >= 0.25:
            caveats.append(
                f"Tech-stop likelihood elevated (~{int(dispatch.tech_stop_probability * 100)}%) "
                "under stated seasonal/payload assumptions."
            )
    elif margin < 200:
        verdict_cap = BrokerVerdict.VIABLE_WITH_COMPROMISES.value
        caveats.append("Margin-tight once payload, reserves, and seasonal pressure are applied.")

    if ctx.catalog_peak_nm <= 0 and ctx.dispatch_factors.nonstop_required:
        verdict_cap = BrokerVerdict.VIABLE_WITH_COMPROMISES.value
        caveats.append("Nonstop feasibility not catalog-authorized — corridor classified only.")

    return AircraftOperationalAssessment(
        model=model,
        effective_practical_nm=eff,
        required_nm=required,
        margin_nm=margin,
        dispatch=dispatch,
        recommended_verdict_cap=verdict_cap,
        operational_caveats=caveats,
    )


def apply_verdict_cap(current_verdict: str, cap: Optional[str]) -> str:
    """Never upgrade verdict above operational cap."""
    if not cap:
        return current_verdict
    order = {
        BrokerVerdict.NOT_OPERATIONALLY_CREDIBLE.value: 0,
        BrokerVerdict.MISSION_RISKY.value: 1,
        BrokerVerdict.VIABLE_WITH_COMPROMISES.value: 2,
        BrokerVerdict.PRIMARY_RECOMMENDATION.value: 3,
    }
    cur = order.get(current_verdict, 2)
    lim = order.get(cap, 2)
    return current_verdict if cur <= lim else cap

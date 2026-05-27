"""
Route feasibility evaluation — practical vs brochure range with NBAA-style margins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.consultant.mission_state import MissionState


@dataclass
class RouteFeasibilityAssessment:
    route_label: str
    distance_nm: float
    brochure_capable: bool
    practical_with_restrictions: bool
    reliably_nonstop: bool
    confidence: float
    westbound_penalty_nm: float = 0.0
    reserve_nm: float = 200.0
    payload_penalty_note: str = ""
    seasonal_note: str = ""
    runway_note: str = ""
    classification: str = "unknown"  # brochure_capable | practical_restricted | reliably_nonstop | not_feasible
    caveats: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_label": self.route_label,
            "distance_nm": self.distance_nm,
            "brochure_capable": self.brochure_capable,
            "practical_with_restrictions": self.practical_with_restrictions,
            "reliably_nonstop": self.reliably_nonstop,
            "westbound_penalty_nm": self.westbound_penalty_nm,
            "reserve_nm": self.reserve_nm,
            "payload_penalty_note": self.payload_penalty_note,
            "seasonal_note": self.seasonal_note,
            "runway_note": self.runway_note,
            "classification": self.classification,
            "caveats": list(self.caveats),
        }


# Verified catalog — shared with route_distance_authority (no heuristic defaults).
from services.mission.route_distance_catalog import VERIFIED_ROUTE_DISTANCE_NM

_ROUTE_DISTANCE_NM: Dict[str, float] = VERIFIED_ROUTE_DISTANCE_NM

_NBAA_RESERVE_NM = 200.0
_WINTER_WESTBOUND_EXTRA = 0.12
_WESTBOUND_EXTRA = 0.08
_MOUNTAIN_PAYLOAD_PENALTY_NM = 350.0


def _normalize_route_label(label: str) -> str:
    s = (label or "").strip().lower()
    s = s.replace("→", "->")
    return re.sub(r"\s+", " ", s)


def estimate_route_distance_nm(route_label: str) -> float:
    """Return verified stage length (nm), or 0.0 when unresolved — never invent distance."""
    from services.mission.route_distance_authority import resolve_route_distance

    resolution = resolve_route_distance(route_label)
    return resolution.distance_nm if resolution.is_verified else 0.0


def resolve_route_distance_detail(route_label: str):
    """Full authority resolution including confidence and source."""
    from services.mission.route_distance_authority import resolve_route_distance

    return resolve_route_distance(route_label)


def _is_westbound_route(route_label: str, mission: MissionState) -> bool:
    if mission.westbound:
        return True
    rl = _normalize_route_label(route_label)
    return bool(
        re.search(r"west\s+coast.*europe|sfo.*paris|los\s+angeles.*london|to\s+europe\b", rl)
        or (re.search(r"→\s*europe", rl) and re.search(r"coast|francisco|angeles", rl))
    )


def assess_route_for_aircraft(
    *,
    route_label: str,
    aircraft_practical_nm: float,
    aircraft_brochure_nm: float,
    mission: MissionState,
    passenger_count: int = 6,
) -> RouteFeasibilityAssessment:
    """
    Evaluate one route against aircraft range capability (nm).
    """
    resolution = resolve_route_distance_detail(route_label)
    dist = resolution.distance_nm
    route_confidence = resolution.confidence if resolution.is_verified else 0.35
    reserve = _NBAA_RESERVE_NM
    if (mission.reserves_requirement or "").lower().find("nbaa") >= 0:
        reserve = _NBAA_RESERVE_NM
    reserve += float(getattr(resolution, "extra_reserve_nm", 0) or 0)

    west_pen = 0.0
    seasonal_note = ""
    try:
        from services.operational.wind_realism import compute_wind_adjustment

        wind = compute_wind_adjustment(
            mission,
            stage_distance_nm=dist,
            route_label=route_label,
        )
        west_pen = wind.total_penalty_nm
        if wind.notes:
            seasonal_note = wind.notes[0]
    except Exception:
        if _is_westbound_route(route_label, mission):
            west_pen = dist * _WESTBOUND_EXTRA
            if (mission.seasonal_constraints or "").lower().find("winter") >= 0 or re.search(
                r"winter", route_label, re.I
            ):
                west_pen += dist * (_WINTER_WESTBOUND_EXTRA - _WESTBOUND_EXTRA)
                seasonal_note = (
                    "Winter westbound headwinds increase fuel burn — treat brochure range as optimistic."
                )

    payload_pen = 0.0
    payload_note = ""
    if passenger_count >= 8:
        payload_pen = 150.0
        payload_note = "Higher passenger count reduces effective range vs brochure payloads."
    if mission.mountain_airport_requirement:
        payload_pen += _MOUNTAIN_PAYLOAD_PENALTY_NM
        payload_note = (payload_note + " Hot/high or mountain departures erode payload-range.").strip()

    required_nm = dist + reserve + west_pen + payload_pen
    brochure_ok = aircraft_brochure_nm >= dist + reserve * 0.5
    practical_ok = aircraft_practical_nm >= required_nm
    reliable_ok = aircraft_practical_nm >= required_nm * 1.08

    caveats: List[str] = []
    if west_pen > 0:
        caveats.append(f"Westbound margin applied (~{int(west_pen)} nm equivalent).")
    if payload_pen > 0:
        caveats.append(payload_note or "Payload/routing restrictions likely.")
    if not resolution.authorize_nonstop_feasibility:
        reliable_ok = False
        if resolution.corridor_classification_only:
            caveats.append(
                "Stage length is geodesic corridor-classified only — "
                "not authorized as verified nonstop feasibility."
            )

    if mission.nonstop_requirement and not reliable_ok:
        caveats.append("Nonstop requirement not met with comfortable NBAA-style margin.")

    if reliable_ok:
        classification = "reliably_nonstop"
    elif practical_ok:
        classification = "practical_restricted"
    elif brochure_ok:
        classification = "brochure_capable"
    else:
        classification = "not_feasible"

    runway_note = ""
    if mission.runway_constraints:
        runway_note = f"Runway constraint noted: {mission.runway_constraints}."

    return RouteFeasibilityAssessment(
        route_label=route_label,
        distance_nm=dist,
        brochure_capable=brochure_ok,
        practical_with_restrictions=practical_ok and not reliable_ok,
        reliably_nonstop=reliable_ok,
        confidence=min(
            route_confidence,
            0.82 if reliable_ok else (0.7 if practical_ok else 0.55),
        ),
        westbound_penalty_nm=west_pen,
        reserve_nm=reserve,
        payload_penalty_note=payload_note,
        seasonal_note=seasonal_note,
        runway_note=runway_note,
        classification=classification,
        caveats=caveats,
    )


def assess_mission_routes(
    mission: MissionState,
    *,
    aircraft_practical_nm: float,
    aircraft_brochure_nm: float,
    passenger_count: Optional[int] = None,
) -> List[RouteFeasibilityAssessment]:
    pax = passenger_count or mission.passenger_count or 6
    routes = mission.routes or []
    if not routes:
        return []
    return [
        assess_route_for_aircraft(
            route_label=r,
            aircraft_practical_nm=aircraft_practical_nm,
            aircraft_brochure_nm=aircraft_brochure_nm,
            mission=mission,
            passenger_count=int(pax),
        )
        for r in routes
    ]


def pick_worst_route_classification(
    assessments: List[RouteFeasibilityAssessment],
) -> str:
    order = ("not_feasible", "brochure_capable", "practical_restricted", "reliably_nonstop", "unknown")
    if not assessments:
        return "unknown"
    ranks = {c: order.index(c) if c in order else 0 for c in order}
    worst = min(assessments, key=lambda a: ranks.get(a.classification, 0))
    return worst.classification

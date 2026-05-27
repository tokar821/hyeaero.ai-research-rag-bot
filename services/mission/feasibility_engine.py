"""
Hard-feasibility mission evaluation — eliminate impossible aircraft before scoring.

Never uses brochure range for go/no-go; ``compute_practical_range`` applies operational reductions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

from services.mission.route_distance_authority import resolve_route_distance


def estimate_route_distance_nm(route_label: str) -> float:
    r = resolve_route_distance(route_label)
    return r.distance_nm if r.is_verified else 0.0
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.mission.models import MissionProfile, PriorityLevel, Route

_NBAA_RESERVE_NM = 200.0
_NONSTOP_MARGIN_FACTOR = 1.08
_WESTBOUND_FACTOR = 0.08
_WINTER_WESTBOUND_EXTRA = 0.12
_PAX_NM_PENALTY = 35.0
_BAGGAGE_NM_PENALTY = 80.0
_MOUNTAIN_AVAILABLE_PENALTY = 280.0

# Mission-required runway (ft) by priority / environment
_RUNWAY_NEED_DEFAULT = 4500
_RUNWAY_NEED_SHORT_FIELD = 4000
_RUNWAY_NEED_MOUNTAIN = 5000


@dataclass
class FeasibilityResult:
    feasible: bool
    elimination_reasons: List[str] = field(default_factory=list)
    practical_range_nm: float = 0.0
    mission_margin_nm: float = 0.0
    operational_risk_level: str = "low"  # low | medium | high | eliminated
    notes: List[str] = field(default_factory=list)
    required_route_nm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible": self.feasible,
            "elimination_reasons": list(self.elimination_reasons),
            "practical_range_nm": round(self.practical_range_nm, 1),
            "mission_margin_nm": round(self.mission_margin_nm, 1),
            "operational_risk_level": self.operational_risk_level,
            "notes": list(self.notes),
            "required_route_nm": round(self.required_route_nm, 1),
        }


def _priority_high(p: PriorityLevel) -> bool:
    return p in (PriorityLevel.HIGH, PriorityLevel.MEDIUM)


def _route_distance_nm(route: Route) -> float:
    return estimate_route_distance_nm(route.label())


def _is_westbound_profile(profile: MissionProfile) -> bool:
    if profile.westbound_sensitive:
        return True
    blob = " ".join(r.label() for r in profile.routes).lower()
    return bool(
        re.search(r"westbound|west\s+coast.*(?:europe|tokyo|paris)", blob)
        or (re.search(r"tokyo|europe|london|paris", blob) and re.search(r"francisco|angeles|seattle", blob))
    )


def _is_winter_westbound_transpacific(profile: MissionProfile) -> bool:
    if not _is_westbound_profile(profile):
        return False
    winter = (profile.seasonal_note or "").lower().find("winter") >= 0
    blob = " ".join(r.label() for r in profile.routes).lower()
    transpacific = bool(re.search(r"tokyo|seoul|beijing|hong\s+kong|asia", blob))
    return winter or transpacific


def required_route_nm_with_margin(
    profile: MissionProfile,
    *,
    route: Optional[Route] = None,
) -> float:
    """
    Stage length required for feasibility (nm) — distance + reserves + penalties + nonstop margin.
    """
    if route is not None:
        dist = _route_distance_nm(route)
    else:
        routes = profile.routes or []
        dist = max((_route_distance_nm(r) for r in routes), default=0.0)

    if dist <= 0:
        return 0.0

    reserve = _NBAA_RESERVE_NM if (
        profile.nbaa_reserve_required
        or (profile.reserves_requirement or "").lower().find("nbaa") >= 0
    ) else _NBAA_RESERVE_NM * 0.85

    pax = profile.passengers or 6
    payload_pen = 0.0
    if pax >= 10:
        payload_pen += 200.0
    elif pax >= 8:
        payload_pen += 120.0
    if _priority_high(profile.baggage_priority):
        payload_pen += 90.0

    west_pen = 0.0
    if _is_westbound_profile(profile):
        if _is_winter_westbound_transpacific(profile):
            west_pen = dist * 0.10
        else:
            west_pen = dist * _WESTBOUND_FACTOR

    if profile.mountain_airport_priority or profile.mountain_airports:
        payload_pen += 320.0

    required = dist + reserve + west_pen + payload_pen
    if profile.nonstop_required:
        margin_factor = 1.03 if dist >= 4500 else _NONSTOP_MARGIN_FACTOR
        required *= margin_factor
    return required


def peak_required_route_nm(profile: MissionProfile) -> float:
    """Worst-case required NM across all mission legs."""
    routes = profile.routes or []
    if not routes:
        return 0.0
    return max(required_route_nm_with_margin(profile, route=r) for r in routes)


def compute_practical_range(
    aircraft: Mapping[str, Any],
    *,
    passengers: int = 6,
    baggage_weight: str = "normal",
    westbound: bool = False,
    nbaa_reserves: bool = True,
    winter_westbound: bool = False,
    mountain: bool = False,
) -> float:
    """
    Operational range available (nm) — baseline practical_nm minus penalties (not brochure).
    """
    baseline = float(aircraft.get("practical_nm") or 0)
    if baseline <= 0:
        return 0.0

    typical = int(aircraft.get("pax_typical") or 6)
    available = baseline

    if passengers > typical:
        available -= min(500.0, (passengers - typical) * _PAX_NM_PENALTY)

    if baggage_weight == "high":
        available -= _BAGGAGE_NM_PENALTY

    # NBAA reserve is applied on the mission-required side, not double-subtracted here.

    cat = str(aircraft.get("category") or "")
    if westbound:
        available -= baseline * (0.03 if cat == "ultra-long" else 0.06)
    if winter_westbound:
        available -= baseline * (0.04 if cat == "ultra-long" else 0.08)

    if mountain:
        available -= _MOUNTAIN_AVAILABLE_PENALTY

    cat = str(aircraft.get("category") or "")
    if cat in ("ultra-long", "large") and mountain:
        available -= 150.0

    return max(available, 0.0)


def _required_runway_ft(profile: MissionProfile) -> float:
    if profile.mountain_airport_priority or profile.mountain_airports:
        return float(_RUNWAY_NEED_MOUNTAIN)
    if _priority_high(profile.short_field_priority) or _priority_high(profile.runway_priority):
        return float(_RUNWAY_NEED_SHORT_FIELD)
    return float(_RUNWAY_NEED_DEFAULT)


def evaluate_mission_feasibility(
    mission_profile: MissionProfile,
    aircraft: Union[str, Mapping[str, Any]],
    *,
    override_experimental: bool = False,
) -> FeasibilityResult:
    """
    Hard elimination evaluation for one aircraft against a mission profile.

    Delegates to :mod:`services.aircraft_feasibility` for conservative pre-LLM gating.
    """
    from services.aircraft_feasibility.engine import evaluate_aircraft_feasibility
    from services.aircraft_feasibility.mission_context import mission_context_from_profile

    if override_experimental:
        pass  # legacy bypass handled below after engine verdict

    ctx = mission_context_from_profile(mission_profile)
    verdict = evaluate_aircraft_feasibility(ctx, aircraft)

    risk = "low"
    if not verdict.feasible:
        risk = "eliminated"
    elif verdict.required_nm > 0 and verdict.margin_nm < verdict.required_nm * 0.12:
        risk = "high"
    elif verdict.required_nm > 0 and verdict.margin_nm < verdict.required_nm * 0.25:
        risk = "medium"

    notes: List[str] = []
    if verdict.required_nm > 0:
        notes.append(
            f"Required ~{int(verdict.required_nm)} nm vs practical available ~{int(verdict.available_nm)} nm."
        )

    feasible = verdict.feasible
    reasons = list(verdict.rejection_reasons)
    if override_experimental and not feasible:
        notes.append("override_experimental=True — feasibility bypass for scoring only.")
        feasible = True
        reasons = []

    return FeasibilityResult(
        feasible=feasible,
        elimination_reasons=reasons,
        practical_range_nm=verdict.available_nm,
        mission_margin_nm=verdict.margin_nm,
        operational_risk_level=risk,
        notes=notes,
        required_route_nm=verdict.required_nm,
    )


def filter_feasible_aircraft(
    mission_profile: MissionProfile,
    models: Optional[List[str]] = None,
    *,
    override_experimental: bool = False,
) -> Dict[str, FeasibilityResult]:
    """Evaluate all candidate models; returns map of model -> FeasibilityResult."""
    candidates = models or list(AIRCRAFT_PROFILES.keys())
    out: Dict[str, FeasibilityResult] = {}
    for model in candidates:
        if model not in AIRCRAFT_PROFILES:
            out[model] = FeasibilityResult(
                feasible=False,
                elimination_reasons=[f"Unknown aircraft: {model}"],
                operational_risk_level="eliminated",
            )
            continue
        out[model] = evaluate_mission_feasibility(
            mission_profile,
            model,
            override_experimental=override_experimental,
        )
    return out


def feasible_models(
    mission_profile: MissionProfile,
    models: Optional[List[str]] = None,
    *,
    override_experimental: bool = False,
) -> List[str]:
    """Models that pass hard feasibility (eliminated aircraft excluded)."""
    results = filter_feasible_aircraft(
        mission_profile, models, override_experimental=override_experimental
    )
    return [m for m, r in results.items() if r.feasible]

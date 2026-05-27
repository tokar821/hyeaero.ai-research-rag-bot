"""
Constraint-based aircraft capability graph.

Deterministic hard filtering before compatibility scoring — not opinion-based ranking.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from services.consultant.route_feasibility import estimate_route_distance_nm
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.mission.feasibility_engine import peak_required_route_nm
from services.mission.models import MissionProfile, PriorityLevel, Route

logger = logging.getLogger(__name__)

_NBAA_RESERVE_NM = 200.0
_WESTBOUND_HEADWIND_THRESHOLD = 0.92
_MISSION_CATEGORY_FIT_MIN = 0.35
_AIRPORT_RUNWAY_LIMIT_FT: Dict[str, float] = {
    "regional": 4500.0,
    "mountain": 5000.0,
    "domestic": 5500.0,
    "international": 6500.0,
}

_SCORE_WEIGHTS = {
    "range_fit": 0.28,
    "payload_fit": 0.18,
    "runway_fit": 0.14,
    "cost_efficiency": 0.18,
    "cabin_fit": 0.12,
    "category_fit": 0.10,
}


class MissionClass(str, Enum):
    DOMESTIC = "domestic"
    TRANSATLANTIC = "transatlantic"
    TRANSPACIFIC = "transpacific"


class AirportType(str, Enum):
    REGIONAL = "regional"
    MOUNTAIN = "mountain"
    DOMESTIC = "domestic"
    INTERNATIONAL = "international"


@dataclass(frozen=True)
class AircraftNode:
    """Capability node — operational constraints, not brochure marketing."""

    model: str
    range_nbaa_reserves: float
    payload_range_curve: Dict[int, float]
    runway_class_support: str
    runway_ft_required: float
    airport_compatibility: Tuple[str, ...]
    westbound_margin_factor: float
    mission_category_fit: Dict[str, float]
    cost_index: float
    cabin_score: float = 0.7
    category: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "range_nbaa_reserves": round(self.range_nbaa_reserves, 1),
            "payload_range_curve": dict(self.payload_range_curve),
            "runway_class_support": self.runway_class_support,
            "runway_ft_required": round(self.runway_ft_required, 0),
            "airport_compatibility": list(self.airport_compatibility),
            "westbound_margin_factor": round(self.westbound_margin_factor, 3),
            "mission_category_fit": dict(self.mission_category_fit),
            "cost_index": round(self.cost_index, 3),
            "cabin_score": round(self.cabin_score, 3),
            "category": self.category,
        }


@dataclass(frozen=True)
class MissionNode:
    """Mission constraint node derived from a turn profile."""

    route_distance_nm: float
    passenger_count: int
    airport_type: str
    westbound_flag: bool
    mission_class: str
    required_route_nm: float
    nonstop_required: bool = False
    mountain_flag: bool = False
    high_payload_flag: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_distance_nm": round(self.route_distance_nm, 1),
            "passenger_count": self.passenger_count,
            "airport_type": self.airport_type,
            "westbound_flag": self.westbound_flag,
            "mission_class": self.mission_class,
            "required_route_nm": round(self.required_route_nm, 1),
            "nonstop_required": self.nonstop_required,
            "mountain_flag": self.mountain_flag,
            "high_payload_flag": self.high_payload_flag,
        }


@dataclass
class ExcludedAircraft:
    model: str
    failed_constraint_reason: str
    pass_fail: str = "fail"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aircraft": self.model,
            "pass_fail": self.pass_fail,
            "failed_constraint_reason": self.failed_constraint_reason,
        }


@dataclass
class RankedAircraft:
    model: str
    total_score: float
    range_fit: float
    payload_fit: float
    runway_fit: float
    cost_efficiency: float
    cabin_fit: float
    category_fit: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "total_score": round(self.total_score, 4),
            "range_fit": round(self.range_fit, 3),
            "payload_fit": round(self.payload_fit, 3),
            "runway_fit": round(self.runway_fit, 3),
            "cost_efficiency": round(self.cost_efficiency, 3),
            "cabin_fit": round(self.cabin_fit, 3),
            "category_fit": round(self.category_fit, 3),
        }


@dataclass
class CapabilityGraphResult:
    mission: MissionNode
    feasible_aircraft_list: List[str]
    excluded_aircraft_list: List[ExcludedAircraft]
    ranked_results: List[RankedAircraft]
    filter_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission": self.mission.to_dict(),
            "feasible_aircraft_list": list(self.feasible_aircraft_list),
            "excluded_aircraft_list": [e.to_dict() for e in self.excluded_aircraft_list],
            "ranked_results": [r.to_dict() for r in self.ranked_results],
            "filter_log": list(self.filter_log),
        }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _priority_high(p: PriorityLevel) -> bool:
    return p in (PriorityLevel.HIGH, PriorityLevel.MEDIUM)


def _infer_mission_class(profile: MissionProfile, route_distance_nm: float) -> str:
    blob = " ".join(r.label() for r in profile.routes).lower()
    if re.search(r"\btransatlantic\b", blob):
        return MissionClass.TRANSATLANTIC.value
    if re.search(r"\btranspacific\b", blob):
        return MissionClass.TRANSPACIFIC.value
    # Short Asia hops (e.g. Tokyo–Seoul) are regional — not transpacific ULR missions.
    if route_distance_nm >= 4200:
        return MissionClass.TRANSPACIFIC.value
    if route_distance_nm >= 3000 and re.search(
        r"(?:san\s+francisco|los\s+angeles|seattle|new\s+york|nyc|west\s+coast).*(?:tokyo|seoul|beijing|hong\s+kong|singapore)"
        r"|(?:tokyo|seoul|beijing|hong\s+kong|singapore).*(?:san\s+francisco|los\s+angeles|seattle|new\s+york|nyc)",
        blob,
    ):
        return MissionClass.TRANSPACIFIC.value
    if route_distance_nm >= 2600 or re.search(r"london|paris|geneva|europe", blob):
        return MissionClass.TRANSATLANTIC.value
    return MissionClass.DOMESTIC.value


def _infer_airport_type(profile: MissionProfile) -> str:
    if profile.mountain_airport_priority or profile.mountain_airports:
        return AirportType.MOUNTAIN.value
    # REGIONAL airport type is for true short/rough field ops — not executive runway-flex preference.
    if _priority_high(profile.short_field_priority) and (
        profile.mountain_airport_priority or profile.mountain_airports
    ):
        return AirportType.REGIONAL.value
    blob = " ".join(r.label() for r in profile.routes).lower()
    if re.search(r"london|paris|geneva|europe|tokyo|asia", blob):
        return AirportType.INTERNATIONAL.value
    return AirportType.DOMESTIC.value


def _category_fit_map(category: str, practical_nm: float) -> Dict[str, float]:
    """Deterministic mission-class fit — super-midsize cannot score high on transpacific."""
    if category == "ultra-long":
        return {
            MissionClass.DOMESTIC.value: 0.75,
            MissionClass.TRANSATLANTIC.value: 0.95,
            MissionClass.TRANSPACIFIC.value: 0.98,
        }
    if category == "large":
        return {
            MissionClass.DOMESTIC.value: 0.85,
            MissionClass.TRANSATLANTIC.value: 0.88,
            MissionClass.TRANSPACIFIC.value: 0.72,
        }
    if category == "super-midsize":
        transpacific = 0.12 if practical_nm < 4000 else 0.28
        return {
            MissionClass.DOMESTIC.value: 0.9,
            MissionClass.TRANSATLANTIC.value: 0.52,
            MissionClass.TRANSPACIFIC.value: transpacific,
        }
    if category == "light":
        return {
            MissionClass.DOMESTIC.value: 0.92,
            MissionClass.TRANSATLANTIC.value: 0.25,
            MissionClass.TRANSPACIFIC.value: 0.08,
        }
    if category == "turboprop":
        return {
            MissionClass.DOMESTIC.value: 0.88,
            MissionClass.TRANSATLANTIC.value: 0.1,
            MissionClass.TRANSPACIFIC.value: 0.05,
        }
    return {
        MissionClass.DOMESTIC.value: 0.7,
        MissionClass.TRANSATLANTIC.value: 0.45,
        MissionClass.TRANSPACIFIC.value: 0.2,
    }


def _runway_class(category: str, runway_ft: float) -> str:
    if runway_ft <= 3200:
        return "A"
    if runway_ft <= 4200:
        return "B"
    if runway_ft <= 5000:
        return "C"
    return "D"


def _payload_curve(typical: int, practical_nm: float, pax_max: int) -> Dict[int, float]:
    curve: Dict[int, float] = {}
    for pax in range(1, pax_max + 3):
        if pax <= typical:
            curve[pax] = practical_nm
        else:
            curve[pax] = max(400.0, practical_nm - (pax - typical) * 45.0)
    return curve


def build_aircraft_node(model: str, spec: Optional[Mapping[str, Any]] = None) -> AircraftNode:
    raw = dict(spec or AIRCRAFT_PROFILES.get(model) or {})
    if not raw:
        raise ValueError(f"Unknown aircraft: {model}")

    practical = float(raw.get("practical_nm") or 0)
    typical = int(raw.get("pax_typical") or 6)
    pax_max = int(raw.get("pax_max_long_range") or typical)
    runway_ft = float(raw.get("runway_ft") or 4500)
    category = str(raw.get("category") or "super-midsize")
    reserve_factor = 0.22 if category == "ultra-long" else 0.4
    range_nbaa = max(400.0, practical - _NBAA_RESERVE_NM * reserve_factor)

    compat: List[str] = [AirportType.DOMESTIC.value, AirportType.INTERNATIONAL.value]
    if float(raw.get("short_field_score") or 0) >= 0.7:
        compat.append(AirportType.REGIONAL.value)
    if float(raw.get("hot_high_score") or 0) >= 0.65:
        compat.append(AirportType.MOUNTAIN.value)

    wb_factor = 0.93 if category == "ultra-long" else (0.84 if category == "large" else 0.76)

    return AircraftNode(
        model=model,
        range_nbaa_reserves=range_nbaa,
        payload_range_curve=_payload_curve(typical, range_nbaa, pax_max),
        runway_class_support=_runway_class(category, runway_ft),
        runway_ft_required=runway_ft,
        airport_compatibility=tuple(dict.fromkeys(compat)),
        westbound_margin_factor=wb_factor,
        mission_category_fit=_category_fit_map(category, practical),
        cost_index=float(raw.get("operating_index") or 0.6),
        cabin_score=float(raw.get("cabin_score") or 0.7),
        category=category,
    )


def build_mission_node(profile: MissionProfile) -> MissionNode:
    from services.mission.models import MissionCategory as ProfileMissionCategory

    routes = profile.routes or []
    stage_lengths = [estimate_route_distance_nm(r.label()) for r in routes] if routes else [0.0]
    route_distance_nm = max(stage_lengths) if stage_lengths else 0.0
    pax = profile.passengers or 6
    required = peak_required_route_nm(profile) if routes else 0.0

    mission_class = _infer_mission_class(profile, route_distance_nm)
    airport_type = _infer_airport_type(profile)
    if route_distance_nm <= 0 and profile.mission_category == ProfileMissionCategory.COMPARISON:
        mission_class = MissionClass.TRANSATLANTIC.value
        airport_type = AirportType.INTERNATIONAL.value

    return MissionNode(
        route_distance_nm=route_distance_nm,
        passenger_count=pax,
        airport_type=airport_type,
        westbound_flag=bool(profile.westbound_sensitive),
        mission_class=mission_class,
        required_route_nm=required,
        nonstop_required=bool(profile.nonstop_required),
        mountain_flag=bool(profile.mountain_airport_priority or profile.mountain_airports),
        high_payload_flag=pax >= 10 or _priority_high(profile.baggage_priority),
    )


def _effective_range_nm(aircraft: AircraftNode, mission: MissionNode) -> float:
    curve = aircraft.payload_range_curve
    pax = mission.passenger_count
    if pax in curve:
        base = curve[pax]
    else:
        keys = sorted(curve.keys())
        base = curve[keys[-1]] if pax > keys[-1] else curve[keys[0]]
    if mission.westbound_flag:
        base *= aircraft.westbound_margin_factor
    if mission.mountain_flag:
        base *= 0.88
    return base


def _log_graph_filter(
    aircraft: str,
    passed: bool,
    reason: str = "",
) -> Dict[str, Any]:
    entry = {
        "aircraft": aircraft,
        "pass_fail": "pass" if passed else "fail",
        "failed_constraint_reason": reason if not passed else "",
    }
    logger.info(
        "AIRCRAFT_GRAPH_FILTER: aircraft=%s pass_fail=%s failed_constraint_reason=%s",
        aircraft,
        entry["pass_fail"],
        entry["failed_constraint_reason"] or "none",
    )
    return entry


def _check_hard_constraints(
    mission: MissionNode,
    aircraft: AircraftNode,
) -> Tuple[bool, str]:
    """Return (pass, failed_constraint_reason)."""
    effective = _effective_range_nm(aircraft, mission)
    required = mission.required_route_nm or mission.route_distance_nm + _NBAA_RESERVE_NM

    # Range + NBAA
    if required > 0 and effective < required:
        return False, (
            f"range_nbaa_reserves: effective ~{int(effective)} nm < required ~{int(required)} nm "
            f"(NBAA reserves applied; not brochure range)."
        )

    # Mission category hard gate (Challenger 350 transpacific / long-haul)
    cat_fit = aircraft.mission_category_fit.get(mission.mission_class, 0.5)
    if cat_fit < _MISSION_CATEGORY_FIT_MIN:
        return False, (
            f"mission_category_fit: {aircraft.model} fit {cat_fit:.2f} < {_MISSION_CATEGORY_FIT_MIN} "
            f"for {mission.mission_class} missions."
        )

    # Runway vs airport class (skip when no route/airport is stated — e.g. cabin-only comparisons)
    if mission.route_distance_nm > 0 or mission.required_route_nm > 0:
        airport_limit = _AIRPORT_RUNWAY_LIMIT_FT.get(mission.airport_type, 5500.0)
        if aircraft.runway_ft_required > airport_limit:
            return False, (
                f"runway_class: aircraft needs ~{int(aircraft.runway_ft_required)} ft > "
                f"mission airport limit ~{int(airport_limit)} ft ({mission.airport_type})."
            )

    if mission.airport_type not in aircraft.airport_compatibility:
        if mission.airport_type == AirportType.MOUNTAIN.value and AirportType.DOMESTIC.value in aircraft.airport_compatibility:
            pass
        else:
            return False, (
                f"airport_compatibility: {aircraft.model} not compatible with "
                f"{mission.airport_type} airport environment."
            )

    # Westbound headwind margin
    if mission.westbound_flag and required > 0:
        ratio = effective / required
        if ratio < _WESTBOUND_HEADWIND_THRESHOLD:
            return False, (
                f"westbound_headwind_margin: effective/required ratio {ratio:.2f} < "
                f"{_WESTBOUND_HEADWIND_THRESHOLD} (headwind margin insufficient)."
            )

    # Transpacific / long-haul: super-midsize hard block
    if mission.mission_class == MissionClass.TRANSPACIFIC.value and aircraft.category == "super-midsize":
        return False, (
            f"mission_class_block: super-midsize ({aircraft.model}) excluded from transpacific missions."
        )

    if (
        mission.route_distance_nm > 0
        and mission.route_distance_nm < 2000
        and aircraft.category == "ultra-long"
    ):
        return False, (
            f"regional_overbuy: ultra-long ({aircraft.model}) excluded for "
            f"~{int(mission.route_distance_nm)} nm stage length."
        )

    if mission.mission_class == MissionClass.TRANSATLANTIC.value and mission.high_payload_flag:
        pax_max = max(aircraft.payload_range_curve.keys())
        if mission.passenger_count > pax_max:
            return False, (
                f"payload_envelope: {mission.passenger_count} pax exceeds long-range envelope ({pax_max})."
            )

    return True, ""


def filter_feasible_aircraft(
    mission: Union[MissionNode, MissionProfile],
    aircraft_list: List[Union[str, AircraftNode]],
) -> Tuple[List[str], List[ExcludedAircraft], List[Dict[str, Any]]]:
    """
    Hard constraint filter — only passing aircraft may be scored.
    """
    m_node = mission if isinstance(mission, MissionNode) else build_mission_node(mission)
    feasible: List[str] = []
    excluded: List[ExcludedAircraft] = []
    log: List[Dict[str, Any]] = []

    for item in aircraft_list:
        if isinstance(item, AircraftNode):
            node = item
            model = node.model
        else:
            model = str(item)
            if model not in AIRCRAFT_PROFILES:
                excluded.append(
                    ExcludedAircraft(model=model, failed_constraint_reason="unknown_aircraft_model")
                )
                log.append(_log_graph_filter(model, False, "unknown_aircraft_model"))
                continue
            node = build_aircraft_node(model)

        ok, reason = _check_hard_constraints(m_node, node)
        log.append(_log_graph_filter(model, ok, reason))
        if ok:
            feasible.append(model)
        else:
            excluded.append(ExcludedAircraft(model=model, failed_constraint_reason=reason))

    return feasible, excluded, log


def score_aircraft(mission: Union[MissionNode, MissionProfile], aircraft: Union[str, AircraftNode]) -> RankedAircraft:
    """Weighted compatibility score (0–1 dimensions) — only for feasible aircraft."""
    m_node = mission if isinstance(mission, MissionNode) else build_mission_node(mission)
    a_node = aircraft if isinstance(aircraft, AircraftNode) else build_aircraft_node(str(aircraft))

    effective = _effective_range_nm(a_node, m_node)
    required = m_node.required_route_nm or m_node.route_distance_nm + _NBAA_RESERVE_NM

    if required > 0:
        ratio = effective / required
        range_fit = _clamp01(0.35 + min(0.65, (ratio - 1.0) * 0.55) if ratio >= 1.0 else ratio * 0.85)
    else:
        range_fit = _clamp01(a_node.range_nbaa_reserves / 5000.0)

    pax = m_node.passenger_count
    max_pax = max(a_node.payload_range_curve.keys())
    payload_fit = _clamp01(1.0 - max(0, pax - max_pax) * 0.12) if pax <= max_pax else _clamp01(max_pax / max(pax, 1))

    airport_limit = _AIRPORT_RUNWAY_LIMIT_FT.get(m_node.airport_type, 5500.0)
    runway_fit = _clamp01(1.0 - max(0, a_node.runway_ft_required - airport_limit * 0.85) / 2500.0)

    cost_efficiency = _clamp01(1.08 - a_node.cost_index)
    cabin_fit = _clamp01(a_node.cabin_score)
    category_fit = _clamp01(a_node.mission_category_fit.get(m_node.mission_class, 0.5))

    total = sum(
        _SCORE_WEIGHTS[k] * v
        for k, v in (
            ("range_fit", range_fit),
            ("payload_fit", payload_fit),
            ("runway_fit", runway_fit),
            ("cost_efficiency", cost_efficiency),
            ("cabin_fit", cabin_fit),
            ("category_fit", category_fit),
        )
    )

    return RankedAircraft(
        model=a_node.model,
        total_score=round(total, 4),
        range_fit=range_fit,
        payload_fit=payload_fit,
        runway_fit=runway_fit,
        cost_efficiency=cost_efficiency,
        cabin_fit=cabin_fit,
        category_fit=category_fit,
    )


class AircraftCapabilityGraph:
    """Graph facade — build nodes, filter, score, rank."""

    def __init__(self) -> None:
        self._aircraft_cache: Dict[str, AircraftNode] = {}

    def aircraft_node(self, model: str) -> AircraftNode:
        if model not in self._aircraft_cache:
            self._aircraft_cache[model] = build_aircraft_node(model)
        return self._aircraft_cache[model]

    def mission_node(self, profile: MissionProfile) -> MissionNode:
        return build_mission_node(profile)

    def evaluate(
        self,
        profile: MissionProfile,
        aircraft_list: Optional[List[str]] = None,
    ) -> CapabilityGraphResult:
        return evaluate_capability_graph(profile, aircraft_list)


def evaluate_capability_graph(
    profile: MissionProfile,
    aircraft_list: Optional[List[str]] = None,
) -> CapabilityGraphResult:
    """
    Full graph pass: filter → score feasible only → rank.
    """
    mission = build_mission_node(profile)
    candidates = aircraft_list or list(AIRCRAFT_PROFILES.keys())
    feasible, excluded, log = filter_feasible_aircraft(mission, candidates)

    ranked: List[RankedAircraft] = []
    for model in feasible:
        ranked.append(score_aircraft(mission, model))
    ranked.sort(key=lambda r: -r.total_score)

    return CapabilityGraphResult(
        mission=mission,
        feasible_aircraft_list=feasible,
        excluded_aircraft_list=excluded,
        ranked_results=ranked,
        filter_log=log,
    )

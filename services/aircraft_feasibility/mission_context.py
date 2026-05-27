"""Normalize mission JSON into a feasibility evaluation context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from services.consultant.route_feasibility import estimate_route_distance_nm
from services.mission.models import MissionProfile, PriorityLevel, Route
from services.mission.normalization import normalize_place

_HIGH_PRIORITY = frozenset({"high", "medium"})


@dataclass(frozen=True)
class FeasibilityMissionContext:
    """Operational mission snapshot for hard feasibility — no aircraft recommendations."""

    passengers: int = 6
    stage_distance_nm: float = 0.0
    route_label: str = ""
    nonstop_required: bool = False
    westbound_sensitive: bool = False
    winter_ops: bool = False
    baggage_high: bool = False
    runway_priority_high: bool = False
    short_runway_ops: bool = False
    mountain_airports: bool = False
    hot_high_ops: bool = False
    transatlantic: bool = False
    transpacific: bool = False
    international_ops: bool = False
    nbaa_reserves: bool = True
    stop_required: bool = False

    @property
    def winter_westbound_transpacific(self) -> bool:
        return self.westbound_sensitive and self.winter_ops and self.transpacific


def _priority_high(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.lower() in _HIGH_PRIORITY
    if isinstance(value, PriorityLevel):
        return value in (PriorityLevel.HIGH, PriorityLevel.MEDIUM)
    return False


def _build_route_label(origin: Optional[str], destinations: Optional[List[str]]) -> str:
    if not origin or not destinations:
        return ""
    dest = destinations[0]
    return f"{normalize_place(origin)} -> {normalize_place(dest)}"


def _infer_oceanic_flags(
    route_label: str,
    stage_nm: float,
    mission: Dict[str, Any],
) -> tuple[bool, bool]:
    blob = route_label.lower()
    transatlantic = bool(mission.get("transatlantic"))
    transpacific = bool(mission.get("transpacific"))

    if not transatlantic:
        transatlantic = bool(
            mission.get("europe")
            or re.search(r"\b(?:london|paris|geneva|europe|transatlantic)\b", blob)
            or (
                stage_nm >= 2600
                and re.search(r"\b(?:new\s+york|nyc|boston|miami|teterboro)\b", blob)
                and re.search(r"\b(?:london|paris|geneva|europe)\b", blob)
            )
            or (
                stage_nm >= 4800
                and re.search(r"\b(?:los\s+angeles|san\s+francisco|west\s+coast)\b", blob)
                and re.search(r"\b(?:london|paris|europe)\b", blob)
            )
        )
    if not transpacific:
        transpacific = bool(
            mission.get("asia")
            or re.search(r"\b(?:tokyo|sydney|honolulu|transpacific|seoul|beijing)\b", blob)
            or (
                stage_nm >= 4200
                and re.search(
                    r"\b(?:san\s+francisco|los\s+angeles|seattle|new\s+york|honolulu)\b",
                    blob,
                )
                and re.search(r"\b(?:tokyo|sydney|seoul|beijing|hong\s+kong)\b", blob)
            )
        )
    return transatlantic, transpacific


def mission_context_from_json(mission: Union[Dict[str, Any], Any]) -> FeasibilityMissionContext:
    """Build context from mission extraction JSON or compatible dict."""
    if hasattr(mission, "model_dump"):
        data: Dict[str, Any] = mission.model_dump(mode="json")
    elif isinstance(mission, dict):
        data = mission
    else:
        data = {}

    passengers = int(data.get("passengers") or 6)
    origin = data.get("origin")
    destinations = data.get("destination") or []
    if isinstance(destinations, str):
        destinations = [destinations]

    route_label = _build_route_label(origin, destinations if isinstance(destinations, list) else None)
    if not route_label and data.get("routes"):
        routes = data["routes"]
        if isinstance(routes, list) and routes:
            first = routes[0]
            if isinstance(first, dict):
                route_label = f"{first.get('origin', '')} -> {first.get('destination', '')}".strip()
            elif isinstance(first, str):
                route_label = first

    stage_nm = estimate_route_distance_nm(route_label) if route_label else 0.0
    transatlantic, transpacific = _infer_oceanic_flags(route_label, stage_nm, data)

    westbound = bool(data.get("westbound_sensitive"))
    if not westbound and route_label:
        rl = route_label.lower()
        westbound = bool(
            re.search(r"westbound", rl)
            or (
                re.search(r"\b(?:san\s+francisco|los\s+angeles|seattle|honolulu)\b", rl)
                and re.search(r"\b(?:tokyo|london|paris|sydney|europe|asia)\b", rl)
            )
        )

    nonstop = bool(data.get("nonstop_required"))
    stop_required = bool(data.get("stop_required"))
    if not stop_required and nonstop is False:
        stop_required = True

    return FeasibilityMissionContext(
        passengers=passengers,
        stage_distance_nm=stage_nm,
        route_label=route_label,
        nonstop_required=nonstop,
        stop_required=stop_required,
        westbound_sensitive=westbound,
        winter_ops=bool(data.get("winter_ops")),
        baggage_high=_priority_high(data.get("baggage_priority")),
        runway_priority_high=_priority_high(data.get("runway_priority")),
        short_runway_ops=bool(data.get("short_runway_ops")),
        mountain_airports=bool(data.get("mountain_airports")),
        hot_high_ops=bool(data.get("hot_high_ops")),
        transatlantic=transatlantic,
        transpacific=transpacific,
        international_ops=bool(data.get("international_ops") or transatlantic or transpacific),
        nbaa_reserves=True,
    )


def mission_context_from_profile(profile: MissionProfile) -> FeasibilityMissionContext:
    """Adapter from legacy :class:`MissionProfile`."""
    routes = profile.routes or []
    route_label = routes[0].label() if routes else ""
    stage_nm = max((estimate_route_distance_nm(r.label()) for r in routes), default=0.0)

    data = {
        "passengers": profile.passengers,
        "origin": routes[0].origin if routes else None,
        "destination": [routes[0].destination] if routes else None,
        "nonstop_required": profile.nonstop_required,
        "westbound_sensitive": profile.westbound_sensitive,
        "winter_ops": (profile.seasonal_note or "").lower().find("winter") >= 0,
        "baggage_priority": profile.baggage_priority.value,
        "runway_priority": profile.runway_priority.value,
        "short_runway_ops": profile.short_field_priority in (PriorityLevel.HIGH, PriorityLevel.MEDIUM),
        "mountain_airports": profile.mountain_airports or profile.mountain_airport_priority,
        "hot_high_ops": profile.mountain_airports or profile.mountain_airport_priority,
        "international_ops": profile.international_ops,
    }
    transatlantic, transpacific = _infer_oceanic_flags(route_label, stage_nm, data)

    stop_required = not profile.nonstop_required

    return FeasibilityMissionContext(
        passengers=int(profile.passengers or 6),
        stage_distance_nm=stage_nm,
        route_label=route_label,
        nonstop_required=profile.nonstop_required,
        stop_required=stop_required,
        westbound_sensitive=profile.westbound_sensitive,
        winter_ops=(profile.seasonal_note or "").lower().find("winter") >= 0,
        baggage_high=profile.baggage_priority in (PriorityLevel.HIGH, PriorityLevel.MEDIUM),
        runway_priority_high=profile.runway_priority in (PriorityLevel.HIGH, PriorityLevel.MEDIUM),
        short_runway_ops=profile.short_field_priority in (PriorityLevel.HIGH, PriorityLevel.MEDIUM),
        mountain_airports=profile.mountain_airports or profile.mountain_airport_priority,
        hot_high_ops=profile.mountain_airports or profile.mountain_airport_priority,
        transatlantic=transatlantic,
        transpacific=transpacific,
        international_ops=profile.international_ops or transatlantic or transpacific,
        nbaa_reserves=profile.nbaa_reserve_required
        or (profile.reserves_requirement or "").lower().find("nbaa") >= 0,
    )


def profile_from_context(ctx: FeasibilityMissionContext) -> MissionProfile:
    """Convert context back to MissionProfile for legacy pipeline bridges."""
    routes: List[Route] = []
    if ctx.route_label and "->" in ctx.route_label:
        left, right = ctx.route_label.split("->", 1)
        routes = [Route(origin=left.strip(), destination=right.strip())]

    return MissionProfile(
        passengers=ctx.passengers,
        routes=routes,
        nonstop_required=ctx.nonstop_required,
        westbound_sensitive=ctx.westbound_sensitive,
        seasonal_note="winter_headwinds" if ctx.winter_ops else None,
        baggage_priority=PriorityLevel.HIGH if ctx.baggage_high else PriorityLevel.NONE,
        runway_priority=PriorityLevel.HIGH if ctx.runway_priority_high else PriorityLevel.NONE,
        short_field_priority=PriorityLevel.HIGH if ctx.short_runway_ops else PriorityLevel.NONE,
        mountain_airports=ctx.mountain_airports,
        mountain_airport_priority=ctx.mountain_airports,
        international_ops=ctx.international_ops,
        nbaa_reserve_required=ctx.nbaa_reserves,
    )

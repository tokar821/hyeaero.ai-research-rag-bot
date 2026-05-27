"""
Adapters between typed MissionProfile and legacy MissionState (ranking/formatters).
"""

from __future__ import annotations

from services.consultant.mission_state import MissionState, normalize_routes
from services.mission.models import MissionProfile, OwnershipMode, PriorityLevel, Route


def mission_state_to_profile(state: MissionState) -> MissionProfile:
    """Rebuild MissionProfile from MissionState for feasibility evaluation."""
    routes = []
    for label in state.routes or []:
        r = Route.from_label(label)
        if r:
            routes.append(r)
    cabin = PriorityLevel.NONE
    if (state.cabin_priority or "").lower() == "high":
        cabin = PriorityLevel.HIGH
    op_cost = PriorityLevel.HIGH if (state.operating_cost_priority or "").lower() == "high" else PriorityLevel.NONE
    bag = PriorityLevel.HIGH if (state.baggage_priority or "").lower() == "high" else PriorityLevel.NONE
    runway = PriorityLevel.HIGH if state.runway_constraints else PriorityLevel.NONE
    own = None
    if state.acquisition_strategy:
        try:
            own = OwnershipMode(state.acquisition_strategy)
        except ValueError:
            pass
    return MissionProfile(
        passengers=state.passenger_count,
        routes=routes,
        nonstop_required=bool(state.nonstop_requirement),
        westbound_sensitive=bool(state.westbound),
        nbaa_reserve_required=(state.reserves_requirement or "").lower().find("nbaa") >= 0,
        runway_priority=runway,
        operating_cost_priority=op_cost,
        cabin_priority=cabin,
        baggage_priority=bag,
        ownership_interest=own,
        ownership_posture=own,
        mountain_airports=bool(state.mountain_airport_requirement),
        mountain_airport_priority=bool(state.mountain_airport_requirement),
        reserves_requirement=state.reserves_requirement,
        preferred_airports=list(state.preferred_airports or []),
        seasonal_note=state.seasonal_constraints,
        short_field_priority=runway,
    )


def mission_profile_to_state(profile: MissionProfile) -> MissionState:
    """Map typed turn profile to MissionState without merging prior turns."""
    ms = MissionState()
    dist = profile.passenger_distribution
    ms.passenger_count = (
        dist.planning_load if dist and dist.planning_load is not None else profile.passengers
    )
    if dist:
        ms.passenger_min = dist.min_pax
        ms.passenger_max = dist.max_pax
        ms.cargo_required = dist.cargo_required or None
    ms.routes = normalize_routes(profile.route_labels())
    ms.mission_type = (
        profile.mission_category.value if profile.mission_category else None
    )
    ms.budget_usd = profile.budget_usd_mid
    ms.westbound = profile.westbound_sensitive if profile.westbound_sensitive else None
    ms.eastbound = profile.eastbound_sensitive if profile.eastbound_sensitive else None
    ms.nonstop_requirement = profile.nonstop_required if profile.nonstop_required else None
    own = profile.ownership_posture or profile.ownership_interest
    if own:
        ms.acquisition_strategy = own.value
    ms.seasonal_constraints = profile.seasonal_note
    ms.mountain_airport_requirement = (
        profile.mountain_airport_priority or profile.mountain_airports
    ) or None
    if profile.short_field_priority in (PriorityLevel.HIGH, PriorityLevel.MEDIUM):
        ms.runway_constraints = "short_field"
    ms.preferred_airports = list(profile.preferred_airports)
    ms.reserves_requirement = profile.reserves_requirement

    if profile.cabin_priority in (PriorityLevel.HIGH, PriorityLevel.MEDIUM):
        ms.cabin_priority = "high"
    if profile.baggage_priority in (PriorityLevel.HIGH, PriorityLevel.MEDIUM):
        ms.baggage_priority = "high"
    if profile.operating_cost_priority == PriorityLevel.HIGH:
        ms.operating_cost_priority = "high"

    ms.turn_index = 1
    return ms

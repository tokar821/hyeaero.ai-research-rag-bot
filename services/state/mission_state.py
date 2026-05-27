"""
Persistent internal MissionState — updated each turn, never shown to users.

Stored under ``hye_persistent_mission_state`` in ``data_used`` / ``conversation_state``.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.consultant.mission_state import MissionState as ConsultantMissionState
from services.consultant.mission_state import normalize_routes
from services.consultant.route_feasibility import estimate_route_distance_nm
from services.mission.adapters import mission_profile_to_state
from services.mission.models import MissionCategory, MissionProfile, OwnershipMode, PriorityLevel, Route

PERSISTENT_MISSION_STATE_KEY = "hye_persistent_mission_state"

# User-facing keys that must never echo this object
_FORBIDDEN_USER_KEYS = frozenset(
    {
        PERSISTENT_MISSION_STATE_KEY,
        "persistent_mission_state",
        "internal_mission_state",
    }
)


class MissionType:
    BUSINESS_TRAVEL = "business_travel"
    ACQUISITION = "acquisition"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"


@dataclass
class MissionPriorities:
    """Normalized priority flags (none / medium / high)."""

    cost: str = "none"
    runway: str = "none"
    luxury: str = "none"
    baggage: str = "none"
    ownership: str = "none"

    def to_dict(self) -> Dict[str, str]:
        return {
            "cost": self.cost,
            "runway": self.runway,
            "luxury": self.luxury,
            "baggage": self.baggage,
            "ownership": self.ownership,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> MissionPriorities:
        if not isinstance(raw, dict):
            return cls()
        out = cls()
        for k in ("cost", "runway", "luxury", "baggage", "ownership"):
            v = str(raw.get(k) or "none").strip().lower()
            if v in ("none", "medium", "high"):
                setattr(out, k, v)
        return out


@dataclass
class MissionState:
    """
    Cross-turn mission accumulator for ranking, feasibility, and decision engines.

    **Never** format or print this object for end users.
    """

    passengers: Optional[int] = None
    routes: List[str] = field(default_factory=list)
    mission_type: str = MissionType.UNKNOWN
    range_requirement_nm: Optional[float] = None
    nonstop_required: bool = False
    westbound: bool = False
    mountain_airports: bool = False
    budget_usd: Optional[float] = None
    priorities: MissionPriorities = field(default_factory=MissionPriorities)
    home_base: Optional[str] = None
    fleet_preferences: List[str] = field(default_factory=list)
    turn_count: int = 0
    clarification_questions_asked: int = 0

    def clone(self) -> MissionState:
        return copy.deepcopy(self)

    def __str__(self) -> str:
        return "<MissionState internal>"

    def __repr__(self) -> str:
        return "<MissionState internal>"

    def to_storage_dict(self) -> Dict[str, Any]:
        """Serialize for ``data_used`` / conversation echo — internal telemetry only."""
        return {
            "schema_version": 1,
            "passengers": self.passengers,
            "routes": list(self.routes),
            "mission_type": self.mission_type,
            "range_requirement_nm": self.range_requirement_nm,
            "nonstop_required": self.nonstop_required,
            "westbound": self.westbound,
            "budget_usd": self.budget_usd,
            "priorities": self.priorities.to_dict(),
            "home_base": self.home_base,
            "fleet_preferences": list(self.fleet_preferences),
            "turn_count": self.turn_count,
            "clarification_questions_asked": self.clarification_questions_asked,
        }

    @classmethod
    def from_storage_dict(cls, raw: Optional[Dict[str, Any]]) -> MissionState:
        if not isinstance(raw, dict):
            return cls()
        ms = cls()
        if raw.get("passengers") is not None:
            try:
                p = int(raw["passengers"])
                if 1 <= p <= 19:
                    ms.passengers = p
            except (TypeError, ValueError):
                pass
        ms.routes = normalize_routes(raw.get("routes"))
        mt = str(raw.get("mission_type") or MissionType.UNKNOWN).strip().lower()
        if mt in (
            MissionType.BUSINESS_TRAVEL,
            MissionType.ACQUISITION,
            MissionType.COMPARISON,
            MissionType.UNKNOWN,
        ):
            ms.mission_type = mt
        if raw.get("range_requirement_nm") is not None:
            try:
                ms.range_requirement_nm = float(raw["range_requirement_nm"])
            except (TypeError, ValueError):
                pass
        ms.nonstop_required = bool(raw.get("nonstop_required"))
        ms.westbound = bool(raw.get("westbound"))
        ms.mountain_airports = bool(raw.get("mountain_airports"))
        if raw.get("budget_usd") is not None:
            try:
                ms.budget_usd = float(raw["budget_usd"])
            except (TypeError, ValueError):
                pass
        ms.priorities = MissionPriorities.from_dict(raw.get("priorities"))
        if raw.get("home_base"):
            ms.home_base = str(raw["home_base"]).strip() or None
        if isinstance(raw.get("fleet_preferences"), list):
            ms.fleet_preferences = [str(x).strip() for x in raw["fleet_preferences"] if str(x).strip()]
        try:
            ms.turn_count = max(0, int(raw.get("turn_count") or 0))
        except (TypeError, ValueError):
            ms.turn_count = 0
        try:
            ms.clarification_questions_asked = max(
                0, int(raw.get("clarification_questions_asked") or 0)
            )
        except (TypeError, ValueError):
            ms.clarification_questions_asked = 0
        return ms


def record_clarification_question_asked(state: MissionState) -> MissionState:
    """Increment cross-turn clarification budget (max one before recommending)."""
    state.clarification_questions_asked = min(
        1, max(0, int(state.clarification_questions_asked or 0)) + 1
    )
    return state


def load_persistent_mission_state(
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> MissionState:
    for src in (data_used, conversation_state):
        if not isinstance(src, dict):
            continue
        raw = src.get(PERSISTENT_MISSION_STATE_KEY)
        if isinstance(raw, dict):
            return MissionState.from_storage_dict(raw)
    return MissionState()


def persist_mission_state_patch(state: MissionState) -> Dict[str, Any]:
    return {PERSISTENT_MISSION_STATE_KEY: state.to_storage_dict()}


def _priority_level_to_str(level: PriorityLevel) -> str:
    if level == PriorityLevel.HIGH:
        return "high"
    if level == PriorityLevel.MEDIUM:
        return "medium"
    return "none"


def _merge_priority(current: str, incoming: PriorityLevel) -> str:
    inc = _priority_level_to_str(incoming)
    rank = {"none": 0, "medium": 1, "high": 2}
    return inc if rank.get(inc, 0) > rank.get(current, 0) else current


def _max_priority(*levels: PriorityLevel) -> PriorityLevel:
    order = {
        PriorityLevel.NONE: 0,
        PriorityLevel.LOW: 0,
        PriorityLevel.MEDIUM: 1,
        PriorityLevel.HIGH: 2,
    }
    best = PriorityLevel.NONE
    for lv in levels:
        if order.get(lv, 0) > order.get(best, 0):
            best = lv
    return best


def _infer_mission_type(profile: MissionProfile, query: str) -> str:
    cat = profile.mission_category
    ql = (query or "").lower()
    if cat == MissionCategory.COMPARISON or re.search(r"\bcompare|versus|vs\.?\b", ql):
        return MissionType.COMPARISON
    if cat in (
        MissionCategory.ACQUISITION_ADVISORY,
        MissionCategory.OWNERSHIP_STRUCTURE,
        MissionCategory.DISPOSITION,
    ):
        return MissionType.ACQUISITION
    if re.search(r"\b(?:buy|acquire|purchase|ownership|fractional)\b", ql):
        return MissionType.ACQUISITION
    if profile.routes or profile.passengers or re.search(
        r"\b(?:recommend|nonstop|route|trip|fly|pax|passengers?)\b", ql
    ):
        return MissionType.BUSINESS_TRAVEL
    return MissionType.UNKNOWN


def _implicit_range_nm(routes: List[str], profile: MissionProfile) -> Optional[float]:
    """Compute range only from validated route labels — never guess without routes."""
    del profile
    labels = normalize_routes(routes)
    if not labels:
        return None
    return max(estimate_route_distance_nm(r) for r in labels)


def _mission_type_rank(mt: str) -> int:
    return {
        MissionType.UNKNOWN: 0,
        MissionType.BUSINESS_TRAVEL: 1,
        MissionType.ACQUISITION: 2,
        MissionType.COMPARISON: 3,
    }.get(mt, 0)


def advance_persistent_mission_state(
    prior: MissionState,
    turn_profile: MissionProfile,
    query: str,
) -> MissionState:
    """
    Update prior state from the current turn — inherit unless explicitly overwritten.
    """
    from services.state.mission_validation import (
        turn_explicitly_sets_passengers,
        turn_explicitly_sets_routes,
        user_requested_mission_reset,
    )

    if user_requested_mission_reset(query):
        state = MissionState(turn_count=max(1, prior.turn_count + 1))
    else:
        state = prior.clone()
        state.turn_count = max(1, prior.turn_count + 1)

    def _route_overlap(cur: List[str], prev: List[str]) -> float:
        if not cur or not prev:
            return 0.0
        a = {r.strip().lower() for r in cur if r}
        b = {r.strip().lower() for r in prev if r}
        if not a or not b:
            return 0.0
        return len(a & b) / max(len(a), 1)

    # Passengers: only change when user states a count this turn
    if turn_explicitly_sets_passengers(query) and turn_profile.passengers is not None:
        state.passengers = turn_profile.passengers

    # Routes: only replace when this turn extracted valid route(s)
    if turn_explicitly_sets_routes(turn_profile):
        new_routes = turn_profile.route_labels()
        prior_routes = list(prior.routes or [])
        ov = _route_overlap(new_routes, prior_routes)
        state.routes = new_routes
        # Mission pivot: when the user provides a new explicit route set that doesn't overlap,
        # do not carry sticky operational constraints from the old mission into the new one.
        if prior_routes and ov < 0.2:
            state.nonstop_required = False
            state.westbound = False
            state.mountain_airports = False
            state.range_requirement_nm = None
            state.priorities.runway = "none"

    from services.state.session_mission_memory import (
        parse_budget_from_query,
        turn_explicitly_sets_budget,
    )

    if turn_explicitly_sets_budget(query, turn_profile):
        if turn_profile.budget_usd_mid is not None:
            state.budget_usd = turn_profile.budget_usd_mid
        else:
            parsed_budget = parse_budget_from_query(query)
            if parsed_budget is not None:
                state.budget_usd = parsed_budget

    # Constraints persist (sticky) unless this turn explicitly pivoted routes above.
    state.nonstop_required = bool(state.nonstop_required or turn_profile.nonstop_required)
    state.westbound = bool(state.westbound or turn_profile.westbound_sensitive)
    state.mountain_airports = bool(
        state.mountain_airports
        or turn_profile.mountain_airport_priority
        or turn_profile.mountain_airports
    )

    from services.state.session_mission_memory import (
        turn_explicitly_sets_home_base,
        turn_explicitly_sets_ownership,
        turn_explicitly_sets_runway_priority,
    )

    if turn_explicitly_sets_home_base(query, turn_profile) and turn_profile.home_base:
        state.home_base = turn_profile.home_base
    if turn_profile.fleet_preferences:
        state.fleet_preferences = list(turn_profile.fleet_preferences)

    from services.state.session_mission_memory import turn_explicitly_sets_mission_type

    inferred_type = _infer_mission_type(turn_profile, query)
    if turn_explicitly_sets_mission_type(query, turn_profile):
        state.mission_type = inferred_type
    elif _mission_type_rank(inferred_type) >= _mission_type_rank(state.mission_type):
        state.mission_type = inferred_type

    state.priorities.cost = _merge_priority(
        state.priorities.cost, turn_profile.operating_cost_priority
    )
    if turn_explicitly_sets_runway_priority(query, turn_profile):
        state.priorities.runway = _priority_level_to_str(
            _max_priority(turn_profile.runway_priority, turn_profile.short_field_priority)
        )
    else:
        state.priorities.runway = _merge_priority(
            state.priorities.runway,
            _max_priority(turn_profile.runway_priority, turn_profile.short_field_priority),
        )
    state.priorities.luxury = _merge_priority(state.priorities.luxury, turn_profile.cabin_priority)
    state.priorities.baggage = _merge_priority(state.priorities.baggage, turn_profile.baggage_priority)
    own = turn_profile.ownership_posture or turn_profile.ownership_interest
    if turn_explicitly_sets_ownership(query, turn_profile) and own and own != OwnershipMode.UNDECIDED:
        state.priorities.ownership = own.value

    implicit = _implicit_range_nm(state.routes, turn_profile)
    state.range_requirement_nm = implicit if implicit is not None else prior.range_requirement_nm

    return state


def persistent_to_mission_profile(state: MissionState) -> MissionProfile:
    """Materialize typed profile for feasibility graph and rankers."""
    routes: List[Route] = []
    for label in state.routes or []:
        r = Route.from_label(label)
        if r:
            routes.append(r)

    profile = MissionProfile(
        passengers=state.passengers,
        routes=routes,
        nonstop_required=state.nonstop_required,
        westbound_sensitive=state.westbound,
        mountain_airport_priority=state.mountain_airports,
        mountain_airports=state.mountain_airports,
        budget_usd_mid=state.budget_usd,
        home_base=state.home_base,
        fleet_preferences=list(state.fleet_preferences),
    )

    if state.priorities.cost == "high":
        profile.operating_cost_priority = PriorityLevel.HIGH
    elif state.priorities.cost == "medium":
        profile.operating_cost_priority = PriorityLevel.MEDIUM

    if state.priorities.runway == "high":
        profile.runway_priority = PriorityLevel.HIGH
        profile.short_field_priority = PriorityLevel.HIGH
    elif state.priorities.runway == "medium":
        profile.runway_priority = PriorityLevel.MEDIUM

    if state.priorities.luxury == "high":
        profile.cabin_priority = PriorityLevel.HIGH
    elif state.priorities.luxury == "medium":
        profile.cabin_priority = PriorityLevel.MEDIUM

    if state.priorities.baggage == "high":
        profile.baggage_priority = PriorityLevel.HIGH

    if state.mission_type == MissionType.COMPARISON:
        profile.mission_category = MissionCategory.COMPARISON
    elif state.mission_type == MissionType.ACQUISITION:
        profile.mission_category = MissionCategory.ACQUISITION_ADVISORY
    elif state.mission_type == MissionType.BUSINESS_TRAVEL:
        profile.mission_category = MissionCategory.POINT_TO_POINT

    if state.priorities.ownership:
        try:
            profile.ownership_posture = OwnershipMode(state.priorities.ownership)
        except ValueError:
            pass

    return profile


def to_consultant_mission_state(state: MissionState) -> ConsultantMissionState:
    """Bridge to legacy consultant dataclass used by rankers/formatters."""
    return mission_profile_to_state(persistent_to_mission_profile(state))


def to_decision_mission_dict(state: MissionState) -> Dict[str, Any]:
    """Shape for ``aircraft_decision_engine`` mission hints."""
    missing: List[str] = []
    if state.passengers is None:
        missing.append("passenger_count")
    leg = state.range_requirement_nm
    if leg is None:
        missing.append("longest_leg_nm")
    budget_m = (state.budget_usd / 1_000_000.0) if state.budget_usd else None
    if budget_m is None:
        missing.append("budget")
    usage = ""
    if state.priorities.ownership == OwnershipMode.CHARTER.value:
        usage = "charter"
    elif state.priorities.ownership in (
        OwnershipMode.FULL_OWNERSHIP.value,
        OwnershipMode.FRACTIONAL.value,
    ):
        usage = "private"
    else:
        missing.append("usage_private_vs_charter")

    return {
        "passengers": state.passengers,
        "longest_leg_nm": leg,
        "budget_millions_usd": budget_m,
        "usage": usage,
        "typical_routes_hint": bool(state.routes),
        "missing_fields": missing,
        "mission_type": state.mission_type,
        "routes": list(state.routes),
        "priorities": state.priorities.to_dict(),
    }


def merge_decision_mission(
    parsed: Dict[str, Any],
    persistent: MissionState,
) -> Dict[str, Any]:
    """Fill gaps in per-turn parse from persistent session state."""
    merged = dict(parsed)
    hint = to_decision_mission_dict(persistent)
    for key in ("passengers", "longest_leg_nm", "budget_millions_usd", "usage", "typical_routes_hint"):
        if merged.get(key) in (None, "", False) and hint.get(key) not in (None, "", False):
            merged[key] = hint[key]
    missing = list(merged.get("missing_fields") or [])
    for field_name in ("passenger_count", "longest_leg_nm", "budget", "usage_private_vs_charter"):
        if field_name in missing:
            check = {
                "passenger_count": "passengers",
                "longest_leg_nm": "longest_leg_nm",
                "budget": "budget_millions_usd",
                "usage_private_vs_charter": "usage",
            }[field_name]
            if merged.get(check) not in (None, "", False):
                missing = [m for m in missing if m != field_name]
    merged["missing_fields"] = missing
    merged["persistent_mission_applied"] = True
    return merged


def sync_persistent_mission_state(
    query: str,
    *,
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    turn_profile: Optional[MissionProfile] = None,
) -> Tuple[MissionState, MissionProfile, "MissionStateConsistencyReport"]:
    """
    Load → update from current turn → validate → return (state, profile, report).
    """
    from services.mission.mission_extractor import extract_mission
    from services.state.mission_validation import validate_mission_state_consistency

    prior = load_persistent_mission_state(conversation_state, data_used)
    turn = turn_profile if turn_profile is not None else extract_mission(query)
    updated = advance_persistent_mission_state(prior, turn, query)
    report = validate_mission_state_consistency(prior, updated, turn, query)
    profile = persistent_to_mission_profile(updated)
    if isinstance(data_used, dict):
        data_used.update(persist_mission_state_patch(updated))
        data_used["mission_state_validation"] = report.to_dict()
    return updated, profile, report


def assert_not_user_visible(payload: Dict[str, Any]) -> None:
    """Guardrail for formatters — internal keys must not leak into answer assembly."""
    for key in _FORBIDDEN_USER_KEYS:
        if key in payload:
            raise ValueError(f"Internal mission state key leaked: {key}")

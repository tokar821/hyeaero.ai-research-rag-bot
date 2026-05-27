"""
Mission state consistency validation — cross-turn inheritance rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mission.models import MissionProfile
from services.state.mission_state import MissionState

# CamelCase alias requested in product spec
validateMissionStateConsistency = None  # set at module bottom

_PAX_EXPLICIT_RE = re.compile(
    r"\b(\d{1,2})\s*(?:pax|passengers?|people|executives?|seats?|souls)\b"
    r"|\b(?:for\s+)?(\d{1,2})\s+pax\b"
    r"|\btravel\s+with\s+(\d{1,2})\b",
    re.I,
)

_ROUTE_ADVISORY_RE = re.compile(
    r"\b(?:recommend|best\s+(?:jet|aircraft|option)|which\s+(?:jet|aircraft)|"
    r"what\s+(?:jet|aircraft)|shortlist|nonstop|feasibility|compare\s+(?!all\s+five))\b",
    re.I,
)

_MISSION_RESET_RE = re.compile(
    r"\b(?:new\s+trip|different\s+route|forget\s+(?:that|the)|start\s+over|"
    r"switching\s+gears|ignore\s+(?:that|the)\s+above|new\s+mission|"
    r"unrelated\s+question|instead\s+(?:fly|route))\b",
    re.I,
)


@dataclass
class MissionStateConsistencyReport:
    is_consistent: bool = True
    needs_route_clarification: bool = False
    clarifying_question: str = ""
    inherited_fields: List[str] = field(default_factory=list)
    updated_fields: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_consistent": self.is_consistent,
            "needs_route_clarification": self.needs_route_clarification,
            "clarifying_question": self.clarifying_question,
            "inherited_fields": list(self.inherited_fields),
            "updated_fields": list(self.updated_fields),
            "issues": list(self.issues),
            "warnings": list(self.warnings),
        }


def turn_explicitly_sets_passengers(query: str) -> bool:
    return bool(_PAX_EXPLICIT_RE.search(query or ""))


def turn_explicitly_sets_routes(turn_profile: MissionProfile) -> bool:
    return bool(turn_profile.routes)


def query_requests_route_advisory(query: str) -> bool:
    return bool(_ROUTE_ADVISORY_RE.search(query or ""))


def query_requires_route_for_advisory(query: str) -> bool:
    """Route is required for mission-fit recommendations, not bare model comparisons or visuals."""
    ql = (query or "").strip()
    try:
        from services.recommendation.query_recommendation_intent import is_visualization_query

        if is_visualization_query(ql):
            return False
    except Exception:
        pass
    if re.search(r"\bcompare\b.*\b(?:vs\.?|versus)\b", ql, re.I | re.S):
        return False
    if re.search(r"\bvs\.?\b|\bversus\b", ql, re.I):
        try:
            from services.consultant.recommendation_engine import detect_models_from_text

            if len(detect_models_from_text(ql)) >= 2:
                return False
        except Exception:
            pass
    if re.search(
        r"\b(?:fractional|full\s+ownership|leaning\s+fractional|overbuying)\b",
        ql,
        re.I,
    ):
        return False
    return query_requests_route_advisory(ql)


def user_requested_mission_reset(query: str) -> bool:
    return bool(_MISSION_RESET_RE.search(query or ""))


def build_route_clarifying_question(state: MissionState) -> str:
    """User-facing clarifier when route is unknown — no guessing."""
    hints: List[str] = []
    if state.passengers:
        hints.append(f"{state.passengers} passengers")
    if state.budget_usd:
        hints.append(f"roughly ${state.budget_usd / 1_000_000:.1f}M budget")
    if state.nonstop_required:
        hints.append("nonstop required")
    if state.westbound:
        hints.append("westbound")
    ctx = f" ({', '.join(hints)})" if hints else ""
    return (
        f"What's the primary city pair{ctx}? "
        "Origin and destination are enough — e.g. Los Angeles to Miami or San Francisco to Tokyo."
    )


def turn_explicitly_sets_home_base(query: str, turn_profile: MissionProfile) -> bool:
    from services.state.session_mission_memory import turn_explicitly_sets_home_base as _impl

    return _impl(query, turn_profile)


def validate_mission_state_consistency(
    prior: MissionState,
    current: MissionState,
    turn_profile: MissionProfile,
    query: str,
) -> MissionStateConsistencyReport:
    """
    Validate cross-turn mission persistence rules after ``advance_persistent_mission_state``.

    Rules enforced:
    - Passengers persist unless the user explicitly states a new count this turn.
    - Routes persist unless the user provides new route(s) this turn.
    - Mission constraints (nonstop, westbound, priorities) inherit from prior unless overwritten.
    - Missing route on an advisory request → clarifying question (no route guessing).
    """
    report = MissionStateConsistencyReport()
    pax_explicit = turn_explicitly_sets_passengers(query)
    routes_explicit = turn_explicitly_sets_routes(turn_profile)

    # --- inheritance audit ---
    if prior.passengers is not None and not pax_explicit:
        if current.passengers != prior.passengers:
            report.issues.append(
                f"passenger_count changed without explicit user input "
                f"({prior.passengers} -> {current.passengers})"
            )
            report.is_consistent = False
        else:
            report.inherited_fields.append("passengers")

    if prior.routes and not routes_explicit:
        if current.routes != prior.routes:
            report.issues.append("routes changed without explicit route in current turn")
            report.is_consistent = False
        else:
            report.inherited_fields.append("routes")

    if prior.nonstop_required and current.nonstop_required == prior.nonstop_required:
        report.inherited_fields.append("nonstop_required")
    if prior.westbound and current.westbound == prior.westbound:
        report.inherited_fields.append("westbound")
    if prior.budget_usd is not None and turn_profile.budget_usd_mid is None:
        if current.budget_usd == prior.budget_usd:
            report.inherited_fields.append("budget_usd")

    if prior.home_base and not turn_explicitly_sets_home_base(query, turn_profile):
        if current.home_base == prior.home_base:
            report.inherited_fields.append("home_base")

    if prior.mission_type and prior.mission_type != "unknown":
        from services.state.session_mission_memory import turn_explicitly_sets_mission_type

        if not turn_explicitly_sets_mission_type(query, turn_profile):
            if current.mission_type == prior.mission_type:
                report.inherited_fields.append("mission_type")

    if prior.priorities.ownership not in ("", "none"):
        from services.state.session_mission_memory import turn_explicitly_sets_ownership

        if not turn_explicitly_sets_ownership(query, turn_profile):
            if current.priorities.ownership == prior.priorities.ownership:
                report.inherited_fields.append("ownership")

    if prior.priorities.runway not in ("", "none"):
        from services.state.session_mission_memory import turn_explicitly_sets_runway_priority

        if not turn_explicitly_sets_runway_priority(query, turn_profile):
            if current.priorities.runway == prior.priorities.runway:
                report.inherited_fields.append("runway")

    if pax_explicit:
        report.updated_fields.append("passengers")
    if routes_explicit:
        report.updated_fields.append("routes")
    if turn_profile.nonstop_required:
        report.updated_fields.append("nonstop_required")
    if turn_profile.westbound_sensitive:
        report.updated_fields.append("westbound")
    if turn_profile.budget_usd_mid is not None:
        report.updated_fields.append("budget_usd")

    # --- route clarification only when route is truly missing ---
    needs_route = query_requires_route_for_advisory(query)
    route_truly_missing = False
    if needs_route:
        from services.recommendation.clarification_decision import (
            route_truly_missing as _route_truly_missing,
        )
        from services.state.mission_state import to_consultant_mission_state

        consultant = to_consultant_mission_state(current)
        route_truly_missing = _route_truly_missing(consultant, query)

    clarifications_asked = max(0, int(getattr(current, "clarification_questions_asked", 0) or 0))
    if route_truly_missing and clarifications_asked < 1:
        report.needs_route_clarification = True
        report.clarifying_question = build_route_clarifying_question(current)
        report.warnings.append("route_missing_advisory_blocked")
    elif route_truly_missing and clarifications_asked >= 1:
        report.warnings.append("route_missing_clarification_budget_exhausted")

    if not current.routes and current.range_requirement_nm:
        report.issues.append("range_requirement_nm set without validated routes")
        report.is_consistent = False
        report.warnings.append("implicit_range_without_route")

    if (
        not routes_explicit
        and not prior.routes
        and route_truly_missing
        and not report.needs_route_clarification
        and clarifications_asked < 1
    ):
        report.needs_route_clarification = True
        report.clarifying_question = build_route_clarifying_question(current)

    return report


# Public camelCase alias
validateMissionStateConsistency = validate_mission_state_consistency

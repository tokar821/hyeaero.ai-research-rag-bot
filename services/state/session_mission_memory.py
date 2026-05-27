"""
Session mission memory — cross-turn persistence for stable mission context.

Persists (via ``hye_persistent_mission_state``):
  - home base
  - passenger count
  - budget
  - preferred mission type
  - ownership preference
  - runway priorities

Current turn always wins when the user explicitly overrides a field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from services.mission.models import MissionCategory, MissionProfile, OwnershipMode, PriorityLevel
from services.state.mission_state import MissionState, MissionType, persistent_to_mission_profile

SESSION_MEMORY_FIELD_KEYS = frozenset(
    {
        "home_base",
        "passengers",
        "budget_usd",
        "mission_type",
        "ownership",
        "runway",
    }
)

_BUDGET_EXPLICIT_RE = re.compile(
    r"\$\s*[\d,.]+(?:\s*(?:m|mm|million|mil))?|\b[\d.]+\s*(?:m|mm|million|mil)\b|"
    r"\bunder\s+\$?\s*[\d,.]+|\bbudget\s+(?:of|around|about|is)?\s*\$?\s*[\d,.]+",
    re.I,
)

_OWNERSHIP_EXPLICIT_RE = re.compile(
    r"\b(?:fractional|full\s+ownership|charter\s+only|leaning\s+fractional|"
    r"prefer\s+(?:fractional|charter)|ownership\s+(?:vs|or))\b",
    re.I,
)

_RUNWAY_EXPLICIT_RE = re.compile(
    r"\b(?:short\s+(?:runway|field)|runway\s+flex|runway\s+priority|"
    r"hot[- ]and[- ]high|mountain\s+airport|aspen|telluride|"
    r"under\s+4[,.]?000\s*ft|short[- ]field)\b",
    re.I,
)

_MISSION_TYPE_EXPLICIT_RE = re.compile(
    r"\b(?:compare\b.*\b(?:vs\.?|versus)|\bvs\.?\b|\bversus\b|"
    r"\b(?:buy|purchase|acquire|acquisition|sell\s+my|disposition))\b",
    re.I,
)


@dataclass
class SessionMissionSnapshot:
    """User-session mission facts (internal telemetry)."""

    home_base: Optional[str] = None
    passengers: Optional[int] = None
    budget_usd: Optional[float] = None
    mission_type: str = MissionType.UNKNOWN
    ownership: str = "none"
    runway_priority: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "home_base": self.home_base,
            "passengers": self.passengers,
            "budget_usd": self.budget_usd,
            "mission_type": self.mission_type,
            "ownership": self.ownership,
            "runway_priority": self.runway_priority,
        }

    @classmethod
    def from_persistent(cls, state: MissionState) -> SessionMissionSnapshot:
        return cls(
            home_base=state.home_base,
            passengers=state.passengers,
            budget_usd=state.budget_usd,
            mission_type=state.mission_type or MissionType.UNKNOWN,
            ownership=(state.priorities.ownership or "none"),
            runway_priority=(state.priorities.runway or "none"),
        )


@dataclass
class SessionMergeResult:
    profile: MissionProfile
    inherited_fields: List[str] = field(default_factory=list)
    overridden_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inherited_fields": list(self.inherited_fields),
            "overridden_fields": list(self.overridden_fields),
        }


def parse_budget_from_query(query: str) -> Optional[float]:
    """Parse acquisition budget from free text when the extractor left budget empty."""
    from services.mission.normalization import parse_budget_usd_mid

    usd = parse_budget_usd_mid(query or "")
    if usd is not None:
        return usd
    m = re.search(
        r"\$\s*([\d,.]+)\s*(m|mm|million|mil)?\b|"
        r"\b([\d,.]+)\s*(m|mm|million|mil)\b",
        query or "",
        re.I,
    )
    if not m:
        return None
    raw = (m.group(1) or m.group(3) or "").replace(",", "")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    suf = (m.group(2) or m.group(4) or "m").lower()
    if suf in ("m", "mm", "million", "mil", ""):
        return val * 1_000_000.0
    if suf == "k":
        return val * 1_000.0
    return val


def turn_explicitly_sets_budget(query: str, turn: MissionProfile) -> bool:
    if turn.budget_usd_mid is not None or (turn.budget_range or "").strip():
        return True
    return bool(_BUDGET_EXPLICIT_RE.search(query or ""))


def turn_explicitly_sets_home_base(query: str, turn: MissionProfile) -> bool:
    if (turn.home_base or "").strip():
        return True
    try:
        from services.memory.mission_memory import detect_home_base

        return bool(detect_home_base(query or ""))
    except Exception:
        return False


def turn_explicitly_sets_ownership(query: str, turn: MissionProfile) -> bool:
    if turn.ownership_interest is not None or (turn.ownership_posture or OwnershipMode.UNDECIDED) != OwnershipMode.UNDECIDED:
        return True
    return bool(_OWNERSHIP_EXPLICIT_RE.search(query or ""))


def turn_explicitly_sets_runway_priority(query: str, turn: MissionProfile) -> bool:
    if turn.runway_priority not in (None, PriorityLevel.NONE, PriorityLevel.LOW):
        return True
    if turn.short_field_priority not in (None, PriorityLevel.NONE, PriorityLevel.LOW):
        return True
    if turn.mountain_airport_priority or turn.mountain_airports:
        return True
    return bool(_RUNWAY_EXPLICIT_RE.search(query or ""))


def turn_explicitly_sets_mission_type(query: str, turn: MissionProfile) -> bool:
    if turn.mission_category not in (None, MissionCategory.GENERAL, MissionCategory.POINT_TO_POINT):
        if turn.mission_category in (
            MissionCategory.COMPARISON,
            MissionCategory.ACQUISITION_ADVISORY,
            MissionCategory.OWNERSHIP_STRUCTURE,
            MissionCategory.DISPOSITION,
        ):
            return True
    return bool(_MISSION_TYPE_EXPLICIT_RE.search(query or ""))


def detect_session_field_overrides(query: str, turn: MissionProfile) -> Set[str]:
    """Fields the user explicitly set this turn — session must not overwrite."""
    from services.state.mission_validation import turn_explicitly_sets_passengers

    overrides: Set[str] = set()
    if turn_explicitly_sets_passengers(query):
        overrides.add("passengers")
    if turn_explicitly_sets_budget(query, turn):
        overrides.add("budget_usd")
    if turn_explicitly_sets_home_base(query, turn):
        overrides.add("home_base")
    if turn_explicitly_sets_ownership(query, turn):
        overrides.add("ownership")
    if turn_explicitly_sets_runway_priority(query, turn):
        overrides.add("runway")
    if turn_explicitly_sets_mission_type(query, turn):
        overrides.add("mission_type")
    return overrides


def _priority_str_to_level(value: str) -> PriorityLevel:
    v = (value or "none").strip().lower()
    if v == "high":
        return PriorityLevel.HIGH
    if v == "medium":
        return PriorityLevel.MEDIUM
    return PriorityLevel.NONE


def _apply_session_mission_type(profile: MissionProfile, mission_type: str) -> None:
    if mission_type == MissionType.COMPARISON:
        profile.mission_category = MissionCategory.COMPARISON
    elif mission_type == MissionType.ACQUISITION:
        profile.mission_category = MissionCategory.ACQUISITION_ADVISORY
    elif mission_type == MissionType.BUSINESS_TRAVEL and profile.mission_category in (
        None,
        MissionCategory.GENERAL,
    ):
        profile.mission_category = MissionCategory.POINT_TO_POINT


def merge_turn_with_session(
    turn: MissionProfile,
    session: MissionState,
    query: str = "",
) -> SessionMergeResult:
    """
    Fill gaps in the current turn from session memory.

    Turn-specific facts (routes, nonstop, westbound) are never inherited here —
    only the six session memory fields. Call after ``merge_memory`` when using both layers.
    """
    from services.memory.mission_memory import _copy_profile

    overrides = detect_session_field_overrides(query, turn)
    inherited: List[str] = []
    merged = _copy_profile(turn)

    if merged.passengers is None and session.passengers is not None and "passengers" not in overrides:
        merged.passengers = session.passengers
        inherited.append("passengers")

    if merged.budget_usd_mid is None and session.budget_usd is not None and "budget_usd" not in overrides:
        merged.budget_usd_mid = session.budget_usd
        inherited.append("budget_usd")

    if (
        not (merged.home_base or "").strip()
        and (session.home_base or "").strip()
        and "home_base" not in overrides
    ):
        merged.home_base = session.home_base
        inherited.append("home_base")

    if (
        merged.ownership_interest is None
        and session.priorities.ownership
        and session.priorities.ownership != "none"
        and "ownership" not in overrides
    ):
        try:
            merged.ownership_interest = OwnershipMode(session.priorities.ownership)
            inherited.append("ownership")
        except ValueError:
            pass

    if (
        merged.runway_priority in (None, PriorityLevel.NONE, PriorityLevel.LOW)
        and session.priorities.runway not in ("", "none")
        and "runway" not in overrides
    ):
        merged.runway_priority = _priority_str_to_level(session.priorities.runway)
        if merged.runway_priority == PriorityLevel.HIGH:
            merged.short_field_priority = PriorityLevel.HIGH
        inherited.append("runway")

    if session.mission_type and session.mission_type != MissionType.UNKNOWN and "mission_type" not in overrides:
        _apply_session_mission_type(merged, session.mission_type)
        inherited.append("mission_type")

    return SessionMergeResult(
        profile=merged,
        inherited_fields=inherited,
        overridden_fields=sorted(overrides),
    )


def build_consultant_mission_with_session(
    query: str,
    *,
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> Tuple[MissionState, MissionProfile, SessionMergeResult]:
    """
    Load session → extract turn → merge → advance persistent store.

    Preferred entry for ranked advisory paths that need cross-turn mission context.
    """
    from services.consultant.mission_state import MissionState as ConsultantMissionState
    from services.mission.mission_extractor import extract_mission
    from services.state.mission_state import (
        advance_persistent_mission_state,
        load_persistent_mission_state,
        persist_mission_state_patch,
        to_consultant_mission_state,
    )

    prior = load_persistent_mission_state(conversation_state, data_used)
    turn = extract_mission(query)
    updated = advance_persistent_mission_state(prior, turn, query)
    merge_result = merge_turn_with_session(turn, updated, query)
    profile = merge_result.profile

    from services.mission.adapters import mission_profile_to_state

    consultant: ConsultantMissionState = mission_profile_to_state(merge_result.profile)

    if isinstance(data_used, dict):
        data_used.update(persist_mission_state_patch(updated))
        data_used["session_mission_memory"] = {
            **merge_result.to_dict(),
            "session_snapshot": SessionMissionSnapshot.from_persistent(updated).to_dict(),
        }

    return consultant, profile, merge_result

"""
MissionState — structured mission container for consultant ranking/formatters.

Turn-isolated extraction lives in ``services.mission.mission_extractor``.
This module keeps normalization helpers and the legacy dataclass shape.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Accidental ``", ".join(str)`` corruption: "S, a, n,  , F, r, ..."
_CORRUPT_ROUTE_JOIN_RE = re.compile(r"(?:^|, )[A-Za-z], [a-z], ")


@dataclass
class MissionConstraint:
    """Legacy audit field; not populated by turn-isolated extraction."""

    key: str
    value: Any
    confidence: float = 0.5
    source: str = "inferred"
    turn_index: int = 0


@dataclass
class MissionStateSnapshot:
    """Legacy snapshot type (unused by turn-isolated flow)."""

    turn_index: int
    captured_at_utc: str
    fields: Dict[str, Any]
    confidences: Dict[str, float]


@dataclass
class MissionState:
    """Structured mission profile for one consultant turn (no cross-turn merge)."""

    passenger_count: Optional[int] = None
    passenger_min: Optional[int] = None
    passenger_max: Optional[int] = None
    cargo_required: Optional[bool] = None
    mission_type: Optional[str] = None
    routes: List[str] = field(default_factory=list)
    westbound: Optional[bool] = None
    eastbound: Optional[bool] = None
    reserves_requirement: Optional[str] = None
    runway_constraints: Optional[str] = None
    baggage_priority: Optional[str] = None
    ownership_goal: Optional[str] = None
    budget_usd: Optional[float] = None
    preferred_airports: List[str] = field(default_factory=list)
    cabin_priority: Optional[str] = None
    operating_cost_priority: Optional[str] = None
    acquisition_strategy: Optional[str] = None
    mountain_airport_requirement: Optional[bool] = None
    international_frequency: Optional[str] = None
    nonstop_requirement: Optional[bool] = None
    seasonal_constraints: Optional[str] = None
    constraints: List[MissionConstraint] = field(default_factory=list)
    snapshots: List[MissionStateSnapshot] = field(default_factory=list)
    turn_index: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passenger_count": self.passenger_count,
            "mission_type": self.mission_type,
            "routes": normalize_routes(self.routes),
            "westbound": self.westbound,
            "eastbound": self.eastbound,
            "reserves_requirement": self.reserves_requirement,
            "runway_constraints": self.runway_constraints,
            "baggage_priority": self.baggage_priority,
            "ownership_goal": self.ownership_goal,
            "budget_usd": self.budget_usd,
            "preferred_airports": list(self.preferred_airports),
            "cabin_priority": self.cabin_priority,
            "operating_cost_priority": self.operating_cost_priority,
            "acquisition_strategy": self.acquisition_strategy,
            "mountain_airport_requirement": self.mountain_airport_requirement,
            "international_frequency": self.international_frequency,
            "nonstop_requirement": self.nonstop_requirement,
            "seasonal_constraints": self.seasonal_constraints,
            "constraints": [],
            "snapshots": [],
            "turn_index": self.turn_index,
            "extraction_mode": "turn_isolated",
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> MissionState:
        if not isinstance(raw, dict):
            return cls()
        ms = cls()
        for k in (
            "passenger_count",
            "mission_type",
            "westbound",
            "eastbound",
            "reserves_requirement",
            "runway_constraints",
            "baggage_priority",
            "ownership_goal",
            "budget_usd",
            "cabin_priority",
            "operating_cost_priority",
            "acquisition_strategy",
            "mountain_airport_requirement",
            "international_frequency",
            "nonstop_requirement",
            "seasonal_constraints",
            "turn_index",
        ):
            if k in raw and raw[k] is not None:
                setattr(ms, k, raw[k])
        ms.routes = normalize_routes(raw.get("routes"))
        if isinstance(raw.get("preferred_airports"), list):
            ms.preferred_airports = [str(x).strip() for x in raw["preferred_airports"] if str(x).strip()]
        return ms


def normalize_routes(value: Any) -> List[str]:
    """Coerce routes to a list of leg labels."""
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s or _CORRUPT_ROUTE_JOIN_RE.search(s):
            return []
        if len(s) < 8 and "->" not in s and "→" not in s:
            return []
        return [s.replace("→", "->")]
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            part = str(item).strip() if item is not None else ""
            if not part or _CORRUPT_ROUTE_JOIN_RE.search(part):
                continue
            if len(part) < 8 and "->" not in part and "→" not in part:
                continue
            out.append(part.replace("→", "->"))
        return list(dict.fromkeys(out))[:12]
    return []


def format_routes_for_display(routes: Any) -> str:
    normalized = normalize_routes(routes)
    return ", ".join(normalized) if normalized else ""


def build_mission_from_current_turn(user_message: str) -> MissionState:
    """Canonical entry: extract mission from current user message only."""
    from services.mission import extract_mission, mission_profile_to_state

    return mission_profile_to_state(extract_mission(user_message))


def build_mission_with_session(
    user_message: str,
    *,
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> MissionState:
    """Extract current turn and merge session memory (passengers, budget, prefs)."""
    from services.state.session_mission_memory import build_consultant_mission_with_session

    mission, _, _ = build_consultant_mission_with_session(
        user_message,
        conversation_state=conversation_state,
        data_used=data_used,
    )
    return mission


def update_mission_state(
    prior: MissionState,
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    client_mission: Optional[Dict[str, Any]] = None,
) -> MissionState:
    """
    Deprecated accumulator — now turn-isolated (prior/history/client ignored).
    """
    if prior and (prior.passenger_count or prior.routes or prior.budget_usd):
        warnings.warn(
            "update_mission_state ignores prior mission state; use build_mission_from_current_turn",
            DeprecationWarning,
            stacklevel=2,
        )
    if history:
        warnings.warn(
            "update_mission_state ignores conversation history",
            DeprecationWarning,
            stacklevel=2,
        )
    return build_mission_from_current_turn(query)


def load_mission_state_from_data_used(data_used: Optional[Dict[str, Any]]) -> MissionState:
    """Read last serialized turn snapshot (display only — not merged into next turn)."""
    du = data_used if isinstance(data_used, dict) else {}
    raw = du.get("consultant_mission_state") or du.get("consultant_mission_profile")
    if isinstance(raw, dict):
        return MissionState.from_dict(raw)
    return MissionState()


def mission_state_confidence_summary(state: MissionState) -> Dict[str, float]:
    return {}

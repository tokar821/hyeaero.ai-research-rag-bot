"""
Typed aviation mission schemas — no stringly-typed route blobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PriorityLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OwnershipMode(str, Enum):
    FRACTIONAL = "fractional"
    FULL_OWNERSHIP = "full_ownership"
    CHARTER = "charter"
    UNDECIDED = "undecided"


class MissionCategory(str, Enum):
    COMPARISON = "comparison"
    ACQUISITION_ADVISORY = "acquisition_advisory"
    OWNERSHIP_STRUCTURE = "ownership_structure"
    ROUTE_FEASIBILITY = "route_feasibility"
    POINT_TO_POINT = "point_to_point"
    DISPOSITION = "disposition"
    SPECS = "specs"
    GENERAL = "general"


@dataclass(frozen=True)
class Route:
    origin: str
    destination: str

    def __post_init__(self) -> None:
        o = (self.origin or "").strip()
        d = (self.destination or "").strip()
        if not o or not d:
            raise ValueError("Route requires non-empty origin and destination")
        object.__setattr__(self, "origin", o)
        object.__setattr__(self, "destination", d)

    def label(self) -> str:
        return f"{self.origin} -> {self.destination}"

    def to_dict(self) -> Dict[str, str]:
        return {"origin": self.origin, "destination": self.destination}

    @classmethod
    def from_label(cls, label: str) -> Optional[Route]:
        """Parse ``Origin -> Destination`` when already validated."""
        s = (label or "").strip().replace("→", "->")
        if "->" not in s:
            return None
        left, right = s.split("->", 1)
        o, d = left.strip(), right.strip()
        if not o or not d:
            return None
        return cls(origin=o, destination=d)


@dataclass
class PassengerDistribution:
    """Variable passenger load — ``passengers`` remains planning_load for legacy paths."""

    min_pax: Optional[int] = None
    max_pax: Optional[int] = None
    planning_load: Optional[int] = None
    typical_pax: Optional[int] = None
    cargo_required: bool = False
    variance_note: str = ""

    @property
    def is_variable(self) -> bool:
        return (
            self.min_pax is not None
            and self.max_pax is not None
            and self.max_pax > self.min_pax
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_pax": self.min_pax,
            "max_pax": self.max_pax,
            "planning_load": self.planning_load,
            "typical_pax": self.typical_pax,
            "cargo_required": self.cargo_required,
            "variance_note": self.variance_note,
            "is_variable": self.is_variable,
        }


@dataclass
class MissionProfile:
    """Deterministic typed mission snapshot for one user turn."""

    passengers: Optional[int] = None
    passenger_distribution: Optional[PassengerDistribution] = None
    routes: List[Route] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    nonstop_required: bool = False
    westbound_sensitive: bool = False
    eastbound_sensitive: bool = False
    cabin_priority: PriorityLevel = PriorityLevel.NONE
    operating_cost_priority: PriorityLevel = PriorityLevel.NONE
    runway_priority: PriorityLevel = PriorityLevel.NONE
    baggage_priority: PriorityLevel = PriorityLevel.NONE
    ownership_interest: Optional[OwnershipMode] = None
    mission_category: Optional[MissionCategory] = None
    budget_range: Optional[str] = None
    budget_usd_mid: Optional[float] = None
    preferred_airports: List[str] = field(default_factory=list)
    seasonal_note: Optional[str] = None
    mountain_airports: bool = False
    mountain_airport_priority: bool = False
    reserves_requirement: Optional[str] = None
    nbaa_reserve_required: bool = False
    short_field_priority: PriorityLevel = PriorityLevel.NONE
    airport_constraints: List[str] = field(default_factory=list)
    ownership_posture: Optional[OwnershipMode] = None
    mission_frequency: Optional[str] = None
    international_ops: bool = False
    home_base: Optional[str] = None
    fleet_preferences: List[str] = field(default_factory=list)
    # Mission-understanding hints for gating (set by apply_understanding_to_profile)
    planning_band_ceiling: Optional[str] = None
    international_jet_floor: bool = False
    balanced_cost_dispatch: bool = False

    def route_labels(self) -> List[str]:
        return [r.label() for r in self.routes]

    def to_dict(self) -> Dict[str, Any]:
        """Structured serialization for API / observability (typed fields only)."""
        return {
            "schema_version": 3,
            "extraction_mode": "turn_isolated",
            "passengers": self.passengers,
            "passenger_distribution": (
                self.passenger_distribution.to_dict()
                if self.passenger_distribution
                else None
            ),
            "routes": [r.to_dict() for r in self.routes],
            "regions": list(self.regions),
            "nonstop_required": self.nonstop_required,
            "westbound_sensitive": self.westbound_sensitive,
            "eastbound_sensitive": self.eastbound_sensitive,
            "cabin_priority": self.cabin_priority.value,
            "operating_cost_priority": self.operating_cost_priority.value,
            "runway_priority": self.runway_priority.value,
            "baggage_priority": self.baggage_priority.value,
            "ownership_interest": (
                self.ownership_interest.value if self.ownership_interest else None
            ),
            "mission_category": (
                self.mission_category.value if self.mission_category else None
            ),
            "budget_range": self.budget_range,
            "budget_usd_mid": self.budget_usd_mid,
            "preferred_airports": list(self.preferred_airports),
            "seasonal_note": self.seasonal_note,
            "mountain_airports": self.mountain_airports,
            "mountain_airport_priority": self.mountain_airport_priority,
            "reserves_requirement": self.reserves_requirement,
            "nbaa_reserve_required": self.nbaa_reserve_required,
            "short_field_priority": self.short_field_priority.value,
            "airport_constraints": list(self.airport_constraints),
            "ownership_posture": (
                (self.ownership_posture or self.ownership_interest).value
                if (self.ownership_posture or self.ownership_interest)
                else None
            ),
            "mission_frequency": self.mission_frequency,
            "international_ops": self.international_ops,
            "home_base": self.home_base,
            "fleet_preferences": list(self.fleet_preferences),
            "memory_merge_applied": False,
        }

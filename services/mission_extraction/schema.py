"""
Strict mission extraction output schema — operational requirements only.

This layer must never recommend aircraft. Validation uses Pydantic (Python equivalent
of Zod); see ``frontend/src/lib/mission-extraction/schema.ts`` for the Zod mirror.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

MissionType = Literal[
    "point_to_point",
    "multi_city",
    "comparison",
    "ownership",
    "feasibility",
    "acquisition",
    "general",
]

PriorityLevel = Literal["low", "medium", "high"]

OwnershipInterest = Literal["fractional", "full_ownership", "charter", "undecided"]

AircraftCategory = Literal[
    "light_jet",
    "midsize",
    "super_midsize",
    "large_cabin",
    "ultra_long_range",
    "turboprop",
    "regional_utility",
]


class MissionExtractionResult(BaseModel):
    """Structured operational requirements extracted from one user message."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    passengers: Optional[int] = Field(default=None, ge=1, le=24)
    origin: Optional[str] = None
    destination: Optional[List[str]] = None
    mission_type: Optional[MissionType] = None
    nonstop_required: Optional[bool] = None
    westbound_sensitive: Optional[bool] = None
    winter_ops: Optional[bool] = None
    runway_priority: Optional[PriorityLevel] = None
    operating_cost_priority: Optional[PriorityLevel] = None
    cabin_priority: Optional[PriorityLevel] = None
    baggage_priority: Optional[PriorityLevel] = None
    ownership_interest: Optional[OwnershipInterest] = None
    annual_hours: Optional[int] = Field(default=None, ge=1, le=2000)
    budget: Optional[float] = Field(default=None, ge=0)
    hot_high_ops: Optional[bool] = None
    mountain_airports: Optional[bool] = None
    short_runway_ops: Optional[bool] = None
    international_ops: Optional[bool] = None
    transatlantic: Optional[bool] = None
    transpacific: Optional[bool] = None
    south_america: Optional[bool] = None
    caribbean: Optional[bool] = None
    europe: Optional[bool] = None
    asia: Optional[bool] = None
    inferred_aircraft_category: Optional[AircraftCategory] = None

    @field_validator("destination")
    @classmethod
    def _destinations_non_empty(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        cleaned = [str(x).strip() for x in v if str(x).strip()]
        return cleaned or None

    @field_validator("origin")
    @classmethod
    def _origin_non_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s or None


def validate_extraction_payload(data: Union[dict, MissionExtractionResult]) -> MissionExtractionResult:
    """Validate arbitrary dict or model — raises ``ValidationError`` on invalid input."""
    if isinstance(data, MissionExtractionResult):
        return data
    return MissionExtractionResult.model_validate(data)

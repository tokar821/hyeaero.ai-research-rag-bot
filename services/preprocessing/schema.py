"""
Pre-processed mission schema — every field is either a concrete value or ``UNKNOWN``.

Never use null for "not stated"; use ``UNKNOWN``. Route fields (origin, destination)
must only be set from validated extraction — never inferred from region keywords alone.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

UNKNOWN = "UNKNOWN"

PriorityValue = Union[Literal["low", "medium", "high"], Literal["UNKNOWN"]]
BoolValue = Union[bool, Literal["UNKNOWN"]]
PassengersValue = Union[int, Literal["UNKNOWN"]]
BudgetValue = Union[float, Literal["UNKNOWN"]]
HoursValue = Union[int, Literal["UNKNOWN"]]
OwnershipValue = Union[
    Literal["fractional", "full_ownership", "charter", "undecided"],
    Literal["UNKNOWN"],
]
StringValue = Union[str, Literal["UNKNOWN"]]


class PreprocessedMission(BaseModel):
    """Structured mission JSON emitted before recommendation / LLM narration."""

    model_config = ConfigDict(extra="forbid")

    passengers: PassengersValue = UNKNOWN
    origin: StringValue = UNKNOWN
    destination: StringValue = UNKNOWN
    nonstop_required: BoolValue = UNKNOWN
    westbound: BoolValue = UNKNOWN
    winter_operation: BoolValue = UNKNOWN
    runway_priority: PriorityValue = UNKNOWN
    operating_cost_priority: PriorityValue = UNKNOWN
    luxury_priority: PriorityValue = UNKNOWN
    budget: BudgetValue = UNKNOWN
    annual_hours: HoursValue = UNKNOWN
    ownership_interest: OwnershipValue = UNKNOWN
    mountain_airport: BoolValue = UNKNOWN
    international: BoolValue = UNKNOWN
    transatlantic: BoolValue = UNKNOWN
    transpacific: BoolValue = UNKNOWN

    # Audit — not part of user-facing mission facts
    route_evidence: str = Field(
        default="none",
        description="validated_route | explicit_from_to | none",
    )
    extraction_notes: list[str] = Field(default_factory=list)

    @field_validator(
        "passengers",
        mode="before",
    )
    @classmethod
    def _passengers_bounds(cls, v: Any) -> Any:
        if v == UNKNOWN or v is None:
            return UNKNOWN
        try:
            n = int(v)
        except (TypeError, ValueError):
            return UNKNOWN
        if 1 <= n <= 24:
            return n
        return UNKNOWN

    @field_validator("budget", mode="before")
    @classmethod
    def _budget_non_negative(cls, v: Any) -> Any:
        if v == UNKNOWN or v is None:
            return UNKNOWN
        try:
            f = float(v)
        except (TypeError, ValueError):
            return UNKNOWN
        return f if f >= 0 else UNKNOWN

    @field_validator("annual_hours", mode="before")
    @classmethod
    def _hours_bounds(cls, v: Any) -> Any:
        if v == UNKNOWN or v is None:
            return UNKNOWN
        try:
            n = int(v)
        except (TypeError, ValueError):
            return UNKNOWN
        if 1 <= n <= 2000:
            return n
        return UNKNOWN

    def to_public_dict(self) -> Dict[str, Any]:
        """Mission fields only (no audit keys) for client / prompt injection."""
        d = self.model_dump(mode="json")
        d.pop("route_evidence", None)
        d.pop("extraction_notes", None)
        return d

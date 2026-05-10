"""
Pydantic schemas for the conversation continuity / intent layering engine.

Serialized under ``data_used["consultant_conversation_state"]["continuity"]``
(or legacy ``consultant_continuity_state``) for client round-trip.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class LockedEntityType(str, Enum):
    TAIL = "tail"
    AIRCRAFT_MODEL = "aircraft_model"
    SERIAL = "serial"
    LISTING = "listing"


class LockedEntity(BaseModel):
    type: LockedEntityType
    value: str = Field(..., min_length=1)
    locked_at_turn_hint: Optional[str] = Field(
        None, description="Fingerprint of query that created the lock (optional)"
    )


class AircraftCategory(str, Enum):
    UNKNOWN = "unknown"
    VLJ = "vlj"
    LIGHT = "light_jet"
    MIDSIZE = "midsize"
    SUPER_MID = "super_midsize"
    LARGE = "large_cabin"
    ULR = "ultra_long_range"


class BuyerDirection(BaseModel):
    size: Optional[Literal["larger", "smaller", "same"]] = None
    luxury: Optional[Literal["higher", "lower", "same"]] = None
    budget_usd_approx: Optional[float] = Field(
        None,
        description="Approximate acquisition budget in USD when parsable",
    )


class RefinementInterpretation(BaseModel):
    """Structured output from the deterministic refinement interpreter."""

    type: str = Field(
        ...,
        description=(
            "e.g. size_upgrade, style_shift, view_change, budget_shift, comparison_anchor, "
            "lifestyle_inference, ambiguous_followup, explicit_reset, none"
        ),
    )
    reference_aircraft: Optional[str] = None
    reference_tail: Optional[str] = None
    preserve_traits: List[str] = Field(default_factory=list)
    remove_traits: List[str] = Field(default_factory=list)
    add_traits: List[str] = Field(default_factory=list)
    requested_view: Optional[str] = None
    inherit_entity: bool = True
    inferred_style_tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class ContinuityResponseMode(str, Enum):
    VISUAL_ONLY = "visual_only"
    SHORT_CAPTION = "short_caption"
    CONSULTANT_MODE = "consultant_mode"
    COMPARISON_MODE = "comparison_mode"
    TECHNICAL_MODE = "technical_mode"


class ConversationContinuityState(BaseModel):
    """
    Canonical continuity snapshot for multi-turn refinement.

    Kept JSON-friendly for ``data_used`` and Next.js echoes.
    """

    schema_version: int = Field(default=1, ge=1, le=16)
    locked_entity: Optional[LockedEntity] = None
    current_aircraft: Optional[str] = None
    current_tail: Optional[str] = None
    current_category: AircraftCategory = AircraftCategory.UNKNOWN
    aircraft_evolution: List[str] = Field(
        default_factory=list,
        description="Most recent last; bounded server-side",
    )
    style_preferences: List[str] = Field(default_factory=list)
    negative_preferences: List[str] = Field(default_factory=list)
    buyer_direction: BuyerDirection = Field(default_factory=BuyerDirection)
    last_requested_view: Optional[str] = None
    response_mode: ContinuityResponseMode = ContinuityResponseMode.CONSULTANT_MODE
    last_refinement: Optional[RefinementInterpretation] = None
    contextual_intent_tags: List[str] = Field(default_factory=list)
    drift_flags: List[str] = Field(default_factory=list)


def continuity_state_from_dict(raw: Optional[Dict[str, Any]]) -> ConversationContinuityState:
    if not isinstance(raw, dict) or not raw:
        return ConversationContinuityState()
    try:
        return ConversationContinuityState.model_validate(raw)
    except Exception:
        return ConversationContinuityState()

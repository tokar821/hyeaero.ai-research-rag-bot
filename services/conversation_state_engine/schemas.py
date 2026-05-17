"""
Centralized conversational memory for multi-turn luxury aviation advisory.

Serialized under ``consultant_conversation_state["conversation_memory"]``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResponseMode(str, Enum):
    CONSULTANT = "consultant_mode"
    IMAGE_SHOWCASE = "image_showcase"
    SHORT_CAPTION = "short_caption"
    VISUAL_ONLY = "visual_only"
    COMPARISON = "comparison_mode"
    TECHNICAL = "technical_mode"


class ConversationGoal(str, Enum):
    UNKNOWN = "unknown"
    EXPLORE = "explore_model"
    VISUAL_GALLERY = "visual_gallery"
    REFINEMENT = "refinement"
    COMPARE = "compare_models"
    MISSION = "mission_advisory"
    OWNERSHIP = "ownership_research"
    SHOPPING = "shopping"


class AircraftCategory(str, Enum):
    UNKNOWN = "unknown"
    VLJ = "vlj"
    LIGHT = "light_jet"
    MIDSIZE = "midsize"
    SUPER_MID = "super_midsize"
    LARGE = "large_cabin"
    ULR = "ultra_long_range"


class ConversationMemoryState(BaseModel):
    """Canonical structured memory across turns."""

    schema_version: int = Field(default=1, ge=1, le=16)
    turn_index: int = Field(default=0, ge=0)

    active_aircraft: Optional[str] = None
    active_tail: Optional[str] = None
    active_category: AircraftCategory = AircraftCategory.UNKNOWN
    active_topic: Optional[str] = None
    response_mode: ResponseMode = ResponseMode.CONSULTANT
    aesthetic_preferences: List[str] = Field(default_factory=list)
    negative_preferences: List[str] = Field(default_factory=list)
    active_budget_usd: Optional[float] = None
    active_budget_label: Optional[str] = None
    active_mission: Optional[str] = None
    comparison_target: Optional[str] = None
    conversation_goal: ConversationGoal = ConversationGoal.UNKNOWN
    last_visual_context: Optional[str] = None

    aircraft_evolution: List[str] = Field(default_factory=list)
    memory_stack: List[str] = Field(
        default_factory=list,
        description="Ordered priority keys (highest first) still active this turn",
    )
    decayed_fields: List[str] = Field(default_factory=list)
    reinforced_fields: List[str] = Field(default_factory=list)

    field_turns: Dict[str, int] = Field(
        default_factory=dict,
        description="Last turn_index when each logical field was set/reinforced",
    )


def memory_from_dict(raw: Optional[Dict[str, Any]]) -> ConversationMemoryState:
    if not isinstance(raw, dict) or not raw:
        return ConversationMemoryState()
    try:
        return ConversationMemoryState.model_validate(raw)
    except Exception:
        return ConversationMemoryState()

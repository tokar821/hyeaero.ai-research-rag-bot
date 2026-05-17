"""
Persistent conversational intent — JSON-friendly state for multi-turn advisory threads.

Serialized under ``data_used["intent_persistence"]`` and nested in
``consultant_conversation_state["intent_persistence"]`` for client echo.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConversationGoal(str, Enum):
    UNKNOWN = "unknown"
    EXPLORE_MODEL = "explore_model"
    VISUAL_GALLERY = "visual_gallery"
    COMPARE_MODELS = "compare_models"
    MISSION_ADVISORY = "mission_advisory"
    OWNERSHIP_RESEARCH = "ownership_research"
    REFINEMENT = "refinement"


class IntentResponseMode(str, Enum):
    CONSULTANT_MODE = "consultant_mode"
    IMAGE_SHOWCASE = "image_showcase"
    SHORT_CAPTION = "short_caption"
    VISUAL_ONLY = "visual_only"
    COMPARISON_MODE = "comparison_mode"
    TECHNICAL_MODE = "technical_mode"


class RoutingDecision(str, Enum):
    """How retrieval / tools should treat this turn."""

    FRESH_RETRIEVAL = "fresh_retrieval"
    INHERIT_CONTEXT = "inherit_context"
    IMAGE_SHOWCASE_CONTINUATION = "image_showcase_continuation"
    REFINEMENT_CONTINUATION = "refinement_continuation"


class PersistentIntentState(BaseModel):
    schema_version: int = Field(default=1, ge=1, le=16)
    active_aircraft: Optional[str] = None
    active_tail: Optional[str] = None
    active_topic: Optional[str] = None
    active_visual_focus: Optional[str] = None
    active_budget_usd: Optional[float] = None
    response_mode: IntentResponseMode = IntentResponseMode.CONSULTANT_MODE
    aesthetic_preferences: List[str] = Field(default_factory=list)
    negative_preferences: List[str] = Field(default_factory=list)
    comparison_target: Optional[str] = None
    current_conversation_goal: ConversationGoal = ConversationGoal.UNKNOWN
    last_refinement_type: Optional[str] = None
    standalone_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


def intent_state_from_dict(raw: Optional[Dict[str, Any]]) -> PersistentIntentState:
    if not isinstance(raw, dict) or not raw:
        return PersistentIntentState()
    try:
        return PersistentIntentState.model_validate(raw)
    except Exception:
        return PersistentIntentState()

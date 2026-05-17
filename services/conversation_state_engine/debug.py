"""Structured debug logging for conversation memory."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .schemas import ConversationMemoryState

logger = logging.getLogger(__name__)


def log_state_transition(
    *,
    previous: Dict[str, Any],
    resolved: ConversationMemoryState,
    inherited_fields: List[str],
    decayed_fields: List[str],
    routing_hint: str = "",
) -> None:
    snapshot = {
        "active_aircraft": resolved.active_aircraft,
        "active_tail": resolved.active_tail,
        "active_category": resolved.active_category.value,
        "response_mode": resolved.response_mode.value,
        "conversation_goal": resolved.conversation_goal.value,
        "last_visual_context": resolved.last_visual_context,
        "aesthetic_preferences": resolved.aesthetic_preferences[-6:],
        "memory_stack": resolved.memory_stack[:8],
    }
    logger.info(
        "conversation_state_engine previous_state=%s resolved_state=%s "
        "inherited_fields=%s decayed_fields=%s routing_hint=%s turn_index=%s",
        _brief(previous),
        snapshot,
        inherited_fields,
        decayed_fields,
        routing_hint or "-",
        resolved.turn_index,
    )


def _brief(d: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "active_aircraft",
        "active_tail",
        "response_mode",
        "conversation_goal",
        "last_visual_context",
        "turn_index",
    )
    return {k: d.get(k) for k in keys if k in d}

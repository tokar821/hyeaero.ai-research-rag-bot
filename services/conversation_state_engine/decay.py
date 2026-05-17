"""Safe decay of stale conversational memory."""

from __future__ import annotations

import copy
from typing import List, Set

from .priority import MEMORY_PRIORITY_STACK
from .schemas import ConversationMemoryState


def apply_memory_decay(state: ConversationMemoryState, *, explicit_reset: bool) -> List[str]:
    """
    Expire low-priority fields that were not reinforced recently.

    Returns list of field keys that were cleared this turn.
    """
    if explicit_reset:
        return []

    decayed: List[str] = []
    turn = int(state.turn_index or 0)
    if turn < 2:
        return decayed

    fts = dict(state.field_turns or {})
    protected: Set[str] = set(state.reinforced_fields or [])

    for key, ttl in MEMORY_PRIORITY_STACK:
        if key in protected:
            continue
        last = fts.get(key)
        if last is None:
            continue
        if turn - int(last) <= int(ttl):
            continue

        if key == "active_aircraft" and state.active_tail:
            continue
        if key == "aesthetic_preferences" and not state.aesthetic_preferences:
            continue

        if key == "active_aircraft":
            state.active_aircraft = None
        elif key == "active_tail":
            state.active_tail = None
        elif key == "response_mode":
            state.response_mode = state.response_mode.CONSULTANT
        elif key == "aesthetic_preferences":
            state.aesthetic_preferences = []
        elif key == "negative_preferences":
            state.negative_preferences = []
        elif key == "active_category":
            state.active_category = state.active_category.UNKNOWN
        elif key == "conversation_goal":
            state.conversation_goal = state.conversation_goal.UNKNOWN
        elif key == "active_budget_usd":
            state.active_budget_usd = None
            state.active_budget_label = None
        elif key == "active_mission":
            state.active_mission = None
        elif key == "comparison_target":
            state.comparison_target = None
        elif key == "active_topic":
            state.active_topic = None
        elif key == "last_visual_context":
            state.last_visual_context = None
        elif key == "aircraft_evolution":
            state.aircraft_evolution = []

        fts.pop(key, None)
        decayed.append(key)

    state.field_turns = fts
    state.decayed_fields = list(dict.fromkeys((state.decayed_fields or []) + decayed))[-24:]
    return decayed

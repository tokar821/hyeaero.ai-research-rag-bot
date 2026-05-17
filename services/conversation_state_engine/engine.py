"""
Conversation State Engine — centralized multi-turn memory orchestration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .debug import log_state_transition
from .decay import apply_memory_decay
from .priority import active_memory_stack
from .prompt_block import format_memory_prompt_block
from .schemas import ConversationMemoryState, memory_from_dict
from .update_rules import apply_update_rules

_RESET_RE = re.compile(
    r"\b(start\s+over|new\s+topic|forget\s+(?:everything|that)|different\s+(?:subject|topic)|reset\b)\b",
    re.I,
)


@dataclass
class ConversationStateBundle:
    state: ConversationMemoryState
    serialized: Dict[str, Any]
    prompt_block: str
    previous_snapshot: Dict[str, Any]
    inherited_fields: List[str]
    decayed_fields: List[str]


def _load_previous(client_state: Optional[Dict[str, Any]]) -> ConversationMemoryState:
    if not isinstance(client_state, dict):
        return ConversationMemoryState()
    mem = client_state.get("conversation_memory")
    if isinstance(mem, dict):
        return memory_from_dict(mem)
    return _legacy_to_memory(client_state)


def _legacy_to_memory(leg: Dict[str, Any]) -> ConversationMemoryState:
    """Bootstrap memory from legacy flat consultant_conversation_state keys."""
    m = ConversationMemoryState()
    m.active_aircraft = (leg.get("current_aircraft_reference") or "").strip() or None
    m.last_visual_context = (leg.get("current_visual_intent") or "").strip() or None
    m.active_budget_label = (leg.get("current_budget") or "").strip() or None
    m.active_mission = (leg.get("current_mission") or "").strip() or None
    if leg.get("user_style"):
        m.aesthetic_preferences = [str(leg["user_style"]).strip()]
    ip = leg.get("intent_persistence")
    if isinstance(ip, dict):
        return memory_from_dict({**m.model_dump(mode="json"), **ip})
    return m


def _diff_inherited(prev: Dict[str, Any], cur: ConversationMemoryState) -> List[str]:
    out: List[str] = []
    cur_d = cur.model_dump(mode="json")
    for key in (
        "active_aircraft",
        "active_tail",
        "active_category",
        "response_mode",
        "conversation_goal",
        "last_visual_context",
        "active_budget_usd",
        "active_mission",
        "comparison_target",
    ):
        if prev.get(key) != cur_d.get(key) and cur_d.get(key) is not None:
            out.append(key)
    if cur.aesthetic_preferences and cur.aesthetic_preferences != prev.get("aesthetic_preferences"):
        out.append("aesthetic_preferences")
    return list(dict.fromkeys(out))[:24]


def run_conversation_state_turn(
    *,
    query: str,
    client_conversation_state: Optional[Dict[str, Any]],
    continuity_serialized: Optional[Dict[str, Any]] = None,
    intent_resolved: Optional[Dict[str, Any]] = None,
    refinement_type: str = "none",
    entity_models: Optional[List[str]] = None,
    user_wants_gallery: bool = False,
    mission_hint: Optional[str] = None,
    routing_hint: str = "",
) -> ConversationStateBundle:
    """
    Single turn: load prior memory → apply updates → decay stale fields → log → return bundle.
    """
    prev_state = _load_previous(client_conversation_state)
    prev_snapshot = prev_state.model_dump(mode="json")

    explicit_reset = bool(_RESET_RE.search(query or "")) or refinement_type == "explicit_reset"
    if explicit_reset:
        state = ConversationMemoryState(turn_index=0)
    else:
        state = ConversationMemoryState.model_validate(prev_snapshot)
        state.turn_index = int(state.turn_index or 0) + 1
        state.decayed_fields = []
        state.reinforced_fields = []

    apply_update_rules(
        state,
        query=query,
        refinement_type=refinement_type,
        continuity=continuity_serialized,
        intent_resolved=intent_resolved,
        legacy_state=client_conversation_state if isinstance(client_conversation_state, dict) else None,
        entity_models=entity_models,
        user_wants_gallery=user_wants_gallery,
        mission_hint=mission_hint,
    )

    decayed = apply_memory_decay(state, explicit_reset=explicit_reset)
    inherited = _diff_inherited(prev_snapshot, state)

    has_map = {
        "active_aircraft": bool(state.active_aircraft),
        "active_tail": bool(state.active_tail),
        "response_mode": state.response_mode.value != "consultant_mode",
        "aesthetic_preferences": bool(state.aesthetic_preferences),
        "negative_preferences": bool(state.negative_preferences),
        "active_category": state.active_category.value != "unknown",
        "conversation_goal": state.conversation_goal.value != "unknown",
        "active_budget_usd": state.active_budget_usd is not None,
        "active_mission": bool(state.active_mission),
        "comparison_target": bool(state.comparison_target),
        "active_topic": bool(state.active_topic),
        "last_visual_context": bool(state.last_visual_context),
        "aircraft_evolution": bool(state.aircraft_evolution),
    }
    state.memory_stack = active_memory_stack(has_map)

    log_state_transition(
        previous=prev_snapshot,
        resolved=state,
        inherited_fields=inherited,
        decayed_fields=decayed,
        routing_hint=routing_hint,
    )

    return ConversationStateBundle(
        state=state,
        serialized=state.model_dump(mode="json"),
        prompt_block=format_memory_prompt_block(state),
        previous_snapshot=prev_snapshot,
        inherited_fields=inherited,
        decayed_fields=decayed,
    )


def sync_legacy_flat_fields(memory: ConversationMemoryState) -> Dict[str, Any]:
    """Map canonical memory → legacy keys for older clients."""
    mode_map = {
        "consultant_mode": "browsing",
        "image_showcase": "shopping",
        "comparison_mode": "comparing",
    }
    return {
        "user_style": memory.aesthetic_preferences[0] if memory.aesthetic_preferences else None,
        "current_aircraft_reference": memory.active_aircraft,
        "current_visual_intent": memory.last_visual_context,
        "current_budget": memory.active_budget_label
        or (f"${int(memory.active_budget_usd / 1_000_000)}M" if memory.active_budget_usd else None),
        "current_mission": memory.active_mission,
        "current_passenger_count": None,
        "current_cabin_preference": None,
        "conversation_mode": mode_map.get(memory.response_mode.value, "browsing"),
    }

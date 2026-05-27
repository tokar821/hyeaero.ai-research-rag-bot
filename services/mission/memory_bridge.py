"""
Bridge turn-isolated extraction with optional safe mission memory.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from services.memory.mission_memory import (
    MissionMemory,
    advance_memory,
    detect_home_base,
    load_mission_memory,
    merge_memory,
    mission_memory_enabled,
)
from services.mission.mission_extractor import extract_mission
from services.mission.models import MissionProfile


def extract_mission_with_memory(
    user_message: str,
    *,
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    memory: Optional[MissionMemory] = None,
) -> Tuple[MissionProfile, MissionProfile, MissionMemory]:
    """
    Extract current turn, optionally merge stable memory, advance memory for next turn.

    Returns ``(turn_only, merged_for_ranking, next_memory)``.
    """
    turn_only = extract_mission(user_message)
    if not (turn_only.home_base or "").strip():
        home = detect_home_base(user_message)
        if home:
            turn_only.home_base = home
    mem = memory if memory is not None else load_mission_memory(conversation_state, data_used)

    if not mission_memory_enabled():
        next_mem = advance_memory(mem, turn_only, user_message=user_message)
        return turn_only, turn_only, next_mem

    merged = merge_memory(turn_only, mem)
    try:
        from services.state.mission_state import load_persistent_mission_state
        from services.state.session_mission_memory import merge_turn_with_session

        session = load_persistent_mission_state(conversation_state, data_used)
        if session.turn_count or any(
            (
                session.passengers,
                session.budget_usd,
                session.home_base,
                session.priorities.ownership not in ("", "none"),
                session.priorities.runway not in ("", "none"),
                session.mission_type != "unknown",
            )
        ):
            session_merge = merge_turn_with_session(merged, session, user_message)
            merged = session_merge.profile
            if isinstance(data_used, dict):
                data_used["session_mission_memory"] = session_merge.to_dict()
    except Exception:
        pass
    next_mem = advance_memory(mem, turn_only, user_message=user_message)
    return turn_only, merged, next_mem

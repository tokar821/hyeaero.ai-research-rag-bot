"""Context drift prevention — prompt fragments + flags."""

from __future__ import annotations

from typing import List

from .schemas import ConversationContinuityState, LockedEntity, LockedEntityType


def continuity_drift_flags(state: ConversationContinuityState, query_empty: bool) -> List[str]:
    flags: List[str] = []
    if state.locked_entity and state.locked_entity.type == LockedEntityType.TAIL:
        flags.append("entity_locked_tail")
    if (state.aircraft_evolution or []):
        flags.append("tracked_aircraft_evolution")
    if (state.style_preferences or state.negative_preferences):
        flags.append("style_memory_active")
    if state.last_requested_view:
        flags.append("visual_follow_through")
    if query_empty:
        flags.append("empty_query_guard")
    return flags


DRIFT_CONTRACT = """**Continuity contract (mandatory — internal routing):**
- Do **not** open with generic \"How can I help?\" introductions; continue the briefing thread.
- Keep the **same focal aircraft/tail/listing** the user anchored unless they clearly switch subjects.
- If `entity_locked_tail` is active, anchor cabin/cockpit/bedroom questions to **that tail** unless they name another.
- If `visual_follow_through` is active with gallery images, prioritize **captions ≤ 2 short sentences**, **avoid raw URL dumps**, emphasize what the visuals show instead of encyclopedic prose.
"""

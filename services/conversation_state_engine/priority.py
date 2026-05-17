"""Memory priority stack — higher entries resist decay longer."""

from __future__ import annotations

from typing import Dict, List, Tuple

# (field_key, ttl_turns_without_reinforcement)
MEMORY_PRIORITY_STACK: List[Tuple[str, int]] = [
    ("active_aircraft", 24),
    ("active_tail", 24),
    ("response_mode", 14),
    ("aesthetic_preferences", 12),
    ("negative_preferences", 12),
    ("active_category", 10),
    ("conversation_goal", 10),
    ("active_budget_usd", 10),
    ("active_mission", 10),
    ("comparison_target", 8),
    ("active_topic", 8),
    ("last_visual_context", 6),
    ("aircraft_evolution", 12),
]

PRIORITY_RANK: Dict[str, int] = {k: i for i, (k, _) in enumerate(MEMORY_PRIORITY_STACK)}


def active_memory_stack(state_has: Dict[str, bool]) -> List[str]:
    """Return priority-ordered keys that currently hold values."""
    out: List[str] = []
    for key, _ttl in MEMORY_PRIORITY_STACK:
        if state_has.get(key):
            out.append(key)
    return out

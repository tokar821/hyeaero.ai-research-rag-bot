"""
Conversation Continuity Layer — multi-turn orchestration for the aviation consultant.

Public entry: :func:`run_continuity_turn`.
"""

from __future__ import annotations

from .orchestrator import ContinuityTurnBundle, run_continuity_turn
from .schemas import ConversationContinuityState, continuity_state_from_dict

__all__ = [
    "ContinuityTurnBundle",
    "ConversationContinuityState",
    "continuity_state_from_dict",
    "run_continuity_turn",
]

"""
Conversation State Engine — persistent structured memory for multi-turn advisory.

Public entry: :func:`run_conversation_state_turn`.
"""

from __future__ import annotations

from .engine import ConversationStateBundle, run_conversation_state_turn, sync_legacy_flat_fields
from .schemas import ConversationMemoryState, memory_from_dict

__all__ = [
    "ConversationMemoryState",
    "ConversationStateBundle",
    "memory_from_dict",
    "run_conversation_state_turn",
    "sync_legacy_flat_fields",
]

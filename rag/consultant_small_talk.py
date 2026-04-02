"""Backward-compatible shim — implementation is in :mod:`rag.conversation_guard`."""

from __future__ import annotations

from rag.conversation_guard import (
    ALL_GREETING_REPLIES,
    GREETING_REPLY,
    IDENTITY_REPLY,
    ConversationGuardResult,
    ConversationMessageType,
    consultant_small_talk_reply,
    evaluate_conversation_guard,
    query_has_aviation_signals,
)

__all__ = [
    "ALL_GREETING_REPLIES",
    "GREETING_REPLY",
    "IDENTITY_REPLY",
    "ConversationGuardResult",
    "ConversationMessageType",
    "consultant_small_talk_reply",
    "evaluate_conversation_guard",
    "query_has_aviation_signals",
]

"""Greeting and identity turns: no retrieval, fixed professional replies."""

from __future__ import annotations

import re
from typing import Optional

_GREETING_EXACT = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "hey there",
        "hi there",
        "hello there",
        "good morning",
        "good afternoon",
        "good evening",
    }
)


def consultant_small_talk_reply(query: str) -> Optional[str]:
    """
    If the latest message is only a greeting or a short identity question, return the
    canonical consultant reply and **skip** SQL / vector / Tavily.
    """
    raw = (query or "").strip()
    if not raw:
        return None
    low = raw.lower()
    greet_norm = low.rstrip("!?.… ").strip()

    if greet_norm in _GREETING_EXACT or (
        len(raw) <= 12 and greet_norm in {"hi", "hello", "hey"}
    ):
        return (
            "Hello! I'm your aircraft research and market consultant. "
            "How can I assist you today?"
        )

    if len(raw) <= 120 and re.match(
        r"^(who are you|what are you|what do you do|introduce yourself)\??\s*$",
        low,
    ):
        return (
            "I'm your aircraft research and market consultant for Hye Aero. "
            "I help answer questions about aircraft capabilities, ownership, specifications, and market insights."
        )

    return None

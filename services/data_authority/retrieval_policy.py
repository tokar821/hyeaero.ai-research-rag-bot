"""
RAG / Tavily retrieval policy — performance specs never from web or vector snippets.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


_SPEC_INTENT_RE = re.compile(
    r"\b(?:"
    r"could\s+(?:a\s+)?\w+.*(?:fly|make|reach)|"
    r"compare\b.*\b(?:vs|versus)|"
    r"nonstop\s+to|"
    r"what\s+aircraft\s+.*(?:survive|fit|recommend)|"
    r"capability\s+verdict|"
    r"nbba\s+reserves?"
    r")\b",
    re.I,
)


def should_suppress_tavily(
    query: str,
    *,
    client_state: Optional[Dict[str, Any]] = None,
    fine_intent: str = "",
) -> bool:
    """Tavily is market/dynamic only — not aircraft performance specs."""
    if isinstance(client_state, dict) and client_state.get("suppress_tavily_for_specs"):
        return True
    fi = (fine_intent or "").lower()
    if fi in (
        "named_aircraft_capability",
        "aircraft_specs",
        "explicit_comparison",
        "mission_feasibility",
    ):
        return True
    if _SPEC_INTENT_RE.search(query or ""):
        return True
    return False


def should_suppress_rag_performance_context(
    query: str,
    *,
    client_state: Optional[Dict[str, Any]] = None,
    fine_intent: str = "",
) -> bool:
    """Block aviacost/listing snippets from substituting for postgres spec authority on vNext turns."""
    if isinstance(client_state, dict) and client_state.get("suppress_rag_performance_specs"):
        return True
    return should_suppress_tavily(query, client_state=client_state, fine_intent=fine_intent)


__all__ = ["should_suppress_tavily", "should_suppress_rag_performance_context"]

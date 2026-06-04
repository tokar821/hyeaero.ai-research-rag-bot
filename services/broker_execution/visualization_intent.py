"""
Route map / diagram visualization intent — broker turns that need generated visuals.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple


_ROUTE_MAP_RE = re.compile(
    r"(?is)\b(?:route\s+map|map\s+of\s+(?:the\s+)?route|create\s+(?:a\s+)?map|"
    r"show\s+(?:me\s+)?(?:the\s+)?route\s+on\s+a\s+map|flight\s+path\s+map)\b"
)


def detect_visualization_intent(query: str) -> Tuple[bool, str]:
    q = (query or "").strip()
    if not q:
        return False, ""
    if _ROUTE_MAP_RE.search(q):
        return True, "route_map"
    if re.search(r"(?is)\b(?:diagram|graphic|chart)\b.*\b(?:route|mission)\b", q):
        return True, "route_map"
    return False, ""


def render_visualization_fallback(query: str, *, kind: str = "route_map") -> str:
    """Honest fallback when image generation is not wired for this turn."""
    if kind == "route_map":
        return (
            "I can't render a live route map in this chat yet. "
            "For Boston → Denver → London, picture two distinct missions: "
            "a U.S. domestic leg (super-mid / large-cabin) and a separate transatlantic leg "
            "(large-cabin or charter). I can size each leg and recommend buy vs charter if you "
            "confirm passengers and whether nonstop is required on each segment."
        )
    return ""

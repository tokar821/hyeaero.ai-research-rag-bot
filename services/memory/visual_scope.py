"""
Visual memory scoping — clear stale aircraft/tail anchors when the mission route changes.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _route_signature(routes: List[str]) -> str:
    parts: List[str] = []
    for r in routes or []:
        s = re.sub(r"\s+", " ", (r or "").strip().lower())
        if s:
            parts.append(s)
    return "|".join(sorted(parts))


def mission_route_changed(
    prior_routes: Optional[List[str]],
    new_routes: Optional[List[str]],
) -> bool:
    """True when the user moved to a materially different mission (not a refinement)."""
    old_sig = _route_signature(list(prior_routes or []))
    new_sig = _route_signature(list(new_routes or []))
    if not new_sig:
        return False
    if not old_sig:
        return False
    return old_sig != new_sig


def clear_visual_memory_patch(
    conversation_state: Optional[Dict[str, Any]],
    *,
    new_routes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Return keys to merge into conversation memory when routes change.

    Prevents N807JS / prior-tail gallery anchors from contaminating unrelated legs.
    """
    if not isinstance(conversation_state, dict):
        return {}
    mem = conversation_state.get("conversation_memory")
    if not isinstance(mem, dict):
        mem = {}

    prior_routes = mem.get("last_mission_routes")
    if isinstance(prior_routes, str):
        prior_routes = [prior_routes]
    if not mission_route_changed(
        prior_routes if isinstance(prior_routes, list) else None,
        new_routes,
    ):
        if new_routes:
            return {"last_mission_routes": list(new_routes)}
        return {}

    patch: Dict[str, Any] = {
        "last_mission_routes": list(new_routes or []),
        "active_aircraft": None,
        "active_tail": None,
        "comparison_target": None,
        "last_visual_context": None,
        "gallery_tail_anchor": None,
        "visual_memory_cleared": True,
    }
    return patch

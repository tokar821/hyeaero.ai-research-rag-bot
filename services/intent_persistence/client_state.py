"""Sanitize echoed client state when the user pivots to a new shopping/visual thread."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional


def sanitize_client_state_for_shopping_pivot(
    client_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Drop inherited aircraft/tail/comparison anchors so continuity does not keep G650
    after \"show modern cabin under $10M\".
    """
    if not isinstance(client_state, dict):
        return client_state
    out = copy.deepcopy(client_state)
    for k in (
        "current_aircraft_reference",
        "current_visual_intent",
        "current_mission",
        "current_passenger_count",
    ):
        out[k] = None
    cont = out.get("continuity")
    if isinstance(cont, dict):
        c = dict(cont)
        c["current_aircraft"] = None
        c["current_tail"] = None
        c["aircraft_evolution"] = []
        c["locked_entity"] = None
        lr = c.get("last_refinement")
        if isinstance(lr, dict):
            c["last_refinement"] = dict(lr)
        out["continuity"] = c
    mem = out.get("conversation_memory")
    if isinstance(mem, dict):
        m = dict(mem)
        m["active_aircraft"] = None
        m["active_tail"] = None
        m["comparison_target"] = None
        m["aircraft_evolution"] = []
        out["conversation_memory"] = m
    ip = out.get("intent_persistence")
    if isinstance(ip, dict):
        p = dict(ip)
        p["active_aircraft"] = None
        p["active_tail"] = None
        p["comparison_target"] = None
        out["intent_persistence"] = p
    return out

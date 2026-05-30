"""
Strategic fleet analysis renderer — operational broker terminology, no shortlist.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.consultant.mission_state import MissionState
from services.mission.mission_understanding_engine import MissionUnderstandingPacket


def format_strategic_analysis_response(
    mission: MissionState,
    packet: Optional[MissionUnderstandingPacket],
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Rich strategic narrative — delegates to prose renderer when metadata available."""
    if isinstance(data_used, dict):
        try:
            from services.mission.center_of_gravity import attach_center_of_gravity_metadata
            from services.mission.procurement_driver import analyze_procurement_drivers
            from services.mission.network_priority import resolve_network_priority

            attach_center_of_gravity_metadata(data_used, query, mission)
            analyze_procurement_drivers(query, mission, data_used=data_used)
            resolve_network_priority(query, mission, data_used=data_used)
        except Exception:
            pass

    payload: Dict[str, Any] = {
        "conflicts": [],
        "operational_domains": [],
        "recommendation": {"ranked_shortlist": False},
    }
    if isinstance(data_used, dict):
        pda = data_used.get("procurement_driver_analysis") or {}
        cog = data_used.get("mission_center_of_gravity") or {}
        if isinstance(pda, dict):
            payload["conflicts"].extend(pda.get("guidance") or [])
        if isinstance(cog, dict):
            for hub in cog.get("primary_hubs") or []:
                payload["operational_domains"].append(f"hub:{hub}")
            if cog.get("episodic_distortion_risk"):
                payload["conflicts"].append("episodic_ulr_distortion_risk")

    if packet is not None:
        ic = dict(packet.inferred_constraints or {})
        if ic.get("incompatible_domains"):
            payload["conflicts"].append("incompatible operational domains")
        if ic.get("westbound_winter_pressure"):
            payload["conflicts"].append("westbound winter reserve-margin pressure")
        if ic.get("continuation_hub_secondary"):
            payload["conflicts"].append("continuation hubs must remain secondary")

    try:
        from services.rendering.prose_renderer_v2 import render_strategic_prose

        text = render_strategic_prose(
            payload,
            mission=mission,
            query=query,
            data_used=data_used,
        )
        if isinstance(data_used, dict):
            data_used["strategic_analysis_renderer"] = True
            data_used["broker_narrative_authoritative"] = True
        return text
    except Exception:
        pass

    lines: List[str] = [
        "## Strategic Fleet Analysis",
        "",
        "This is a fleet-structure and utilization question — not a single-aircraft shopping exercise.",
        "",
    ]
    if isinstance(data_used, dict):
        data_used["strategic_analysis_renderer"] = True
        data_used["broker_narrative_authoritative"] = True
    return "\n".join(lines)


__all__ = ["format_strategic_analysis_response"]

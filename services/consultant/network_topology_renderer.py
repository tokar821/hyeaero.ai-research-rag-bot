"""
Network topology / hierarchy renderer — no aircraft shortlist.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from services.consultant.mission_state import MissionState
from services.orchestration.hierarchy_weighting import (
    attach_hierarchy_weighting_metadata,
    detect_dominant_mission,
    format_hierarchy_weighting_section,
)


def format_network_topology_response(
    mission: MissionState,
    *,
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
    packet: Any = None,
) -> str:
    hw = detect_dominant_mission(packet, query=query, data_used=data_used)
    attach_hierarchy_weighting_metadata(data_used, hw)

    lines = [
        "## NETWORK STRUCTURE",
        "",
        "This is a hierarchy/topology question — the correct answer is *how to represent the network*, not which aircraft to buy.",
        "",
        format_hierarchy_weighting_section(hw),
        "",
        "### Rule of representation",
        "- **Dominant utilization band** drives fleet sizing and dispatch-reliability optimization.",
        "- **Peak capability legs** are treated as an overlay (charter, second tail, or conditional capability), unless you explicitly state they are primary/weekly/majority utilization.",
    ]
    if isinstance(data_used, dict):
        data_used["broker_narrative_authoritative"] = True
        data_used["network_topology_renderer"] = True
    return "\n".join(lines).strip()


__all__ = ["format_network_topology_response"]


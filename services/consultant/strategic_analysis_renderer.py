"""
Strategic fleet analysis renderer — operational broker terminology, no shortlist.

Replaces generic OPERATIONAL SYNTHESIS for fleet / structural tradeoff queries.
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
    lines: List[str] = [
        "## STRATEGIC ANALYSIS",
        "",
        "This is a fleet-structure and utilization question — not a single-aircraft shopping exercise.",
        "",
    ]

    ic: Dict[str, Any] = {}
    if packet is not None:
        ic = dict(packet.inferred_constraints or {})

    dom = str(ic.get("dominant_utilization") or "").strip()
    if dom:
        lines.append(f"- **Dominant utilization:** {dom.replace('_', ' ')}")
    if ic.get("incompatible_domains"):
        lines.append("- **Utilization conflict:** incompatible operational domains in one profile.")
    if ic.get("westbound_winter_pressure") or ic.get("westbound_winter"):
        lines.append("- **Reserve-margin conflict:** westbound winter transatlantic pressure on dispatch reliability.")
    if ic.get("short_runway_likely") or ic.get("runway_over_cabin"):
        lines.append("- **Runway incompatibility:** field or short-runway legs conflict with large-cabin ULR assumptions.")
    if ic.get("continuation_hub_secondary"):
        lines.append(
            "- **Continuation hubs (secondary only):** Dubai/Singapore-style nodes must not override origin integrity."
        )

    if mission.routes:
        lines.append(f"- **Peak corridor under review:** {mission.routes[0]}")

    lines.extend(
        [
            "",
            "### Structural assessment",
            "- **Dispatch mismatch risk** if one airframe is forced across incompatible corridor classes.",
            "- **Fleet segmentation requirement** likely — mixed utilization rarely survives single-platform optimization.",
            "- **Maintenance profile divergence** between high-cycle domestic and long-stage international legs.",
            "- **Scheduling incoherence** when occasional ULR legs drive procurement but hours stay regional.",
            "",
            "No ranked acquisition shortlist is produced unless you explicitly request aircraft recommendations.",
        ]
    )

    if isinstance(data_used, dict):
        data_used["strategic_analysis_renderer"] = True
        data_used["broker_narrative_authoritative"] = True

    return "\n".join(lines)


__all__ = ["format_strategic_analysis_response"]

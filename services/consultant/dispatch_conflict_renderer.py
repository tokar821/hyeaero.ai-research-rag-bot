"""
Dispatch mismatch language — explicit conflict framing for orchestration output.

When industrial + executive, Arctic + transatlantic, cargo + pax, etc. coexist,
the response MUST state dispatch mismatch / utilization conflict plainly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.mission.mission_understanding_engine import MissionUnderstandingPacket


def mission_has_operational_conflict(
    packet: Optional[MissionUnderstandingPacket],
    data_used: Optional[Dict[str, Any]] = None,
) -> bool:
    if packet is None and not isinstance(data_used, dict):
        return False
    ic: Dict[str, Any] = {}
    if packet is not None:
        ic = dict(packet.inferred_constraints or {})
    if ic.get("incompatible_mission_bands") or ic.get("multi_hard_domain_mission"):
        return True
    if ic.get("passenger_load_variable") and ic.get("cargo_over_cabin"):
        return True
    if isinstance(data_used, dict):
        kernel = data_used.get("mission_authority_kernel") or {}
        if isinstance(kernel, dict) and kernel.get("structural_decomposition"):
            return True
        structural = data_used.get("structural_decomposition") or {}
        if isinstance(structural, dict) and structural.get("required"):
            return True
        resolution = data_used.get("mission_structure_resolution") or {}
        if isinstance(resolution, dict) and resolution.get("decomposition_required"):
            return True
    return False


def format_dispatch_conflict_block(
    packet: Optional[MissionUnderstandingPacket],
    *,
    data_used: Optional[Dict[str, Any]] = None,
    query: str = "",
) -> str:
    """Render explicit dispatch / utilization conflict language."""
    if not mission_has_operational_conflict(packet, data_used):
        return ""

    lines: List[str] = ["Operational Conflict Summary"]
    lines.append("- dispatch mismatch: single-aircraft dispatch reliability cannot span these bands")
    lines.append("- utilization conflict: primary utilization and peak-leg requirements diverge")
    lines.append("- operational incompatibility: runway, payload, and season constraints do not align")
    lines.append("- fleet segmentation required: treat as separate operational domains, not one platform")

    ic: Dict[str, Any] = {}
    if packet is not None:
        ic = dict(packet.inferred_constraints or {})
    if ic.get("industrial_hard_domain") or ic.get("industrial_airport_access"):
        lines.append("- industrial field access conflicts with executive transoceanic expectations")
    if ic.get("arctic_hard_domain"):
        lines.append("- Arctic gravel reliability conflicts with long-range executive dispatch")
    if ic.get("passenger_load_variable"):
        lines.append("- passenger/cargo variability prevents stable cabin configuration")

    ql = (query or "").lower()
    if "dispatch" in ql or "reliability" in ql or "suffered" in ql:
        lines.append("- prior single-aircraft strategy likely caused the dispatch failures described")

    return "\n".join(lines)


__all__ = ["format_dispatch_conflict_block", "mission_has_operational_conflict"]

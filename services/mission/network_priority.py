"""
Network priority — primary hubs vs episodic routes for planning hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mission.center_of_gravity import detect_center_of_gravity
from services.mission.utilization_weighting import compute_utilization_weighting


@dataclass
class NetworkPriorityResult:
    primary_hubs: List[str] = field(default_factory=list)
    episodic_routes: List[str] = field(default_factory=list)
    planning_priority: List[str] = field(default_factory=list)
    do_not_procure_around: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_hubs": list(self.primary_hubs),
            "episodic_routes": list(self.episodic_routes),
            "planning_priority": list(self.planning_priority),
            "do_not_procure_around": list(self.do_not_procure_around),
        }


def resolve_network_priority(
    query: str,
    mission: Any = None,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> NetworkPriorityResult:
    cog = detect_center_of_gravity(query, mission)
    util = compute_utilization_weighting(mission, query=query, data_used=data_used)

    result = NetworkPriorityResult(
        primary_hubs=list(cog.primary_hubs),
        episodic_routes=list(cog.episodic_nodes),
        do_not_procure_around=list(cog.episodic_nodes),
    )

    if util.dominant_route:
        result.planning_priority.append(f"dominant_corridor:{util.dominant_route}")
    if cog.domestic_dominant:
        result.planning_priority.insert(0, "domestic_executive_core")
    if cog.episodic_distortion_risk:
        result.planning_priority.append("continuation_secondary_only")

    if isinstance(data_used, dict):
        data_used["network_priority"] = result.to_dict()

    return result


__all__ = ["NetworkPriorityResult", "resolve_network_priority"]

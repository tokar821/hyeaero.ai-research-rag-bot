"""
Phase 2 structural synthesis gate — segment authority, hierarchy, verdict sync, suppression.

Applied after mission graph build, before authority kernel render.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from services.consultant.mission_state import MissionState
from services.mission.mission_graph import MissionGraph, save_mission_graph
from services.mission.mission_structure_resolution import (
    MissionStructureResolution,
    build_mission_structure_resolution,
    save_structure_resolution,
)
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.mission.models import MissionProfile
from services.mission.planning_hierarchy_resolver import (
    PLANNING_HIERARCHY_KEY,
    apply_peak_dominance_to_graph,
    resolve_planning_hierarchy,
)
from services.mission.recommendation_suppression import (
    RecommendationSuppressionPolicy,
    build_recommendation_suppression_policy,
)
from services.mission.segment_authority import (
    SegmentAuthority,
    filter_renderable_segments,
)

PHASE2_STRUCTURAL_SYNTHESIS_KEY = "phase2_structural_synthesis"
AUTHORITATIVE_OPERATIONAL_KERNEL_KEY = "authoritative_operational_kernel"


def apply_phase2_structural_synthesis(
    graph: MissionGraph,
    packet: Optional[MissionUnderstandingPacket],
    profile: MissionProfile,
    mission: MissionState,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    feasible_models=None,
) -> Tuple[MissionGraph, MissionStructureResolution, List[SegmentAuthority], RecommendationSuppressionPolicy]:
    """
    Enforce segment integrity, peak hierarchy, structure resolution, and suppression policy.
    """
    routes = list(profile.route_labels() or mission.routes or [])
    stage_nm = {lbl: _estimate_route_nm(lbl) for lbl in routes}

    hierarchy = resolve_planning_hierarchy(graph, routes)
    graph = apply_peak_dominance_to_graph(graph, hierarchy)

    graph.segments, authorities = filter_renderable_segments(
        graph.segments,
        stage_nm_by_route=stage_nm,
    )

    resolution = build_mission_structure_resolution(
        packet,
        graph,
        profile=profile,
        mission=mission,
        query=query,
        data_used=data_used,
        feasible_models=feasible_models,
    )
    if resolution.decomposition_required:
        graph.structural_incompatibility = True

    suppression = build_recommendation_suppression_policy(
        resolution, packet, query=query, data_used=data_used
    )

    if isinstance(data_used, dict):
        save_structure_resolution(data_used, resolution)
        save_mission_graph(data_used, graph)
        data_used[PLANNING_HIERARCHY_KEY] = hierarchy.to_dict()
        data_used[PHASE2_STRUCTURAL_SYNTHESIS_KEY] = {
            "segment_authorities": [a.to_dict() for a in authorities],
            "renderable_segment_count": len(graph.segments),
            "suppression": suppression.to_dict(),
        }
        data_used["recommendation_suppression"] = suppression.to_dict()
        if packet and resolution.decomposition_required:
            packet.inferred_constraints["defer_global_shortlist"] = True
            packet.inferred_constraints.setdefault(
                "defer_global_shortlist_reason",
                resolution.decomposition_reason,
            )

    return graph, resolution, authorities, suppression


def freeze_authoritative_operational_kernel(
    data_used: Optional[Dict[str, Any]],
    kernel_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Immutable operational kernel snapshot — renderer may append, not replace."""
    if isinstance(data_used, dict):
        existing = data_used.get(AUTHORITATIVE_OPERATIONAL_KERNEL_KEY)
        if isinstance(existing, dict):
            return existing
        data_used[AUTHORITATIVE_OPERATIONAL_KERNEL_KEY] = dict(kernel_dict)
    return dict(kernel_dict)


def _estimate_route_nm(label: str) -> float:
    try:
        from services.consultant.route_feasibility import estimate_route_distance_nm

        return float(estimate_route_distance_nm(label) or 0)
    except Exception:
        return 0.0


__all__ = [
    "AUTHORITATIVE_OPERATIONAL_KERNEL_KEY",
    "PHASE2_STRUCTURAL_SYNTHESIS_KEY",
    "apply_phase2_structural_synthesis",
    "freeze_authoritative_operational_kernel",
]

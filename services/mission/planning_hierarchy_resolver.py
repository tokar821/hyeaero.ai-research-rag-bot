"""
Planning hierarchy — peak continuation leg drives synthesis order and structure verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mission.mission_graph import MissionGraph, MissionSegmentProfile
from services.mission.segment_kinds import SegmentKind

PLANNING_HIERARCHY_KEY = "planning_hierarchy"


@dataclass
class PlanningHierarchy:
    peak_leg: str = ""
    continuation_legs: List[str] = field(default_factory=list)
    supporting_legs: List[str] = field(default_factory=list)
    utilization_legs: List[str] = field(default_factory=list)
    peak_segment_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_leg": self.peak_leg,
            "continuation_legs": list(self.continuation_legs),
            "supporting_legs": list(self.supporting_legs),
            "utilization_legs": list(self.utilization_legs),
            "peak_segment_id": self.peak_segment_id,
        }


def _estimate_route_nm(label: str) -> float:
    try:
        from services.consultant.route_feasibility import estimate_route_distance_nm

        return float(estimate_route_distance_nm(label) or 0)
    except Exception:
        return 0.0


def resolve_planning_hierarchy(
    graph: MissionGraph,
    routes: List[str],
) -> PlanningHierarchy:
    """Classify legs by planning authority — peak ULR/continuation first."""
    route_nm = [(lbl, _estimate_route_nm(lbl)) for lbl in routes]
    hierarchy = PlanningHierarchy(peak_segment_id=graph.peak_segment_id)

    peak_seg = next((s for s in graph.segments if s.is_peak_planning), None)
    if peak_seg and peak_seg.route_labels:
        hierarchy.peak_leg = peak_seg.route_labels[0]
        hierarchy.peak_segment_id = peak_seg.segment_id

    if not hierarchy.peak_leg and route_nm:
        hierarchy.peak_leg = max(route_nm, key=lambda x: x[1])[0]

    continuation_kinds = {
        SegmentKind.ULR_CONTINUATION,
        SegmentKind.PACIFIC_ULR,
        SegmentKind.TRANSATLANTIC_EXECUTIVE,
    }
    for seg in graph.segments:
        for lbl in seg.route_labels:
            if lbl == hierarchy.peak_leg:
                continue
            if seg.kind in continuation_kinds:
                if lbl not in hierarchy.continuation_legs:
                    hierarchy.continuation_legs.append(lbl)
            elif seg.kind == SegmentKind.DOMESTIC_EXECUTIVE:
                if lbl not in hierarchy.utilization_legs:
                    hierarchy.utilization_legs.append(lbl)
            else:
                if lbl not in hierarchy.supporting_legs:
                    hierarchy.supporting_legs.append(lbl)

    for lbl, nm in sorted(route_nm, key=lambda x: -x[1]):
        if lbl == hierarchy.peak_leg:
            continue
        if nm >= 2800 and lbl not in hierarchy.continuation_legs:
            hierarchy.continuation_legs.append(lbl)
        elif nm < 1500 and lbl not in hierarchy.utilization_legs:
            hierarchy.utilization_legs.append(lbl)
        elif lbl not in hierarchy.supporting_legs:
            hierarchy.supporting_legs.append(lbl)

    return hierarchy


def apply_peak_dominance_to_graph(
    graph: MissionGraph,
    hierarchy: PlanningHierarchy,
) -> MissionGraph:
    """Mark peak segment and reorder segment list — peak first."""
    if not graph.segments:
        return graph
    peak_id = hierarchy.peak_segment_id or graph.peak_segment_id
    for seg in graph.segments:
        seg.is_peak_planning = seg.segment_id == peak_id and bool(peak_id)
    if not peak_id and hierarchy.peak_leg:
        best_nm = 0.0
        for seg in graph.segments:
            seg_nm = max((_estimate_route_nm(r) for r in seg.route_labels), default=0.0)
            if seg_nm > best_nm:
                best_nm = seg_nm
                peak_id = seg.segment_id
        for seg in graph.segments:
            seg.is_peak_planning = seg.segment_id == peak_id
        graph.peak_segment_id = peak_id

    peak_first = sorted(
        graph.segments,
        key=lambda s: (0 if s.is_peak_planning else 1, s.segment_id),
    )
    graph.segments = peak_first
    graph.peak_segment_id = peak_id or graph.peak_segment_id
    return graph


def format_peak_route_display(hierarchy: PlanningHierarchy, fallback: str) -> str:
    if hierarchy.peak_leg:
        others = (
            hierarchy.continuation_legs[:2]
            + hierarchy.supporting_legs[:2]
            + hierarchy.utilization_legs[:2]
        )
        others = [o for o in others if o != hierarchy.peak_leg][:3]
        if others:
            return f"{hierarchy.peak_leg} (peak planning); other legs: {'; '.join(others)}"
        return hierarchy.peak_leg
    return fallback


__all__ = [
    "PLANNING_HIERARCHY_KEY",
    "PlanningHierarchy",
    "apply_peak_dominance_to_graph",
    "format_peak_route_display",
    "resolve_planning_hierarchy",
]

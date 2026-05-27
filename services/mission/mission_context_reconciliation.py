"""
Mission context reconciliation — strict separation before memory merge.

Layers:
  1. Current turn extraction (routes, pax, priorities from this message only)
  2. Persistent broker memory (posture + validated continuity only)
  3. Prior operational graph (only when continuity confidence is high)

A previous mission must NOT alter route synthesis unless explicitly validated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from services.mission.models import MissionProfile
from services.mission.mission_operational_graph import MissionOperationalGraph

_CONTINUITY_APPLY_THRESHOLD = 0.72

# Geography tokens extracted from route labels / band text for overlap checks.
_REGION_TOKENS = (
    ("caribbean", ("caribbean", "miami", "nassau", "bahamas", "island")),
    ("europe", ("london", "paris", "berlin", "moscow", "geneva", "frankfurt", "europe", "uk")),
    ("pacific", ("tokyo", "seoul", "hong kong", "pacific", "asia", "japan")),
    ("middle_east", ("dubai", "riyadh", "doha", "middle east", "abu dhabi")),
    ("mountain", ("aspen", "telluride", "jackson", "mountain", "sun valley")),
    ("domestic_us", ("dallas", "new york", "nyc", "los angeles", "chicago", "boston", "teterboro")),
)


@dataclass
class MissionContinuityAssessment:
    """Whether prior session structure may influence this turn."""

    continuity_confidence: float = 0.0
    mission_pivot: bool = False
    reason: str = ""
    apply_structural_memory: bool = False
    apply_posture_memory: bool = True
    current_regions: Set[str] = field(default_factory=set)
    prior_regions: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "continuity_confidence": round(self.continuity_confidence, 3),
            "mission_pivot": self.mission_pivot,
            "reason": self.reason,
            "apply_structural_memory": self.apply_structural_memory,
            "apply_posture_memory": self.apply_posture_memory,
            "current_regions": sorted(self.current_regions),
            "prior_regions": sorted(self.prior_regions),
        }


def _regions_from_text(text: str) -> Set[str]:
    low = (text or "").lower()
    found: Set[str] = set()
    for region, tokens in _REGION_TOKENS:
        if any(t in low for t in tokens):
            found.add(region)
    return found


def _regions_from_routes(route_labels: List[str]) -> Set[str]:
    return _regions_from_text(" ".join(route_labels))


def _regions_from_broker_memory(broker_memory: Optional[Dict[str, Any]]) -> Set[str]:
    if not isinstance(broker_memory, dict):
        return set()
    blob = " ".join(broker_memory.get("recurring_routes") or [])
    blob += " " + " ".join(broker_memory.get("operational_bands") or [])
    blob += " " + str(broker_memory.get("travel_pattern") or "")
    return _regions_from_text(blob)


def _regions_from_graph(graph: Optional[MissionOperationalGraph]) -> Set[str]:
    if graph is None:
        return set()
    blob = " ".join(graph.operational_bands or [])
    blob += f" {graph.corridor_type} {graph.travel_pattern}"
    return _regions_from_text(blob)


def _route_overlap(current: List[str], prior: List[str]) -> float:
    if not current or not prior:
        return 0.0
    cur = {r.strip().lower() for r in current if r}
    pri = {r.strip().lower() for r in prior if r}
    if not cur or not pri:
        return 0.0
    overlap = len(cur & pri) / max(len(cur), 1)
    return overlap


def assess_mission_continuity(
    query: str,
    profile: MissionProfile,
    *,
    broker_memory: Optional[Dict[str, Any]] = None,
    prior_graph: Optional[MissionOperationalGraph] = None,
) -> MissionContinuityAssessment:
    """
    Decide if prior mission structure may merge into the current turn.

    Structural memory (bands, corridor, fleet flags) requires high confidence.
    Posture memory (nonstop preference, cost philosophy) may apply at lower confidence.
    """
    current_routes = profile.route_labels()
    current_regions = _regions_from_routes(current_routes)
    if not current_regions:
        current_regions = _regions_from_text(query)

    prior_routes: List[str] = []
    if isinstance(broker_memory, dict):
        prior_routes = [str(r) for r in (broker_memory.get("recurring_routes") or []) if r]

    prior_regions = _regions_from_broker_memory(broker_memory) | _regions_from_graph(prior_graph)

    assessment = MissionContinuityAssessment(
        current_regions=current_regions,
        prior_regions=prior_regions,
    )

    if not prior_regions and not prior_routes and (
        prior_graph is None or not prior_graph.operational_bands
    ):
        assessment.continuity_confidence = 1.0
        assessment.reason = "no_prior_mission_structure"
        assessment.apply_structural_memory = False
        assessment.apply_posture_memory = bool(broker_memory)
        return assessment

    if not current_routes and not current_regions:
        assessment.continuity_confidence = 0.55
        assessment.reason = "current_turn_ambiguous_allow_light_posture_only"
        assessment.apply_structural_memory = False
        assessment.apply_posture_memory = True
        return assessment

    route_ov = _route_overlap(current_routes, prior_routes)
    region_ov = 0.0
    if current_regions and prior_regions:
        region_ov = len(current_regions & prior_regions) / max(len(current_regions), 1)

    confidence = max(route_ov, region_ov * 0.85)
    if current_routes:
        confidence = max(confidence, 0.15 + route_ov * 0.7 + region_ov * 0.3)

    assessment.continuity_confidence = min(1.0, confidence)

    if current_regions and prior_regions and not (current_regions & prior_regions):
        assessment.mission_pivot = True
        assessment.reason = (
            f"geography_pivot: current={sorted(current_regions)} prior={sorted(prior_regions)}"
        )
        assessment.apply_structural_memory = False
        assessment.apply_posture_memory = True
        return assessment

    if current_routes and prior_routes and route_ov < 0.15 and region_ov < 0.2:
        assessment.mission_pivot = True
        assessment.reason = "route_pivot_no_overlap_with_prior_recurring_routes"
        assessment.apply_structural_memory = False
        assessment.apply_posture_memory = True
        return assessment

    if confidence >= _CONTINUITY_APPLY_THRESHOLD:
        assessment.reason = "continuity_validated"
        assessment.apply_structural_memory = True
        assessment.apply_posture_memory = True
    else:
        assessment.reason = f"low_continuity_{confidence:.2f}_posture_only"
        assessment.apply_structural_memory = False
        assessment.apply_posture_memory = True

    return assessment


def reconcile_broker_memory_for_turn(
    memory: Any,
    packet: Any,
    continuity: MissionContinuityAssessment,
) -> Any:
    """
    Update broker memory from current turn packet.

    On mission pivot: replace mission-specific fields instead of accumulating.
    """
    from services.session.broker_memory import BrokerMemory, update_broker_memory_from_understanding

    if not isinstance(memory, BrokerMemory):
        memory = BrokerMemory()

    if continuity.mission_pivot:
        memory.operational_bands = []
        memory.incompatible_bands = False
        memory.fleet_strategy_required = False
        memory.travel_pattern = ""
        memory.recurring_routes = []

    return update_broker_memory_from_understanding(memory, packet)


def validate_understanding_packet(packet: Any) -> List[str]:
    """Structured completeness checks before formatting."""
    issues: List[str] = []
    if packet is None:
        return ["missing_packet"]
    routes = []
    if hasattr(packet, "explicit_constraints"):
        routes = (packet.explicit_constraints or {}).get("routes") or []
    elif isinstance(packet, dict):
        routes = (packet.get("explicit_constraints") or {}).get("routes") or []

    synthesis = getattr(packet, "operational_synthesis", None) or (
        packet.get("operational_synthesis") if isinstance(packet, dict) else ""
    )
    if not (synthesis or "").strip():
        issues.append("empty_operational_synthesis")

    corridor = getattr(packet, "corridor_type", None) or (
        packet.get("corridor_type") if isinstance(packet, dict) else ""
    )
    if routes and corridor in ("unknown", "", None):
        issues.append("routes_without_corridor")

    return issues

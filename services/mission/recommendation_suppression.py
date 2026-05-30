"""
Recommendation suppression — no generic jet dumps while structure is unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from services.consultant.recommendation_engine import AircraftRecommendation
from services.mission.mission_authority_kernel import MissionAuthorityKernel
from services.mission.mission_structure_resolution import MissionStructureResolution
from services.mission.mission_understanding_engine import MissionUnderstandingPacket

_GENERIC_DUMP_MODELS = frozenset(
    {
        "global 7500",
        "g650",
        "g650er",
        "gulfstream g650",
        "gulfstream g650er",
        "falcon 8x",
        "bombardier global 7500",
    }
)

STRUCTURE_GUIDANCE = (
    "This is not a model-selection problem first — it is a mission decomposition problem."
)


@dataclass
class RecommendationSuppressionPolicy:
    suppress_aircraft_specificity: bool
    permits_aircraft_specificity: bool
    reason: str
    render_class_bands_only: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suppress_aircraft_specificity": self.suppress_aircraft_specificity,
            "permits_aircraft_specificity": self.permits_aircraft_specificity,
            "reason": self.reason,
            "render_class_bands_only": self.render_class_bands_only,
        }


def build_recommendation_suppression_policy(
    resolution: MissionStructureResolution,
    packet: Optional[MissionUnderstandingPacket],
    kernel: Optional[MissionAuthorityKernel] = None,
) -> RecommendationSuppressionPolicy:
    """Determine whether model-level recommendations may render."""
    reasons: List[str] = []
    suppress = False

    if resolution.decomposition_required:
        suppress = True
        reasons.append(resolution.decomposition_reason or "decomposition_required")

    if packet:
        ic = packet.inferred_constraints or {}
        if ic.get("incompatible_mission_bands"):
            suppress = True
            reasons.append("incompatible_domains")
        if ic.get("defer_global_shortlist"):
            suppress = True
            reasons.append("governance_or_portfolio_unresolved")
        if ic.get("passenger_load_variable") and ic.get("cargo_over_cabin"):
            suppress = True
            reasons.append("planning_hierarchy_unstable")

    if kernel and kernel.structural_decomposition and kernel.single_aircraft_forbidden:
        suppress = True
        reasons.append("portfolio_decomposition_active")

    if kernel and kernel.route_certainty_degraded and len(kernel.segments) >= 2:
        suppress = True
        reasons.append("continuation_dominance_unresolved")

    return RecommendationSuppressionPolicy(
        suppress_aircraft_specificity=suppress,
        permits_aircraft_specificity=not suppress,
        reason="; ".join(dict.fromkeys(reasons)) if reasons else "",
        render_class_bands_only=suppress,
    )


def filter_suppressed_recommendations(
    recommendations: Sequence[AircraftRecommendation],
    policy: RecommendationSuppressionPolicy,
) -> List[AircraftRecommendation]:
    if not policy.suppress_aircraft_specificity:
        return list(recommendations)
    return []


def is_generic_dump_model(model: str) -> bool:
    return (model or "").strip().lower() in _GENERIC_DUMP_MODELS


__all__ = [
    "STRUCTURE_GUIDANCE",
    "RecommendationSuppressionPolicy",
    "build_recommendation_suppression_policy",
    "filter_suppressed_recommendations",
    "is_generic_dump_model",
]

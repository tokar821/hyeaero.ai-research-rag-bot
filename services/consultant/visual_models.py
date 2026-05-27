"""
Visualization-ready schemas (no rendering) for future range maps, charts, and cards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.consultant.comparison_engine import StructuredComparison
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.route_feasibility import RouteFeasibilityAssessment


@dataclass
class RangeMapModel:
    aircraft_model: str
    origin_label: str
    destination_label: str
    practical_radius_nm: float
    brochure_radius_nm: float
    classification: str
    confidence: float  # internal only

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "range_map",
            "aircraft_model": self.aircraft_model,
            "origin_label": self.origin_label,
            "destination_label": self.destination_label,
            "practical_radius_nm": self.practical_radius_nm,
            "brochure_radius_nm": self.brochure_radius_nm,
            "classification": self.classification,
        }


@dataclass
class PayloadRangeChartPoint:
    passengers: int
    practical_nm: float
    brochure_nm: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passengers": self.passengers,
            "practical_nm": self.practical_nm,
            "brochure_nm": self.brochure_nm,
        }


@dataclass
class PayloadRangeChartModel:
    aircraft_model: str
    points: List[PayloadRangeChartPoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "payload_range_chart",
            "aircraft_model": self.aircraft_model,
            "points": [p.to_dict() for p in self.points],
        }


@dataclass
class CabinLayoutModel:
    aircraft_model: str
    category: str
    stand_up_cabin: bool
    typical_pax: int
    cabin_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "cabin_layout",
            "aircraft_model": self.aircraft_model,
            "category": self.category,
            "stand_up_cabin": self.stand_up_cabin,
            "typical_pax": self.typical_pax,
            "cabin_score": self.cabin_score,
        }


@dataclass
class MissionReachabilityModel:
    route_label: str
    distance_nm: float
    aircraft_models: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "mission_reachability_map",
            "route_label": self.route_label,
            "distance_nm": self.distance_nm,
            "aircraft_models": list(self.aircraft_models),
        }


@dataclass
class VisualIntelligenceBundle:
    range_maps: List[RangeMapModel] = field(default_factory=list)
    payload_range_charts: List[PayloadRangeChartModel] = field(default_factory=list)
    cabin_layouts: List[CabinLayoutModel] = field(default_factory=list)
    comparison_cards: List[Dict[str, Any]] = field(default_factory=list)
    mission_reachability: List[MissionReachabilityModel] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "range_maps": [r.to_dict() for r in self.range_maps],
            "payload_range_charts": [c.to_dict() for c in self.payload_range_charts],
            "cabin_layouts": [c.to_dict() for c in self.cabin_layouts],
            "comparison_cards": list(self.comparison_cards),
            "mission_reachability": [m.to_dict() for m in self.mission_reachability],
        }


def build_visual_intelligence_bundle(
    mission: MissionState,
    recommendations: List[AircraftRecommendation],
    route_assessments: List[RouteFeasibilityAssessment],
    comparison: Optional[StructuredComparison] = None,
) -> VisualIntelligenceBundle:
    from services.consultant.recommendation_engine import _AIRCRAFT_PROFILES

    bundle = VisualIntelligenceBundle()

    if comparison:
        bundle.comparison_cards = list(comparison.visual_normalized.get("comparison_cards") or [])

    for rec in recommendations[:4]:
        prof = _AIRCRAFT_PROFILES.get(rec.model) or {}
        bundle.cabin_layouts.append(
            CabinLayoutModel(
                aircraft_model=rec.model,
                category=rec.category,
                stand_up_cabin=float(prof.get("cabin_score") or 0) >= 0.75,
                typical_pax=int(prof.get("pax_typical") or 8),
                cabin_score=float(prof.get("cabin_score") or 0),
            )
        )
        pax = mission.passenger_count or int(prof.get("pax_typical") or 8)
        points = []
        for p in range(max(4, pax - 2), pax + 3):
            penalty = max(0, (p - int(prof.get("pax_typical") or 8)) * 120)
            points.append(
                PayloadRangeChartPoint(
                    passengers=p,
                    practical_nm=max(800, float(prof.get("practical_nm") or 0) - penalty),
                    brochure_nm=max(1000, float(prof.get("brochure_nm") or 0) - penalty * 0.8),
                )
            )
        bundle.payload_range_charts.append(
            PayloadRangeChartModel(aircraft_model=rec.model, points=points)
        )

    for ra in route_assessments[:3]:
        for rec in recommendations[:3]:
            prof = _AIRCRAFT_PROFILES.get(rec.model) or {}
            parts = (ra.route_label or "").split("→")
            origin = parts[0].strip() if parts else "Origin"
            dest = parts[1].strip() if len(parts) > 1 else "Destination"
            bundle.range_maps.append(
                RangeMapModel(
                    aircraft_model=rec.model,
                    origin_label=origin,
                    destination_label=dest,
                    practical_radius_nm=float(prof.get("practical_nm") or 0),
                    brochure_radius_nm=float(prof.get("brochure_nm") or 0),
                    classification=ra.classification,
                    confidence=ra.confidence,
                )
            )
        bundle.mission_reachability.append(
            MissionReachabilityModel(
                route_label=ra.route_label,
                distance_nm=ra.distance_nm,
                aircraft_models=[
                    {
                        "model": r.model,
                        "classification": ra.classification,
                        "fit": r.fit or "Good Fit",
                    }
                    for r in recommendations[:5]
                ],
            )
        )

    return bundle

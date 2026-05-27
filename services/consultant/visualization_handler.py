"""
Direct visualization generation from extracted entities — no purchase shortlists.

Follow-ups are allowed only when origin is truly missing (range / reachability maps)
or aircraft is truly missing (cabin / layout visuals).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from services.consultant.comparison_engine import build_structured_comparison
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import (
    AircraftRecommendation,
    _AIRCRAFT_PROFILES,
    detect_models_from_text,
)
from services.consultant.route_feasibility import RouteFeasibilityAssessment, assess_mission_routes
from services.consultant.visual_models import VisualIntelligenceBundle, build_visual_intelligence_bundle
from services.recommendation.clarification_decision import effective_route_labels, infer_route_labels_from_query
from services.recommendation.query_recommendation_intent import (
    QueryRecommendationIntent,
    classify_query_recommendation_intent,
)


class VisualizationKind(str, Enum):
    RANGE_MAP = "range_map"
    REACHABLE_CITIES = "reachable_cities"
    CABIN_LAYOUT = "cabin_layout"
    COMPARE_LAYOUTS = "compare_layouts"
    CABIN_GRAPHIC = "cabin_graphic"
    GENERIC = "generic_visual"


_RANGE_MAP_RE = re.compile(r"\b(?:range\s+map|rangemap)\b", re.I)
_REACHABLE_RE = re.compile(
    r"\b(?:reachable\s+cities?|cities?\s+(?:reachable|within\s+range)|within\s+range\s+of)\b",
    re.I,
)
_COMPARE_LAYOUT_RE = re.compile(
    r"\b(?:compare\s+layouts?|layout\s+comparison|cabin\s+comparison|compare\s+cabins?)\b",
    re.I,
)
_CABIN_GRAPHIC_RE = re.compile(
    r"\b(?:cabin\s+graphic|cabin\s+layout|layout\s+graphic|interior\s+layout)\b",
    re.I,
)
_MAP_OR_VISUALIZE_RE = re.compile(r"\b(?:\bmap\b|visuali[sz]e|visuali[sz]ation)\b", re.I)
_FROM_ORIGIN_RE = re.compile(
    r"\bfrom\s+([A-Za-z][A-Za-z0-9\s\-]{1,40}?)(?:\s+to|\s+for|\s+with|\s*$|\,|\.)",
    re.I,
)
_TO_ROUTE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9\s\-]{1,30}?)\s+to\s+([A-Za-z][A-Za-z0-9\s\-]{1,30}?)(?:\s+for|\s+with|\s*$|\,|\.)",
    re.I,
)


def _resolve_place_label(raw: str) -> str:
    token = (raw or "").strip()
    if not token:
        return ""
    try:
        from services.mission.route_extractor import resolve_place

        place, conf = resolve_place(token)
        if place is not None and conf >= 0.5:
            return place.canonical
    except Exception:
        pass
    return token.title()


def _parse_route_from_query(query: str) -> Tuple[str, str, str]:
    """Return (origin, destination, route_label) when inferable."""
    m = _TO_ROUTE_RE.search(query or "")
    if m:
        origin = _resolve_place_label(m.group(1).strip())
        dest = _resolve_place_label(m.group(2).strip())
        if origin and dest:
            return origin, dest, f"{origin} -> {dest}"
    return "", "", ""


@dataclass
class VisualizationEntities:
    aircraft_models: List[str] = field(default_factory=list)
    routes: List[str] = field(default_factory=list)
    origin_label: str = ""
    destination_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aircraft_models": list(self.aircraft_models),
            "routes": list(self.routes),
            "origin_label": self.origin_label,
            "destination_label": self.destination_label,
        }


@dataclass
class VisualizationTurnResult:
    bundle: VisualIntelligenceBundle
    kind: VisualizationKind
    entities: VisualizationEntities
    followup_needed: bool = False
    followup_message: str = ""
    caption: str = ""
    recommendations: List[AircraftRecommendation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "entities": self.entities.to_dict(),
            "followup_needed": self.followup_needed,
            "followup_message": self.followup_message,
            "caption": self.caption,
            "visual_models": self.bundle.to_dict(),
        }


def is_visualization_turn(query: str, *, data_used: Optional[Dict[str, Any]] = None) -> bool:
    du = data_used if isinstance(data_used, dict) else {}
    stored = str(du.get("query_recommendation_intent") or "").strip()
    if stored == QueryRecommendationIntent.VISUALIZATION_REQUEST.value:
        return True
    return classify_query_recommendation_intent(query).intent == QueryRecommendationIntent.VISUALIZATION_REQUEST


def classify_visualization_kind(query: str) -> VisualizationKind:
    ql = (query or "").lower()
    if _COMPARE_LAYOUT_RE.search(ql):
        return VisualizationKind.COMPARE_LAYOUTS
    if _REACHABLE_RE.search(ql):
        return VisualizationKind.REACHABLE_CITIES
    if _RANGE_MAP_RE.search(ql):
        return VisualizationKind.RANGE_MAP
    if _CABIN_GRAPHIC_RE.search(ql):
        return VisualizationKind.CABIN_LAYOUT
    if _MAP_OR_VISUALIZE_RE.search(ql):
        if re.search(r"\b(?:cabin|interior|layout|galley)\b", ql):
            return VisualizationKind.CABIN_LAYOUT
        return VisualizationKind.RANGE_MAP
    return VisualizationKind.GENERIC


def _models_from_context(
    query: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> List[str]:
    models = detect_models_from_text(query or "")
    if models:
        return list(dict.fromkeys(models))

    if history:
        for turn in reversed(history):
            if not isinstance(turn, dict):
                continue
            if str(turn.get("role") or "").lower() != "user":
                continue
            found = detect_models_from_text(str(turn.get("content") or ""))
            if found:
                return list(dict.fromkeys(found))

    if isinstance(conversation_state, dict):
        mem = conversation_state.get("conversation_memory") or {}
        anchor = str(mem.get("active_aircraft") or mem.get("comparison_target") or "").strip()
        if anchor:
            found = detect_models_from_text(anchor)
            if found:
                return list(dict.fromkeys(found))
            if anchor in _AIRCRAFT_PROFILES:
                return [anchor]

    du = data_used if isinstance(data_used, dict) else {}
    for key in ("consultant_recommendations",):
        recs = du.get(key) or []
        if isinstance(recs, list) and recs:
            out = [str(r.get("model") or "").strip() for r in recs if isinstance(r, dict)]
            out = [m for m in out if m]
            if out:
                return list(dict.fromkeys(out))
    pipe = du.get("deterministic_recommendation_pipeline") or {}
    if isinstance(pipe, dict):
        recs = pipe.get("recommendations") or []
        if isinstance(recs, list) and recs:
            out = [str(r.get("model") or "").strip() for r in recs if isinstance(r, dict)]
            out = [m for m in out if m]
            if out:
                return list(dict.fromkeys(out))

    return []


def _extract_origin_label(query: str, mission: MissionState) -> str:
    origin, _, _ = _parse_route_from_query(query)
    if origin:
        return origin

    routes = effective_route_labels(mission, query)
    if routes:
        parts = re.split(r"\s*(?:->|→)\s*", routes[0], maxsplit=1)
        if parts and parts[0].strip():
            return parts[0].strip()

    m = _FROM_ORIGIN_RE.search(query or "")
    if m:
        return m.group(1).strip()

    try:
        from services.mission.route_extractor import extract_routes

        extractions = extract_routes(query or "")
        if extractions:
            return extractions[0].route.origin
    except Exception:
        pass

    if mission.preferred_airports:
        return mission.preferred_airports[0]

    return ""


def _extract_destination_label(query: str, mission: MissionState) -> str:
    _, dest, _ = _parse_route_from_query(query)
    if dest:
        return dest

    routes = effective_route_labels(mission, query)
    if routes:
        parts = re.split(r"\s*(?:->|→)\s*", routes[0], maxsplit=1)
        if len(parts) > 1 and parts[1].strip():
            return parts[1].strip()

    try:
        from services.mission.route_extractor import extract_routes

        extractions = extract_routes(query or "")
        if extractions:
            return extractions[0].route.destination
    except Exception:
        pass

    return ""


def extract_visualization_entities(
    query: str,
    mission: MissionState,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> VisualizationEntities:
    origin, dest, route_label = _parse_route_from_query(query)
    routes = effective_route_labels(mission, query) or infer_route_labels_from_query(query)
    if not routes and route_label:
        routes = [route_label]
    return VisualizationEntities(
        aircraft_models=_models_from_context(
            query,
            history=history,
            conversation_state=conversation_state,
            data_used=data_used,
        ),
        routes=list(routes),
        origin_label=origin or _extract_origin_label(query, mission),
        destination_label=dest or _extract_destination_label(query, mission),
    )


def _visualization_followup(kind: VisualizationKind, entities: VisualizationEntities) -> Tuple[bool, str]:
    needs_origin = kind in (VisualizationKind.RANGE_MAP, VisualizationKind.REACHABLE_CITIES)
    if needs_origin and not (entities.origin_label or entities.routes):
        return True, "Which city or airport should I anchor the map from?"

    needs_aircraft = kind in (
        VisualizationKind.CABIN_LAYOUT,
        VisualizationKind.CABIN_GRAPHIC,
        VisualizationKind.COMPARE_LAYOUTS,
    )
    if kind == VisualizationKind.COMPARE_LAYOUTS:
        if len(entities.aircraft_models) < 2:
            if not entities.aircraft_models:
                return True, "Which two aircraft should I compare cabin layouts for?"
            return True, "Name the second aircraft for a side-by-side layout comparison."
        return False, ""

    if needs_aircraft and not entities.aircraft_models:
        return True, "Which aircraft model should I show the cabin layout for?"

    return False, ""


def _stub_recommendations(models: List[str]) -> List[AircraftRecommendation]:
    recs: List[AircraftRecommendation] = []
    for i, model in enumerate(models[:5]):
        prof = _AIRCRAFT_PROFILES.get(model) or {}
        recs.append(
            AircraftRecommendation(
                model=model,
                category=str(prof.get("category") or "business_jet"),
                total_score=max(0.5, 1.0 - i * 0.05),
                confidence=0.85,
                rank=i + 1,
                fit="Good Fit",
            )
        )
    return recs


def _mission_for_visualization(entities: VisualizationEntities, mission: MissionState) -> MissionState:
    routes = list(entities.routes)
    if not routes and entities.origin_label:
        dest = entities.destination_label or "Practical range envelope"
        routes = [f"{entities.origin_label} -> {dest}"]
    if routes and routes != (mission.routes or []):
        return MissionState(
            passenger_count=mission.passenger_count,
            mission_type=mission.mission_type,
            routes=routes,
            westbound=mission.westbound,
            eastbound=mission.eastbound,
            reserves_requirement=mission.reserves_requirement,
            runway_constraints=mission.runway_constraints,
            baggage_priority=mission.baggage_priority,
            ownership_goal=mission.ownership_goal,
            budget_usd=mission.budget_usd,
            preferred_airports=list(mission.preferred_airports),
            cabin_priority=mission.cabin_priority,
            operating_cost_priority=mission.operating_cost_priority,
            acquisition_strategy=mission.acquisition_strategy,
            mountain_airport_requirement=mission.mountain_airport_requirement,
            international_frequency=mission.international_frequency,
            nonstop_requirement=mission.nonstop_requirement,
            seasonal_constraints=mission.seasonal_constraints,
            constraints=list(mission.constraints),
            snapshots=list(mission.snapshots),
            turn_index=mission.turn_index,
        )
    return mission


def _build_route_assessments(
    mission: MissionState,
    recommendations: List[AircraftRecommendation],
) -> List[RouteFeasibilityAssessment]:
    if not mission.routes or not recommendations:
        return []
    assessments: List[RouteFeasibilityAssessment] = []
    for rec in recommendations[:4]:
        prof = _AIRCRAFT_PROFILES.get(rec.model) or {}
        assessments.extend(
            assess_mission_routes(
                mission,
                aircraft_practical_nm=float(prof.get("practical_nm") or 3000),
                aircraft_brochure_nm=float(prof.get("brochure_nm") or 3500),
            )[: len(mission.routes)]
        )
    return assessments


def _caption_for_visualization(
    kind: VisualizationKind,
    entities: VisualizationEntities,
    recommendations: List[AircraftRecommendation],
) -> str:
    models = ", ".join(entities.aircraft_models[:3]) or (
        ", ".join(r.model for r in recommendations[:3]) if recommendations else "selected aircraft"
    )
    origin = entities.origin_label or (entities.routes[0].split("->")[0].strip() if entities.routes else "")
    dest = entities.destination_label
    if kind == VisualizationKind.RANGE_MAP:
        if origin and dest:
            return f"Range map: {models} — {origin} to {dest}."
        if origin:
            return f"Range map from {origin} for {models}."
        return f"Range map for {models}."
    if kind == VisualizationKind.REACHABLE_CITIES:
        return f"Reachable cities from {origin} for {models}."
    if kind == VisualizationKind.COMPARE_LAYOUTS:
        return f"Cabin layout comparison: {models}."
    if kind in (VisualizationKind.CABIN_LAYOUT, VisualizationKind.CABIN_GRAPHIC):
        return f"Cabin layout graphic for {models}."
    return f"Visualization ready for {models}."


def run_visualization_turn(
    query: str,
    *,
    mission: MissionState,
    history: Optional[List[Dict[str, str]]] = None,
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> VisualizationTurnResult:
    kind = classify_visualization_kind(query)
    entities = extract_visualization_entities(
        query,
        mission,
        history=history,
        conversation_state=conversation_state,
        data_used=data_used,
    )

    needs_followup, followup_msg = _visualization_followup(kind, entities)
    if needs_followup:
        return VisualizationTurnResult(
            bundle=VisualIntelligenceBundle(),
            kind=kind,
            entities=entities,
            followup_needed=True,
            followup_message=followup_msg,
        )

    recommendations = _stub_recommendations(entities.aircraft_models)
    if not recommendations and kind in (VisualizationKind.RANGE_MAP, VisualizationKind.REACHABLE_CITIES):
        recommendations = _stub_recommendations(["Gulfstream G650"])

    viz_mission = _mission_for_visualization(entities, mission)
    route_assessments = _build_route_assessments(viz_mission, recommendations)

    comparison = None
    if kind == VisualizationKind.COMPARE_LAYOUTS and len(entities.aircraft_models) >= 2:
        comparison = build_structured_comparison(
            entities.aircraft_models,
            viz_mission,
            recommendations=recommendations,
        )

    bundle = build_visual_intelligence_bundle(
        viz_mission,
        recommendations,
        route_assessments,
        comparison=comparison,
    )

    return VisualizationTurnResult(
        bundle=bundle,
        kind=kind,
        entities=entities,
        caption=_caption_for_visualization(kind, entities, recommendations),
        recommendations=recommendations,
    )


def build_visualization_authority_block(result: VisualizationTurnResult) -> str:
    if result.followup_needed:
        return (
            "[VISUALIZATION INTENT — ENTITY GAP]\n"
            f"Visualization kind: {result.kind.value}.\n"
            f"Ask exactly one focused question: {result.followup_message}\n"
            "Do not ask about passengers, budget, or purchase preferences."
        )
    lines = [
        "[VISUALIZATION INTENT — DIRECT RENDER]",
        f"Kind: {result.kind.value}.",
        f"Entities: {result.entities.to_dict()}",
        "Generate the visualization from extracted entities — no purchase shortlist.",
        "Do not ask follow-up questions unless the user message lacked required origin or aircraft.",
    ]
    if result.caption:
        lines.append(f"Caption anchor: {result.caption}")
    return "\n".join(lines)

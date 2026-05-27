"""
Clarification decision logic — ask only when ambiguity materially changes the recommendation.

Suppress follow-ups when mission category, aircraft class, longest leg, and ranking
confidence are already sufficient (e.g. Dallas, New York, London, 15 passengers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.route_feasibility import estimate_route_distance_nm
from services.mission.aviation_places import AviationPlace
from services.mission.route_extractor import extract_routes, resolve_place
from services.recommendation.mission_ranker import (
    MissionCategory,
    classify_mission_category,
    mission_max_leg_nm,
)

# Leader score at or above this → decisive shortlist (aligned with assign_fit_tiers)
RECOMMENDATION_CONFIDENCE_THRESHOLD = 0.52
# Top-two within this gap → category still unclear at the margin
RECOMMENDATION_SCORE_GAP_THRESHOLD = 0.12

_DOMESTIC_VS_OCEANIC_RE = re.compile(
    r"\b(?:mostly\s+)?domestic(?:\s+u\.?s\.?)?|"
    r"frequent(?:ly)?\s+transoceanic|transoceanic|"
    r"primarily\s+(?:domestic|international)|"
    r"mostly\s+(?:domestic|international)\b",
    re.I,
)
_SHORT_RUNWAY_UNPINNED_RE = re.compile(
    r"\bshort\s+(?:runway|field)|runway\s+flex|under\s+4[,.]?000\s*ft\b",
    re.I,
)
_US_ONLY_CITY_LIST_RE = re.compile(
    r"^[a-z0-9\s,.-]+$",
    re.I,
)


@dataclass
class MissionClarificationNeeds:
    """Gaps that should trigger a single focused follow-up in advisor copy."""

    needs_route: bool = False
    needs_passenger_count: bool = False
    needs_budget: bool = False
    needs_category_usage: bool = False
    needs_runway_detail: bool = False

    @property
    def any(self) -> bool:
        return (
            self.needs_route
            or self.needs_passenger_count
            or self.needs_budget
            or self.needs_category_usage
            or self.needs_runway_detail
        )


def infer_route_labels_from_query(query: str) -> List[str]:
    """Validated route labels from the current user message (including city lists)."""
    extractions = extract_routes(query or "")
    return [f"{e.route.origin} -> {e.route.destination}" for e in extractions]


def effective_route_labels(mission: MissionState, query: str = "") -> List[str]:
    """Persisted routes, else routes inferable from the current turn."""
    if mission.routes:
        return list(mission.routes)
    return infer_route_labels_from_query(query)


def mission_with_effective_routes(mission: MissionState, query: str = "") -> MissionState:
    """Mission snapshot with inferred routes applied for classification/ranking."""
    routes = effective_route_labels(mission, query)
    if routes == (mission.routes or []):
        return mission
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


def _budget_materially_ambiguous(mission: MissionState) -> bool:
    budget = mission.budget_usd
    if not budget or budget <= 0:
        return False
    if 7_000_000 <= budget <= 12_000_000:
        return True
    if 17_000_000 <= budget <= 23_000_000:
        return True
    return False


def _places_from_route_labels(routes: Sequence[str]) -> List[AviationPlace]:
    places: List[AviationPlace] = []
    seen: set[str] = set()
    for label in routes:
        for part in re.split(r"\s*->\s*|\s+to\s+", label, maxsplit=1, flags=re.I):
            trimmed = part.strip()
            if not trimmed:
                continue
            place, conf = resolve_place(trimmed)
            if place and conf >= 0.72 and place.canonical.lower() not in seen:
                seen.add(place.canonical.lower())
                places.append(place)
    return places


def _is_international_place(place: AviationPlace) -> bool:
    if place.kind == "region" and (place.region or "").lower() in (
        "europe",
        "asia-pacific",
        "middle east",
        "oceania",
        "latin america",
        "caribbean",
        "transatlantic",
    ):
        return True
    return bool(place.country and place.country != "US")


def longest_leg_reasonably_inferable(mission: MissionState, query: str = "") -> bool:
    effective = mission_with_effective_routes(mission, query)
    if mission_max_leg_nm(effective) > 0:
        return True
    ql = (query or "").lower()
    if re.search(
        r"\b(?:transatlantic|transoceanic|transpacific|nonstop\s+to\s+(?:tokyo|london|paris)|"
        r"san\s+francisco\s+to\s+tokyo|westbound)\b",
        ql,
    ):
        return True
    return False


def mission_category_obvious(mission: MissionState, query: str = "") -> bool:
    effective = mission_with_effective_routes(mission, query)
    category = classify_mission_category(effective)
    if category != MissionCategory.GENERAL_ADVISORY:
        return True
    if effective.nonstop_requirement or effective.westbound:
        return True
    if effective.passenger_count is not None and effective.passenger_count >= 12:
        return True
    places = _places_from_route_labels(effective_route_labels(mission, query))
    if any(_is_international_place(p) for p in places):
        return True
    if mission_max_leg_nm(effective) >= 1700:
        return True
    return False


def aircraft_class_inferable(mission: MissionState, query: str = "") -> bool:
    if mission_category_obvious(mission, query):
        return True
    budget = mission.budget_usd
    if budget and (budget < 5_000_000 or budget >= 25_000_000):
        return True
    if budget and not _budget_materially_ambiguous(mission):
        return True
    if (mission.operating_cost_priority or "").lower() == "high":
        return True
    if (mission.cabin_priority or "").lower() == "high":
        return True
    if mission.mountain_airport_requirement:
        return True
    return False


def recommendation_confidence_sufficient(
    recommendations: Optional[List[AircraftRecommendation]],
) -> bool:
    if not recommendations:
        return False
    viable = [r for r in recommendations if not r.avoid]
    if not viable:
        return False
    top = viable[0]
    if top.total_score < RECOMMENDATION_CONFIDENCE_THRESHOLD:
        return False
    if len(viable) >= 2:
        gap = top.total_score - viable[1].total_score
        if gap < RECOMMENDATION_SCORE_GAP_THRESHOLD and top.total_score < 0.62:
            return False
    return True


def _is_us_multi_city_list_query(query: str) -> bool:
    """Comma-separated US cities without an explicit origin→destination pair."""
    ql = (query or "").strip()
    if "," not in ql:
        return False
    if re.search(r"\bto\b|\b->\b|→", ql, re.I):
        return False
    if re.search(
        r"\b(?:london|paris|geneva|tokyo|seoul|europe|asia|caribbean|mexico|dubai)\b",
        ql,
        re.I,
    ):
        return False
    working = re.sub(
        r",?\s*\d{1,2}\s*(?:passengers?|pax|people|executives?|seats?).*$",
        "",
        ql,
        flags=re.I,
    ).strip()
    if "," not in working:
        return False
    if not _US_ONLY_CITY_LIST_RE.match(working.split("\n")[0][:200]):
        return False
    segments = [s.strip() for s in working.split(",") if s.strip()]
    for seg in segments:
        place, conf = resolve_place(seg)
        if place and conf >= 0.72 and _is_international_place(place):
            return False
    return True


def category_usage_fundamentally_ambiguous(mission: MissionState, query: str = "") -> bool:
    """
    US-only multi-stop itinerary where domestic vs transoceanic usage changes class.
    """
    if _DOMESTIC_VS_OCEANIC_RE.search(query or ""):
        return False
    if _is_us_multi_city_list_query(query):
        return True
    effective = mission_with_effective_routes(mission, query)
    routes = effective.routes or []
    if len(routes) < 1:
        return False
    places = _places_from_route_labels(routes)
    if len(places) < 2:
        return False
    if any(_is_international_place(p) for p in places):
        return False
    if mission.passenger_count is not None and mission.passenger_count >= 12:
        return False
    if mission_max_leg_nm(effective) >= 1700:
        return False
    if effective.mountain_airport_requirement:
        return False
    return False


def runway_constraint_materially_ambiguous(mission: MissionState, query: str = "") -> bool:
    """Short-field priority without a pinned airport — class selection shifts."""
    if mission.mountain_airport_requirement:
        return False
    if not _SHORT_RUNWAY_UNPINNED_RE.search(query or ""):
        return False
    if mission.runway_constraints:
        return False
    effective = mission_with_effective_routes(mission, query)
    blob = " ".join(effective.routes).lower()
    if re.search(r"\b(aspen|telluride|jackson|sun\s+valley|teb|teterboro)\b", blob):
        return False
    return True


def _regional_mission_inferable_from_query(query: str) -> bool:
    ql = (query or "").lower()
    return bool(
        re.search(
            r"\b(?:east\s+coast|us\s+east\s+coast|caribbean|miami|aspen|ski\s+trip|"
            r"mountain|runway\s+flex|high[- ]cycle|transcon)\b",
            ql,
        )
    )


def route_truly_missing(mission: MissionState, query: str = "") -> bool:
    if effective_route_labels(mission, query):
        return False
    if longest_leg_reasonably_inferable(mission, query):
        return False
    if _regional_mission_inferable_from_query(query):
        return False
    if _is_ownership_economics_query(query):
        return False
    if aircraft_class_inferable(mission, query) and mission.passenger_count is not None:
        return False
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        if re.search(r"\bvs\.?\b|\bversus\b", query or "", re.I) and len(
            detect_models_from_text(query)
        ) >= 2:
            return False
    except Exception:
        pass
    return True


def _is_ownership_economics_query(query: str) -> bool:
    ql = (query or "").lower()
    return bool(
        re.search(
            r"\b(?:fractional|full\s+ownership|ownership\s+vs|hours\s+(?:a|per)\s+year|"
            r"leaning\s+fractional|overbuying)\b",
            ql,
        )
    )


def mission_clarification_needs(
    mission: MissionState,
    query: str = "",
    recommendations: Optional[List[AircraftRecommendation]] = None,
    *,
    clarifications_already_asked: int = 0,
) -> MissionClarificationNeeds:
    from services.recommendation.question_necessity_engine import (
        evaluate_question_necessity,
    )

    return evaluate_question_necessity(
        mission,
        query,
        recommendations,
        clarifications_already_asked=clarifications_already_asked,
    ).needs


def build_category_usage_question() -> str:
    return "Mostly domestic U.S. or frequent transoceanic? That split changes which aircraft class is the right default."


def build_runway_detail_question() -> str:
    return (
        "Which airports are driving the runway constraint — mountain strips like Aspen, "
        "or shorter metro fields? That changes whether you need a light jet or a larger wing."
    )


def build_clarification_questions(needs: MissionClarificationNeeds) -> List[str]:
    """At most one follow-up — highest materiality first (matches question necessity engine)."""
    if needs.needs_route:
        return []
    if needs.needs_category_usage:
        return [build_category_usage_question()]
    if needs.needs_runway_detail:
        return [build_runway_detail_question()]
    if needs.needs_passenger_count:
        return ["How many passengers are you typically moving on this mission?"]
    if needs.needs_budget:
        from services.recommendation.question_necessity_engine import (
            build_budget_question,
        )

        return [build_budget_question()]
    return []


def mission_well_defined(
    mission: MissionState,
    query: str = "",
    recommendations: Optional[List[AircraftRecommendation]] = None,
    *,
    clarifications_already_asked: int = 0,
) -> bool:
    from services.recommendation.question_necessity_engine import (
        mission_well_defined_from_engine,
    )

    return mission_well_defined_from_engine(
        mission,
        query,
        recommendations,
        clarifications_already_asked=clarifications_already_asked,
    )


def mission_maps_to_category(mission: MissionState, query: str = "") -> bool:
    """Backward-compatible alias — category obvious enough to recommend without more intake."""
    return mission_category_obvious(mission, query) or aircraft_class_inferable(
        mission, query
    )

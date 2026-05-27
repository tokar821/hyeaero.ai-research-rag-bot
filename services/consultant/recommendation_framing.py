"""
Recommendation framing — senior advisor structure for complex missions.

Order for larger/complex missions:
  1. Aircraft category (class)
  2. Operational reality
  3. Specific models

Single-model anchoring only when fit is overwhelmingly obvious or constraints
strongly favor one airframe.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.route_feasibility import RouteFeasibilityAssessment
from services.recommendation.mission_ranker import MissionCategory, classify_mission_category, mission_max_leg_nm

_CATEGORY_CLASS_LABELS = {
    MissionCategory.REGIONAL_UTILITY: "light jet to midsize",
    MissionCategory.MOUNTAIN_AIRPORT: "midsize with strong field performance",
    MissionCategory.COAST_TO_COAST: "super-midsize to large-cabin",
    MissionCategory.TRANSATLANTIC_EXECUTIVE: "large-cabin / long-range",
    MissionCategory.ULTRA_LONG_RANGE: "ultra-long-range",
    MissionCategory.GENERAL_ADVISORY: "the right cabin class for how you fly",
}

_PROFILE_CATEGORY_ORDER = (
    "light",
    "midsize",
    "super-midsize",
    "large",
    "ultra-long",
)

_PROFILE_CATEGORY_LABELS = {
    "light": "light jet",
    "midsize": "midsize",
    "super-midsize": "super-midsize",
    "large": "large-cabin",
    "ultra-long": "ultra-long-range",
}

_SCORE_GAP_OBVIOUS = 0.15
_SCORE_TOP_OBVIOUS = 0.72
_SCORE_TOP_DOMINANT = 0.78


def _normalize_profile_category(raw: str) -> str:
    return (raw or "").strip().lower().replace("_", "-")


def _class_label_from_recommendations(recs: Sequence[AircraftRecommendation]) -> Optional[str]:
    cats: List[str] = []
    for rec in recs[:4]:
        c = _normalize_profile_category(rec.category)
        if c and c not in cats:
            cats.append(c)
    if not cats:
        return None
    if len(cats) == 1:
        return _PROFILE_CATEGORY_LABELS.get(cats[0], cats[0])
    indices = [_PROFILE_CATEGORY_ORDER.index(c) for c in cats if c in _PROFILE_CATEGORY_ORDER]
    if not indices:
        return cats[0]
    lo = _PROFILE_CATEGORY_ORDER[min(indices)]
    hi = _PROFILE_CATEGORY_ORDER[max(indices)]
    if lo == hi:
        return _PROFILE_CATEGORY_LABELS.get(lo, lo)
    return f"{_PROFILE_CATEGORY_LABELS.get(lo, lo)} to {_PROFILE_CATEGORY_LABELS.get(hi, hi)}"


def mission_category_class_label(
    mission: MissionState,
    mission_category: MissionCategory,
    recommendations: Sequence[AircraftRecommendation],
) -> str:
    from_profile = _class_label_from_recommendations(recommendations)
    base = _CATEGORY_CLASS_LABELS.get(mission_category, "the right cabin class")
    if from_profile and mission_category == MissionCategory.GENERAL_ADVISORY:
        return from_profile
    if from_profile and mission_category in (
        MissionCategory.COAST_TO_COAST,
        MissionCategory.TRANSATLANTIC_EXECUTIVE,
    ):
        if " to " in from_profile or from_profile != base:
            return from_profile
    return base


def mission_is_complex(
    mission: MissionState,
    mission_category: MissionCategory,
    recommendations: Sequence[AircraftRecommendation],
    route_assessments: Sequence[RouteFeasibilityAssessment],
) -> bool:
    routes = mission.routes or []
    pax = mission.passenger_count or 0
    max_leg = mission_max_leg_nm(mission)

    if len(routes) >= 2:
        return True
    if pax >= 10:
        return True
    if mission_category in (
        MissionCategory.ULTRA_LONG_RANGE,
        MissionCategory.TRANSATLANTIC_EXECUTIVE,
    ):
        return True
    if max_leg >= 2800:
        return True
    if mission.westbound and max_leg >= 2000:
        return True
    if mission.nonstop_requirement and max_leg >= 1700:
        return True
    if mission.mountain_airport_requirement and max_leg >= 1200:
        return True

    classes = {_normalize_profile_category(r.category) for r in recommendations[:3]}
    classes.discard("")
    if len(classes) >= 2:
        return True

    if route_assessments and any(
        a.classification in ("not_feasible", "practical_restricted") for a in route_assessments
    ):
        return True

    if len(recommendations) >= 2:
        gap = recommendations[0].total_score - recommendations[1].total_score
        if gap < 0.10 and recommendations[0].total_score < 0.68:
            return True

    return False


def should_anchor_single_model(
    mission: MissionState,
    recommendations: Sequence[AircraftRecommendation],
    mission_category: MissionCategory,
) -> bool:
    if not recommendations:
        return False

    top = recommendations[0]
    second = recommendations[1] if len(recommendations) > 1 else None
    gap = (top.total_score - second.total_score) if second else 1.0

    if top.total_score >= _SCORE_TOP_DOMINANT:
        return True
    if top.total_score >= _SCORE_TOP_OBVIOUS and gap >= _SCORE_GAP_OBVIOUS:
        return True

    if mission.mountain_airport_requirement and top.total_score >= 0.65:
        if gap >= 0.12:
            return True

    routes = mission.routes or []
    pax = mission.passenger_count or 0
    if (
        mission_category == MissionCategory.REGIONAL_UTILITY
        and len(routes) == 1
        and 0 < pax <= 8
        and top.total_score >= 0.62
        and gap >= 0.10
    ):
        return True

    return False


def use_tiered_advisor_framing(
    mission: MissionState,
    recommendations: Sequence[AircraftRecommendation],
    route_assessments: Sequence[RouteFeasibilityAssessment],
    *,
    mission_category: Optional[MissionCategory] = None,
) -> bool:
    if not recommendations:
        return False
    category = mission_category or classify_mission_category(mission)
    if should_anchor_single_model(mission, recommendations, category):
        return False
    return mission_is_complex(mission, category, recommendations, route_assessments)


def build_operational_reality_block(
    mission: MissionState,
    route_assessments: Sequence[RouteFeasibilityAssessment],
    *,
    operational_context: str = "",
) -> str:
    parts: List[str] = []
    ctx = (operational_context or "").strip()

    if route_assessments:
        a = route_assessments[0]
        if a.classification == "not_feasible":
            parts.append(
                f"{a.route_label} is a demanding leg on fuel and payload — "
                "even large-cabin jets need conservative planning."
            )
        elif a.caveats:
            clean = a.caveats[0].rstrip(".")
            if clean and len(clean) > 12:
                parts.append(f"On {a.route_label}, {clean.lower()}.")

    if mission.nonstop_requirement and mission.westbound:
        parts.append(
            "Westbound Pacific nonstop work is where brochure range stops mattering — "
            "you need real payload-margin aircraft, not marketing NM."
        )
    elif mission.nonstop_requirement and ctx and "transatlantic" in ctx:
        parts.append(
            "Transatlantic nonstop drives large-cabin range and winter headwind margin — "
            "super-mids can work on some city pairs but not all year-round."
        )
    elif mission.mountain_airport_requirement:
        parts.append(
            "Hot-and-high and short runways eliminate most large-cabin defaults — "
            "field performance and climb matter more than cabin length."
        )
    elif ctx and ctx != "how you're actually using the airplane":
        parts.append(f"Operationally, this is {ctx} — class choice follows from that reality.")

    pax = mission.passenger_count
    if pax and pax >= 10 and not any("passenger" in p.lower() for p in parts):
        parts.append(
            f"At {pax} passengers, cabin volume and baggage matter as much as range — "
            "that's what pushes you out of midsize territory."
        )

    if not parts:
        return ""
    return " ".join(parts[:2]).strip()


def build_category_framing_line(
    mission: MissionState,
    mission_category: MissionCategory,
    recommendations: Sequence[AircraftRecommendation],
    *,
    route_phrase: str = "",
    operational_context: str = "",
) -> str:
    class_label = mission_category_class_label(mission, mission_category, recommendations)
    route = (route_phrase or "").strip()
    pax = mission.passenger_count
    ctx = (operational_context or "this trip").strip()

    if route and pax:
        return (
            f"For {pax} passengers on {route}, you're shopping {class_label} — "
            "not a single-model decision until the operational picture is clear."
        )
    if route:
        return (
            f"On {route}, you're in {class_label} territory — "
            "the right answer is a class decision first, then a tail number."
        )
    if pax:
        return (
            f"With {pax} passengers and {ctx}, {class_label} is the band — "
            "I'd set the mission profile before you commit to one airframe."
        )
    return (
        f"For {ctx}, {class_label} is where serious operators land — "
        "I'd set the class before picking a single model."
    )


def build_model_transition_line(models: Sequence[str], seed: str = "") -> str:
    import hashlib

    options = [
        "Airframes I'd pressure-test in that class:",
        "Specific models that earn a serious look once the class is set:",
        "Where I'd narrow after the operational frame is clear:",
        "Names I'd put on the desk in that category:",
    ]
    if not models:
        return options[0]
    digest = hashlib.sha256((seed or "|".join(models)).encode()).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def build_tiered_conclusion(
    model_names: Sequence[str],
    tradeoff_block: str,
    *,
    seed: str = "",
) -> str:
    import hashlib

    if tradeoff_block:
        base = tradeoff_block.rstrip(".")
        closes = [
            f"{base} Lock the class first, then choose the tail that matches outfit, runway, and hours.",
            f"{base} I'd confirm class fit before committing to one airframe.",
        ]
        digest = hashlib.sha256(f"{seed}:tiered_close".encode()).hexdigest()
        return closes[int(digest[:8], 16) % len(closes)]

    models = [m for m in model_names if m][:3]
    if len(models) >= 2:
        digest = hashlib.sha256(f"{seed}:tiered_default".encode()).hexdigest()
        opts = [
            f"Pressure-test {' and '.join(models[:2])} (and {models[2]} if you want a third data point) before you lock one tail.",
            "Any of these can work — pick the one whose operating economics and runway story match how you'll actually fly.",
        ]
        return opts[int(digest[:8], 16) % len(opts)]
    if models:
        return f"{models[0]} is a credible default once the class fits — confirm runway and outfit before you wire money."
    return ""

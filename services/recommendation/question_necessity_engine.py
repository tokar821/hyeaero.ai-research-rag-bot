"""
Question Necessity Engine — ask follow-ups only when answers materially change outcomes.

Materiality dimensions (only these justify a question):
  - aircraft_category — cabin class / platform band
  - mission_feasibility — whether the mission can be flown as stated
  - ownership_structure — full vs fractional, capital envelope, DOC framing

Do NOT ask when route, passengers, and mission category are already sufficient to recommend
with reasonable assumptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.recommendation.clarification_decision import MissionClarificationNeeds

if TYPE_CHECKING:
    from services.recommendation import clarification_decision as _cd_mod

# Inference confidence thresholds (0–1)
ROUTE_CONFIDENT = 0.72
PASSENGER_CONFIDENT = 0.85
MISSION_CATEGORY_CONFIDENT = 0.70
AIRCRAFT_CLASS_CONFIDENT = 0.68
RECOMMENDATION_READY = 0.62

_MAX_QUESTIONS = 1

_PAX_EXPLICIT_RE = re.compile(
    r"\b(\d{1,2})\s*(?:pax|passengers?|people|executives?|seats?|souls)\b"
    r"|\b(?:for\s+)?(\d{1,2})\s+pax\b"
    r"|\btravel\s+with\s+(\d{1,2})\b",
    re.I,
)

_ACQUISITION_INTENT_RE = re.compile(
    r"\b(?:buy|purchase|acquire|acquisition|what\s+(?:jet|aircraft)\s+should|"
    r"best\s+(?:jet|aircraft)\s+for|recommend|shortlist|budget)\b",
    re.I,
)

_OWNERSHIP_STRUCTURE_RE = re.compile(
    r"\b(?:fractional|full\s+ownership|ownership\s+vs|hours\s+(?:a|per)\s+year|"
    r"leaning\s+fractional|charter\s+vs\s+own)\b",
    re.I,
)


class MaterialityDimension(str, Enum):
    AIRCRAFT_CATEGORY = "aircraft_category"
    MISSION_FEASIBILITY = "mission_feasibility"
    OWNERSHIP_STRUCTURE = "ownership_structure"


@dataclass
class InferenceScores:
    """How confidently the pipeline can infer mission facts without another question."""

    route_confidence: float = 0.0
    passenger_confidence: float = 0.0
    mission_category_confidence: float = 0.0
    aircraft_class_confidence: float = 0.0
    recommendation_readiness: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "route_confidence": round(self.route_confidence, 3),
            "passenger_confidence": round(self.passenger_confidence, 3),
            "mission_category_confidence": round(self.mission_category_confidence, 3),
            "aircraft_class_confidence": round(self.aircraft_class_confidence, 3),
            "recommendation_readiness": round(self.recommendation_readiness, 3),
        }


@dataclass
class QuestionCandidate:
    key: str
    text: str
    materiality: MaterialityDimension
    reason: str
    priority: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "text": self.text,
            "materiality": self.materiality.value,
            "reason": self.reason,
            "priority": self.priority,
        }


@dataclass
class QuestionNecessityReport:
    should_ask_any: bool = False
    should_block_recommendation: bool = False
    questions: List[str] = field(default_factory=list)
    needs: MissionClarificationNeeds = field(default_factory=MissionClarificationNeeds)
    inference: InferenceScores = field(default_factory=InferenceScores)
    suppress_reasons: List[str] = field(default_factory=list)
    candidates: List[QuestionCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_ask_any": self.should_ask_any,
            "should_block_recommendation": self.should_block_recommendation,
            "questions": list(self.questions),
            "needs": {
                "needs_route": self.needs.needs_route,
                "needs_passenger_count": self.needs.needs_passenger_count,
                "needs_budget": self.needs.needs_budget,
                "needs_category_usage": self.needs.needs_category_usage,
                "needs_runway_detail": self.needs.needs_runway_detail,
            },
            "inference": self.inference.to_dict(),
            "suppress_reasons": list(self.suppress_reasons),
            "candidates": [c.to_dict() for c in self.candidates],
        }


def _cd() -> "_cd_mod":
    from services.recommendation import clarification_decision as cd

    return cd


def _passenger_explicit_in_query(query: str) -> bool:
    return bool(_PAX_EXPLICIT_RE.search(query or ""))


def score_route_inference(mission: MissionState, query: str = "") -> float:
    cd = _cd()
    if cd.effective_route_labels(mission, query):
        return 0.95
    if cd.longest_leg_reasonably_inferable(mission, query):
        return 0.88
    if cd._regional_mission_inferable_from_query(query):
        return 0.75
    if cd._is_ownership_economics_query(query):
        return 0.70
    return 0.0


def score_passenger_inference(mission: MissionState, query: str = "") -> float:
    from services.recommendation.mission_ranker import mission_max_leg_nm

    cd = _cd()
    if mission.passenger_count is not None and mission.passenger_count > 0:
        if _passenger_explicit_in_query(query):
            return 0.98
        return 0.90
    if _passenger_explicit_in_query(query):
        return 0.92
    effective = cd.mission_with_effective_routes(mission, query)
    if cd.mission_category_obvious(mission, query) and mission_max_leg_nm(effective) >= 1700:
        return 0.55
    return 0.0


def score_mission_category_inference(mission: MissionState, query: str = "") -> float:
    from services.recommendation.mission_ranker import mission_max_leg_nm

    cd = _cd()
    if cd.mission_category_obvious(mission, query):
        return 0.92
    effective = cd.mission_with_effective_routes(mission, query)
    if mission_max_leg_nm(effective) >= 1200:
        return 0.78
    if mission.nonstop_requirement or mission.westbound:
        return 0.85
    return 0.35


def score_aircraft_class_inference(mission: MissionState, query: str = "") -> float:
    cd = _cd()
    if cd.aircraft_class_inferable(mission, query):
        return 0.88
    if mission.budget_usd and not cd._budget_materially_ambiguous(mission):
        return 0.82
    return 0.40


def score_recommendation_readiness(
    mission: MissionState,
    query: str,
    recommendations: Optional[List[AircraftRecommendation]],
) -> float:
    cd = _cd()
    route_s = score_route_inference(mission, query)
    cat_s = score_mission_category_inference(mission, query)
    class_s = score_aircraft_class_inference(mission, query)
    pax_s = score_passenger_inference(mission, query)

    base = 0.25 * route_s + 0.30 * cat_s + 0.25 * class_s + 0.20 * pax_s
    if cd.recommendation_confidence_sufficient(recommendations):
        base = min(1.0, base + 0.15)
    if cd.route_truly_missing(mission, query):
        return min(base, 0.35)
    return base


def recommendation_possible_with_assumptions(
    inference: InferenceScores,
    mission: MissionState,
    query: str,
    recommendations: Optional[List[AircraftRecommendation]],
) -> bool:
    cd = _cd()
    if cd.route_truly_missing(mission, query):
        return False
    if inference.route_confidence < ROUTE_CONFIDENT:
        return False
    if inference.mission_category_confidence < MISSION_CATEGORY_CONFIDENT:
        if inference.aircraft_class_confidence < AIRCRAFT_CLASS_CONFIDENT:
            return False
    if inference.recommendation_readiness >= RECOMMENDATION_READY:
        return True
    if cd.recommendation_confidence_sufficient(recommendations):
        return True
    return (
        inference.mission_category_confidence >= MISSION_CATEGORY_CONFIDENT
        and inference.aircraft_class_confidence >= AIRCRAFT_CLASS_CONFIDENT
    )


def _is_acquisition_advisory(query: str) -> bool:
    return bool(_ACQUISITION_INTENT_RE.search(query or ""))


def _budget_material_to_category(mission: MissionState, query: str) -> bool:
    cd = _cd()
    if mission.budget_usd is None:
        if _is_acquisition_advisory(query) and _is_ownership_structure_query(query):
            return True
        return False
    if not cd._budget_materially_ambiguous(mission):
        return False
    return not cd.aircraft_class_inferable(mission, query)


def _is_ownership_structure_query(query: str) -> bool:
    cd = _cd()
    return bool(_OWNERSHIP_STRUCTURE_RE.search(query or "")) or cd._is_ownership_economics_query(
        query
    )


def build_budget_question() -> str:
    return "What's your approximate acquisition budget?"


def mission_inferable(inference: InferenceScores, mission: MissionState, query: str) -> bool:
    """Route + mission category are clear enough to size the mission without another question."""
    cd = _cd()
    if inference.route_confidence < ROUTE_CONFIDENT and cd.route_truly_missing(mission, query):
        return False
    if inference.mission_category_confidence >= MISSION_CATEGORY_CONFIDENT:
        return True
    if cd.mission_category_obvious(mission, query):
        return True
    if inference.aircraft_class_confidence >= AIRCRAFT_CLASS_CONFIDENT:
        return True
    return False


def category_recommendation_ready(
    inference: InferenceScores,
    mission: MissionState,
    query: str,
    recommendations: Optional[List[AircraftRecommendation]] = None,
) -> bool:
    """Enough signal to place the mission in an aircraft category band without more intake."""
    cd = _cd()
    if cd.category_usage_fundamentally_ambiguous(mission, query):
        return False
    if cd.mission_category_obvious(mission, query):
        return True
    if inference.mission_category_confidence >= MISSION_CATEGORY_CONFIDENT:
        return True
    if cd.aircraft_class_inferable(mission, query):
        return True
    if recommendation_possible_with_assumptions(inference, mission, query, recommendations):
        return True
    return False


def feasibility_uncertain(
    inference: InferenceScores,
    mission: MissionState,
    query: str,
    recommendations: Optional[List[AircraftRecommendation]] = None,
) -> bool:
    """Mission feasibility or class selection is not yet defensible without one more fact."""
    cd = _cd()
    if cd.route_truly_missing(mission, query):
        return True
    if cd.runway_constraint_materially_ambiguous(mission, query):
        return True
    if recommendations is not None and recommendations and not cd.recommendation_confidence_sufficient(
        recommendations
    ):
        return True
    if inference.recommendation_readiness < RECOMMENDATION_READY:
        if not recommendation_possible_with_assumptions(
            inference, mission, query, recommendations
        ):
            return True
    return False


def question_materially_changes_recommendation(
    key: str,
    mission: MissionState,
    query: str,
    inference: InferenceScores,
) -> bool:
    """Whether answering this gap would change the recommended aircraft category or feasibility."""
    cd = _cd()
    if key == "route":
        return cd.route_truly_missing(mission, query)
    if key == "category_usage":
        return cd.category_usage_fundamentally_ambiguous(mission, query)
    if key == "runway_detail":
        return cd.runway_constraint_materially_ambiguous(mission, query)
    if key == "passengers":
        return (
            mission.passenger_count is None
            and inference.passenger_confidence < PASSENGER_CONFIDENT
        )
    if key == "budget":
        return _budget_material_to_category(mission, query)
    return False


def should_suppress_followups(
    inference: InferenceScores,
    mission: MissionState,
    query: str,
    recommendations: Optional[List[AircraftRecommendation]] = None,
    *,
    clarifications_already_asked: int = 0,
) -> tuple[bool, List[str]]:
    """
    Do not ask follow-ups when route is known, mission is inferable, and category placement is ready.
    """
    reasons: List[str] = []
    if clarifications_already_asked >= _MAX_QUESTIONS:
        reasons.append("clarification_budget_exhausted")
        return True, reasons

    cd = _cd()
    if inference.route_confidence >= ROUTE_CONFIDENT:
        reasons.append("route_already_known")
    elif not cd.route_truly_missing(mission, query):
        reasons.append("route_already_known")

    if mission_inferable(inference, mission, query):
        reasons.append("mission_inferable")

    if category_recommendation_ready(inference, mission, query, recommendations):
        reasons.append("category_recommendation_ready")

    if recommendation_possible_with_assumptions(inference, mission, query, recommendations):
        reasons.append("recommendation_viable_with_assumptions")

    suppress = (
        not cd.route_truly_missing(mission, query)
        and mission_inferable(inference, mission, query)
        and category_recommendation_ready(inference, mission, query, recommendations)
    )
    return suppress, reasons


def build_route_question(mission: MissionState) -> str:
    from services.state.mission_state import MissionState as PersistentMissionState
    from services.state.mission_validation import build_route_clarifying_question

    ps = PersistentMissionState(
        passengers=mission.passenger_count,
        budget_usd=mission.budget_usd,
        nonstop_required=bool(mission.nonstop_requirement),
        westbound=bool(mission.westbound),
    )
    return build_route_clarifying_question(ps)


def evaluate_question_necessity(
    mission: MissionState,
    query: str = "",
    recommendations: Optional[List[AircraftRecommendation]] = None,
    *,
    clarifications_already_asked: int = 0,
) -> QuestionNecessityReport:
    """
    Decide whether to ask follow-up questions and which gaps are material.

    Ask only when feasibility is uncertain or the answer would materially change the
    recommendation. Returns at most one question; after one clarification, recommend with
    stated assumptions.
    """
    cd = _cd()
    report = QuestionNecessityReport()
    ql = (query or "").strip()
    asked = max(0, int(clarifications_already_asked or 0))

    try:
        from services.consultant.visualization_handler import is_visualization_turn

        if is_visualization_turn(ql):
            report.suppress_reasons.append("visualization_turn")
            return report
    except Exception:
        pass

    try:
        from services.state.mission_validation import query_requires_route_for_advisory

        if not query_requires_route_for_advisory(ql):
            report.suppress_reasons.append("not_route_advisory_intent")
            return report
    except Exception:
        pass

    inference = InferenceScores(
        route_confidence=score_route_inference(mission, ql),
        passenger_confidence=score_passenger_inference(mission, ql),
        mission_category_confidence=score_mission_category_inference(mission, ql),
        aircraft_class_confidence=score_aircraft_class_inference(mission, ql),
        recommendation_readiness=score_recommendation_readiness(mission, ql, recommendations),
    )
    report.inference = inference

    suppress, suppress_reasons = should_suppress_followups(
        inference,
        mission,
        ql,
        recommendations,
        clarifications_already_asked=asked,
    )
    report.suppress_reasons.extend(suppress_reasons)

    if asked >= _MAX_QUESTIONS:
        return report

    can_recommend = recommendation_possible_with_assumptions(
        inference, mission, ql, recommendations
    )
    uncertain = feasibility_uncertain(inference, mission, ql, recommendations)

    candidates: List[QuestionCandidate] = []

    if cd.route_truly_missing(mission, ql):
        if suppress:
            report.suppress_reasons.append("route_missing_but_mission_sufficient")
            return report
        report.questions = [build_route_question(mission)]
        report.needs = MissionClarificationNeeds(needs_route=True)
        report.should_ask_any = True
        report.should_block_recommendation = True
        report.candidates = [
            QuestionCandidate(
                key="route",
                text=report.questions[0],
                materiality=MaterialityDimension.MISSION_FEASIBILITY,
                reason="route_missing_changes_feasibility_and_class",
                priority=100,
            )
        ]
        return report

    if suppress:
        return report

    if (
        inference.passenger_confidence < PASSENGER_CONFIDENT
        and mission.passenger_count is None
        and not category_recommendation_ready(inference, mission, ql, recommendations)
    ):
        candidates.append(
            QuestionCandidate(
                key="passengers",
                text="How many passengers are you typically moving on this mission?",
                materiality=MaterialityDimension.AIRCRAFT_CATEGORY,
                reason="passenger_load_changes_cabin_class",
                priority=60,
            )
        )
    elif inference.passenger_confidence >= PASSENGER_CONFIDENT:
        report.suppress_reasons.append("passenger_count_inferred_confidently")

    if cd.category_usage_fundamentally_ambiguous(mission, ql):
        candidates.append(
            QuestionCandidate(
                key="category_usage",
                text=cd.build_category_usage_question(),
                materiality=MaterialityDimension.AIRCRAFT_CATEGORY,
                reason="domestic_vs_transoceanic_changes_default_class",
                priority=85,
            )
        )

    if cd.runway_constraint_materially_ambiguous(mission, ql):
        candidates.append(
            QuestionCandidate(
                key="runway_detail",
                text=cd.build_runway_detail_question(),
                materiality=MaterialityDimension.MISSION_FEASIBILITY,
                reason="unpinned_runway_changes_field_performance_class",
                priority=80,
            )
        )

    if _budget_material_to_category(mission, ql) and not can_recommend:
        candidates.append(
            QuestionCandidate(
                key="budget",
                text=build_budget_question(),
                materiality=MaterialityDimension.OWNERSHIP_STRUCTURE,
                reason="budget_band_changes_aircraft_category_or_ownership_path",
                priority=70,
            )
        )
    elif _budget_material_to_category(mission, ql) and _is_ownership_structure_query(ql):
        candidates.append(
            QuestionCandidate(
                key="budget",
                text=build_budget_question(),
                materiality=MaterialityDimension.OWNERSHIP_STRUCTURE,
                reason="ownership_structure_requires_capital_envelope",
                priority=75,
            )
        )
    elif mission.budget_usd and not cd._budget_materially_ambiguous(mission):
        report.suppress_reasons.append("budget_not_materially_ambiguous")

    candidates = [
        c
        for c in candidates
        if question_materially_changes_recommendation(c.key, mission, ql, inference)
        and (uncertain or c.key in ("category_usage", "budget"))
    ]

    candidates.sort(key=lambda c: -c.priority)
    report.candidates = candidates

    selected = candidates[:_MAX_QUESTIONS]
    report.questions = [c.text for c in selected]
    report.should_ask_any = bool(selected)
    report.should_block_recommendation = any(c.key == "route" for c in selected)

    needs = MissionClarificationNeeds()
    for c in selected:
        if c.key == "route":
            needs.needs_route = True
        elif c.key == "passengers":
            needs.needs_passenger_count = True
        elif c.key == "budget":
            needs.needs_budget = True
        elif c.key == "category_usage":
            needs.needs_category_usage = True
        elif c.key == "runway_detail":
            needs.needs_runway_detail = True
    report.needs = needs

    return report


def mission_well_defined_from_engine(
    mission: MissionState,
    query: str = "",
    recommendations: Optional[List[AircraftRecommendation]] = None,
    *,
    clarifications_already_asked: int = 0,
) -> bool:
    report = evaluate_question_necessity(
        mission,
        query,
        recommendations,
        clarifications_already_asked=clarifications_already_asked,
    )
    if report.should_ask_any:
        return False
    cd = _cd()
    if not (query or "").strip():
        return (
            score_mission_category_inference(mission, query) >= MISSION_CATEGORY_CONFIDENT
            or bool(cd.effective_route_labels(mission, query))
        )
    return recommendation_possible_with_assumptions(
        report.inference, mission, query, recommendations
    )

"""
Weighted aircraft recommendation engine — mission-fit scoring without hardcoded winners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.consultant.mission_state import MissionState
from services.consultant.route_feasibility import assess_mission_routes


@dataclass
class RecommendationScore:
    dimension: str
    score: float  # 0..1
    weight: float
    weighted: float
    note: str = ""


@dataclass
class RecommendationExplanation:
    summary: str
    strengths: List[str] = field(default_factory=list)
    penalties: List[str] = field(default_factory=list)
    operational_caveats: List[str] = field(default_factory=list)
    why_it_fits: List[str] = field(default_factory=list)
    operational_compromises: List[str] = field(default_factory=list)
    why_alternatives_lost: List[str] = field(default_factory=list)


@dataclass
class AircraftRecommendation:
    model: str
    category: str
    total_score: float  # internal sort key only — not user-facing
    confidence: float  # internal telemetry only
    rank: int
    avoid: bool = False
    fit: str = ""  # Strong Fit | Good Fit | Partial Fit | Not Recommended
    fit_verdict: str = ""  # BEST FIT | CONDITIONAL FIT | NOT A FIT (broker-style)
    suitability_score: float = 0.0
    economics_score: float = 0.0
    operational_flexibility_score: float = 0.0
    mission_conflict_penalty: float = 0.0
    scores: List[RecommendationScore] = field(default_factory=list)
    explanation: Optional[RecommendationExplanation] = None

    def to_dict(self) -> Dict[str, Any]:
        from services.recommendation.fit_policy import (
            fit_tier_for_dimension,
            normalize_fit_label,
            score_to_fit_label,
        )

        fit = normalize_fit_label(self.fit or score_to_fit_label(self.total_score, avoid=self.avoid))
        payload: Dict[str, Any] = {
            "model": self.model,
            "category": self.category,
            "fit": fit,
            "fit_verdict": self.fit_verdict or None,
            "avoid": self.avoid,
            "suitability_score": round(float(self.suitability_score), 4) or None,
            "economics_score": round(float(self.economics_score), 4) or None,
            "operational_flexibility_score": round(float(self.operational_flexibility_score), 4) or None,
            "mission_conflict_penalty": round(float(self.mission_conflict_penalty), 4) or None,
            "scores": [
                {
                    "dimension": s.dimension,
                    "fit": fit_tier_for_dimension(s.score),
                    "note": s.note,
                }
                for s in self.scores
            ],
            "explanation": (
                {
                    "summary": self.explanation.summary,
                    "strengths": list(self.explanation.strengths),
                    "penalties": list(self.explanation.penalties),
                    "operational_caveats": list(
                        self.explanation.operational_compromises
                        or self.explanation.operational_caveats
                    ),
                    "why_it_fits": list(self.explanation.why_it_fits or self.explanation.strengths),
                    "operational_compromises": list(
                        self.explanation.operational_compromises
                        or self.explanation.operational_caveats
                    ),
                    "why_alternatives_lost": list(self.explanation.why_alternatives_lost),
                }
                if self.explanation
                else None
            ),
        }
        return payload


from services.mission.aircraft_profiles import AIRCRAFT_PROFILES as _AIRCRAFT_PROFILES

_DIMENSION_WEIGHTS: Dict[str, float] = {
    "range_fit": 0.14,
    "payload_margin": 0.08,
    "runway_flexibility": 0.07,
    "operating_cost": 0.09,
    "cabin_comfort": 0.1,
    "baggage_volume": 0.06,
    "dispatch_reliability": 0.08,
    "resale_strength": 0.06,
    "pilot_workload": 0.05,
    "ownership_efficiency": 0.07,
    "mission_margin": 0.12,
    "westbound_margin": 0.08,
    "airport_compatibility": 0.1,
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _budget_band_usd(budget: Optional[float]) -> Tuple[float, float]:
    if budget is None:
        return 5_000_000.0, 50_000_000.0
    return budget * 0.6, budget * 1.15


def _category_budget_fit(category: str, budget: Optional[float]) -> float:
    if budget is None:
        return 0.7
    mid = {
        "light": 8_000_000.0,
        "super-midsize": 18_000_000.0,
        "large": 28_000_000.0,
        "ultra-long": 45_000_000.0,
    }.get(category, 20_000_000.0)
    lo, hi = _budget_band_usd(budget)
    if lo <= mid <= hi:
        return 0.95
    if mid < lo:
        return _clamp01(0.5 + (mid / max(lo, 1)) * 0.4)
    return _clamp01(0.9 - (mid - hi) / max(hi, 1) * 0.5)


def _score_dimension(
    name: str,
    raw: float,
    weight: float,
    note: str = "",
) -> RecommendationScore:
    w = weight * _DIMENSION_WEIGHTS.get(name, 0.05)
    s = _clamp01(raw)
    return RecommendationScore(dimension=name, score=s, weight=w, weighted=s * w, note=note)


def score_aircraft_for_mission(
    model: str,
    profile: Dict[str, Any],
    mission: MissionState,
    *,
    route_assessments: Optional[List[Any]] = None,
) -> AircraftRecommendation:
    pax = mission.passenger_count or 6
    brochure = float(profile["brochure_nm"])
    practical = float(profile["practical_nm"])
    scores: List[RecommendationScore] = []

    routes = mission.routes or []
    if routes:
        assessments = route_assessments or assess_mission_routes(
            mission,
            aircraft_practical_nm=practical,
            aircraft_brochure_nm=brochure,
            passenger_count=pax,
        )
        if assessments:
            rel = sum(1 for a in assessments if a.reliably_nonstop) / len(assessments)
            prac = sum(1 for a in assessments if a.practical_with_restrictions or a.reliably_nonstop) / len(
                assessments
            )
            range_fit = rel * 0.7 + prac * 0.25
            west = sum(1 for a in assessments if a.westbound_penalty_nm > 0) / len(assessments)
            west_margin = _clamp01(1.0 - west * (0.3 if practical >= 5000 else 0.6))
        else:
            range_fit = _clamp01(practical / 3500.0)
            west_margin = 0.7
    else:
        range_fit = _clamp01(practical / 3200.0)
        west_margin = 0.75

    scores.append(_score_dimension("range_fit", range_fit, 1.0))
    scores.append(_score_dimension("westbound_margin", west_margin, 1.0))
    payload_margin = _clamp01(1.0 - max(0, pax - profile["pax_typical"]) * 0.08)
    scores.append(_score_dimension("payload_margin", payload_margin, 1.0))

    runway_need = 5000 if mission.mountain_airport_requirement else 4200
    runway_score = _clamp01(1.0 - max(0, runway_need - profile["runway_ft"]) / 2500.0)
    scores.append(_score_dimension("runway_flexibility", runway_score, 1.0))

    op_cost = _clamp01(1.1 - float(profile["operating_index"]))
    if (mission.operating_cost_priority or "") == "high":
        op_cost = _clamp01(op_cost * 1.1)
    scores.append(_score_dimension("operating_cost", op_cost, 1.0))

    cabin = float(profile["cabin_score"])
    if (mission.cabin_priority or "") == "high":
        cabin = _clamp01(cabin * 1.05)
    scores.append(_score_dimension("cabin_comfort", cabin, 1.0))
    scores.append(_score_dimension("baggage_volume", float(profile["baggage_score"]), 1.0))
    scores.append(_score_dimension("dispatch_reliability", float(profile["dispatch_score"]), 1.0))
    scores.append(_score_dimension("resale_strength", float(profile["resale_score"]), 1.0))
    scores.append(_score_dimension("pilot_workload", float(profile["pilot_workload"]), 1.0))
    scores.append(_score_dimension("ownership_efficiency", float(profile["ownership_efficiency"]), 1.0))
    scores.append(_score_dimension("mission_margin", range_fit * 0.6 + payload_margin * 0.4, 1.0))
    scores.append(_score_dimension("airport_compatibility", runway_score * 0.7 + payload_margin * 0.3, 1.0))

    total_weight = sum(s.weight for s in scores) or 1.0
    total_score = sum(s.weighted for s in scores) / total_weight

    budget_fit = _category_budget_fit(str(profile["category"]), mission.budget_usd)
    total_score = _clamp01(total_score * 0.85 + budget_fit * 0.15)

    avoid = False
    penalties: List[str] = []
    if routes and route_assessments:
        worst = [a for a in route_assessments if a.classification == "not_feasible"]
        if worst and len(worst) == len(route_assessments):
            avoid = True
            penalties.append("Does not meet practical nonstop margin on stated route(s).")
    if mission.budget_usd and budget_fit < 0.45:
        avoid = True
        penalties.append("Acquisition economics likely misaligned with stated budget.")

    strengths: List[str] = []
    if range_fit >= 0.8:
        strengths.append("Strong mission range margin for stated route(s).")
    if cabin >= 0.85 and (mission.cabin_priority or "") == "high":
        strengths.append("Cabin experience aligns with premium cabin priority.")
    if float(profile["dispatch_score"]) >= 0.85:
        strengths.append("Dispatch reliability track record is a differentiator in this class.")

    from services.recommendation.fit_policy import score_to_fit_label as _fit_label

    expl = RecommendationExplanation(
        summary=f"{model} — {_fit_label(total_score, avoid=avoid)} for this mission.",
        strengths=strengths[:4],
        penalties=penalties[:4],
        operational_caveats=(
            [a.caveats[0] for a in (route_assessments or []) if a.caveats][:3]
            if route_assessments
            else []
        ),
    )

    conf = 0.55 + 0.35 * min(1.0, len(mission.routes) * 0.25 + (1 if mission.passenger_count else 0) * 0.2)
    if mission.budget_usd:
        conf += 0.05

    return AircraftRecommendation(
        model=model,
        category=str(profile["category"]),
        total_score=round(total_score, 4),
        confidence=_clamp01(conf),
        rank=0,
        avoid=avoid,
        fit=_fit_label(total_score, avoid=avoid),
        scores=scores,
        explanation=expl,
    )


def recommendation_from_storage_dict(raw: Dict[str, Any]) -> AircraftRecommendation:
    """Rebuild ``AircraftRecommendation`` from pipeline ``to_dict()`` payload."""
    from services.recommendation.fit_policy import normalize_fit_label

    expl_raw = raw.get("explanation") if isinstance(raw.get("explanation"), dict) else None
    expl = None
    if expl_raw:
        expl = RecommendationExplanation(
            summary=str(expl_raw.get("summary") or ""),
            strengths=list(expl_raw.get("strengths") or []),
            penalties=list(expl_raw.get("penalties") or []),
            operational_caveats=list(
                expl_raw.get("operational_compromises")
                or expl_raw.get("operational_caveats")
                or []
            ),
            why_it_fits=list(expl_raw.get("why_it_fits") or []),
            operational_compromises=list(expl_raw.get("operational_compromises") or []),
            why_alternatives_lost=list(expl_raw.get("why_alternatives_lost") or []),
        )
    return AircraftRecommendation(
        model=str(raw.get("model") or ""),
        category=str(raw.get("category") or ""),
        total_score=0.55,
        confidence=0.6,
        rank=int(raw.get("rank") or 0),
        avoid=bool(raw.get("avoid")),
        fit=normalize_fit_label(str(raw.get("fit") or "")),
        scores=[],
        explanation=expl,
    )


def recommendations_from_storage(
    items: List[Dict[str, Any]],
) -> List[AircraftRecommendation]:
    return [recommendation_from_storage_dict(x) for x in items if isinstance(x, dict) and x.get("model")]


def rank_aircraft_recommendations(
    mission: MissionState,
    *,
    candidate_models: Optional[List[str]] = None,
    max_results: int = 3,
) -> List[AircraftRecommendation]:
    """
    Score and rank aircraft for mission-fit via operational mission ranker.
    """
    from services.recommendation.mission_ranker import rank_missions

    _category, recs, _feas, _audit = rank_missions(
        mission,
        candidate_models=candidate_models,
        max_results=max_results,
    )
    return recs


def detect_models_from_text(text: str) -> List[str]:
    try:
        from rag.consultant_query_expand import _detect_models

        return list(_detect_models(text or ""))
    except Exception:
        return []

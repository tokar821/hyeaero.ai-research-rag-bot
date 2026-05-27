"""
Weighted aircraft scoring — deterministic mission-fit before broker narration.

Nine scored dimensions, five penalty classes, and broker-style fit verdicts.
Ranking applies only inside the surviving operational band — not global cross-category scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import RecommendationScore
from services.consultant.route_feasibility import pick_worst_route_classification
from services.recommendation.mission_ranker import MissionCategory

from services.broker.broker_verdicts import BrokerVerdict, verdict_from_operational_signals

VERDICT_BEST_FIT = BrokerVerdict.PRIMARY_RECOMMENDATION.value
VERDICT_CONDITIONAL_FIT = BrokerVerdict.VIABLE_WITH_COMPROMISES.value
VERDICT_MISSION_RISKY = BrokerVerdict.MISSION_RISKY.value
VERDICT_NOT_A_FIT = BrokerVerdict.NOT_OPERATIONALLY_CREDIBLE.value

_SCORE_DIMENSIONS = (
    "range_realism",
    "runway_performance",
    "dispatch_reliability",
    "operating_economics",
    "cabin_comfort",
    "baggage",
    "resale_liquidity",
    "maintenance_network",
    "passenger_count_fit",
)

_PENALTY_KEYS = (
    "overkill_aircraft",
    "insufficient_range",
    "poor_runway_capability",
    "weak_support_network",
    "high_operating_cost_mismatch",
)

_MAX_PENALTY_DEDUCTION = 0.48


class ScoringDimension(str, Enum):
    RANGE_REALISM = "range_realism"
    RUNWAY_PERFORMANCE = "runway_performance"
    DISPATCH_RELIABILITY = "dispatch_reliability"
    OPERATING_ECONOMICS = "operating_economics"
    CABIN_COMFORT = "cabin_comfort"
    BAGGAGE = "baggage"
    RESALE_LIQUIDITY = "resale_liquidity"
    MAINTENANCE_NETWORK = "maintenance_network"
    PASSENGER_COUNT_FIT = "passenger_count_fit"


_WEIGHTS_BY_MISSION: Dict[MissionCategory, Dict[str, float]] = {
    MissionCategory.REGIONAL_UTILITY: {
        "range_realism": 0.14,
        "runway_performance": 0.12,
        "dispatch_reliability": 0.10,
        "operating_economics": 0.15,
        "cabin_comfort": 0.08,
        "baggage": 0.08,
        "resale_liquidity": 0.07,
        "maintenance_network": 0.09,
        "passenger_count_fit": 0.11,
    },
    MissionCategory.MOUNTAIN_AIRPORT: {
        "range_realism": 0.10,
        "runway_performance": 0.18,
        "dispatch_reliability": 0.10,
        "operating_economics": 0.10,
        "cabin_comfort": 0.07,
        "baggage": 0.08,
        "resale_liquidity": 0.06,
        "maintenance_network": 0.10,
        "passenger_count_fit": 0.08,
    },
    MissionCategory.COAST_TO_COAST: {
        "range_realism": 0.16,
        "runway_performance": 0.09,
        "dispatch_reliability": 0.10,
        "operating_economics": 0.12,
        "cabin_comfort": 0.10,
        "baggage": 0.08,
        "resale_liquidity": 0.07,
        "maintenance_network": 0.09,
        "passenger_count_fit": 0.10,
    },
    MissionCategory.TRANSATLANTIC_EXECUTIVE: {
        "range_realism": 0.18,
        "runway_performance": 0.06,
        "dispatch_reliability": 0.10,
        "operating_economics": 0.08,
        "cabin_comfort": 0.12,
        "baggage": 0.08,
        "resale_liquidity": 0.08,
        "maintenance_network": 0.10,
        "passenger_count_fit": 0.10,
    },
    MissionCategory.ULTRA_LONG_RANGE: {
        "range_realism": 0.20,
        "runway_performance": 0.05,
        "dispatch_reliability": 0.10,
        "operating_economics": 0.07,
        "cabin_comfort": 0.11,
        "baggage": 0.08,
        "resale_liquidity": 0.08,
        "maintenance_network": 0.11,
        "passenger_count_fit": 0.10,
    },
    MissionCategory.GENERAL_ADVISORY: {
        "range_realism": 0.14,
        "runway_performance": 0.10,
        "dispatch_reliability": 0.10,
        "operating_economics": 0.11,
        "cabin_comfort": 0.10,
        "baggage": 0.08,
        "resale_liquidity": 0.08,
        "maintenance_network": 0.10,
        "passenger_count_fit": 0.11,
    },
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _route_blob(mission: MissionState) -> str:
    return " ".join((mission.routes or [])).lower()


def _is_westbound_mission(mission: MissionState) -> bool:
    if mission.westbound:
        return True
    blob = _route_blob(mission)
    return bool(
        re.search(r"westbound|west\s+coast.*europe|sfo.*paris|san\s+francisco.*tokyo", blob)
        or (
            re.search(r"tokyo|europe|london|paris", blob)
            and re.search(r"francisco|angeles|seattle", blob)
        )
    )


def _mission_max_leg_nm(mission: MissionState) -> float:
    from services.consultant.route_feasibility import estimate_route_distance_nm

    routes = mission.routes or []
    if not routes:
        return 0.0
    return max(estimate_route_distance_nm(r) for r in routes)


@dataclass
class ScoringPenalty:
    """Penalty deduction applied after weighted dimension average."""

    key: str
    magnitude: float  # 0..~0.25 per penalty
    reason: str


@dataclass
class WeightedAircraftScore:
    """Full scoring output for one aircraft on one mission."""

    model: str
    category: str
    total_score: float
    dimension_scores: List[RecommendationScore] = field(default_factory=list)
    penalties: List[ScoringPenalty] = field(default_factory=list)
    fit_verdict: str = VERDICT_CONDITIONAL_FIT
    fit_explanation: str = ""
    tradeoffs: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    avoid: bool = False
    overbuying_factor: float = 0.0
    range_margin_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "category": self.category,
            "total_score": round(self.total_score, 4),
            "fit_verdict": self.fit_verdict,
            "fit_explanation": self.fit_explanation,
            "tradeoffs": list(self.tradeoffs),
            "strengths": list(self.strengths),
            "penalties": [
                {"key": p.key, "magnitude": round(p.magnitude, 3), "reason": p.reason}
                for p in self.penalties
            ],
            "dimensions": [
                {
                    "dimension": s.dimension,
                    "score": round(s.score, 3),
                    "weight": s.weight,
                    "note": s.note,
                }
                for s in self.dimension_scores
            ],
        }


def _score_dim(name: str, raw: float, weight: float, note: str = "") -> RecommendationScore:
    s = _clamp01(raw)
    return RecommendationScore(dimension=name, score=s, weight=weight, weighted=s * weight, note=note)


def maintenance_network_score(profile: Dict[str, Any]) -> float:
    """OEM / MRO support proxy — catalog override or dispatch + resale blend."""
    if profile.get("maintenance_network_score") is not None:
        return _clamp01(float(profile["maintenance_network_score"]))
    dispatch = float(profile.get("dispatch_score") or 0.7)
    resale = float(profile.get("resale_score") or 0.7)
    workload = float(profile.get("pilot_workload") or 0.6)
    return _clamp01(0.50 * dispatch + 0.35 * resale + 0.15 * (1.0 - workload * 0.35))


def _runway_need_ft(mission: MissionState) -> int:
    need = 5200 if mission.mountain_airport_requirement else 4500
    if re.search(r"short\s+runway|aspen|telluride", _route_blob(mission), re.I):
        return max(need, 5000)
    return need


def _compute_range_realism(
    *,
    practical_avail: float,
    required_peak: float,
    brochure: float,
    mission: MissionState,
    profile: Dict[str, Any],
    route_assessments: Optional[Sequence[Any]],
) -> Tuple[float, str, float]:
    """Returns (score, note, margin_ratio)."""
    margin_ratio = 1.0
    if required_peak > 0:
        margin_ratio = practical_avail / required_peak
        if margin_ratio >= 1.12:
            score = _clamp01(0.88 + min(0.12, (margin_ratio - 1.12) * 0.3))
        elif margin_ratio >= 1.0:
            score = _clamp01(0.72 + (margin_ratio - 1.0) * 1.3)
        elif margin_ratio >= 0.92:
            score = _clamp01(0.45 + (margin_ratio - 0.92) * 3.0)
        else:
            score = _clamp01(margin_ratio * 0.5)
        note = (
            f"Practical ~{int(practical_avail)} nm vs ~{int(required_peak)} nm required "
            f"(NBAA reserves, payload, winds — not brochure {int(brochure)} nm)."
        )
    else:
        score = _clamp01(practical_avail / 4000.0) * 0.6 + 0.25
        note = "No route stated — range scored from class practical capability."

    if _is_westbound_mission(mission):
        wb = _clamp01(practical_avail / max(required_peak or 5000, 1))
        if str(profile.get("category")) == "ultra-long":
            wb = _clamp01(wb * 1.05 + 0.08)
        elif str(profile.get("category")) in ("light",):
            wb *= 0.35
        score = _clamp01(score * 0.65 + wb * 0.35)
        note += " Westbound / headwind margin included in range realism."

    if route_assessments:
        reliable = sum(1 for a in route_assessments if a.reliably_nonstop) / len(route_assessments)
        practical_ok = sum(
            1 for a in route_assessments if a.practical_with_restrictions or a.reliably_nonstop
        ) / len(route_assessments)
        score = _clamp01(score * 0.5 + reliable * 0.35 + practical_ok * 0.15)
        if pick_worst_route_classification(route_assessments) == "not_feasible":
            score *= 0.35

    return score, note, margin_ratio


def _collect_penalties(
    *,
    model: str,
    profile: Dict[str, Any],
    mission: MissionState,
    mission_category: MissionCategory,
    max_leg: float,
    required_peak: float,
    practical_avail: float,
    margin_ratio: float,
    runway_score: float,
    op_index: float,
    maintenance: float,
    route_assessments: Optional[Sequence[Any]],
) -> List[ScoringPenalty]:
    from services.recommendation.mission_ranker import overbuying_penalty_factor

    penalties: List[ScoringPenalty] = []

    plan_ceiling = None
    try:
        from services.mission.adapters import mission_state_to_profile

        mp = mission_state_to_profile(mission)
        plan_ceiling = mp.planning_band_ceiling
    except Exception:
        pass

    ob_factor, ob_reason = overbuying_penalty_factor(
        mission_category,
        max_leg,
        {**profile, "model": model},
        required_nm=required_peak,
        practical_available=practical_avail,
        passenger_count=int(mission.passenger_count or 6),
        planning_band_ceiling=plan_ceiling,
    )
    if ob_factor >= 0.18 and ob_reason:
        penalties.append(
            ScoringPenalty(
                key="overkill_aircraft",
                magnitude=min(0.28, ob_factor * 0.55),
                reason=ob_reason,
            )
        )

    if required_peak > 0 and margin_ratio < 0.92:
        mag = _clamp01((0.92 - margin_ratio) * 0.55 + 0.08)
        penalties.append(
            ScoringPenalty(
                key="insufficient_range",
                magnitude=mag,
                reason=(
                    f"Practical range margin tight (~{int(practical_avail)} nm available vs "
                    f"~{int(required_peak)} nm required with reserves)."
                ),
            )
        )

    if runway_score < 0.48:
        penalties.append(
            ScoringPenalty(
                key="poor_runway_capability",
                magnitude=_clamp01(0.22 - runway_score * 0.2),
                reason="Runway or hot/high footprint is weak for stated airport environment.",
            )
        )

    if maintenance < 0.52:
        penalties.append(
            ScoringPenalty(
                key="weak_support_network",
                magnitude=_clamp01(0.18 - maintenance * 0.15),
                reason="Parts, MRO, and dispatch support network are thinner than peers in this class.",
            )
        )

    op_mismatch = False
    if (mission.operating_cost_priority or "").lower() == "high" and op_index >= 0.78:
        if mission_category in (
            MissionCategory.REGIONAL_UTILITY,
            MissionCategory.COAST_TO_COAST,
        ):
            op_mismatch = True
    if mission_category == MissionCategory.REGIONAL_UTILITY and op_index >= 0.85:
        op_mismatch = True
    if op_mismatch:
        penalties.append(
            ScoringPenalty(
                key="high_operating_cost_mismatch",
                magnitude=min(0.22, (op_index - 0.65) * 0.35),
                reason="Direct operating cost is high relative to mission economics priorities.",
            )
        )

    cat = str(profile.get("category") or "").lower()
    blob = " ".join(mission.routes or []).lower()
    intl_floor = False
    try:
        from services.mission.adapters import mission_state_to_profile

        mp = mission_state_to_profile(mission)
        intl_floor = bool(mp.international_jet_floor or mp.balanced_cost_dispatch)
    except Exception:
        pass
    if intl_floor and cat in ("light", "turboprop") and re.search(
        r"\b(?:london|paris|berlin|moscow|europe|frankfurt)\b", blob, re.I
    ):
        penalties.append(
            ScoringPenalty(
                key="international_jet_floor",
                magnitude=0.38,
                reason=(
                    "Europe / intercontinental mission — light-jet band lacks winter margin, "
                    "pressurization, and dispatch practicality despite cost priority."
                ),
            )
        )
    pax = int(mission.passenger_count or 6)
    executive_regional = (
        (pax >= 8 or (pax >= 6 and (mission.cabin_priority or "").lower() == "high"))
        and not mission.mountain_airport_requirement
        and (
            "caribbean" in blob
            or "south america" in blob
            or (mission.operating_cost_priority or "").lower() != "high"
        )
    )
    if executive_regional and cat in ("turboprop",):
        penalties.append(
            ScoringPenalty(
                key="executive_cabin_floor",
                magnitude=0.42,
                reason=(
                    "Executive shuttle load — turboprop/utility economics are below the "
                    "pressurized jet dispatch and cabin floor for this profile."
                ),
            )
        )
    if executive_regional and cat == "light" and op_index < 0.52:
        penalties.append(
            ScoringPenalty(
                key="executive_cabin_floor",
                magnitude=0.28,
                reason=(
                    "Light-jet utility band is thin for 8+ executive passengers on "
                    "regional international legs — super-mid dispatch is the planning floor."
                ),
            )
        )

    if route_assessments and all(a.classification == "not_feasible" for a in route_assessments):
        penalties.append(
            ScoringPenalty(
                key="insufficient_range",
                magnitude=0.32,
                reason="Does not meet practical nonstop margin on stated route(s).",
            )
        )

    return penalties


def assign_fit_verdict(
    total_score: float,
    *,
    avoid: bool,
    penalties: Sequence[ScoringPenalty],
    margin_nm: float = 0.0,
) -> str:
    if avoid:
        return VERDICT_NOT_A_FIT
    total_pen = sum(p.magnitude for p in penalties)
    severe_range = any(
        p.key == "insufficient_range" and p.magnitude >= 0.20 for p in penalties
    )
    if severe_range or margin_nm < 0:
        return verdict_from_operational_signals(
            hard_feasible=False,
            margin_nm=margin_nm,
            penalty_total=total_pen,
        ).value
    if total_score < 0.38 or total_pen >= 0.32:
        return verdict_from_operational_signals(
            hard_feasible=True,
            margin_nm=margin_nm,
            penalty_total=total_pen,
        ).value
    # Mission-risky is a penalty-driven verdict, not purely a small-margin label.
    if total_pen >= 0.28:
        return VERDICT_MISSION_RISKY
    broker = verdict_from_operational_signals(
        hard_feasible=True,
        margin_nm=max(margin_nm, 400 if total_score >= 0.68 else 200),
        penalty_total=total_pen,
    )
    if broker == BrokerVerdict.PRIMARY_RECOMMENDATION:
        # Allow primary recommendations when margin is close but not extreme.
        # Over-tight caps here can incorrectly re-label workable missions
        # as MISSION-RISKY for modest margin deficits.
        if total_score < 0.55 or margin_nm < 200:
            return VERDICT_MISSION_RISKY
    return broker.value


def build_fit_explanation(
    model: str,
    verdict: str,
    strengths: Sequence[str],
    penalties: Sequence[ScoringPenalty],
) -> str:
    if verdict == VERDICT_BEST_FIT:
        lead = strengths[0] if strengths else "Range, runway, and operating profile align with the mission."
        return f"{model} — {lead}"
    if verdict == VERDICT_NOT_A_FIT:
        reason = penalties[0].reason if penalties else "Operational margins are too thin for a confident recommendation."
        return f"{model} — {VERDICT_NOT_A_FIT}: {reason}"
    if verdict == VERDICT_MISSION_RISKY:
        reason = penalties[0].reason if penalties else "Margin-tight on stated leg."
        return f"{model} — {VERDICT_MISSION_RISKY}: {reason}"
    if penalties:
        return f"{model} — {VERDICT_CONDITIONAL_FIT}: {penalties[0].reason}"
    return f"{model} — Viable with operational tradeoffs; verify payload, runway, and cost assumptions."


def score_aircraft_weighted(
    model: str,
    profile: Dict[str, Any],
    mission: MissionState,
    *,
    mission_category: MissionCategory,
    route_assessments: Optional[Sequence[Any]] = None,
    operational_context: Optional[Any] = None,
) -> WeightedAircraftScore:
    """
    Score one aircraft against a mission using weighted dimensions and penalty deductions.
    """
    from services.recommendation.mission_ranker import (
        overbuying_penalty_factor,
        practical_available_nm,
        required_mission_nm,
    )

    pax = int(mission.passenger_count or 6)
    brochure = float(profile.get("brochure_nm") or 0)
    max_leg = _mission_max_leg_nm(mission)
    routes = mission.routes or []

    required_peak = 0.0
    practical_avail = practical_available_nm(profile, mission=mission)
    if operational_context is not None:
        try:
            from services.operational.mission_operational_assessment import assess_aircraft_operational

            op = assess_aircraft_operational(model, profile, operational_context)
            required_peak = op.required_nm
            practical_avail = op.effective_practical_nm
        except Exception:
            pass
    if required_peak <= 0 and routes:
        required_peak = max(
            required_mission_nm(mission, route_label=r, passenger_count=pax)[0] for r in routes
        )

    weights = dict(_WEIGHTS_BY_MISSION.get(mission_category, _WEIGHTS_BY_MISSION[MissionCategory.GENERAL_ADVISORY]))
    w_sum = sum(weights.values()) or 1.0
    weights = {k: v / w_sum for k, v in weights.items()}

    range_score, range_note, margin_ratio = _compute_range_realism(
        practical_avail=practical_avail,
        required_peak=required_peak,
        brochure=brochure,
        mission=mission,
        profile=profile,
        route_assessments=route_assessments,
    )

    typical = int(profile.get("pax_typical") or 8)
    if pax <= typical:
        pax_score = _clamp01(0.85 + (typical - pax) * 0.02)
        pax_note = f"Typical {typical}-seat layout covers {pax} passengers."
    else:
        pax_score = _clamp01(1.0 - (pax - typical) * 0.09)
        pax_note = f"{pax} passengers exceed typical {typical}-seat configuration."

    runway_need = _runway_need_ft(mission)
    runway_ft = float(profile.get("runway_ft") or 5000)
    runway_score = _clamp01(1.0 - max(0, runway_need - runway_ft) / 2200.0)
    if str(profile.get("category")) == "light" and mission.mountain_airport_requirement:
        runway_score *= 0.55
    short_field = float(profile.get("short_field_score") or 0.5)
    if runway_need >= 5000:
        runway_score = _clamp01(runway_score * 0.7 + short_field * 0.3)

    op_index = float(profile.get("operating_index") or 0.7)
    op_score = _clamp01(1.08 - op_index)
    if (mission.operating_cost_priority or "").lower() == "high":
        op_score = _clamp01(op_score * 1.12)
    if mission_category == MissionCategory.REGIONAL_UTILITY:
        op_score = _clamp01(op_score * 1.08)

    cabin = float(profile.get("cabin_score") or 0.7)
    if (mission.cabin_priority or "").lower() == "high":
        cabin = _clamp01(cabin * 1.08)

    bag_score = float(profile.get("baggage_score") or 0.6)
    if (mission.baggage_priority or "").lower() == "high":
        bag_score = _clamp01(bag_score * 1.1)

    dispatch = float(profile.get("dispatch_score") or 0.75)
    if operational_context is not None:
        try:
            from services.operational.mission_operational_assessment import assess_aircraft_operational

            op_dispatch = assess_aircraft_operational(model, profile, operational_context)
            dispatch = op_dispatch.dispatch.reliability_score
            if not op_dispatch.dispatch.works_reliably and op_dispatch.dispatch.technically_possible:
                dispatch = min(dispatch, 0.58)
        except Exception:
            pass
    resale = float(profile.get("resale_score") or 0.7)
    maintenance = maintenance_network_score(profile)

    dimensions: List[RecommendationScore] = [
        _score_dim("range_realism", range_score, weights["range_realism"], range_note),
        _score_dim(
            "runway_performance",
            runway_score,
            weights["runway_performance"],
            f"Runway need ~{runway_need} ft vs aircraft ~{int(runway_ft)} ft.",
        ),
        _score_dim("dispatch_reliability", dispatch, weights["dispatch_reliability"]),
        _score_dim(
            "operating_economics",
            op_score,
            weights["operating_economics"],
            "Lower operating index favors high-cycle missions.",
        ),
        _score_dim("cabin_comfort", cabin, weights["cabin_comfort"]),
        _score_dim("baggage", bag_score, weights["baggage"]),
        _score_dim("resale_liquidity", resale, weights["resale_liquidity"]),
        _score_dim(
            "maintenance_network",
            maintenance,
            weights["maintenance_network"],
            "OEM / MRO support and parts network.",
        ),
        _score_dim("passenger_count_fit", pax_score, weights["passenger_count_fit"], pax_note),
    ]

    total_weight = sum(s.weight for s in dimensions) or 1.0
    base_total = sum(s.weighted for s in dimensions) / total_weight

    penalties = _collect_penalties(
        model=model,
        profile=profile,
        mission=mission,
        mission_category=mission_category,
        max_leg=max_leg,
        required_peak=required_peak,
        practical_avail=practical_avail,
        margin_ratio=margin_ratio,
        runway_score=runway_score,
        op_index=op_index,
        maintenance=maintenance,
        route_assessments=route_assessments,
    )
    penalty_deduction = min(_MAX_PENALTY_DEDUCTION, sum(p.magnitude for p in penalties))
    total_score = _clamp01(base_total - penalty_deduction)

    ob_factor, _ = overbuying_penalty_factor(
        mission_category,
        max_leg,
        {**profile, "model": model},
        required_nm=required_peak,
        practical_available=practical_avail,
    )

    avoid = False
    if route_assessments and all(a.classification == "not_feasible" for a in route_assessments):
        avoid = True
    if ob_factor >= 0.5 and mission_category == MissionCategory.REGIONAL_UTILITY:
        avoid = True
    if margin_ratio < 0.85 and required_peak > 0:
        avoid = avoid or ob_factor >= 0.45

    strengths: List[str] = []
    tradeoffs: List[str] = []

    if range_score >= 0.78 and required_peak > 0:
        strengths.append(
            f"Practical range supports the mission (~{int(practical_avail)} nm vs ~{int(required_peak)} nm required)."
        )
    if runway_score >= 0.8 and mission.mountain_airport_requirement:
        strengths.append("Runway and field performance suit mountain or hot/high operations.")
    if op_score >= 0.75 and mission_category == MissionCategory.REGIONAL_UTILITY:
        strengths.append("Operating economics fit regional utility missions.")
    if cabin >= 0.82:
        strengths.append("Cabin comfort is strong for this passenger load.")
    if dispatch >= 0.85:
        strengths.append("Dispatch reliability and support are above class average.")

    for p in penalties:
        if p.key == "overkill_aircraft":
            tradeoffs.append(p.reason)
        elif p.key == "high_operating_cost_mismatch":
            tradeoffs.append(p.reason)
        elif p.key in ("insufficient_range", "poor_runway_capability"):
            tradeoffs.append(p.reason)

    if str(profile.get("category")) == "ultra-long" and mission_category != MissionCategory.ULTRA_LONG_RANGE:
        tradeoffs.append("Higher acquisition, crew, and airport fees than a right-sized jet.")
    if pax_score < 0.65:
        tradeoffs.append("Passenger load may force fuel stops or reduced baggage on longest legs.")

    margin_nm = max(0.0, practical_avail - required_peak) if required_peak else practical_avail * 0.15
    verdict = assign_fit_verdict(
        total_score,
        avoid=avoid,
        penalties=penalties,
        margin_nm=margin_nm,
    )
    explanation = build_fit_explanation(model, verdict, strengths, penalties)

    return WeightedAircraftScore(
        model=model,
        category=str(profile.get("category") or ""),
        total_score=round(total_score, 4),
        dimension_scores=dimensions,
        penalties=penalties,
        fit_verdict=verdict,
        fit_explanation=explanation,
        tradeoffs=tradeoffs[:6],
        strengths=strengths[:4],
        avoid=avoid,
        overbuying_factor=ob_factor,
        range_margin_ratio=margin_ratio,
    )


def weighted_verdict_to_qualitative_fit(verdict: str, *, avoid: bool = False) -> str:
    """Map broker verdict to internal qualitative fit labels."""
    from services.recommendation.fit_policy import (
        FIT_GOOD,
        FIT_NOT_RECOMMENDED,
        FIT_PARTIAL,
        FIT_STRONG,
    )

    if avoid or verdict == VERDICT_NOT_A_FIT:
        return FIT_NOT_RECOMMENDED
    if verdict == VERDICT_BEST_FIT:
        return FIT_STRONG
    if verdict == VERDICT_CONDITIONAL_FIT:
        return FIT_GOOD
    if verdict == VERDICT_MISSION_RISKY:
        return FIT_PARTIAL
    return FIT_PARTIAL


__all__ = [
    "VERDICT_BEST_FIT",
    "VERDICT_CONDITIONAL_FIT",
    "VERDICT_MISSION_RISKY",
    "VERDICT_NOT_A_FIT",
    "ScoringDimension",
    "ScoringPenalty",
    "WeightedAircraftScore",
    "assign_fit_verdict",
    "build_fit_explanation",
    "maintenance_network_score",
    "score_aircraft_weighted",
    "weighted_verdict_to_qualitative_fit",
]

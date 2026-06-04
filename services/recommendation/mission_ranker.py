"""
Mission-fit aircraft ranker — operational realism, overbuying penalties, tradeoff outputs.

Never scores brochure range directly; uses practical NM with NBAA / payload / wind margins.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import (
    AircraftRecommendation,
    RecommendationExplanation,
    RecommendationScore,
)
from services.mission.adapters import mission_state_to_profile
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES as _AIRCRAFT_PROFILES
from services.mission.feasibility_engine import FeasibilityResult, filter_feasible_aircraft
from services.mission.models import MissionProfile
from services.consultant.route_feasibility import (
    _MOUNTAIN_PAYLOAD_PENALTY_NM,
    _NBAA_RESERVE_NM,
    _WESTBOUND_EXTRA,
    _WINTER_WESTBOUND_EXTRA,
    assess_mission_routes,
    estimate_route_distance_nm,
    pick_worst_route_classification,
)

# Re-export for tests / intelligence layer
AIRCRAFT_PROFILES = _AIRCRAFT_PROFILES

_ULTRA_LONG_MODELS = frozenset(
    {"Falcon 8X", "Gulfstream G650", "Gulfstream G650ER", "Global 7500", "Global 6500"}
)
_MOUNTAIN_FORBIDDEN_MODELS = frozenset(
    _ULTRA_LONG_MODELS | {"Gulfstream G650", "Citation CJ2"}
)
_LIGHT_MODELS = frozenset({"Citation CJ2", "Citation CJ4", "Learjet 75"})


class MissionCategory(str, Enum):
    TRANSATLANTIC_EXECUTIVE = "transatlantic_executive"
    REGIONAL_UTILITY = "regional_utility"
    MOUNTAIN_AIRPORT = "mountain_airport"
    COAST_TO_COAST = "coast_to_coast"
    ULTRA_LONG_RANGE = "ultra_long_range"
    GENERAL_ADVISORY = "general_advisory"


_CATEGORY_WEIGHTS: Dict[MissionCategory, Dict[str, float]] = {
    MissionCategory.REGIONAL_UTILITY: {
        "route_realism": 0.17,
        "passenger_load": 0.1,
        "runway_performance": 0.13,
        "operating_economics": 0.15,
        "winter_westbound_margin": 0.04,
        "baggage_practicality": 0.08,
        "dispatch_reliability": 0.09,
        "ownership_economics": 0.11,
        "overbuying_penalty": 0.08,
        "modern_operational_fit": 0.09,
    },
    MissionCategory.MOUNTAIN_AIRPORT: {
        "route_realism": 0.11,
        "passenger_load": 0.08,
        "runway_performance": 0.19,
        "operating_economics": 0.1,
        "winter_westbound_margin": 0.06,
        "baggage_practicality": 0.09,
        "dispatch_reliability": 0.11,
        "ownership_economics": 0.08,
        "overbuying_penalty": 0.13,
        "modern_operational_fit": 0.09,
    },
    MissionCategory.COAST_TO_COAST: {
        "route_realism": 0.15,
        "passenger_load": 0.1,
        "runway_performance": 0.09,
        "operating_economics": 0.13,
        "winter_westbound_margin": 0.09,
        "baggage_practicality": 0.08,
        "dispatch_reliability": 0.09,
        "ownership_economics": 0.09,
        "overbuying_penalty": 0.11,
        "modern_operational_fit": 0.1,
    },
    MissionCategory.TRANSATLANTIC_EXECUTIVE: {
        "route_realism": 0.19,
        "passenger_load": 0.1,
        "runway_performance": 0.06,
        "operating_economics": 0.07,
        "winter_westbound_margin": 0.13,
        "baggage_practicality": 0.08,
        "dispatch_reliability": 0.09,
        "ownership_economics": 0.07,
        "overbuying_penalty": 0.15,
        "modern_operational_fit": 0.1,
    },
    MissionCategory.ULTRA_LONG_RANGE: {
        "route_realism": 0.21,
        "passenger_load": 0.1,
        "runway_performance": 0.05,
        "operating_economics": 0.06,
        "winter_westbound_margin": 0.13,
        "baggage_practicality": 0.08,
        "dispatch_reliability": 0.09,
        "ownership_economics": 0.07,
        "overbuying_penalty": 0.17,
        "modern_operational_fit": 0.1,
    },
    MissionCategory.GENERAL_ADVISORY: {
        "route_realism": 0.13,
        "passenger_load": 0.1,
        "runway_performance": 0.09,
        "operating_economics": 0.11,
        "winter_westbound_margin": 0.09,
        "baggage_practicality": 0.08,
        "dispatch_reliability": 0.09,
        "ownership_economics": 0.09,
        "overbuying_penalty": 0.14,
        "modern_operational_fit": 0.12,
    },
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _route_blob(mission: MissionState) -> str:
    return " ".join((mission.routes or [])).lower()


def mission_max_leg_nm(mission: MissionState) -> float:
    routes = mission.routes or []
    if not routes:
        return 0.0
    return max(estimate_route_distance_nm(r) for r in routes)


def _is_westbound_mission(mission: MissionState) -> bool:
    if mission.westbound:
        return True
    blob = _route_blob(mission)
    return bool(
        re.search(r"westbound|west\s+coast.*europe|sfo.*paris|san\s+francisco.*tokyo", blob)
        or (re.search(r"tokyo|europe|london|paris", blob) and re.search(r"francisco|angeles|seattle", blob))
    )


def _is_transpacific(mission: MissionState) -> bool:
    blob = _route_blob(mission)
    if re.search(r"\btranspacific\b", blob):
        return True
    max_leg = mission_max_leg_nm(mission)
    if max_leg > 0 and max_leg < 3500:
        return False
    return bool(
        max_leg >= 4200
        and re.search(r"tokyo|seoul|singapore|beijing|hong\s+kong", blob)
        and re.search(r"san\s+francisco|los\s+angeles|seattle|new\s+york|nyc|west\s+coast", blob)
    )


def classify_mission_category(mission: MissionState) -> MissionCategory:
    """Infer operational mission class from routes and constraints."""
    if mission.mountain_airport_requirement or re.search(
        r"aspen|telluride|jackson|sun\s+valley|hot/high|hot\s+and\s+high|mountain\s+airport",
        _route_blob(mission),
        re.I,
    ):
        return MissionCategory.MOUNTAIN_AIRPORT

    max_leg = mission_max_leg_nm(mission)
    blob = _route_blob(mission)

    if max_leg >= 4800 or (_is_transpacific(mission) and max_leg >= 4200):
        return MissionCategory.ULTRA_LONG_RANGE
    if max_leg >= 2800 or re.search(r"london|paris|geneva|europe", blob):
        return MissionCategory.TRANSATLANTIC_EXECUTIVE
    if max_leg >= 1700:
        return MissionCategory.COAST_TO_COAST
    if max_leg > 0 and max_leg < 1200:
        return MissionCategory.REGIONAL_UTILITY
    if max_leg >= 1200:
        return MissionCategory.COAST_TO_COAST
    return MissionCategory.GENERAL_ADVISORY


def required_mission_nm(
    mission: MissionState,
    *,
    route_label: str,
    passenger_count: int,
) -> Tuple[float, float, str]:
    """
    Practical mission distance (nm) — never brochure.

    Returns ``(required_nm, westbound_penalty_nm, notes)``.
    """
    dist = estimate_route_distance_nm(route_label)
    reserve = _NBAA_RESERVE_NM
    west_pen = 0.0
    notes: List[str] = []

    if _is_westbound_mission(mission) or re.search(r"westbound", route_label, re.I):
        west_pen = dist * _WESTBOUND_EXTRA
        if (mission.seasonal_constraints or "").lower().find("winter") >= 0 or re.search(
            r"winter", route_label, re.I
        ):
            west_pen += dist * (_WINTER_WESTBOUND_EXTRA - _WESTBOUND_EXTRA)
            notes.append("Winter westbound headwind margin applied.")

    payload_pen = 0.0
    if passenger_count >= 10:
        payload_pen += 200.0
        notes.append("High passenger count reduces effective range.")
    elif passenger_count >= 8:
        payload_pen += 120.0
    if (mission.baggage_priority or "") == "high":
        payload_pen += 80.0
        notes.append("Baggage priority reduces payload-range.")
    if mission.mountain_airport_requirement:
        payload_pen += _MOUNTAIN_PAYLOAD_PENALTY_NM
        notes.append("Mountain / hot-high runway erodes range.")

    required = dist + reserve + west_pen + payload_pen
    return required, west_pen, "; ".join(notes)


def practical_available_nm(profile: Dict[str, Any], *, mission: MissionState) -> float:
    """Operational range budget — practical NM only (no brochure)."""
    practical = float(profile["practical_nm"])
    pax = mission.passenger_count or 6
    if pax > int(profile["pax_typical"]):
        practical -= min(400.0, (pax - profile["pax_typical"]) * 35.0)
    if mission.mountain_airport_requirement:
        practical -= 250.0
    return max(practical * 0.85, 800.0)


def overbuying_penalty_factor(
    mission_category: MissionCategory,
    max_leg_nm: float,
    profile: Dict[str, Any],
    *,
    required_nm: float,
    practical_available: float,
    passenger_count: int = 6,
    planning_band_ceiling: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Penalize ultra-long / large jets on short regional missions and modest loads.
    """
    cat = str(profile.get("category") or "")
    practical = float(profile["practical_nm"])
    ratio = practical_available / max(required_nm, 1.0) if required_nm > 0 else 2.0

    penalty = 0.0
    reason = ""

    ceiling = (planning_band_ceiling or "").lower().replace("-", "_")
    modest_load = passenger_count <= 4

    if cat == "ultra-long":
        if modest_load and max_leg_nm < 5200:
            penalty = max(penalty, 0.62)
            reason = (
                f"{passenger_count} passengers on ~{int(max_leg_nm)} nm stage — "
                "ULR-class platform is capital and cabin oversizing; super-mid + supplemental charter "
                "is the rational planning frame."
            )
        elif ceiling == "super_midsize":
            penalty = max(penalty, 0.58)
            reason = "Mission understanding caps planning at super-mid — ULR-class is overspec."
        if max_leg_nm < 1200 or mission_category == MissionCategory.REGIONAL_UTILITY:
            penalty = 0.55
            reason = "Ultra-long-range jet is operational overbuy for a short regional leg."
        elif max_leg_nm < 2000:
            penalty = 0.42
            reason = "Ultra-long-range economics and runway footprint are misaligned with this stage length."
        elif max_leg_nm < 3200 and ratio > 1.35:
            penalty = 0.28
            reason = "Excess range capability trades away operating economics on this mission."
        elif ratio > 1.5:
            penalty = 0.18
            reason = "Range surplus is larger than the mission requires — expect higher DOC and fees."

    elif cat == "large" and max_leg_nm < 900:
        penalty = 0.35
        reason = "Large-cabin aircraft is heavier than needed for this short hop."
    elif cat == "large" and max_leg_nm < 1500 and mission_category == MissionCategory.REGIONAL_UTILITY:
        penalty = 0.22
        reason = "Large-cabin footprint is hard to justify on regional utility missions."

    if profile.get("model") in _ULTRA_LONG_MODELS and mission_category == MissionCategory.REGIONAL_UTILITY:
        penalty = max(penalty, 0.5)

    if cat in ("super-midsize", "midsize") and mission_category == MissionCategory.ULTRA_LONG_RANGE:
        if max_leg_nm >= 4000:
            penalty = max(penalty, 0.52)
            reason = reason or "Super-midsize is undersized for ultra-long-range mission class."
        elif max_leg_nm >= 3200:
            penalty = max(penalty, 0.38)
            reason = reason or "Super-midsize lacks reliable margin for this stage length."

    if cat in ("super-midsize",) and mission_category == MissionCategory.TRANSATLANTIC_EXECUTIVE:
        if max_leg_nm >= 3200 and required_nm > 0 and practical_available < required_nm * 1.08:
            penalty = max(penalty, 0.35)
            reason = reason or "Practical range too tight for consistent transatlantic nonstop."

    return _clamp01(penalty), reason


def _score_dim(name: str, raw: float, weight: float, note: str = "") -> RecommendationScore:
    s = _clamp01(raw)
    return RecommendationScore(dimension=name, score=s, weight=weight, weighted=s * weight, note=note)


def score_aircraft_for_mission_ranked(
    model: str,
    profile: Dict[str, Any],
    mission: MissionState,
    *,
    mission_category: MissionCategory,
    route_assessments: Optional[List[Any]] = None,
    operational_context: Optional[Any] = None,
) -> AircraftRecommendation:
    from services.recommendation.aircraft_archetype_weighting import modern_operational_fit_for_ranking
    from services.recommendation.weighted_aircraft_scoring import (
        score_aircraft_weighted,
        weighted_verdict_to_qualitative_fit,
    )

    weighted = score_aircraft_weighted(
        model,
        profile,
        mission,
        mission_category=mission_category,
        route_assessments=route_assessments,
        operational_context=operational_context,
    )

    scores: List[RecommendationScore] = list(weighted.dimension_scores)
    legacy_w = _CATEGORY_WEIGHTS.get(mission_category, _CATEGORY_WEIGHTS[MissionCategory.GENERAL_ADVISORY])

    range_dim = next((s for s in scores if s.dimension == "range_realism"), None)
    runway_dim = next((s for s in scores if s.dimension == "runway_performance"), None)
    op_dim = next((s for s in scores if s.dimension == "operating_economics"), None)
    bag_dim = next((s for s in scores if s.dimension == "baggage"), None)

    if range_dim:
        scores.append(
            _score_dim(
                "route_realism",
                range_dim.score,
                legacy_w.get("route_realism", 0.14),
                range_dim.note,
            )
        )
    if bag_dim:
        scores.append(
            _score_dim(
                "baggage_practicality",
                bag_dim.score,
                legacy_w.get("baggage_practicality", 0.08),
                bag_dim.note,
            )
        )
    pax_dim = next((s for s in scores if s.dimension == "passenger_count_fit"), None)
    if pax_dim:
        scores.append(
            _score_dim(
                "passenger_load",
                pax_dim.score,
                legacy_w.get("passenger_load", 0.1),
                pax_dim.note,
            )
        )

    own_score = float(profile.get("ownership_efficiency") or 0.65)
    if (mission.acquisition_strategy or "") in ("fractional", "charter"):
        own_score = _clamp01(own_score * 1.05)
    scores.append(
        _score_dim(
            "ownership_economics",
            own_score,
            legacy_w.get("ownership_economics", 0.09),
        )
    )

    wb_score = 0.72
    wb_note = "No westbound penalty on stated mission."
    if _is_westbound_mission(mission):
        practical_avail = practical_available_nm(profile, mission=mission)
        routes = mission.routes or []
        pax = mission.passenger_count or 6
        required_peak = 0.0
        if routes:
            required_peak = max(
                required_mission_nm(mission, route_label=r, passenger_count=pax)[0] for r in routes
            )
        wb_score = _clamp01(practical_avail / max(required_peak or 5000, 1))
        wb_note = "Westbound / headwind-sensitive mission — margin weighted heavily."
    scores.append(
        _score_dim(
            "winter_westbound_margin",
            wb_score,
            legacy_w.get("winter_westbound_margin", 0.08),
            wb_note,
        )
    )

    ob_score = _clamp01(1.0 - weighted.overbuying_factor)
    ob_reason = next((p.reason for p in weighted.penalties if p.key == "overkill_aircraft"), "")
    scores.append(
        _score_dim(
            "overbuying_penalty",
            ob_score,
            legacy_w.get("overbuying_penalty", 0.1),
            ob_reason or "Right-sized for stage length.",
        )
    )

    mod_score, mod_note = modern_operational_fit_for_ranking(model, profile)
    scores.append(
        _score_dim(
            "modern_operational_fit",
            mod_score,
            legacy_w.get("modern_operational_fit", 0.08),
            mod_note,
        )
    )

    penalties = [p.reason for p in weighted.penalties]
    compromises = list(weighted.tradeoffs)
    if route_assessments:
        for a in route_assessments:
            compromises.extend(a.caveats[:2])

    op_assessment = None
    if operational_context is not None:
        try:
            from services.operational.mission_operational_assessment import (
                assess_aircraft_operational,
                apply_verdict_cap,
            )

            op_assessment = assess_aircraft_operational(model, profile, operational_context)
            compromises.extend(op_assessment.operational_caveats[:4])
            weighted.fit_verdict = apply_verdict_cap(
                weighted.fit_verdict,
                op_assessment.recommended_verdict_cap,
            )
            if not op_assessment.dispatch.works_reliably and op_assessment.dispatch.technically_possible:
                penalties.append(
                    "Dispatch reliability degrades under seasonal/payload pressure — "
                    "technically possible, not reliably dispatchable."
                )
        except Exception:
            pass

    why_fits = weighted.strengths[:3] if weighted.strengths else [
        f"{model} covers the stage length and passenger load without stretching the operation beyond what the trip requires."
    ]

    expl = RecommendationExplanation(
        summary=weighted.fit_explanation,
        strengths=why_fits,
        penalties=penalties[:4],
        operational_caveats=compromises[:5],
        why_it_fits=why_fits,
        operational_compromises=compromises[:5],
        why_alternatives_lost=[],
    )

    routes = mission.routes or []
    conf = 0.52 + 0.28 * min(1.0, len(routes) * 0.3 + (0.15 if mission.passenger_count else 0))
    if mission_category != MissionCategory.GENERAL_ADVISORY:
        conf += 0.08

    qual_fit = weighted_verdict_to_qualitative_fit(weighted.fit_verdict, avoid=weighted.avoid)

    return AircraftRecommendation(
        model=model,
        category=weighted.category,
        total_score=weighted.total_score,
        confidence=_clamp01(conf),
        rank=0,
        avoid=weighted.avoid,
        fit=qual_fit,
        fit_verdict=weighted.fit_verdict,
        scores=scores,
        explanation=expl,
    )


def _attach_alternative_loss_reasons(recs: List[AircraftRecommendation]) -> None:
    if len(recs) < 2:
        return
    top = recs[0]
    if not top.explanation:
        return
    dim_leader: Dict[str, float] = {}
    for r in recs[:6]:
        for s in r.scores:
            dim_leader[s.dimension] = max(dim_leader.get(s.dimension, 0), s.score)

    for alt in recs[1:5]:
        if not alt.explanation:
            continue
        losses: List[str] = []
        for s in alt.scores:
            lead = dim_leader.get(s.dimension, 0)
            if s.score < lead - 0.12 and s.dimension in (
                "route_realism",
                "overbuying_penalty",
                "operating_economics",
                "runway_performance",
                "winter_westbound_margin",
                "modern_operational_fit",
            ):
                if s.note:
                    losses.append(f"{s.dimension.replace('_', ' ')}: {s.note}")
                else:
                    from services.recommendation.fit_policy import fit_tier_for_dimension

                    losses.append(
                        f"{s.dimension.replace('_', ' ')}: {fit_tier_for_dimension(s.score)} vs "
                        f"{fit_tier_for_dimension(lead)} on the lead option."
                    )
        alt.explanation.why_alternatives_lost = losses[:3]
        if top.explanation and losses:
            top.explanation.why_alternatives_lost.append(
                f"vs {alt.model}: " + losses[0][:120]
            )
    top.explanation.why_alternatives_lost = top.explanation.why_alternatives_lost[:4]


def _eliminated_recommendation(
    model: str,
    feasibility: FeasibilityResult,
    profile: Dict[str, Any],
) -> AircraftRecommendation:
    """Infeasible aircraft — excluded from best-fit; surfaced only as avoided."""
    reasons = feasibility.elimination_reasons or ["Mission feasibility check failed."]
    expl = RecommendationExplanation(
        summary=f"{model} — Not Recommended (hard feasibility).",
        strengths=[],
        penalties=reasons[:4],
        operational_caveats=feasibility.notes[:3],
        why_it_fits=[],
        operational_compromises=reasons[:3],
        why_alternatives_lost=reasons[:2],
    )
    return AircraftRecommendation(
        model=model,
        category=str(profile.get("category") or ""),
        total_score=0.0,
        confidence=0.35,
        rank=0,
        avoid=True,
        fit="Not Recommended",
        scores=[],
        explanation=expl,
    )


def rank_missions(
    mission: MissionState,
    *,
    candidate_models: Optional[List[str]] = None,
    max_results: int = 6,
    mission_profile: Optional[MissionProfile] = None,
    override_experimental: bool = False,
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    query: str = "",
) -> Tuple[
    MissionCategory,
    List[AircraftRecommendation],
    Dict[str, FeasibilityResult],
    Optional[Any],
]:
    """
    Rank aircraft for mission-fit after hard feasibility elimination.

    Infeasible aircraft never occupy top recommendation slots unless ``override_experimental``.
    Returns ``(category, recommendations, feasibility_map, selection_audit)``.
    """
    from services.recommendation.diversity_controls import (
        RecommendationSelectionAudit,
        apply_diversity_controls,
        apply_hard_feasibility_gate,
    )

    profile = mission_profile or mission_state_to_profile(mission)
    models = candidate_models or list(_AIRCRAFT_PROFILES.keys())

    try:
        from services.aircraft_truth import filter_truth_verified_models

        models = filter_truth_verified_models(models)
    except Exception:
        pass

    from services.recommendation.aircraft_category_gating import (
        GatedMissionCategory,
        apply_mission_category_gating,
    )

    category_gate = apply_mission_category_gating(
        mission,
        models,
        mission_profile=profile,
    )
    category = category_gate.legacy_category
    if (
        mission.mountain_airport_requirement
        and category_gate.category == GatedMissionCategory.SUPER_MIDSIZE
    ):
        category = MissionCategory.MOUNTAIN_AIRPORT
    models = category_gate.candidate_models

    from services.recommendation.aircraft_category_gating import (
        category_exclusion_feasibility_results,
    )

    feasibility_map: Dict[str, FeasibilityResult] = category_exclusion_feasibility_results(
        category_gate
    )

    if isinstance(data_used, dict):
        data_used["mission_category_gate"] = category_gate.to_dict()

    max_leg = mission_max_leg_nm(mission)
    if mission.mountain_airport_requirement or category == MissionCategory.MOUNTAIN_AIRPORT:
        models = [m for m in models if m not in _MOUNTAIN_FORBIDDEN_MODELS]

    from services.consultant.recommendation_engine import apply_budget_gate

    models = apply_budget_gate(mission, list(models))

    # When the pipeline passes an explicit feasible shortlist, trust it — do not re-eliminate
    # every candidate on a second overbuy pass (e.g. Tokyo–Seoul killing all large-cabin survivors).
    if candidate_models is not None:
        feasible_list = list(models)
    else:
        in_band_feas = filter_feasible_aircraft(
            profile,
            models,
            override_experimental=override_experimental,
        )
        feasibility_map.update(in_band_feas)
        feasible_list = [
            m for m in models if in_band_feas.get(m) and in_band_feas[m].feasible
        ]
    if not feasible_list and models and candidate_models is not None:
        try:
            from services.recommendation.hard_mission_elimination import (
                detect_hard_elimination_context,
                hard_gate_allowlist,
            )

            hard_ctx = detect_hard_elimination_context(profile)
            allowlist = hard_gate_allowlist(profile) if hard_ctx else None
            if allowlist is not None and set(models).issubset(set(allowlist)):
                feasible_list = list(models)
                for model in feasible_list:
                    if feasibility_map.get(model) and feasibility_map[model].feasible:
                        continue
                    spec = _AIRCRAFT_PROFILES.get(model) or {}
                    feasibility_map[model] = FeasibilityResult(
                        feasible=True,
                        practical_range_nm=float(spec.get("practical_nm") or 0),
                        operational_risk_level="high",
                        notes=[
                            "Pipeline hard ULR gate — ranked with elevated operational risk on winter westbound legs."
                        ],
                        required_route_nm=hard_ctx.required_route_nm if hard_ctx else 0.0,
                    )
        except Exception:
            pass
    eliminated_list = [m for m in models if m in feasibility_map and not feasibility_map[m].feasible]

    required_peak = 0.0
    if mission.routes:
        pax = mission.passenger_count or 6
        for r in mission.routes:
            req, _, _ = required_mission_nm(mission, route_label=r, passenger_count=pax)
            required_peak = max(required_peak, req)

    hard_rejected: List[Any] = []
    feasible_list, hard_rejected = apply_hard_feasibility_gate(
        feasible_list,
        mission,
        profile,
        category,
        required_peak_nm=required_peak,
        skip_feasibility_engine=candidate_models is not None,
    )

    route_labels = list(mission.routes or [])
    operational_context = None
    try:
        from services.mission.route_distance_authority import (
            mission_route_blocks_ranking,
            peak_verified_stage_nm,
            resolve_mission_route_authority,
        )

        blocks_ranking, route_resolutions = mission_route_blocks_ranking(route_labels)
        verified_peak = peak_verified_stage_nm(route_resolutions)
        if verified_peak > 0:
            max_leg = verified_peak
            required_peak = max(required_peak, verified_peak)
        if isinstance(data_used, dict):
            data_used["route_distance_authority"] = [r.to_dict() for r in route_resolutions]
            data_used["route_blocks_ranking"] = blocks_ranking
        if blocks_ranking and route_labels:
            peak_catalog = peak_catalog_stage_nm(route_resolutions)
            if peak_catalog < 4000 and peak_verified_stage_nm(route_resolutions) < 4000:
                return category, [], feasibility_map, None

        from services.operational.mission_operational_assessment import (
            build_mission_operational_context,
        )

        operational_context = build_mission_operational_context(
            mission,
            profile,
            query=query,
            route_resolutions=route_resolutions,
        )
        if isinstance(data_used, dict):
            data_used["mission_operational_context"] = operational_context.to_dict()
    except Exception:
        route_resolutions = None
        operational_context = None

    model_categories = {
        m.lower(): str((_AIRCRAFT_PROFILES.get(m) or {}).get("category") or "")
        for m in feasible_list
    }
    try:
        from services.elimination.corridor_elimination import apply_corridor_hard_elimination

        corridor_result = apply_corridor_hard_elimination(
            feasible_list,
            profile,
            model_categories=model_categories,
            route_resolutions=route_resolutions,
        )
        for m in corridor_result.eliminated:
            feasibility_map[m] = FeasibilityResult(
                feasible=False,
                elimination_reasons=[corridor_result.reasons.get(m, "Corridor elimination")],
                operational_risk_level="eliminated",
            )
        feasible_list = corridor_result.survivors
        if isinstance(data_used, dict):
            data_used["corridor_hard_elimination"] = corridor_result.to_dict()
    except Exception:
        pass

    try:
        from services.airport.airport_operational_constraints import (
            apply_airport_constraint_elimination,
        )

        specs = {m: _AIRCRAFT_PROFILES.get(m) or {} for m in feasible_list}
        airport_result = apply_airport_constraint_elimination(
            feasible_list,
            route_labels=route_labels,
            model_specs=specs,
        )
        for m in airport_result.eliminated:
            feasibility_map[m] = FeasibilityResult(
                feasible=False,
                elimination_reasons=[airport_result.reasons.get(m, "Airport constraint elimination")],
                operational_risk_level="eliminated",
            )
        feasible_list = airport_result.survivors
        if isinstance(data_used, dict):
            data_used["airport_constraint_elimination"] = airport_result.to_dict()
    except Exception:
        pass

    try:
        from services.elimination.operational_band import (
            determine_operational_band,
            filter_models_to_operational_band,
        )

        band_result = determine_operational_band(
            profile,
            feasible_list,
            distance_nm=max_leg or required_peak,
            model_categories=model_categories,
        )
        from services.elimination.conditional_downgrade import (
            apply_conditional_elimination_map,
            feasibility_for_soft_elimination,
        )

        band_reasons = dict(band_result.elimination_reasons)
        for m in list(band_result.downgraded):
            reason = band_reasons.get(m, "Operational band compromise")
            feasibility_map[m] = feasibility_for_soft_elimination(
                m,
                reason,
                distance_nm=float(max_leg or required_peak or 0),
            )

        hard_elim, soft_elim, soft_labels = apply_conditional_elimination_map(
            feasibility_map,
            eliminated=list(band_result.eliminated),
            reasons=band_reasons,
            distance_nm=float(max_leg or required_peak or 0),
            elimination_kind="band",
            seasonal=bool(getattr(profile, "seasonal_note", None)),
        )
        band_result.eliminated = hard_elim
        for m in soft_elim:
            if m not in band_result.downgraded:
                band_result.downgraded.append(m)
            band_result.compromise_labels[m] = soft_labels.get(
                m, "VIABLE WITH COMPROMISES"
            )

        feasible_list = filter_models_to_operational_band(
            feasible_list, band_result, include_downgraded=True
        )
        if isinstance(data_used, dict):
            data_used["operational_band_elimination"] = band_result.to_dict()
    except Exception:
        pass

    if not feasible_list:
        return category, [], feasibility_map, None

    scored: List[AircraftRecommendation] = []
    for model in feasible_list:
        spec = _AIRCRAFT_PROFILES.get(model)
        if not spec:
            continue
        ra = None
        if mission.routes:
            ra = assess_mission_routes(
                mission,
                aircraft_practical_nm=float(spec["practical_nm"]),
                aircraft_brochure_nm=float(spec["brochure_nm"]),
                passenger_count=mission.passenger_count,
            )
        scored.append(
            score_aircraft_for_mission_ranked(
                model,
                spec,
                mission,
                mission_category=category,
                route_assessments=ra,
                operational_context=operational_context,
            )
        )

    aircraft_op_traces: List[Dict[str, Any]] = []
    if operational_context is not None:
        try:
            from services.operational.mission_operational_assessment import assess_aircraft_operational

            for rec in scored:
                spec = _AIRCRAFT_PROFILES.get(rec.model) or {}
                aircraft_op_traces.append(
                    assess_aircraft_operational(rec.model, spec, operational_context).to_dict()
                )
            if isinstance(data_used, dict):
                data_used["aircraft_operational_assessments"] = aircraft_op_traces
        except Exception:
            pass

    from services.recommendation.aircraft_archetype_weighting import modern_operational_fit_score

    def _rank_sort_key(rec: AircraftRecommendation) -> tuple:
        mod = modern_operational_fit_score(rec.model)
        for s in rec.scores:
            if s.dimension == "modern_operational_fit":
                mod = s.score
                break
        return (-rec.total_score, -mod)

    scored.sort(key=_rank_sort_key)

    ordered, selection_audit = apply_diversity_controls(
        scored,
        mission,
        mission_category=category,
        mission_profile=profile,
        max_results=max_results,
        conversation_state=conversation_state,
        data_used=data_used,
        hard_rejected=hard_rejected,
    )
    from services.recommendation.fit_policy import assign_fit_tiers

    assign_fit_tiers(ordered)
    for i, r in enumerate(ordered, start=1):
        r.rank = i
        r.avoid = False

    try:
        if route_resolutions:
            from services.broker.broker_verdicts import BrokerVerdict
            from services.mission.geodesic_policy import mission_forbids_primary_verdict

            if mission_forbids_primary_verdict(route_resolutions):
                primary = BrokerVerdict.PRIMARY_RECOMMENDATION.value
                viable_alt = BrokerVerdict.VIABLE_WITH_COMPROMISES.value
                for r in ordered:
                    if (r.fit_verdict or "") == primary:
                        r.fit_verdict = viable_alt
    except Exception:
        pass

    _attach_alternative_loss_reasons(ordered)
    return category, ordered, feasibility_map, selection_audit

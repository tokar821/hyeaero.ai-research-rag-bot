"""
Tier downgrade recovery — never return an empty shortlist after filtering.

When ULR / inappropriate aircraft are stripped, re-rank from lower tier bands:
super-midsize → midsize → light (economics-weighted fallback).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

_ULR_CATEGORIES = frozenset({"ultra-long", "ultra_long"})
_ULR_MODELS = frozenset(
    {
        "Global 7500",
        "Global 6500",
        "Global 8000",
        "Gulfstream G650",
        "Gulfstream G650ER",
        "Gulfstream G700",
        "Gulfstream G800",
        "Falcon 8X",
        "Falcon 10X",
    }
)

_TIER_LADDER: Tuple[Tuple[str, ...], ...] = (
    ("super-midsize", "midsize"),
    ("midsize",),
    ("light",),
)


def _models_for_categories(categories: Sequence[str]) -> List[str]:
    cats = {c.lower() for c in categories}
    return [
        model
        for model, spec in AIRCRAFT_PROFILES.items()
        if str(spec.get("category") or "").lower() in cats
    ]


def _exclude_ulr_models(models: List[str]) -> List[str]:
    return [m for m in models if m not in _ULR_MODELS]


def _economics_exclude_ulr(query: str, data_used: Optional[Dict[str, Any]]) -> bool:
    ql = (query or "").lower()
    if any(
        w in ql
        for w in (
            "economics",
            "operating cost",
            "prestige",
            "charter about",
            "costs are too high",
            "cost too high",
            "too high",
            "shuttle",
            "large long-range",
            "long-range jets",
        )
    ):
        return True
    if "mostly" in ql and any(
        w in ql for w in ("shuttle", "la–sf", "la-sf", "la–seattle", "la-seattle")
    ):
        return True
    if isinstance(data_used, dict):
        hw = data_used.get("hierarchy_weighting") or {}
        if isinstance(hw, dict) and hw.get("dominant_utilization"):
            dom = str(hw["dominant_utilization"]).lower()
            if any(w in dom for w in ("domestic", "corridor", "regional", "hawaii", "caribbean")):
                return True
    return False


def _profile_fallback_shortlist(
    mission: MissionState,
    *,
    max_results: int = 5,
    exclude_ulr: bool = True,
    data_used: Optional[Dict[str, Any]] = None,
) -> List[AircraftRecommendation]:
    """Deterministic economics-ranked fallback — no hallucinated models."""
    from services.recommendation.hack_v1_constraint_kernel import (
        filter_models_by_hack_v1,
        hack_v1_constraint_empty,
        hack_v1_permanent_exclusions,
    )

    if hack_v1_constraint_empty(data_used):
        return []

    from services.recommendation.mission_ranker import (
        MissionCategory,
        classify_mission_category,
        score_aircraft_for_mission_ranked,
    )

    category = classify_mission_category(mission)
    models = _models_for_categories(("super-midsize", "midsize", "light"))
    if exclude_ulr:
        models = _exclude_ulr_models(models)
    models = filter_models_by_hack_v1(models, data_used)
    if not models:
        return []

    scored: List[AircraftRecommendation] = []
    for model in models:
        spec = AIRCRAFT_PROFILES.get(model)
        if not spec:
            continue
        try:
            scored.append(
                score_aircraft_for_mission_ranked(
                    model,
                    spec,
                    mission,
                    mission_category=category,
                )
            )
        except Exception:
            from services.recommendation.weighted_aircraft_scoring import score_aircraft_weighted

            weighted = score_aircraft_weighted(
                model, spec, mission, mission_category=category
            )
            scored.append(
                AircraftRecommendation(
                    model=model,
                    category=str(spec.get("category") or ""),
                    total_score=weighted.total_score,
                    confidence=0.55,
                    rank=0,
                    fit="Partial Fit",
                    fit_verdict="VIABLE WITH COMPROMISES",
                )
            )

    scored.sort(key=lambda r: (-r.total_score, r.model))
    exclusions = hack_v1_permanent_exclusions(data_used)
    out: List[AircraftRecommendation] = []
    for i, rec in enumerate(scored, start=1):
        if rec.model in exclusions:
            continue
        rec.rank = len(out) + 1
        rec.avoid = False
        out.append(rec)
        if len(out) >= max_results:
            break
    return out


def tier_downgrade_recovery(
    mission: MissionState,
    query: str,
    *,
    prior_recommendations: Sequence[AircraftRecommendation],
    data_used: Optional[Dict[str, Any]] = None,
    max_results: int = 5,
    exclude_ulr: Optional[bool] = None,
) -> Tuple[List[AircraftRecommendation], str]:
    """
    Re-rank from lower aircraft tiers when filtering emptied the shortlist.

    Returns (recommendations, recovery_tier_label).
    """
    from services.recommendation.hack_v1_constraint_kernel import (
        hack_v1_constraint_empty,
        filter_models_by_hack_v1,
        hack_v1_permanent_exclusions,
    )

    try:
        from services.orchestration.orchestration_router_v2 import (
            orchestration_v2_blocks_tier_fallback,
        )

        if orchestration_v2_blocks_tier_fallback(data_used):
            if isinstance(data_used, dict):
                data_used["tier_downgrade_recovery"] = {
                    "tier": "blocked",
                    "source": "orchestration_v2",
                    "count": 0,
                }
            return [], "orchestration_v2_blocked"
    except Exception:
        pass

    if prior_recommendations:
        exclusions = hack_v1_permanent_exclusions(data_used) if data_used else frozenset()
        viable = [
            r
            for r in prior_recommendations
            if not r.avoid
            and r.model not in exclusions
            and r.model not in _ULR_MODELS
            and str(getattr(r, "category", "") or "").lower() not in _ULR_CATEGORIES
        ]
        if viable:
            return list(viable[:max_results]), "unchanged"

    if exclude_ulr is None:
        exclude_ulr = _economics_exclude_ulr(query, data_used)

    if hack_v1_constraint_empty(data_used):
        if isinstance(data_used, dict):
            data_used["tier_downgrade_recovery"] = {
                "tier": "blocked",
                "source": "hack_v1_empty",
                "count": 0,
            }
        return [], "hack_v1_empty"

    from services.recommendation.hack_v1_constraint_kernel import load_hack_v1_result
    from services.recommendation.mission_ranker import rank_missions

    # Hard ban: do not inject light jets as fallback hallucinations.
    banned_injection = {"Citation CJ2", "Citation CJ4", "Learjet 75"}

    hack_feasible: Optional[set[str]] = None
    hack_loaded = load_hack_v1_result(data_used)
    if hack_loaded is not None and hack_loaded.feasible_aircraft_list:
        hack_feasible = set(hack_loaded.feasible_aircraft_list)
        if exclude_ulr:
            hack_feasible = {m for m in hack_feasible if m not in _ULR_MODELS}
        if not hack_feasible:
            hack_feasible = None

    for tiers in _TIER_LADDER:
        models = _models_for_categories(tiers)
        if exclude_ulr:
            models = _exclude_ulr_models(models)
        models = filter_models_by_hack_v1(models, data_used)
        if hack_feasible is not None:
            models = [m for m in models if m in hack_feasible]
        if not models:
            continue

        _category, recs, _feas, _audit = rank_missions(
            mission,
            candidate_models=models,
            max_results=max_results,
            data_used=data_used,
            query=query,
        )
        viable = [r for r in recs if not r.avoid]
        viable = [r for r in viable if r.model not in banned_injection]
        if viable:
            if isinstance(data_used, dict):
                data_used["tier_downgrade_recovery"] = {
                    "tier": tiers[0],
                    "source": "rank_missions",
                    "count": len(viable),
                }
            return viable[:max_results], tiers[0]

    # Profile fallback eliminated: empty shortlist is not permission to hallucinate feasibility.
    if isinstance(data_used, dict):
        data_used["tier_downgrade_recovery"] = {
            "tier": "blocked",
            "source": "profile_fallback_eliminated",
            "count": 0,
        }
    return [], "profile_fallback_eliminated"


__all__ = ["tier_downgrade_recovery"]

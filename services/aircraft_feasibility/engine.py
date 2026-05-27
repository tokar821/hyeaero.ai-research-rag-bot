"""
Deterministic aircraft feasibility engine — runs BEFORE any LLM recommendation.

Hard-eliminates aircraft that cannot realistically perform the mission.
Never soft-penalizes impossible aircraft; they are rejected entirely.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Union

from services.aircraft_feasibility.hard_reject import apply_hard_reject_rules
from services.aircraft_feasibility.route_realism_validator import validate_route_realism
from services.aircraft_feasibility.mission_context import (
    FeasibilityMissionContext,
    mission_context_from_json,
    mission_context_from_profile,
    profile_from_context,
)
from services.aircraft_feasibility.payload_range import compute_payload_adjusted_range
from services.aircraft_feasibility.range_margin import compute_mission_range_requirement
from services.aircraft_feasibility.schema import AircraftFeasibilityVerdict
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.mission.models import MissionProfile

# nm equivalent penalty when runway requirement exceeds airport limit (hot/high takeoff roll)
_RUNWAY_NM_PENALTY_PER_500FT = 120.0


def _resolve_aircraft(
    aircraft: Union[str, Mapping[str, Any]],
) -> tuple[str, Dict[str, Any]] | tuple[None, None]:
    if isinstance(aircraft, str):
        spec = AIRCRAFT_PROFILES.get(aircraft)
        if not spec:
            return None, None
        return aircraft, {**spec, "model": aircraft}
    spec = dict(aircraft)
    model = str(spec.get("model") or "unknown")
    return model, spec


def _runway_penalty_nm(
    aircraft: Mapping[str, Any],
    mission: FeasibilityMissionContext,
) -> float:
    """Extra nm burden when runway footprint exceeds mission airport limit."""
    from services.aircraft_feasibility.hard_reject import _runway_limit_ft

    runway_ft = float(aircraft.get("runway_ft") or 9999)
    limit_ft = _runway_limit_ft(mission)
    if runway_ft <= limit_ft:
        return 0.0
    excess_ft = runway_ft - limit_ft
    return (excess_ft / 500.0) * _RUNWAY_NM_PENALTY_PER_500FT


def evaluate_aircraft_feasibility(
    mission: Union[Dict[str, Any], FeasibilityMissionContext, MissionProfile, Any],
    aircraft: Union[str, Mapping[str, Any]],
    *,
    performance_db: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> AircraftFeasibilityVerdict:
    """
    Hard feasibility evaluation for one aircraft against mission JSON or context.

    Parameters
    ----------
    mission:
        Mission extraction JSON, :class:`FeasibilityMissionContext`, or :class:`MissionProfile`.
    aircraft:
        Catalog model name or performance spec dict (must include ``practical_nm``).
    performance_db:
        Optional override for :data:`AIRCRAFT_PROFILES`.
    """
    if isinstance(mission, FeasibilityMissionContext):
        ctx = mission
    elif isinstance(mission, MissionProfile):
        ctx = mission_context_from_profile(mission)
    elif hasattr(mission, "model_dump"):
        ctx = mission_context_from_json(mission.model_dump(mode="json"))
    else:
        ctx = mission_context_from_json(mission if isinstance(mission, dict) else {})

    db = performance_db or AIRCRAFT_PROFILES
    model, spec = _resolve_aircraft(aircraft)
    if spec is None:
        if isinstance(aircraft, str) and aircraft in db:
            model, spec = aircraft, {**db[aircraft], "model": aircraft}
        else:
            return AircraftFeasibilityVerdict.rejected(
                f"Unknown aircraft model: {aircraft}",
                reserves_satisfied=True,
            )

    realism = validate_route_realism(ctx)
    if not realism.realistic and ctx.stage_distance_nm > 0:
        return AircraftFeasibilityVerdict.rejected(
            *realism.issues or ["Route failed realism validation."],
            stage_distance_nm=realism.stage_distance_nm,
        )

    requirement = compute_mission_range_requirement(ctx)
    adjusted = compute_payload_adjusted_range(spec, ctx)
    runway_pen = _runway_penalty_nm(spec, ctx)

    # Runway penalty reduces effective available range for margin check
    effective_available = max(adjusted.available_nm - runway_pen, 0.0)
    required = requirement.total_required_nm + runway_pen
    margin = effective_available - required if required > 0 else effective_available

    reasons = apply_hard_reject_rules(
        model=model,
        aircraft=spec,
        mission=ctx,
        requirement=requirement,
        adjusted=adjusted,
        runway_penalty_nm=runway_pen,
    )

    # Re-check margin with runway penalty applied
    if required > 0 and effective_available < required and not reasons:
        reasons.append(
            f"Effective available ~{int(effective_available)} nm < required ~{int(required)} nm "
            f"after runway/payload/winter deductions."
        )

    feasible = len(reasons) == 0

    return AircraftFeasibilityVerdict(
        feasible=feasible,
        rejection_reasons=reasons,
        payload_penalty=adjusted.payload_penalty_nm,
        runway_penalty=runway_pen,
        winter_penalty=adjusted.winter_penalty_nm,
        reserves_satisfied=requirement.reserves_satisfied,
        required_nm=required,
        available_nm=effective_available,
        margin_nm=margin,
        stage_distance_nm=ctx.stage_distance_nm,
    )


def filter_feasible_aircraft(
    mission: Union[Dict[str, Any], FeasibilityMissionContext, MissionProfile, Any],
    models: Optional[List[str]] = None,
    *,
    performance_db: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, AircraftFeasibilityVerdict]:
    """Evaluate all candidates; eliminated aircraft have ``feasible=False``."""
    db = performance_db or AIRCRAFT_PROFILES
    candidates = models or list(db.keys())
    return {
        model: evaluate_aircraft_feasibility(
            mission,
            model,
            performance_db=db,
        )
        for model in candidates
    }


def feasible_model_names(
    mission: Union[Dict[str, Any], FeasibilityMissionContext, MissionProfile, Any],
    models: Optional[List[str]] = None,
    *,
    performance_db: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[str]:
    """Models that pass hard feasibility."""
    results = filter_feasible_aircraft(mission, models, performance_db=performance_db)
    return [m for m, v in results.items() if v.feasible]

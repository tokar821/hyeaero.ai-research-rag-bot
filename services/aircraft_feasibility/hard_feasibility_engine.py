"""
Hard feasibility engine — practical range vs mission distance, NOT A FIT labeling.

For every aircraft evaluated:
  - Mission required nm uses NBAA IFR reserves, westbound/winter, payload, baggage, mountain.
  - Available nm uses catalog ``practical_nm`` only (never brochure).
  - If practical available < required → feasible=False → ``NOT A FIT``.

Integrates route realism validation and ultra-long corridor rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

from services.aircraft_feasibility.engine import evaluate_aircraft_feasibility, filter_feasible_aircraft
from services.aircraft_feasibility.mission_context import (
    FeasibilityMissionContext,
    mission_context_from_json,
    mission_context_from_profile,
)
from services.aircraft_feasibility.route_realism_validator import (
    RouteRealismResult,
    validate_route_realism,
)
from services.aircraft_feasibility.schema import AircraftFeasibilityVerdict
from services.mission.models import MissionProfile

VERDICT_NOT_A_FIT = "NOT A FIT"
VERDICT_GOOD_FIT = "GOOD FIT"
VERDICT_CONDITIONAL_FIT = "CONDITIONAL FIT"


@dataclass
class HardFeasibilityAssessment:
    """Per-aircraft hard feasibility with broker-style fit verdict."""

    model: str
    feasible: bool
    fit_verdict: str
    verdict: AircraftFeasibilityVerdict
    route_realism: Optional[RouteRealismResult] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "model": self.model,
            "feasible": self.feasible,
            "fitVerdict": self.fit_verdict,
            "verdict": self.verdict.to_dict(),
        }
        if self.route_realism is not None:
            out["routeRealism"] = self.route_realism.to_dict()
        return out


def _fit_verdict_from_verdict(verdict: AircraftFeasibilityVerdict) -> str:
    if not verdict.feasible:
        return VERDICT_NOT_A_FIT
    if verdict.margin_nm < 150 and verdict.required_nm > 0:
        return VERDICT_CONDITIONAL_FIT
    return VERDICT_GOOD_FIT


def validate_mission_route_realism(
    mission: Union[Dict[str, Any], FeasibilityMissionContext, MissionProfile, Any],
) -> RouteRealismResult:
    """Run route realism validator once per mission."""
    ctx = _to_context(mission)
    return validate_route_realism(ctx)


def assess_aircraft_hard_feasibility(
    mission: Union[Dict[str, Any], FeasibilityMissionContext, MissionProfile, Any],
    aircraft: Union[str, Mapping[str, Any]],
    *,
    route_realism: Optional[RouteRealismResult] = None,
) -> HardFeasibilityAssessment:
    """
    Hard feasibility for one aircraft — practical range vs mission distance.

    Never recommends when practical available < required (brochure not used).
    """
    ctx = _to_context(mission)
    realism = route_realism if route_realism is not None else validate_route_realism(ctx)
    model = aircraft if isinstance(aircraft, str) else str(aircraft.get("model") or "unknown")
    verdict = evaluate_aircraft_feasibility(ctx, aircraft)
    fit = _fit_verdict_from_verdict(verdict)
    return HardFeasibilityAssessment(
        model=model,
        feasible=verdict.feasible,
        fit_verdict=fit,
        verdict=verdict,
        route_realism=realism,
    )


def assess_all_aircraft_hard_feasibility(
    mission: Union[Dict[str, Any], FeasibilityMissionContext, MissionProfile, Any],
    models: Optional[List[str]] = None,
) -> Dict[str, HardFeasibilityAssessment]:
    """Evaluate full catalog or subset; infeasible aircraft marked NOT A FIT."""
    ctx = _to_context(mission)
    realism = validate_route_realism(ctx)
    raw = filter_feasible_aircraft(ctx, models)
    out: Dict[str, HardFeasibilityAssessment] = {}
    for model, verdict in raw.items():
        out[model] = HardFeasibilityAssessment(
            model=model,
            feasible=verdict.feasible,
            fit_verdict=_fit_verdict_from_verdict(verdict),
            verdict=verdict,
            route_realism=realism,
        )
    return out


def feasible_models_hard(
    mission: Union[Dict[str, Any], FeasibilityMissionContext, MissionProfile, Any],
    models: Optional[List[str]] = None,
) -> List[str]:
    """Models that pass hard feasibility (excludes NOT A FIT)."""
    return [m for m, a in assess_all_aircraft_hard_feasibility(mission, models).items() if a.feasible]


def _to_context(
    mission: Union[Dict[str, Any], FeasibilityMissionContext, MissionProfile, Any],
) -> FeasibilityMissionContext:
    if isinstance(mission, FeasibilityMissionContext):
        return mission
    if isinstance(mission, MissionProfile):
        return mission_context_from_profile(mission)
    return mission_context_from_json(mission if isinstance(mission, dict) else {})

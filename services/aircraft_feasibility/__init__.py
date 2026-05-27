"""
Deterministic aircraft feasibility engine — pre-LLM hard elimination.

Uses practical range (not brochure), NBAA IFR reserves, and operational penalties.
Infeasible aircraft are hard-rejected and labeled NOT A FIT — never ranked or recommended.

Public API:
  - :func:`evaluate_aircraft_feasibility`
  - :func:`assess_aircraft_hard_feasibility`
  - :func:`validate_mission_route_realism`
  - :class:`AircraftFeasibilityVerdict`
"""

from services.aircraft_feasibility.engine import (
    evaluate_aircraft_feasibility,
    feasible_model_names,
    filter_feasible_aircraft,
)
from services.aircraft_feasibility.hard_feasibility_engine import (
    VERDICT_NOT_A_FIT,
    HardFeasibilityAssessment,
    assess_aircraft_hard_feasibility,
    assess_all_aircraft_hard_feasibility,
    feasible_models_hard,
    validate_mission_route_realism,
)
from services.aircraft_feasibility.mission_context import (
    FeasibilityMissionContext,
    mission_context_from_json,
    mission_context_from_profile,
    profile_from_context,
)
from services.aircraft_feasibility.route_realism_validator import (
    RouteRealismResult,
    validate_route_realism,
)
from services.aircraft_feasibility.schema import AircraftFeasibilityVerdict

__all__ = [
    "AircraftFeasibilityVerdict",
    "FeasibilityMissionContext",
    "HardFeasibilityAssessment",
    "RouteRealismResult",
    "VERDICT_NOT_A_FIT",
    "assess_aircraft_hard_feasibility",
    "assess_all_aircraft_hard_feasibility",
    "evaluate_aircraft_feasibility",
    "feasible_models_hard",
    "filter_feasible_aircraft",
    "feasible_model_names",
    "mission_context_from_json",
    "mission_context_from_profile",
    "profile_from_context",
    "validate_mission_route_realism",
    "validate_route_realism",
]

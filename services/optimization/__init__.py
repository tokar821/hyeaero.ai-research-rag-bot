"""Multi-criteria decision optimization (Phase 23)."""

from services.optimization.multi_criteria_decision_engine import (
    AircraftOptimizationScore,
    DecisionProfile,
    OptimizationResult,
    attach_optimization_result_if_enabled,
    build_optimization_result,
    decision_optimization_enabled,
    infer_buyer_profile,
    optimize_aircraft_ranking,
)

__all__ = [
    "AircraftOptimizationScore",
    "DecisionProfile",
    "OptimizationResult",
    "attach_optimization_result_if_enabled",
    "build_optimization_result",
    "decision_optimization_enabled",
    "infer_buyer_profile",
    "optimize_aircraft_ranking",
]

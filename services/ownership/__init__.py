"""Aircraft lifecycle ownership intelligence (Phase 25)."""

from services.ownership.aircraft_lifecycle_ownership_engine import (
    OwnershipIntelligenceReport,
    attach_ownership_intelligence_if_enabled,
    build_ownership_intelligence,
    compare_ownership_profiles,
    estimate_depreciation_curve,
    estimate_future_resale_value,
    evaluate_ownership_risk,
    ownership_intelligence_enabled,
)

__all__ = [
    "OwnershipIntelligenceReport",
    "attach_ownership_intelligence_if_enabled",
    "build_ownership_intelligence",
    "compare_ownership_profiles",
    "estimate_depreciation_curve",
    "estimate_future_resale_value",
    "evaluate_ownership_risk",
    "ownership_intelligence_enabled",
]

"""Acquisition intelligence — liquidity, ownership risk, resale, maintenance, fleet age."""

from services.acquisition.acquisition_advisory import (
    AcquisitionIntelligenceBundle,
    build_acquisition_intelligence,
    format_acquisition_intelligence_block,
)
from services.acquisition.fleet_age_analysis import analyze_fleet_age
from services.acquisition.maintenance_event_estimator import estimate_maintenance_events
from services.acquisition.market_liquidity import assess_market_liquidity
from services.acquisition.ownership_risk import assess_ownership_risk
from services.acquisition.resale_strength import assess_resale_strength

__all__ = [
    "AcquisitionIntelligenceBundle",
    "analyze_fleet_age",
    "build_acquisition_intelligence",
    "format_acquisition_intelligence_block",
    "assess_market_liquidity",
    "assess_ownership_risk",
    "assess_resale_strength",
    "estimate_maintenance_events",
]

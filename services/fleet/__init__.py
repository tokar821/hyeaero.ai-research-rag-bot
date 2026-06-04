"""Fleet portfolio strategy intelligence (Phase 26)."""

from services.fleet.fleet_portfolio_strategy_engine import (
    FleetInput,
    FleetPortfolioStrategyReport,
    analyze_fleet_redundancy,
    attach_fleet_portfolio_strategy_if_enabled,
    build_fleet_portfolio_strategy,
    build_mission_coverage_map,
    fleet_portfolio_strategy_enabled,
    identify_fleet_gaps,
    optimize_fleet_structure,
    rank_aircraft_for_replacement,
)

__all__ = [
    "FleetInput",
    "FleetPortfolioStrategyReport",
    "analyze_fleet_redundancy",
    "attach_fleet_portfolio_strategy_if_enabled",
    "build_fleet_portfolio_strategy",
    "build_mission_coverage_map",
    "fleet_portfolio_strategy_enabled",
    "identify_fleet_gaps",
    "optimize_fleet_structure",
    "rank_aircraft_for_replacement",
]

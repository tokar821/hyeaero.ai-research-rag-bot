"""Multi-domain operational fleet composition."""

from services.fleet.fleet_composition import (
    FleetCompositionPlan,
    FleetRoleAssignment,
    MissionSegment,
    MissionSegmentRole,
    build_fleet_composition_plan,
    detect_multi_aircraft_mission,
    format_fleet_composition_block,
    merge_fleet_into_recommendations,
)
from services.fleet.fleet_domain_analysis import (
    MultiDomainAnalysis,
    OperationalDomain,
    SegmentationTrigger,
    analyze_multi_domain_operational_problem,
)
from services.fleet.fleet_invariant import (
    assert_fleet_invariants,
    enforce_fleet_elimination_invariant,
)

__all__ = [
    "FleetCompositionPlan",
    "FleetRoleAssignment",
    "MissionSegment",
    "MissionSegmentRole",
    "MultiDomainAnalysis",
    "OperationalDomain",
    "SegmentationTrigger",
    "analyze_multi_domain_operational_problem",
    "assert_fleet_invariants",
    "build_fleet_composition_plan",
    "detect_multi_aircraft_mission",
    "enforce_fleet_elimination_invariant",
    "format_fleet_composition_block",
    "merge_fleet_into_recommendations",
]

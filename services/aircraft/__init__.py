"""Aircraft knowledge authority — canonical specs and validation."""

from services.aircraft.aircraft_authority_service import (
    AircraftAuthorityRecord,
    build_authoritative_comparison_dataset,
    build_authoritative_market_context,
    get_aircraft_authority_record,
    get_authority_profile_dict,
    resolve_aircraft_alias,
    validate_aircraft_claim,
)

__all__ = [
    "AircraftAuthorityRecord",
    "build_authoritative_comparison_dataset",
    "build_authoritative_market_context",
    "get_aircraft_authority_record",
    "get_authority_profile_dict",
    "resolve_aircraft_alias",
    "validate_aircraft_claim",
]

"""Aircraft truth validation — verified specs only."""

from services.aircraft_truth.constants import (
    FORBIDDEN_UNVERIFIED_CLAIM_KEYS,
    UNVERIFIED_AIRCRAFT_MESSAGE,
)
from services.aircraft_truth.validator import (
    AircraftTruthResult,
    VerifiedAircraftFacts,
    extract_verified_facts,
    filter_truth_verified_models,
    format_verified_comparison_snippets,
    format_verified_spec_block,
    is_forbidden_unverified_claim,
    reject_forbidden_claims,
    resolve_aircraft_profile,
    unverified_response_for_model,
    validate_aircraft_truth,
)

__all__ = [
    "AircraftTruthResult",
    "FORBIDDEN_UNVERIFIED_CLAIM_KEYS",
    "UNVERIFIED_AIRCRAFT_MESSAGE",
    "VerifiedAircraftFacts",
    "extract_verified_facts",
    "filter_truth_verified_models",
    "format_verified_comparison_snippets",
    "format_verified_spec_block",
    "is_forbidden_unverified_claim",
    "reject_forbidden_claims",
    "resolve_aircraft_profile",
    "unverified_response_for_model",
    "validate_aircraft_truth",
]

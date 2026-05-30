"""Aircraft truth validator — user-facing and internal constants."""

from __future__ import annotations

from services.broker.broker_language import broker_refusal_message

UNVERIFIED_AIRCRAFT_MESSAGE = broker_refusal_message(context="aircraft_specs")

# Phase 5 unified intent — terse refusals without advisory pivot or mission follow-up.
UNIFIED_FACT_UNVERIFIED_MESSAGE = (
    "This specification is not available in verified catalog data."
)
UNIFIED_CATALOG_MISS_MESSAGE = "Aircraft not found in verified catalog."
UNIFIED_CAPABILITY_UNVERIFIED_MESSAGE = (
    "Cannot evaluate corridor feasibility without verified model data."
)
UNIFIED_COMPARISON_INSUFFICIENT_MESSAGE = (
    "Insufficient verified aircraft data to produce a comparison for those aircraft."
)

# Speculative claims blocked unless explicitly verified in catalog + supplement.
FORBIDDEN_UNVERIFIED_CLAIM_KEYS = frozenset(
    {
        "acquisition_price",
        "ask_price",
        "market_price",
        "typical_market_price",
        "list_price",
        "payload_capability",
        "payload_lb",
        "max_payload_lb",
        "runway_performance",
        "takeoff_distance_ft",
        "landing_distance_ft",
        "nonstop_capability",
        "can_do_nonstop",
        "nonstop_to",
    }
)

TRUTH_FIELD_MAX_PASSENGERS = "max_passengers"
TRUTH_FIELD_PRACTICAL_RANGE = "practical_range_nm"
TRUTH_FIELD_RUNWAY_CLASS = "runway_class"
TRUTH_FIELD_BAGGAGE_VOLUME = "baggage_volume_cu_ft"
TRUTH_FIELD_OPERATING_CATEGORY = "operating_category"

REQUIRED_TRUTH_FIELDS = (
    TRUTH_FIELD_MAX_PASSENGERS,
    TRUTH_FIELD_PRACTICAL_RANGE,
    TRUTH_FIELD_RUNWAY_CLASS,
    TRUTH_FIELD_BAGGAGE_VOLUME,
    TRUTH_FIELD_OPERATING_CATEGORY,
)

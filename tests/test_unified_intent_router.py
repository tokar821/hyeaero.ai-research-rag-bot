"""Unified intent router — Phase 5 Step 1 classification tests."""

from services.routing.unified_intent_router import (
    PROMOTION_CONTRACT_VERSION,
    UnifiedExecutionPath,
    UnifiedIntent,
    UnifiedIntentRoute,
    UnifiedSecondaryIntent,
    build_unified_intent_shadow,
    classify_unified_intent,
    get_secondary_intent_promotion_contract,
    validate_unified_intent_route_invariants,
)

_SHADOW_REQUIRED_KEYS = frozenset(
    {
        "qri_intent",
        "unified_intent_primary",
        "unified_intent_secondary",
        "model_resolved",
        "model_confidence",
        "field_detected",
        "intent_confidence",
    }
)


def test_falcon_8x_seats_is_aircraft_fact():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    assert route.intent == UnifiedIntent.AIRCRAFT_FACT
    assert route.field == "seats"
    assert route.model == "Falcon 8X"
    assert route.confidence >= 0.85
    assert route.model_confidence >= 0.7


def test_praetor_600_baggage_is_aircraft_fact():
    route = classify_unified_intent("What is the baggage capacity of a Praetor 600?")
    assert route.intent == UnifiedIntent.AIRCRAFT_FACT
    assert route.field == "baggage"
    assert route.model == "Praetor 600"
    assert route.model_confidence >= 0.7


def test_challenger_3500_worth_is_market_fact():
    route = classify_unified_intent("What is a Challenger 3500 worth?")
    assert route.intent == UnifiedIntent.AIRCRAFT_MARKET_FACT
    assert route.field in ("worth", "value", "price")
    assert route.model == "Challenger 350"
    assert route.model_confidence >= 0.7


def test_longitude_sfo_paris_is_not_fact_path():
    route = classify_unified_intent("Can Longitude fly SFO to Paris?")
    assert route.intent == UnifiedIntent.OTHER
    assert route.secondary_intent == UnifiedSecondaryIntent.AIRCRAFT_CAPABILITY_LIKELY


def test_mixed_seats_and_route_stays_other():
    route = classify_unified_intent("How many seats + can it fly NYC Paris?")
    assert route.intent == UnifiedIntent.OTHER
    assert route.secondary_intent == UnifiedSecondaryIntent.AIRCRAFT_CAPABILITY_LIKELY


def test_ambiguous_longitude_seats_falls_back_to_other():
    route = classify_unified_intent("Longitude seats?")
    assert route.intent == UnifiedIntent.OTHER
    assert route.field == "seats"
    assert route.model is None
    assert route.model_confidence < 0.7
    assert "model_confidence_below_threshold" in route.signals


def test_model_confidence_threshold_blocks_fact_routing():
    route = classify_unified_intent("Longitude seats?")
    shadow = build_unified_intent_shadow(route, "payload_range_analysis", enforce_fact=True)
    assert shadow["unified_intent_primary"] == "other"
    assert shadow["model_resolved"] is None
    assert shadow["would_enforce_fact_path"] is False


def test_comparison_query_shadow_secondary():
    route = classify_unified_intent("Compare Longitude vs Legacy 600")
    assert route.intent == UnifiedIntent.OTHER
    assert route.secondary_intent == UnifiedSecondaryIntent.AIRCRAFT_COMPARISON_LIKELY


def test_shadow_metadata_completeness():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    shadow = build_unified_intent_shadow(route, "payload_range_analysis")
    assert _SHADOW_REQUIRED_KEYS.issubset(shadow.keys())
    assert shadow["qri_intent"] == "payload_range_analysis"
    assert shadow["unified_intent_primary"] == "aircraft_fact"
    assert shadow["field_detected"] == "seats"
    assert shadow["model_resolved"] == "Falcon 8X"
    assert shadow["model_confidence"] >= 0.7
    assert shadow["intent_confidence"] >= 0.85


def test_promotion_contract_capability_query():
    route = classify_unified_intent("Can Longitude fly SFO to Paris?")
    contract = get_secondary_intent_promotion_contract(route)
    assert contract["contract_version"] == PROMOTION_CONTRACT_VERSION
    assert contract["capability_promotable"] is True
    assert contract["comparison_promotable"] is False
    assert contract["mission_promotable"] is False
    assert contract["execution_target"] == "capability_responder"
    assert contract["primary_intent_frozen"] == "other"


def test_promotion_contract_no_secondary():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    contract = get_secondary_intent_promotion_contract(route)
    assert contract["capability_promotable"] is False
    assert contract["comparison_promotable"] is False
    assert contract["mission_promotable"] is False
    assert contract["execution_target"] is None


def test_route_invariants_valid_fact_route():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    result = validate_unified_intent_route_invariants(route)
    assert result["valid"] is True
    assert result["violations"] == []


def test_route_invariants_invalid_fact_without_model():
    route = UnifiedIntentRoute(
        intent=UnifiedIntent.AIRCRAFT_FACT,
        model=None,
        field="seats",
        confidence=0.9,
        model_confidence=0.0,
        execution_path=UnifiedExecutionPath.NONE,
    )
    result = validate_unified_intent_route_invariants(route)
    assert result["valid"] is False
    assert "fact_or_market_requires_resolved_model" in result["violations"]


def test_citation_longitude_range_routes_to_fact():
    route = classify_unified_intent("What is the range of a Citation Longitude?")
    assert route.intent == UnifiedIntent.AIRCRAFT_FACT
    assert route.execution_path == UnifiedExecutionPath.AIRCRAFT_FACT
    assert route.model == "Citation Longitude"


def test_g550_market_value_routes_to_market():
    route = classify_unified_intent("What is the market value of a Gulfstream G550?")
    assert route.intent == UnifiedIntent.AIRCRAFT_MARKET_FACT
    assert route.execution_path == UnifiedExecutionPath.AIRCRAFT_MARKET_FACT


def test_citation_latitude_sell_for_routes_to_market():
    route = classify_unified_intent("What does a Citation Latitude sell for?")
    assert route.intent == UnifiedIntent.AIRCRAFT_MARKET_FACT
    assert route.field == "price"


def test_g650_capability_not_comparison_secondary():
    route = classify_unified_intent("Can a Gulfstream G650 fly SFO to Tokyo?")
    assert route.execution_path == UnifiedExecutionPath.CAPABILITY
    assert route.secondary_intent == UnifiedSecondaryIntent.AIRCRAFT_CAPABILITY_LIKELY


def test_phenom_alternative_consider_instead():
    route = classify_unified_intent("What aircraft should I consider instead of a Phenom 300?")
    assert route.execution_path == UnifiedExecutionPath.ALTERNATIVE


def test_legacy_500_comparison_insufficient_path():
    route = classify_unified_intent("Challenger 3500 vs Legacy 500 — which is more efficient?")
    assert route.execution_path == UnifiedExecutionPath.COMPARISON

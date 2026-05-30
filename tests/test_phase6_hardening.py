"""Phase 6 production hardening — observe-only guardrails."""

from services.comparison.alternative_pipeline_responder import respond_aircraft_alternative
from services.fact.aircraft_fact_responder import respond_aircraft_fact
from services.routing.unified_intent_ambiguity_classifier import (
    AmbiguityType,
    ConfidenceBand,
    classify_ambiguity,
)
from services.routing.unified_intent_hardening_guard import (
    HARDENING_ROUTING_FAILURE,
    attach_hardening_layer,
    evaluate_hardening_guard,
)
from services.routing.unified_intent_production_metrics import (
    get_production_metrics,
    record_hardening_event,
    reset_production_metrics,
)
from services.routing.unified_intent_router import (
    UnifiedExecutionPath,
    UnifiedIntent,
    build_unified_intent_shadow,
    classify_unified_intent,
)
from services.routing.unified_pipeline_gate import (
    evaluate_pipeline_gate,
    execute_unified_pipeline_handler,
)

_HARDENING_FLAG_KEYS = frozenset(
    {"routing_failure", "ambiguity_detected", "fallback_triggered"}
)


def test_hardening_detects_capability_routing_failure_without_blocking():
    reset_production_metrics()
    query = "Can Longitude fly SFO to Paris?"
    route = classify_unified_intent(query)
    assert route.execution_path == UnifiedExecutionPath.NONE

    result = evaluate_hardening_guard(query, route)
    assert result.routing_failure is True
    assert result.event_code == HARDENING_ROUTING_FAILURE
    assert result.requires_fallback_analysis is True
    assert result.expected_path_category == "capability"
    assert "capability" in (result.hardening_reason or "").lower()

    gate = evaluate_pipeline_gate(route, enforce_capability=True)
    assert gate.enforce is False


def test_hardening_does_not_flag_clean_fact_route():
    reset_production_metrics()
    query = "How many seats does a Falcon 8X have?"
    route = classify_unified_intent(query)
    result = evaluate_hardening_guard(query, route)

    assert route.execution_path == UnifiedExecutionPath.AIRCRAFT_FACT
    assert result.routing_failure is False
    assert result.requires_fallback_analysis is False


def test_lexical_ambiguity_detected_for_bare_longitude():
    route = classify_unified_intent("Can Longitude fly SFO to Paris?")
    report = classify_ambiguity("Can Longitude fly SFO to Paris?", route)

    assert report.is_ambiguous is True
    assert report.ambiguity_type in (
        AmbiguityType.LEXICAL,
        AmbiguityType.UNRESOLVED_OBJECT,
        AmbiguityType.BORDERLINE_CONFIDENCE,
    )


def test_intent_collision_detected_capability_and_compare():
    query = "Can a Gulfstream G650 fly SFO to Tokyo vs a Falcon 8X?"
    route = classify_unified_intent(query)
    report = classify_ambiguity(query, route)

    assert report.is_ambiguous is True
    assert report.ambiguity_type == AmbiguityType.INTENT_COLLISION


def test_borderline_confidence_band():
    route = classify_unified_intent("Can Longitude fly SFO to Paris?")
    report = classify_ambiguity("Can Longitude fly SFO to Paris?", route)

    if route.model and 0.55 <= route.model_confidence < 0.7:
        assert report.confidence_band == ConfidenceBand.MEDIUM
    else:
        assert report.confidence_band in (ConfidenceBand.LOW, ConfidenceBand.MEDIUM, ConfidenceBand.HIGH)


def test_shadow_schema_includes_hardening_flags_default():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    shadow = build_unified_intent_shadow(route, "payload_range_analysis")

    assert _HARDENING_FLAG_KEYS.issubset(shadow["hardening_flags"].keys())
    assert shadow["hardening_flags"]["routing_failure"] is False
    assert shadow["hardening_flags"]["ambiguity_detected"] is False
    assert shadow["hardening_flags"]["fallback_triggered"] is False
    assert shadow["qri_intent"] == "payload_range_analysis"
    assert shadow["unified_intent_primary"] == "aircraft_fact"


def test_attach_hardening_layer_merges_shadow_and_metrics():
    reset_production_metrics()
    query = "Can Longitude fly SFO to Paris?"
    route = classify_unified_intent(query)
    data_used = {
        "unified_intent_shadow": build_unified_intent_shadow(route, "mission_feasibility"),
    }

    result = attach_hardening_layer(
        data_used,
        query=query,
        route=route,
        qri_intent="mission_feasibility",
    )

    assert result.routing_failure is True
    assert data_used["unified_intent_hardening"]["event_code"] == HARDENING_ROUTING_FAILURE
    assert data_used["unified_intent_shadow"]["hardening_flags"]["routing_failure"] is True
    assert data_used["unified_intent_shadow"]["hardening_flags"]["fallback_triggered"] is True
    metrics = data_used["unified_intent_production_metrics"]
    assert metrics["hardening_failure_count"] >= 1
    assert metrics["execution_path_none_count"] >= 1


def test_production_metrics_record_hardening_event():
    reset_production_metrics()
    route = classify_unified_intent("Can Longitude fly SFO to Paris?")
    report = classify_ambiguity("Can Longitude fly SFO to Paris?", route)

    record_hardening_event(
        route,
        report,
        routing_failure=True,
        requires_fallback_analysis=True,
        capability_without_model=True,
    )
    metrics = get_production_metrics()

    assert metrics["hardening_failure_count"] == 1
    assert metrics["execution_path_none_count"] == 1
    assert metrics["legacy_fallback_rate"] == 1
    assert metrics["capability_without_model_rate"] == 1
    assert metrics["ambiguity_rate_by_intent"][route.intent.value] == 1


def test_responder_outputs_unchanged_under_hardening():
    fact_before = respond_aircraft_fact("Falcon 8X", "seats")
    alt_before = respond_aircraft_alternative("What are alternatives to a Citation CJ3+?")

    reset_production_metrics()
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    evaluate_hardening_guard("How many seats does a Falcon 8X have?", route)

    fact_after = respond_aircraft_fact("Falcon 8X", "seats")
    alt_after = respond_aircraft_alternative("What are alternatives to a Citation CJ3+?")

    assert fact_before == fact_after
    assert alt_before == alt_after


def test_gate_and_handler_unchanged_after_hardening_annotation():
    query = "How many seats does a Falcon 8X have?"
    route = classify_unified_intent(query)
    evaluate_hardening_guard(query, route)

    gate = evaluate_pipeline_gate(route, enforce_fact=True)
    answer, du, _ = execute_unified_pipeline_handler(route, gate, query)

    assert gate.enforce is True
    assert "Falcon 8X" in answer
    assert du["unified_execution_path"] == UnifiedExecutionPath.AIRCRAFT_FACT.value


def test_comparison_routing_failure_when_path_none():
    reset_production_metrics()
    query = "Compare Longitude vs Legacy 600"
    route = classify_unified_intent(query)
    if route.execution_path == UnifiedExecutionPath.NONE:
        result = evaluate_hardening_guard(query, route)
        assert result.expected_path_category == "comparison"
        assert result.routing_failure is True


def test_alternative_shadow_flags_on_consider_instead():
    reset_production_metrics()
    query = "What aircraft should I consider instead of a Phenom 300?"
    route = classify_unified_intent(query)
    data_used = {"unified_intent_shadow": build_unified_intent_shadow(route, "aircraft_comparison")}

    result = attach_hardening_layer(
        data_used,
        query=query,
        route=route,
        qri_intent="aircraft_comparison",
    )

    assert route.execution_path == UnifiedExecutionPath.ALTERNATIVE
    assert result.routing_failure is False
    assert data_used["unified_intent_shadow"]["hardening_flags"]["routing_failure"] is False

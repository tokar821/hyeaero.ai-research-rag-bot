"""Phase 5 production safety — regression and drift control tests."""

from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.unified_intent_drift_monitor import detect_intent_drift
from services.routing.unified_intent_flag_validator import validate_flag_consistency
from services.routing.unified_intent_router import UnifiedExecutionPath, classify_unified_intent
from services.routing.unified_pipeline_gate import evaluate_pipeline_gate
from services.telemetry.unified_intent_telemetry import (
    build_shadow_normalized,
    build_unified_telemetry_event,
    evaluate_drift_alerts,
    record_unified_intent_telemetry,
    reset_telemetry_counters,
)


def test_execution_path_matches_gate_path_fact():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    gate = evaluate_pipeline_gate(route, enforce_fact=True)
    assert route.execution_path == gate.execution_path
    assert route.execution_path == UnifiedExecutionPath.AIRCRAFT_FACT


def test_execution_path_matches_gate_path_capability():
    route = classify_unified_intent("Can a Falcon 8X fly nonstop from New York to London?")
    gate = evaluate_pipeline_gate(route, enforce_capability=True)
    assert route.execution_path == gate.execution_path == UnifiedExecutionPath.CAPABILITY


def test_execution_path_matches_gate_path_comparison():
    q = "Compare Challenger 650 vs Praetor 600"
    route = classify_unified_intent(q)
    gate = evaluate_pipeline_gate(route, enforce_comparison=True)
    assert route.execution_path == gate.execution_path == UnifiedExecutionPath.COMPARISON


def test_fact_query_not_mission_execution_path():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    assert route.execution_path == UnifiedExecutionPath.AIRCRAFT_FACT


def test_capability_not_comparison_path():
    route = classify_unified_intent("Can a Falcon 8X fly nonstop from New York to London?")
    assert route.execution_path != UnifiedExecutionPath.COMPARISON


def test_comparison_not_none_for_explicit_compare():
    route = classify_unified_intent("Compare Challenger 650 vs Praetor 600")
    assert route.execution_path == UnifiedExecutionPath.COMPARISON


def test_flag_consistency_valid_fact_enforcement():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    result = validate_flag_consistency(route, enforce_fact=True)
    assert result.valid is True


def test_flag_consistency_invalid_fact_with_capability_flag():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    result = validate_flag_consistency(route, enforce_capability=True)
    assert result.valid is False
    assert result.invalid_combinations


def test_flag_consistency_invalid_comparison_with_fact_flag():
    route = classify_unified_intent("Compare Challenger 650 vs Praetor 600")
    result = validate_flag_consistency(route, enforce_fact=True)
    assert result.valid is False


def test_drift_monitor_router_gate_aligned():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    gate = evaluate_pipeline_gate(route, enforce_fact=True)
    drift = detect_intent_drift(route, qri_intent="payload_range_analysis", gate_execution_path=gate.execution_path.value)
    assert "router_vs_gate" not in drift["mismatch_type"]


def test_shadow_normalized_payload_shape():
    normalized = build_shadow_normalized(
        router_execution_path="aircraft_fact",
        gate_execution_path="aircraft_fact",
        qri_intent="payload_range_analysis",
        alignment_status="aligned",
        divergence_reason=None,
    )
    assert normalized["alignment_status"] == "aligned"
    assert normalized["router_execution_path"] == normalized["gate_execution_path"]


def test_drift_alert_pipeline_runs():
    reset_telemetry_counters()
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    qri = classify_query_recommendation_intent("How many seats does a Falcon 8X have?")
    drift = detect_intent_drift(route, qri_intent=qri.intent.value)
    record_unified_intent_telemetry(
        build_unified_telemetry_event(
            query="How many seats does a Falcon 8X have?",
            route_model=route.model,
            execution_path=route.execution_path.value,
            gate_path=route.execution_path.value,
            qri_intent=qri.intent.value,
            router_intent=route.intent.value,
            shadow_mode=True,
            drift_detected=bool(drift.get("is_mismatch")),
            latency_ms=1.0,
            drift_event=drift,
        )
    )
    alerts = evaluate_drift_alerts()
    assert isinstance(alerts, list)

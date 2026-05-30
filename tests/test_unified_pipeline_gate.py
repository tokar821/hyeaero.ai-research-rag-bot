"""Unified pipeline gate — authority lock tests."""

from services.routing.unified_intent_execution import (
    should_enforce_alternative_path,
    should_enforce_capability_path,
    should_enforce_comparison_path,
    should_enforce_fact_path,
)
from services.routing.unified_intent_router import UnifiedExecutionPath, classify_unified_intent
from services.routing.unified_pipeline_gate import evaluate_pipeline_gate


def test_execution_path_on_fact_query():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    assert route.execution_path == UnifiedExecutionPath.AIRCRAFT_FACT
    assert should_enforce_fact_path(route) is True


def test_execution_path_on_capability_query():
    route = classify_unified_intent("Can a Falcon 8X fly nonstop from New York to London?")
    assert route.execution_path == UnifiedExecutionPath.CAPABILITY
    assert should_enforce_capability_path(route) is True


def test_execution_path_on_comparison_query():
    q = "Compare Challenger 650 vs Praetor 600"
    route = classify_unified_intent(q)
    assert route.execution_path == UnifiedExecutionPath.COMPARISON
    assert should_enforce_comparison_path(route) is True


def test_execution_path_on_alternative_query():
    q = "What are credible alternatives to a Gulfstream G650?"
    route = classify_unified_intent(q)
    assert route.execution_path == UnifiedExecutionPath.ALTERNATIVE
    assert should_enforce_alternative_path(route) is True


def test_pipeline_gate_requires_flag_and_authority():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    off = evaluate_pipeline_gate(route, enforce_fact=False)
    on = evaluate_pipeline_gate(route, enforce_fact=True)
    assert off.execution_path == UnifiedExecutionPath.AIRCRAFT_FACT
    assert off.enforce is False
    assert on.enforce is True


def test_pipeline_gate_does_not_reinterpret_query():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    decision = evaluate_pipeline_gate(route, enforce_comparison=True)
    assert decision.enforce is False

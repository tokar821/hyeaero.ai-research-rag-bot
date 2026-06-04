# Phase 31 CI Gate Runner (backend/)
param(
    [ValidateSet("smoke", "merge", "nightly", "stress")]
    [string]$Gate = "merge"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot/..

switch ($Gate) {
    "smoke" {
        pytest -m smoke tests/test_semantic_intent_lock_engine.py tests/test_phase15_fail_closed.py tests/test_deterministic_execution_guard.py tests/test_authority_dispatch.py tests/test_intent_conflict_resolution.py tests/test_intent_execution_trace.py -q
    }
    "merge" {
        pytest -m merge_gate tests/test_semantic_intent_lock_engine.py tests/test_phase15_fail_closed.py tests/test_deterministic_execution_guard.py tests/test_authority_dispatch.py tests/test_intent_conflict_resolution.py tests/test_intent_execution_trace.py tests/test_aircraft_authority_service.py tests/integration/test_intent_lock_guard_integration.py tests/routing/test_budget_constraint_matrix.py tests/akal/test_akal_truth_matrix.py tests/test_execution_replay_engine.py tests/fail_closed/test_fail_closed_matrix.py tests/replay/test_replay_determinism_v2.py -q
    }
    "nightly" {
        pytest -m deterministic tests/test_semantic_intent_lock_engine.py tests/test_phase15_fail_closed.py tests/test_deterministic_execution_guard.py tests/test_authority_dispatch.py tests/test_intent_conflict_resolution.py tests/test_intent_execution_trace.py tests/test_aircraft_authority_service.py tests/test_consultant_evaluator.py tests/test_multi_criteria_decision_engine.py tests/test_fleet_portfolio_strategy_engine.py tests/test_aircraft_market_intelligence_engine.py tests/test_executive_intelligence_synthesis_engine.py tests/test_aircraft_lifecycle_ownership_engine.py tests/test_unified_intent_router.py tests/test_execution_replay_engine.py tests/e2e tests/integration/test_intent_lock_guard_integration.py tests/routing tests/advisory/test_advisory_routing_isolation.py tests/fail_closed/test_fail_closed_matrix.py tests/akal/test_akal_truth_matrix.py tests/replay -q
    }
    "stress" {
        pytest -m stress tests/routing/test_adversarial_routing.py tests/e2e/test_consultant_retrieval_intent_lock_e2e.py tests/replay/test_pipeline_reproducibility.py tests/replay/test_replay_determinism_v2.py -q
    }
}

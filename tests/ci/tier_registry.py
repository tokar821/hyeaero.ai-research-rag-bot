"""
Phase 31 — Deterministic test tier registry.

Tier hierarchy:
  smoke ⊂ merge_gate ⊂ deterministic
  stress is orthogonal (may overlap deterministic)

Tier 0 — smoke:        critical path, target <30s
Tier 1 — merge_gate:   pre-merge CI, target <2min
Tier 2 — deterministic: full nightly suite (478 tests)
Tier 3 — stress:       100x replay, adversarial, heavy E2E
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

# ---------------------------------------------------------------------------
# Tier 0 — Smoke (critical routing smoke)
# ---------------------------------------------------------------------------
SMOKE_FILES: Set[str] = {
    "test_semantic_intent_lock_engine.py",
    "test_phase15_fail_closed.py",
    "test_deterministic_execution_guard.py",
    "test_authority_dispatch.py",
    "test_intent_conflict_resolution.py",
    "test_intent_execution_trace.py",
}

# ---------------------------------------------------------------------------
# Tier 1 — Merge gate (unit + dispatch-level; no E2E retrieval)
# ---------------------------------------------------------------------------
MERGE_GATE_FILES: Set[str] = SMOKE_FILES | {
    "test_aircraft_authority_service.py",
    "test_intent_lock_guard_integration.py",
    "test_budget_constraint_matrix.py",
    "test_akal_truth_matrix.py",
    "test_execution_replay_engine.py",
    "test_fail_closed_matrix.py",
}

# Fast replay/hash tests included in merge gate (module-level, no retrieval bundle).
MERGE_GATE_TEST_NAMES: Set[str] = {
    "test_intent_lock_identical_across_runs",
    "test_whitespace_normalization_stable_hash",
    "test_deterministic_evaluation_id_module_level",
    "test_execution_trace_v2_module_level_deterministic",
    "test_origin_query_hash_deterministic",
    "test_lock_timestamp_deterministic_not_wall_clock",
}

# ---------------------------------------------------------------------------
# Tier 3 — Stress (100x loops, adversarial matrices, heavy E2E)
# ---------------------------------------------------------------------------
STRESS_FILES: Set[str] = {
    "test_adversarial_routing.py",
    "test_consultant_retrieval_intent_lock_e2e.py",
    "test_pipeline_reproducibility.py",
}

STRESS_TEST_NAME_FRAGMENTS: tuple[str, ...] = (
    "100x",
)

# Retrieval-based replay tests — tier 2 deterministic only (excluded from merge gate).
MERGE_GATE_EXCLUDED_TEST_NAMES: Set[str] = {
    "test_dispatch_authority_id_stable",
    "test_execution_trace_v2_trace_id_stable",
    "test_evaluation_id_stable",
    "test_final_output_hash_stable_when_answer_identical",
    "test_evaluator_attach_idempotent",
}

# ---------------------------------------------------------------------------
# Tier 2 — Full deterministic (advisory + router + E2E guard leaks, etc.)
# ---------------------------------------------------------------------------
DETERMINISTIC_FILES: Set[str] = MERGE_GATE_FILES | STRESS_FILES | {
    "test_consultant_evaluator.py",
    "test_multi_criteria_decision_engine.py",
    "test_fleet_portfolio_strategy_engine.py",
    "test_aircraft_market_intelligence_engine.py",
    "test_executive_intelligence_synthesis_engine.py",
    "test_aircraft_lifecycle_ownership_engine.py",
    "test_unified_intent_router.py",
    "test_advisory_routing_isolation.py",
    "test_conversation_guard_leaks.py",
    "test_replay_determinism_v2.py",
    "test_production_validation.py",
}

# Phase 32 production validation — nightly only (500-query audit is slow).
PRODUCTION_VALIDATION_EXCLUDED_FROM_MERGE: Set[str] = {
    "test_production_validation.py",
}


def _basename(path: str) -> str:
    return Path(path).name


def classify_item(nodeid: str) -> dict[str, bool]:
    """Return tier flags for a pytest item nodeid."""
    file_part = nodeid.split("::")[0]
    name = file_part.replace("\\", "/").split("/")[-1]
    test_name = nodeid.split("::")[-1] if "::" in nodeid else ""

    is_deterministic = name in DETERMINISTIC_FILES
    is_stress = name in STRESS_FILES or any(f in test_name for f in STRESS_TEST_NAME_FRAGMENTS)
    is_smoke = name in SMOKE_FILES
    is_merge = name in MERGE_GATE_FILES and not is_stress
    if name == "test_replay_determinism_v2.py":
        if test_name in MERGE_GATE_TEST_NAMES:
            is_merge = True
        elif test_name in MERGE_GATE_EXCLUDED_TEST_NAMES:
            is_merge = False
        elif any(f in test_name for f in STRESS_TEST_NAME_FRAGMENTS):
            is_merge = False
            is_stress = True
        else:
            is_merge = False
    if test_name in MERGE_GATE_EXCLUDED_TEST_NAMES:
        is_merge = False
    if name in PRODUCTION_VALIDATION_EXCLUDED_FROM_MERGE:
        is_merge = False

    return {
        "smoke": is_smoke,
        "merge_gate": is_merge,
        "deterministic": is_deterministic,
        "stress": is_stress,
    }


def apply_tier_markers(items: Iterable) -> None:
    """Apply pytest markers based on tier registry."""
    import pytest

    for item in items:
        flags = classify_item(item.nodeid)
        if flags["deterministic"]:
            item.add_marker(pytest.mark.deterministic)
        if flags["smoke"]:
            item.add_marker(pytest.mark.smoke)
        if flags["merge_gate"]:
            item.add_marker(pytest.mark.merge_gate)
        if flags["stress"]:
            item.add_marker(pytest.mark.stress)


# Inventory for documentation / reports.
TIER_INVENTORY: list[dict[str, str]] = [
    {"file": "tests/test_semantic_intent_lock_engine.py", "tier": "0,1,2", "notes": "IntentLock core"},
    {"file": "tests/test_phase15_fail_closed.py", "tier": "0,1,2", "notes": "Phase 15 fail-closed"},
    {"file": "tests/test_deterministic_execution_guard.py", "tier": "0,1,2", "notes": "Guard bypass"},
    {"file": "tests/test_authority_dispatch.py", "tier": "0,1,2", "notes": "Authority dispatch"},
    {"file": "tests/test_intent_conflict_resolution.py", "tier": "0,1,2", "notes": "ICRL"},
    {"file": "tests/test_intent_execution_trace.py", "tier": "0,1,2", "notes": "Execution trace"},
    {"file": "tests/test_aircraft_authority_service.py", "tier": "1,2", "notes": "AKAL legacy"},
    {"file": "tests/integration/test_intent_lock_guard_integration.py", "tier": "1,2", "notes": "IntentLock guard"},
    {"file": "tests/routing/test_budget_constraint_matrix.py", "tier": "1,2", "notes": "Budget matrix"},
    {"file": "tests/akal/test_akal_truth_matrix.py", "tier": "1,2", "notes": "AKAL truth matrix"},
    {"file": "tests/test_execution_replay_engine.py", "tier": "1,2", "notes": "Replay engine"},
    {"file": "tests/fail_closed/test_fail_closed_matrix.py", "tier": "1,2", "notes": "Fail-closed (dispatch-level)"},
    {"file": "tests/replay/test_replay_determinism_v2.py", "tier": "1,2,3", "notes": "Hash tests=1; retrieval=2; 100x=3"},
    {"file": "tests/test_consultant_evaluator.py", "tier": "2", "notes": "Evaluation"},
    {"file": "tests/test_multi_criteria_decision_engine.py", "tier": "2", "notes": "Optimization advisory"},
    {"file": "tests/test_fleet_portfolio_strategy_engine.py", "tier": "2", "notes": "Fleet advisory"},
    {"file": "tests/test_aircraft_market_intelligence_engine.py", "tier": "2", "notes": "Market advisory"},
    {"file": "tests/test_executive_intelligence_synthesis_engine.py", "tier": "2", "notes": "Synthesis advisory"},
    {"file": "tests/test_aircraft_lifecycle_ownership_engine.py", "tier": "2", "notes": "Ownership advisory"},
    {"file": "tests/test_unified_intent_router.py", "tier": "2", "notes": "Unified router"},
    {"file": "tests/advisory/test_advisory_routing_isolation.py", "tier": "2", "notes": "Advisory isolation"},
    {"file": "tests/e2e/test_conversation_guard_leaks.py", "tier": "2", "notes": "Guard leak E2E"},
    {"file": "tests/routing/test_adversarial_routing.py", "tier": "3", "notes": "Adversarial matrix"},
    {"file": "tests/e2e/test_consultant_retrieval_intent_lock_e2e.py", "tier": "3", "notes": "Full retrieval E2E"},
    {"file": "tests/e2e/test_broker_certification_suite.py", "tier": "3", "notes": "Phase 48 broker certification"},
    {"file": "tests/e2e/test_broker_certification_v2.py", "tier": "3", "notes": "Phase 50 broker certification V2"},
    {"file": "tests/e2e/retrieval_accuracy_suite.py", "tier": "3", "notes": "Phase 51 retrieval benchmark"},
    {"file": "tests/e2e/recommendation_accuracy_suite.py", "tier": "3", "notes": "Phase 51 recommendation benchmark"},
    {"file": "tests/e2e/listing_realism_suite.py", "tier": "3", "notes": "Phase 51 listing realism benchmark"},
    {"file": "tests/test_broker_audit_phase51.py", "tier": "1,2", "notes": "Phase 51 audit unit tests"},
    {"file": "tests/test_alias_expansion_engine.py", "tier": "1,2", "notes": "Phase 53 alias expansion"},
    {"file": "tests/e2e/real_aircraft_benchmark.py", "tier": "3", "notes": "Phase 53 real aircraft benchmark"},
    {"file": "tests/e2e/listing_validation_suite.py", "tier": "3", "notes": "Phase 53 listing validation"},
    {"file": "tests/e2e/tail_investigation_suite.py", "tier": "3", "notes": "Phase 53 tail accuracy"},
    {"file": "tests/e2e/market_recommendation_audit.py", "tier": "3", "notes": "Phase 53 recommendation bias audit"},
    {"file": "tests/e2e/production_query_replay_suite.py", "tier": "3", "notes": "Phase 53 production replay"},
    {"file": "tests/replay/test_pipeline_reproducibility.py", "tier": "3", "notes": "Full pipeline reproduction"},
    {"file": "tests/production_validation/test_production_validation.py", "tier": "2", "notes": "Phase 32 broker QA (nightly)"},
]

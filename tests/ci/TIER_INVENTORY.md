# Phase 31 — Test Tier Inventory

## Tier Hierarchy

```
Tier 0  smoke        ⊂  Tier 1  merge_gate  ⊂  Tier 2  deterministic
Tier 3  stress       (orthogonal — heavy E2E, 100× replay, adversarial)
```

Markers are applied automatically via `tests/ci/tier_registry.py` + `tests/conftest.py`.

---

## Classification Table

| Test File | Tier 0 | Tier 1 | Tier 2 | Tier 3 | Notes |
|-----------|:------:|:------:|:------:|:------:|-------|
| `test_semantic_intent_lock_engine.py` | ✓ | ✓ | ✓ | | IntentLock core (8) |
| `test_phase15_fail_closed.py` | ✓ | ✓ | ✓ | | Fail-closed guard (3) |
| `test_deterministic_execution_guard.py` | ✓ | ✓ | ✓ | | Phase 15 guard (11) |
| `test_authority_dispatch.py` | ✓ | ✓ | ✓ | | Authority dispatch (8) |
| `test_intent_conflict_resolution.py` | ✓ | ✓ | ✓ | | ICRL (9) |
| `test_intent_execution_trace.py` | ✓ | ✓ | ✓ | | Trace v1 (6) |
| `test_aircraft_authority_service.py` | | ✓ | ✓ | | AKAL legacy (13) |
| `integration/test_intent_lock_guard_integration.py` | | ✓ | ✓ | | IntentLock guard (14) |
| `routing/test_budget_constraint_matrix.py` | | ✓ | ✓ | | Budget matrix (51) |
| `akal/test_akal_truth_matrix.py` | | ✓ | ✓ | | AKAL truth (88) |
| `test_execution_replay_engine.py` | | ✓ | ✓ | | Replay engine (8) |
| `fail_closed/test_fail_closed_matrix.py` | | ✓ | ✓ | | Fail-closed matrix (19) |
| `replay/test_replay_determinism_v2.py` | | partial | ✓ | partial | 6 hash=merge; 22 retrieval=2; 3×100=stress |
| `test_consultant_evaluator.py` | | | ✓ | | Evaluation (11) |
| `test_multi_criteria_decision_engine.py` | | | ✓ | | Optimization (14) |
| `test_fleet_portfolio_strategy_engine.py` | | | ✓ | | Fleet (12) |
| `test_aircraft_market_intelligence_engine.py` | | | ✓ | | Market (15) |
| `test_executive_intelligence_synthesis_engine.py` | | | ✓ | | Synthesis (8) |
| `test_aircraft_lifecycle_ownership_engine.py` | | | ✓ | | Ownership (10) |
| `test_unified_intent_router.py` | | | ✓ | | Unified router (19) |
| `advisory/test_advisory_routing_isolation.py` | | | ✓ | | Advisory isolation (15) |
| `e2e/test_conversation_guard_leaks.py` | | | ✓ | | Guard E2E (2) |
| `routing/test_adversarial_routing.py` | | | ✓ | ✓ | Adversarial (28) |
| `e2e/test_consultant_retrieval_intent_lock_e2e.py` | | | ✓ | ✓ | Full E2E (23) |
| `replay/test_pipeline_reproducibility.py` | | | ✓ | ✓ | Pipeline repro (41) |

---

## Gate Commands

| Gate | Config | Command |
|------|--------|---------|
| Tier 0 Smoke | `tests/ci/smoke_gate.ini` | `pytest -m smoke -q` |
| Tier 1 Merge | `tests/ci/merge_gate.ini` | `pytest -m merge_gate -q` |
| Tier 2 Nightly | `tests/ci/nightly_gate.ini` | `pytest -m deterministic -q` |
| Tier 3 Stress | `tests/ci/stress_gate.ini` | `pytest -m stress -q` |

---

## Coverage Estimate

| Tier | Tests | Runtime (est.) | Coverage Focus |
|------|-------|----------------|----------------|
| 0 Smoke | 45 | **~27s** | Routing spine: lock → dispatch → guard → ICRL |
| 1 Merge | 242 | **~80s** | + AKAL, budget, guard integration, hash unit tests |
| 2 Nightly | 478 | ~8–10min | + advisory, E2E, adversarial, pipeline, retrieval replay |
| 3 Stress | 95 | ~10–15min | 100× loops, adversarial, full E2E matrices |

---

## Marker Rules

- `smoke` ⊂ `merge_gate` ⊂ `deterministic`
- `stress` is separate; stress tests may also carry `deterministic`
- Retrieval-based replay tests excluded from `merge_gate`
- `@pytest.mark.stress` on 100× replay tests (replaces deprecated `slow`)
